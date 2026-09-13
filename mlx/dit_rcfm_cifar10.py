import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import math
    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx.data.datasets import load_cifar10

    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Rectified Conditional Flow Matching with a Diffusion Transformer on CIFAR-10 (MLX)

    ## Research Goal

    Train a **class-conditional generative model** on **CIFAR-10** with a
    **Diffusion Transformer (DiT)** backbone and *compare three training
    regimes* under an identical model, loss family, and Euler ODE sampler:

    1. **Vanilla Conditional Flow Matching (CFM)** — the linear
       interpolation path `x_t = (1-t) x_0 + t x_1` with `x_0 ~ N(0, I)`
       sampled independently of `x_1`, and MSE loss on the constant
       target velocity `v = x_1 - x_0`.
    2. **CFM with Minibatch Optimal-Transport (OT) coupling** — for
       every minibatch, solve a balanced assignment between a fresh
       Gaussian batch `x_0` and the real batch `x_1` (squared-Euclidean
       cost, exact solver `scipy.optimize.linear_sum_assignment`) and
       reorder `x_0` to be optimally coupled with `x_1` before applying
       the same CFM loss.
    3. **Rectified Flow with Reflow** — train a first "gen1" model with
       vanilla CFM, then generate a synthetic dataset of `(x_0, x_1)`
       pairs by running the gen1 ODE map on fresh Gaussian noise, and
       train a **second** model on those already-coupled pairs with the
       x0 held fixed to its paired noise (the *reflow* step). This is
       expected to yield **straighter** trajectories at inference time.

    We then quantitatively and visually demonstrate that reflow's
    trajectories are indeed straighter than gen1's (via a curvature /
    step-count MSE metric) and compare the three regimes side-by-side.

    ### Notebook Outline

    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation
    4. Model definition (shared DiT + flow-matching utilities)
    5. Training — 5a Vanilla CFM, 5b OT-CFM, 5c Rectified Flow + Reflow
    6. Optional hyperparameter search (vanilla CFM only)
    7. Validation & 5-fold cross-validation (vanilla CFM only)
    8. Final verification & comparison
       - 8a. ODE generation progression
       - 8b. Straightness proof for reflow
       - 8c. Three-way comparison (loss curves, samples, table)
    9. Save trained models
    10. Load an existing trained model
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration
    """)
    return


@app.function
def cifar10_class_names() -> list:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


@app.function
def load_cifar10_arrays(
    root: str = "../data/cifar10",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    train_buf = load_cifar10(root=root, train=True)
    test_buf = load_cifar10(root=root, train=False)
    train_all = train_buf.batch(len(train_buf))[0]
    test_all = test_buf.batch(len(test_buf))[0]
    x_train = np.asarray(train_all["image"]).astype(np.float32) / 255.0
    y_train = np.asarray(train_all["label"]).astype(np.int32)
    x_test = np.asarray(test_all["image"]).astype(np.float32) / 255.0
    y_test = np.asarray(test_all["label"]).astype(np.int32)
    return x_train, y_train, x_test, y_test, cifar10_class_names()


@app.cell
def _():
    x_train_np, y_train_np, x_test_np, y_test_np, class_names = load_cifar10_arrays("../data/cifar10")
    return class_names, x_test_np, x_train_np, y_test_np, y_train_np


@app.cell
def _(mo, x_test_np, x_train_np):
    mo.md(f"""
    ### Dataset overview

    CIFAR-10 is loaded via `mlx.data.datasets.load_cifar10` from
    `../data/cifar10/` (each split's Buffer is collapsed into a single
    full batch, then materialised to NumPy). Each image is a `float32`
    array shaped `(32, 32, 3)` in `[0, 1]`; labels are `int32` in `[0, 9]`.

    | Split | Size | Shape |
    |-------|------|-------|
    | Train (raw) | {x_train_np.shape[0]:,} | {tuple(x_train_np.shape[1:])} |
    | Test | {x_test_np.shape[0]:,} | {tuple(x_test_np.shape[1:])} |
    """)
    return


@app.function
def plot_sample_grid(images: np.ndarray, labels: np.ndarray, class_names: list, n_show: int = 40, cols: int = 8):
    rows = int(math.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        if i < n_show:
            ax.imshow(np.clip(images[i], 0.0, 1.0))
            ax.set_title(class_names[int(labels[i])], fontsize=8)
        ax.axis("off")
    fig.suptitle("CIFAR-10 sample images", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, x_train_np, y_train_np):
    plot_sample_grid(x_train_np, y_train_np, class_names, n_show=40)
    return


@app.function
def plot_class_distribution(labels: np.ndarray, class_names: list):
    counts = np.bincount(labels.astype(np.int64), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(class_names)), counts, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 class distribution (train)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, y_train_np):
    plot_class_distribution(y_train_np, class_names)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def make_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    val_fraction: float = 0.1,
):
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    n_total = x_train_norm.shape[0]
    n_val = int(n_total * val_fraction)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    x_tr = mx.array(x_train_norm[tr_idx])
    y_tr = mx.array(y_train[tr_idx].astype(np.int32))
    x_val = mx.array(x_train_norm[val_idx])
    y_val = mx.array(y_train[val_idx].astype(np.int32))
    x_te = mx.array(x_test_norm)
    y_te = mx.array(y_test.astype(np.int32))
    return x_tr, y_tr, x_val, y_val, x_te, y_te


@app.cell
def _(x_test_np, x_train_np, y_test_np, y_train_np):
    x_tr, y_tr, x_val, y_val, x_te, y_te = make_datasets(
        x_train_np, y_train_np, x_test_np, y_test_np, val_fraction=0.1
    )
    return x_te, x_tr, x_val, y_te, y_tr, y_val


@app.function
def make_batches(x: mx.array, y: mx.array, batch_size: int = 128, shuffle: bool = True) -> list:
    n = x.shape[0]
    if shuffle:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_idx = mx.array(idx[start:end].astype(np.int32))
        batches.append((x[b_idx], y[b_idx]))
    return batches


@app.function
def make_reflow_batches(x0: mx.array, x1: mx.array, y: mx.array, batch_size: int = 128, shuffle: bool = True) -> list:
    n = x0.shape[0]
    if shuffle:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_idx = mx.array(idx[start:end].astype(np.int32))
        batches.append((x0[b_idx], x1[b_idx], y[b_idx]))
    return batches


@app.cell
def _(mo, x_tr, y_tr):
    _sample_batches = make_batches(x_tr, y_tr, batch_size=128, shuffle=True)
    _xb, _yb = _sample_batches[0]
    mo.md(
        f"""
        ### One-batch check

        - Number of training batches (bs=128): **{len(_sample_batches):,}**
        - `x_batch.shape` = `{tuple(_xb.shape)}`, dtype = `{_xb.dtype}`
        - `y_batch.shape` = `{tuple(_yb.shape)}`, dtype = `{_yb.dtype}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition (shared DiT backbone + flow-matching utilities)
    """)
    return


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / max(half, 1))

    def __call__(self, t: mx.array) -> mx.array:
        return mx.concatenate(
            [mx.sin(t[:, None] * self.freqs[None, :]), mx.cos(t[:, None] * self.freqs[None, :])],
            axis=-1,
        )


@app.class_definition
class AdaptiveLayerNormV1(nn.Module):
    def __init__(self, dim: int = 256, cond_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        scale, shift = mx.split(self.proj(nn.silu(cond))[:, None, :], 2, axis=-1)
        return self.norm(x) * (1.0 + scale) + shift


@app.class_definition
class PatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x).reshape(x.shape[0], -1, self.proj.weight.shape[0])


@app.class_definition
class DiTBlockV1(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, cond_dim: int = 256):
        super().__init__()
        self.attn_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.attn = nn.MultiHeadAttention(dim, num_heads)
        self.mlp_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        h = self.attn_norm(x, cond)
        x = x + self.attn(h, h, h)
        return x + self.mlp(self.mlp_norm(x, cond))


@app.class_definition
class UnpatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 256, out_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.grid = image_size // patch_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        b = x.shape[0]
        p, c, g = self.patch_size, self.out_channels, self.grid
        h = self.proj(x).reshape(b, g, g, p, p, c)
        return h.transpose(0, 1, 3, 2, 4, 5).reshape(b, g * p, g * p, c)


@app.class_definition
class DiffusionTransformerV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2
        self.patchify = PatchifyV1(patch_size, embed_dim, in_channels)
        self.pos_embed = mx.zeros((1, num_patches, embed_dim))
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.blocks = [DiTBlockV1(embed_dim, num_heads, mlp_dim, embed_dim) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatchify = UnpatchifyV1(patch_size, embed_dim, in_channels, image_size)

    def __call__(self, x: mx.array, t: mx.array, y: mx.array) -> mx.array:
        h = self.patchify(x) + self.pos_embed
        cond = self.time_embed(t) + self.class_embed(y)
        for block in self.blocks:
            h = block(h, cond)
        return self.unpatchify(self.final_norm(h))


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def sample_noise_like(x1: mx.array) -> mx.array:
    return mx.random.normal(shape=x1.shape)


@app.function
def compute_flow_loss(model: nn.Module, x1: mx.array, x0: mx.array, y: mx.array) -> mx.array:
    t = mx.random.uniform(shape=(x1.shape[0],))
    t_view = t.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = model(x_t, t, y)
    return mx.mean((v_pred - v_target) ** 2)


@app.function
def minibatch_ot_pairing(x0: mx.array, x1: mx.array) -> tuple:
    x0_np = np.array(x0).reshape(x0.shape[0], -1).astype(np.float64)
    x1_np = np.array(x1).reshape(x1.shape[0], -1).astype(np.float64)
    diffs = x0_np[:, None, :] - x1_np[None, :, :]
    cost = np.sum(diffs * diffs, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.argsort(col_ind)
    x0_reordered = x0[mx.array(perm.astype(np.int32))]
    total_cost = float(cost[row_ind, col_ind].sum())
    return x0_reordered, total_cost


@app.function
def euler_solve(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = model(x, t, y)
        x = x + dt * v
        mx.eval(x)
    return x


@app.function
def euler_solve_trajectory(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> list:
    dt = 1.0 / num_steps
    x = x0
    trajectory = [x]
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = model(x, t, y)
        x = x + dt * v
        mx.eval(x)
        trajectory.append(x)
    return trajectory


@app.function
def compute_path_straightness(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> float:
    trajectory = euler_solve_trajectory(model, x0, y, num_steps)
    x_final = trajectory[-1]
    total_disp = x_final - x0
    avg_step_disp = total_disp / num_steps
    total_dev = 0.0
    for i in range(num_steps):
        step_disp = trajectory[i + 1] - trajectory[i]
        dev = step_disp - avg_step_disp
        total_dev += float(mx.mean(dev * dev).item())
    return total_dev / num_steps


@app.function
def evaluate_model(model: nn.Module, batches: list) -> float:
    total = 0.0
    n = 0
    for xb, yb in batches:
        x0 = sample_noise_like(xb)
        loss = compute_flow_loss(model, xb, x0, yb)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.function
def run_train_epoch_cfm(model: nn.Module, optimizer, batches: list) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    epoch_loss = 0.0
    n = 0
    for xb, yb in batches:
        x0 = sample_noise_like(xb)
        loss, grads = loss_and_grad_fn(model, xb, x0, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def run_train_epoch_ot_cfm(model: nn.Module, optimizer, batches: list) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    epoch_loss = 0.0
    n = 0
    for xb, yb in batches:
        x0 = sample_noise_like(xb)
        x0_paired, _ = minibatch_ot_pairing(x0, xb)
        loss, grads = loss_and_grad_fn(model, xb, x0_paired, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def run_train_epoch_reflow(model: nn.Module, optimizer, batches: list) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    epoch_loss = 0.0
    n = 0
    for x0b, x1b, yb in batches:
        loss, grads = loss_and_grad_fn(model, x1b, x0b, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def build_dit_model(num_layers: int = 6) -> DiffusionTransformerV1:
    model = DiffusionTransformerV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=num_layers,
        dropout=0.0,
    )
    mx.eval(model.parameters())
    return model


@app.function
def train_dit_regime(
    x_tr: mx.array,
    y_tr: mx.array,
    x_val: mx.array,
    y_val: mx.array,
    num_layers: int,
    lr: float,
    wd: float,
    batch_size: int,
    epochs: int,
    epoch_fn,
    progress_cb=None,
) -> tuple:
    model = build_dit_model(num_layers=num_layers)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=wd)
    val_batches = make_batches(x_val, y_val, batch_size=batch_size, shuffle=False)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_batches = make_batches(x_tr, y_tr, batch_size=batch_size, shuffle=True)
        tl = epoch_fn(model, optimizer, train_batches)
        vl = evaluate_model(model, val_batches)
        train_losses.append(tl)
        val_losses.append(vl)
        if progress_cb is not None:
            progress_cb(epoch, epochs, tl, vl)
    return model, train_losses, val_losses


@app.function
def build_reflow_dataset(
    model_gen1: nn.Module,
    num_samples: int,
    num_steps: int,
    y_labels_pool: mx.array,
    batch_size: int = 128,
) -> tuple:
    x0_chunks = []
    x1_chunks = []
    y_chunks = []
    total = 0
    n_pool = y_labels_pool.shape[0]
    while total < num_samples:
        cur = min(batch_size, num_samples - total)
        x0 = mx.random.normal(shape=(cur, 32, 32, 3))
        idx = np.random.randint(0, n_pool, size=cur).astype(np.int32)
        y = y_labels_pool[mx.array(idx)]
        x1 = euler_solve(model_gen1, x0, y, num_steps)
        mx.eval(x1)
        x0_chunks.append(x0)
        x1_chunks.append(x1)
        y_chunks.append(y)
        total += cur
    return mx.concatenate(x0_chunks, axis=0), mx.concatenate(x1_chunks, axis=0), mx.concatenate(y_chunks, axis=0)


@app.function
def train_reflow_stage(
    x0_rf: mx.array,
    x1_rf: mx.array,
    y_rf: mx.array,
    x_val: mx.array,
    y_val: mx.array,
    num_layers: int,
    lr: float,
    wd: float,
    batch_size: int,
    epochs: int,
    progress_cb=None,
) -> tuple:
    model = build_dit_model(num_layers=num_layers)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=wd)
    val_batches = make_batches(x_val, y_val, batch_size=batch_size, shuffle=False)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        batches = make_reflow_batches(x0_rf, x1_rf, y_rf, batch_size=batch_size, shuffle=True)
        tl = run_train_epoch_reflow(model, optimizer, batches)
        vl = evaluate_model(model, val_batches)
        train_losses.append(tl)
        val_losses.append(vl)
        if progress_cb is not None:
            progress_cb(epoch, epochs, tl, vl)
    return model, train_losses, val_losses


@app.cell
def _(mo):
    mo.md("""
    ### Model Architecture — `DiffusionTransformerV1`

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Patch embedding | `PatchifyV1` (Conv2d, stride=patch) | `(B, N, D)` where `N = (H/p)^2` |
    | Positional embedding | learnable table `(1, N, D)` | `(B, N, D)` |
    | Time embedding | `SinusoidalTimestepEmbeddingV1` + MLP | `(B, D)` |
    | Class embedding | `nn.Embedding(C+1, D)` (spare slot reserved) | `(B, D)` |
    | Backbone | `DiTBlockV1 x num_layers` (AdaLN attn + AdaLN MLP) | `(B, N, D)` |
    | Head | `LayerNorm` + `UnpatchifyV1` | `(B, H, W, C)` |

    Defaults: `image_size=32, patch_size=4, embed_dim=256, num_heads=8,
    mlp_dim=512, num_layers=6`.

    ### Path-straightness metric

    `compute_path_straightness(model, x0, y, num_steps)` runs the Euler
    ODE solver and records the trajectory. For a *perfectly straight*
    path, every per-step displacement equals the average
    `(x_final - x_0) / num_steps`. We report

    `S = (1 / num_steps) * sum_i mean((x_{i+1} - x_i) - (x_final - x_0)/num_steps)^2`

    which is **zero for a perfectly straight trajectory** and grows with
    curvature. Lower is better. Rectified flow's reflow step is
    expected to reduce this metric compared to the initial gen1 model.
    """)
    return


@app.cell
def _():
    default_model = build_dit_model(num_layers=6)
    default_param_count = count_parameters(default_model)
    return (default_param_count,)


@app.cell
def _(default_param_count, mo):
    mo.md(f"""
    **Default model parameter count**: `{default_param_count:,}`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training (three regimes, each independently gated)

    Every regime uses the same `DiffusionTransformerV1` architecture and
    the same linear-path flow-matching MSE loss (`compute_flow_loss`).
    The only difference is how `x_0` is generated for each training
    pair:

    - **5a Vanilla CFM** — `x_0` is a fresh Gaussian sample per step
      (independent of `x_1`).
    - **5b OT-CFM** — `x_0` is fresh Gaussian noise **reordered** by
      exact minibatch OT so it is optimally coupled with `x_1`.
    - **5c Rectified Flow with Reflow** — Stage 1 trains a "gen1" model
      exactly like 5a; Stage 2 builds a synthetic pair dataset by
      running the gen1 Euler ODE on fresh noise, then trains a fresh
      model on those *fixed* pairs.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 5a — Vanilla Conditional Flow Matching
    """)
    return


@app.cell
def _(mo):
    lr_vanilla_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="LR (vanilla)",
    )
    bs_vanilla_ui = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch size (vanilla)",
    )
    wd_vanilla_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight decay (vanilla)",
    )
    epochs_vanilla_ui = mo.ui.slider(1, 60, value=15, step=1, label="Epochs (vanilla)")
    num_layers_vanilla_ui = mo.ui.slider(1, 12, value=6, step=1, label="DiT layers (vanilla)")
    train_btn_vanilla = mo.ui.run_button(label="Train Vanilla CFM")
    mo.vstack(
        [
            mo.hstack([lr_vanilla_ui, bs_vanilla_ui, wd_vanilla_ui]),
            mo.hstack([epochs_vanilla_ui, num_layers_vanilla_ui]),
            train_btn_vanilla,
        ]
    )
    return (
        bs_vanilla_ui,
        epochs_vanilla_ui,
        lr_vanilla_ui,
        num_layers_vanilla_ui,
        train_btn_vanilla,
        wd_vanilla_ui,
    )


@app.cell
def _(
    bs_vanilla_ui,
    epochs_vanilla_ui,
    lr_vanilla_ui,
    mo,
    num_layers_vanilla_ui,
    train_btn_vanilla,
    wd_vanilla_ui,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses_vanilla = []
    val_losses_vanilla = []
    trained_model_vanilla = None
    if not train_btn_vanilla.value:
        mo.output.replace(mo.md("Click **Train Vanilla CFM** to begin training."))
    else:
        def _cb_vanilla(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**Vanilla CFM Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model_vanilla, train_losses_vanilla, val_losses_vanilla = train_dit_regime(
            x_tr,
            y_tr,
            x_val,
            y_val,
            num_layers=num_layers_vanilla_ui.value,
            lr=lr_vanilla_ui.value,
            wd=wd_vanilla_ui.value,
            batch_size=bs_vanilla_ui.value,
            epochs=epochs_vanilla_ui.value,
            epoch_fn=run_train_epoch_cfm,
            progress_cb=_cb_vanilla,
        )
        mo.output.replace(
            mo.md(
                f"**Vanilla CFM training complete!** Final train "
                f"`{train_losses_vanilla[-1]:.4f}` | val `{val_losses_vanilla[-1]:.4f}`"
            )
        )
    return train_losses_vanilla, trained_model_vanilla, val_losses_vanilla


@app.cell
def _(mo):
    mo.md("""
    ### 5b — CFM with Minibatch Optimal-Transport Coupling
    """)
    return


@app.cell
def _(mo):
    lr_ot_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="LR (OT)",
    )
    bs_ot_ui = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch size (OT)",
    )
    wd_ot_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight decay (OT)",
    )
    epochs_ot_ui = mo.ui.slider(1, 60, value=15, step=1, label="Epochs (OT)")
    num_layers_ot_ui = mo.ui.slider(1, 12, value=6, step=1, label="DiT layers (OT)")
    train_btn_ot = mo.ui.run_button(label="Train OT-CFM")
    mo.vstack(
        [
            mo.hstack([lr_ot_ui, bs_ot_ui, wd_ot_ui]),
            mo.hstack([epochs_ot_ui, num_layers_ot_ui]),
            train_btn_ot,
        ]
    )
    return (
        bs_ot_ui,
        epochs_ot_ui,
        lr_ot_ui,
        num_layers_ot_ui,
        train_btn_ot,
        wd_ot_ui,
    )


@app.cell
def _(
    bs_ot_ui,
    epochs_ot_ui,
    lr_ot_ui,
    mo,
    num_layers_ot_ui,
    train_btn_ot,
    wd_ot_ui,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses_ot = []
    val_losses_ot = []
    trained_model_ot = None
    if not train_btn_ot.value:
        mo.output.replace(mo.md("Click **Train OT-CFM** to begin training."))
    else:
        def _cb_ot(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**OT-CFM Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model_ot, train_losses_ot, val_losses_ot = train_dit_regime(
            x_tr,
            y_tr,
            x_val,
            y_val,
            num_layers=num_layers_ot_ui.value,
            lr=lr_ot_ui.value,
            wd=wd_ot_ui.value,
            batch_size=bs_ot_ui.value,
            epochs=epochs_ot_ui.value,
            epoch_fn=run_train_epoch_ot_cfm,
            progress_cb=_cb_ot,
        )
        mo.output.replace(
            mo.md(
                f"**OT-CFM training complete!** Final train "
                f"`{train_losses_ot[-1]:.4f}` | val `{val_losses_ot[-1]:.4f}`"
            )
        )
    return train_losses_ot, trained_model_ot, val_losses_ot


@app.cell
def _(mo):
    mo.md("""
    ### 5c — Rectified Flow with Reflow

    Stage 1 trains a "gen1" model with vanilla CFM. Stage 2 uses gen1
    to synthesize `(x_0, x_1)` pairs by ODE integration, then trains a
    second model on those *fixed* pairs — that model's ODE map has
    straighter trajectories.

    **Reflow dataset size matters far more than epoch count.** Stage 2
    trains on a *fixed* synthetic set, unlike Stage 1/vanilla/OT-CFM
    which see a fresh Gaussian `x_0` every step against the *full*
    45k-image training set (~5,265 gradient steps at the default
    15 epochs / batch 128). The default `DiffusionTransformerV1` has
    ~4.9M parameters; a reflow dataset of only 2,048 pairs gives just
    ~240 steps at 15 epochs (under 5% of the real-data step count) —
    nowhere near enough for a model this size to do anything but
    memorize those 2,048 fixed pairs, which reads as "very bad,
    low-diversity, incoherent" samples. Grow **dataset size** first;
    only add epochs once the dataset is large enough that more epochs
    means more coverage rather than more repetition.
    """)
    return


@app.cell
def _(mo):
    lr_rf_gen1_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="LR (RF gen1)",
    )
    bs_rf_gen1_ui = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch size (RF gen1)",
    )
    wd_rf_gen1_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight decay (RF gen1)",
    )
    epochs_rf_gen1_ui = mo.ui.slider(1, 60, value=15, step=1, label="Epochs (RF gen1)")
    num_layers_rf_gen1_ui = mo.ui.slider(1, 12, value=6, step=1, label="DiT layers (RF gen1)")
    train_btn_rf_gen1 = mo.ui.run_button(label="Train Stage 1 (gen1)")
    mo.vstack(
        [
            mo.md("#### Stage 1 — gen1 (vanilla CFM, will be reflowed)"),
            mo.hstack([lr_rf_gen1_ui, bs_rf_gen1_ui, wd_rf_gen1_ui]),
            mo.hstack([epochs_rf_gen1_ui, num_layers_rf_gen1_ui]),
            train_btn_rf_gen1,
        ]
    )
    return (
        bs_rf_gen1_ui,
        epochs_rf_gen1_ui,
        lr_rf_gen1_ui,
        num_layers_rf_gen1_ui,
        train_btn_rf_gen1,
        wd_rf_gen1_ui,
    )


@app.cell
def _(
    bs_rf_gen1_ui,
    epochs_rf_gen1_ui,
    lr_rf_gen1_ui,
    mo,
    num_layers_rf_gen1_ui,
    train_btn_rf_gen1,
    wd_rf_gen1_ui,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses_rf_gen1 = []
    val_losses_rf_gen1 = []
    trained_model_rf_gen1 = None
    if not train_btn_rf_gen1.value:
        mo.output.replace(mo.md("Click **Train Stage 1 (gen1)** to begin training the base model that will be reflowed."))
    else:
        def _cb_gen1(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**RF gen1 Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model_rf_gen1, train_losses_rf_gen1, val_losses_rf_gen1 = train_dit_regime(
            x_tr,
            y_tr,
            x_val,
            y_val,
            num_layers=num_layers_rf_gen1_ui.value,
            lr=lr_rf_gen1_ui.value,
            wd=wd_rf_gen1_ui.value,
            batch_size=bs_rf_gen1_ui.value,
            epochs=epochs_rf_gen1_ui.value,
            epoch_fn=run_train_epoch_cfm,
            progress_cb=_cb_gen1,
        )
        mo.output.replace(
            mo.md(
                f"**RF gen1 training complete!** Final train "
                f"`{train_losses_rf_gen1[-1]:.4f}` | val `{val_losses_rf_gen1[-1]:.4f}`"
            )
        )
    return (trained_model_rf_gen1,)


@app.cell
def _(mo):
    lr_reflow_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="LR (reflow)",
    )
    bs_reflow_ui = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch size (reflow)",
    )
    wd_reflow_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight decay (reflow)",
    )
    epochs_reflow_ui = mo.ui.slider(1, 60, value=15, step=1, label="Epochs (reflow)")
    num_layers_reflow_ui = mo.ui.slider(1, 12, value=6, step=1, label="DiT layers (reflow)")
    num_samples_reflow_ui = mo.ui.slider(
        2048, 45056, value=16384, step=2048, label="Reflow dataset size"
    )
    gen_steps_reflow_ui = mo.ui.slider(
        10, 100, value=50, step=5, label="ODE steps to build reflow pairs"
    )
    train_btn_reflow = mo.ui.run_button(label="Build reflow dataset + Train Stage 2")
    mo.vstack(
        [
            mo.md("#### Stage 2 — reflow (train a fresh model on gen1's ODE map pairs)"),
            mo.md(
                "`Reflow dataset size` trades build time for sample quality: "
                "the ~4.9M-param default DiT needs a large, fixed pair set to "
                "avoid memorizing it (see note above) — 16,384 is a tractable "
                "middle ground; push toward the 45,056 max (matching the real "
                "training-set size) for the closest match to vanilla/OT-CFM's "
                "per-epoch step count."
            ),
            mo.hstack([lr_reflow_ui, bs_reflow_ui, wd_reflow_ui]),
            mo.hstack([epochs_reflow_ui, num_layers_reflow_ui]),
            mo.hstack([num_samples_reflow_ui, gen_steps_reflow_ui]),
            train_btn_reflow,
        ]
    )
    return (
        bs_reflow_ui,
        epochs_reflow_ui,
        gen_steps_reflow_ui,
        lr_reflow_ui,
        num_layers_reflow_ui,
        num_samples_reflow_ui,
        train_btn_reflow,
        wd_reflow_ui,
    )


@app.cell
def _(
    bs_reflow_ui,
    epochs_reflow_ui,
    gen_steps_reflow_ui,
    lr_reflow_ui,
    mo,
    num_layers_reflow_ui,
    num_samples_reflow_ui,
    train_btn_reflow,
    trained_model_rf_gen1,
    wd_reflow_ui,
    x_val,
    y_tr,
    y_val,
):
    train_losses_reflow = []
    val_losses_reflow = []
    trained_model_reflow = None
    if trained_model_rf_gen1 is None:
        mo.output.replace(mo.md("_Train the Stage 1 (gen1) model above first._"))
    elif not train_btn_reflow.value:
        mo.output.replace(
            mo.md(
                "Click **Build reflow dataset + Train Stage 2** to generate "
                "the synthetic `(x_0, x_1)` pairs from gen1 and train the "
                "reflow model."
            )
        )
    else:
        mo.output.replace(mo.md("Building reflow dataset from gen1..."))
        x0_rf, x1_rf, y_rf = build_reflow_dataset(
            trained_model_rf_gen1,
            num_samples=num_samples_reflow_ui.value,
            num_steps=gen_steps_reflow_ui.value,
            y_labels_pool=y_tr,
            batch_size=bs_reflow_ui.value,
        )
        def _cb_reflow(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**Reflow Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model_reflow, train_losses_reflow, val_losses_reflow = train_reflow_stage(
            x0_rf,
            x1_rf,
            y_rf,
            x_val,
            y_val,
            num_layers=num_layers_reflow_ui.value,
            lr=lr_reflow_ui.value,
            wd=wd_reflow_ui.value,
            batch_size=bs_reflow_ui.value,
            epochs=epochs_reflow_ui.value,
            progress_cb=_cb_reflow,
        )
        mo.output.replace(
            mo.md(
                f"**Reflow training complete!** Reflow dataset size "
                f"`{num_samples_reflow_ui.value}`. Final train "
                f"`{train_losses_reflow[-1]:.4f}` | val `{val_losses_reflow[-1]:.4f}`"
            )
        )
    return train_losses_reflow, trained_model_reflow, val_losses_reflow


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (Optional, vanilla CFM only)

    Scope note: to bound compute, hyperparameter search only sweeps the
    vanilla CFM baseline as the representative regime, not all three.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_config_cfm(
    x_tr_arr: mx.array,
    y_tr_arr: mx.array,
    x_val_arr: mx.array,
    y_val_arr: mx.array,
    lr: float,
    num_layers: int,
    n_epochs: int,
    batch_size: int,
) -> float:
    model = build_dit_model(num_layers=num_layers)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    val_batches = make_batches(x_val_arr, y_val_arr, batch_size=batch_size, shuffle=False)
    for _ in range(n_epochs):
        train_batches = make_batches(x_tr_arr, y_tr_arr, batch_size=batch_size, shuffle=True)
        run_train_epoch_cfm(model, optimizer, train_batches)
    return evaluate_model(model, val_batches)


@app.cell
def _(hp_search_cb, mo, x_tr, x_val, y_tr, y_val):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    _search_space = {"lr": [1e-4, 3e-4], "num_layers": [4, 6, 8]}
    _hp_epochs = 3
    _hp_batch_size = 128
    hp_results = []
    _sub_n = min(6000, x_tr.shape[0])
    _sub_x = x_tr[:_sub_n]
    _sub_y = y_tr[:_sub_n]
    for _lr in _search_space["lr"]:
        for _nl in _search_space["num_layers"]:
            _vl = run_hp_config_cfm(_sub_x, _sub_y, x_val, y_val, _lr, _nl, _hp_epochs, _hp_batch_size)
            hp_results.append({"lr": _lr, "num_layers": _nl, "val_loss": round(_vl, 4)})
            mo.output.replace(mo.md(f"lr={_lr}, num_layers={_nl} -> val={_vl:.4f}"))
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation (vanilla CFM only)

    Scope note: CV is scoped to the vanilla CFM regime as the
    representative baseline to keep local Apple-Silicon compute
    bounded (three regimes x k folds x epochs would be too much).
    """)
    return


@app.cell
def _(bs_vanilla_ui, mo, trained_model_vanilla, x_te, y_te):
    if trained_model_vanilla is None:
        _out = mo.md("_Train the vanilla CFM model first (Section 5a) before evaluating on the test set._")
    else:
        _test_batches = make_batches(x_te, y_te, batch_size=bs_vanilla_ui.value, shuffle=False)
        test_loss_vanilla = evaluate_model(trained_model_vanilla, _test_batches)
        _out = mo.md(f"**Vanilla CFM test set flow-matching loss**: `{test_loss_vanilla:.4f}`")
    _out
    return


@app.function
def run_cv_fold_cfm(
    x_fold: mx.array,
    y_fold: mx.array,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    lr: float,
    num_layers: int,
    n_epochs: int,
    batch_size: int,
) -> float:
    tr_i = mx.array(train_idx.astype(np.int32))
    va_i = mx.array(val_idx.astype(np.int32))
    x_tr_fold = x_fold[tr_i]
    y_tr_fold = y_fold[tr_i]
    x_va_fold = x_fold[va_i]
    y_va_fold = y_fold[va_i]
    model = build_dit_model(num_layers=num_layers)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    val_batches = make_batches(x_va_fold, y_va_fold, batch_size=batch_size, shuffle=False)
    for _ in range(n_epochs):
        train_batches = make_batches(x_tr_fold, y_tr_fold, batch_size=batch_size, shuffle=True)
        run_train_epoch_cfm(model, optimizer, train_batches)
    return evaluate_model(model, val_batches)


@app.cell
def _(mo, trained_model_vanilla, x_tr, y_tr):
    if trained_model_vanilla is None:
        _out = mo.md("_Train vanilla CFM first, then k-fold cross-validation results will appear here._")
    else:
        _cv_n = min(5000, x_tr.shape[0])
        _cv_x = x_tr[:_cv_n]
        _cv_y = y_tr[:_cv_n]
        _k = 5
        _rng = np.random.default_rng(seed=0)
        _perm = _rng.permutation(_cv_n)
        _folds = np.array_split(_perm, _k)
        cv_fold_losses = []
        for _f in range(_k):
            _val_idx = _folds[_f]
            _train_idx = np.concatenate([_folds[j] for j in range(_k) if j != _f])
            _vl = run_cv_fold_cfm(_cv_x, _cv_y, _train_idx, _val_idx, 3e-4, 6, 2, 128)
            cv_fold_losses.append(_vl)
            mo.output.replace(mo.md(f"Fold {_f + 1}/{_k} — val loss: {_vl:.4f}"))
        _mean = float(np.mean(cv_fold_losses))
        _std = float(np.std(cv_fold_losses))
        cv_results = {"fold_losses": cv_fold_losses, "mean": _mean, "std": _std}
        _out = mo.md(
            f"**{len(cv_fold_losses)}-Fold CV flow-matching loss**: `{_mean:.4f} ± {_std:.4f}` "
            f"(folds: {[round(v, 4) for v in cv_fold_losses]})"
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Final Verification & Comparison
    """)
    return


@app.function
def denormalize_cifar(x: mx.array) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    img = np.array(x) * std + mean
    return np.clip(img, 0.0, 1.0)


@app.function
def plot_loss_curves(curves: dict):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, losses in curves.items():
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, "-o", lw=2, ms=4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss (MSE)")
    ax.set_title("Training curves — three regimes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_generated_grid(
    model: nn.Module,
    num_steps: int,
    class_names: list,
    num_per_class: int = 2,
    title_prefix: str = "Generated samples",
    seed: int = 0,
):
    n_classes = len(class_names)
    total = n_classes * num_per_class
    labels = np.repeat(np.arange(n_classes, dtype=np.int32), num_per_class)
    y = mx.array(labels)
    mx.random.seed(seed)
    x0 = mx.random.normal(shape=(total, 32, 32, 3))
    x1 = euler_solve(model, x0, y, num_steps)
    mx.eval(x1)
    imgs = denormalize_cifar(x1)
    cols = num_per_class
    rows = n_classes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    for i in range(total):
        r, c = divmod(i, cols)
        ax = axes[r, c] if cols > 1 else axes[r]
        ax.imshow(imgs[i])
        if c == 0:
            ax.set_ylabel(class_names[r], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{title_prefix} (Euler, {num_steps} steps)", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def plot_euler_progression(
    model: nn.Module,
    class_names: list,
    num_steps: int = 50,
    num_samples: int = 6,
    frame_ts: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 1,
):
    mx.random.seed(seed)
    idx = np.random.randint(0, len(class_names), size=num_samples).astype(np.int32)
    y = mx.array(idx)
    x0 = mx.random.normal(shape=(num_samples, 32, 32, 3))
    trajectory = euler_solve_trajectory(model, x0, y, num_steps)
    frame_steps = [int(round(t * num_steps)) for t in frame_ts]
    cols = len(frame_steps)
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    for r in range(rows):
        for c, step in enumerate(frame_steps):
            ax = axes[r, c] if rows > 1 and cols > 1 else (axes[c] if rows == 1 else axes[r])
            img = denormalize_cifar(trajectory[step])[r]
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"t={step / num_steps:.2f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(class_names[int(idx[r])], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Euler ODE generation progression ({num_steps} steps)", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def compute_step_count_mse(
    model: nn.Module,
    step_list: list,
    ref_steps: int = 100,
    num_samples: int = 32,
    seed: int = 7,
) -> list:
    mx.random.seed(seed)
    y = mx.array(np.random.randint(0, 10, size=num_samples).astype(np.int32))
    x0 = mx.random.normal(shape=(num_samples, 32, 32, 3))
    x_ref = euler_solve(model, x0, y, ref_steps)
    mx.eval(x_ref)
    mses = []
    for ns in step_list:
        x_ns = euler_solve(model, x0, y, ns)
        mx.eval(x_ns)
        mses.append(float(mx.mean((x_ns - x_ref) ** 2).item()))
    return mses


@app.function
def plot_step_count_mse(
    step_list: list,
    mse_gen1: list,
    mse_reflow: list,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(step_list, mse_gen1, "b-o", lw=2, ms=6, label="gen1 (before reflow)")
    ax.plot(step_list, mse_reflow, "g-s", lw=2, ms=6, label="reflow (after)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Euler steps")
    ax.set_ylabel("MSE vs high-step reference")
    ax.set_title("Path-straightness proof: MSE(x@k-steps vs x@ref-steps) — lower and flatter is straighter")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


@app.cell
def _(mo):
    mo.md("""
    ### 8a. ODE generation progression

    Pick any of the three trained models to watch it denoise from
    Gaussian noise to a CIFAR-10 image via the Euler ODE.
    """)
    return


@app.cell
def _(mo):
    progression_model_ui = mo.ui.dropdown(
        options=["vanilla", "ot", "reflow"],
        value="vanilla",
        label="Model to visualize",
    )
    progression_steps_ui = mo.ui.slider(10, 200, value=50, step=5, label="Euler steps")
    progression_btn = mo.ui.run_button(label="Show ODE progression")
    mo.vstack([mo.hstack([progression_model_ui, progression_steps_ui]), progression_btn])
    return progression_btn, progression_model_ui, progression_steps_ui


@app.function
def pick_model(name: str, m_vanilla, m_ot, m_reflow):
    if name == "vanilla":
        return m_vanilla
    if name == "ot":
        return m_ot
    if name == "reflow":
        return m_reflow
    return None


@app.cell
def _(
    class_names,
    mo,
    progression_btn,
    progression_model_ui,
    progression_steps_ui,
    trained_model_ot,
    trained_model_reflow,
    trained_model_vanilla,
):
    _sel = pick_model(progression_model_ui.value, trained_model_vanilla, trained_model_ot, trained_model_reflow)
    if _sel is None:
        _out = mo.md(f"_The selected model `{progression_model_ui.value}` has not been trained yet._")
    elif not progression_btn.value:
        _out = mo.md("Click **Show ODE progression** to render the trajectory.")
    else:
        _out = plot_euler_progression(_sel, class_names, num_steps=progression_steps_ui.value)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### 8b. Straightness proof — gen1 vs reflow

    We compare the reflowed model against its own gen1 ancestor with
    two measurements:

    1. **Path-straightness metric `S`** (defined in Section 4): mean
       squared deviation of per-step Euler displacements from a
       straight-line path. Lower = straighter.
    2. **Step-count MSE**: with a *fixed* `x_0` and class, how close
       does an Euler solution at `k` steps get to a high-step reference
       (100 steps)? Straighter paths let the low-step Euler solution
       match the reference more closely, so the MSE curve for reflow
       should stay closer to zero for small `k`.
    """)
    return


@app.cell
def _(mo, trained_model_reflow, trained_model_rf_gen1):
    if trained_model_rf_gen1 is None or trained_model_reflow is None:
        _out = mo.md("_Train both Stage 1 (gen1) and Stage 2 (reflow) first (Section 5c)._")
    else:
        _y = mx.array(np.random.randint(0, 10, size=32).astype(np.int32))
        mx.random.seed(11)
        _x0 = mx.random.normal(shape=(32, 32, 32, 3))
        s_gen1 = compute_path_straightness(trained_model_rf_gen1, _x0, _y, num_steps=50)
        s_reflow = compute_path_straightness(trained_model_reflow, _x0, _y, num_steps=50)
        _improve = 100.0 * (s_gen1 - s_reflow) / max(s_gen1, 1e-12)
        _out = mo.md(
            f"""
            | Model | Path-straightness `S` (lower is straighter) |
            |-------|---------------------------------------------|
            | gen1 (before reflow) | `{s_gen1:.6f}` |
            | reflow (after) | `{s_reflow:.6f}` |
            | **Relative reduction** | **`{_improve:.1f}%`** |

            A positive relative reduction confirms that reflow produces
            straighter ODE trajectories than its gen1 ancestor.
            """
        )
    _out
    return


@app.cell
def _(mo, trained_model_reflow, trained_model_rf_gen1):
    if trained_model_rf_gen1 is None or trained_model_reflow is None:
        _out = mo.md("_Train both gen1 and reflow to see the step-count MSE curve._")
    else:
        _steps = [1, 2, 5, 10, 25, 50]
        _mse_gen1 = compute_step_count_mse(trained_model_rf_gen1, _steps, ref_steps=100)
        _mse_reflow = compute_step_count_mse(trained_model_reflow, _steps, ref_steps=100)
        _out = plot_step_count_mse(_steps, _mse_gen1, _mse_reflow)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### 8c. Three-way comparison — training curves, samples, table
    """)
    return


@app.cell
def _(mo, train_losses_ot, train_losses_reflow, train_losses_vanilla):
    _curves = {
        "vanilla CFM": train_losses_vanilla,
        "OT-CFM": train_losses_ot,
        "reflow (stage 2)": train_losses_reflow,
    }
    if not any(_curves.values()):
        _out = mo.md("_Train at least one regime to see loss curves._")
    else:
        _out = plot_loss_curves(_curves)
    _out
    return


@app.cell
def _(class_names, mo, trained_model_vanilla):
    if trained_model_vanilla is None:
        _out = mo.md("_Vanilla CFM not trained yet._")
    else:
        _out = plot_generated_grid(trained_model_vanilla, num_steps=50, class_names=class_names, title_prefix="Vanilla CFM samples")
    _out
    return


@app.cell
def _(class_names, mo, trained_model_ot):
    if trained_model_ot is None:
        _out = mo.md("_OT-CFM not trained yet._")
    else:
        _out = plot_generated_grid(trained_model_ot, num_steps=50, class_names=class_names, title_prefix="OT-CFM samples")
    _out
    return


@app.cell
def _(class_names, mo, trained_model_reflow):
    if trained_model_reflow is None:
        _out = mo.md("_Reflow model not trained yet._")
    else:
        _out = plot_generated_grid(trained_model_reflow, num_steps=50, class_names=class_names, title_prefix="Reflow samples")
    _out
    return


@app.function
def summarize_regime(name: str, model, train_losses: list, val_losses: list, x_te_arr: mx.array, y_te_arr: mx.array) -> dict:
    row = {"regime": name, "trained": model is not None}
    if model is None:
        row["params"] = None
        row["final_train"] = None
        row["final_val"] = None
        row["test_loss"] = None
        row["straightness"] = None
        return row
    row["params"] = count_parameters(model)
    row["final_train"] = round(float(train_losses[-1]), 4) if train_losses else None
    row["final_val"] = round(float(val_losses[-1]), 4) if val_losses else None
    _tb = make_batches(x_te_arr, y_te_arr, batch_size=128, shuffle=False)
    row["test_loss"] = round(evaluate_model(model, _tb), 4)
    mx.random.seed(99)
    _x0 = mx.random.normal(shape=(16, 32, 32, 3))
    _y = mx.array(np.random.randint(0, 10, size=16).astype(np.int32))
    row["straightness"] = round(compute_path_straightness(model, _x0, _y, num_steps=50), 6)
    return row


@app.cell
def _(
    mo,
    train_losses_ot,
    train_losses_reflow,
    train_losses_vanilla,
    trained_model_ot,
    trained_model_reflow,
    trained_model_vanilla,
    val_losses_ot,
    val_losses_reflow,
    val_losses_vanilla,
    x_te,
    y_te,
):
    if trained_model_vanilla is None and trained_model_ot is None and trained_model_reflow is None:
        _out = mo.md("_Train at least one regime to see the comparison table._")
    else:
        comparison_rows = [
            summarize_regime("vanilla CFM", trained_model_vanilla, train_losses_vanilla, val_losses_vanilla, x_te, y_te),
            summarize_regime("OT-CFM", trained_model_ot, train_losses_ot, val_losses_ot, x_te, y_te),
            summarize_regime("reflow (stage 2)", trained_model_reflow, train_losses_reflow, val_losses_reflow, x_te, y_te),
        ]
        _out = mo.ui.table(comparison_rows)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### Trade-off summary

    - **Vanilla CFM** is the simplest and cheapest to train per step —
      no coupling, no ODE integration in the loop. Its ODE at inference
      is typically the most curved of the three, so it needs many
      Euler steps for good samples.
    - **OT-CFM** adds a per-minibatch balanced assignment
      (`O(B^3)`-ish for the exact Hungarian solver; runtime dominated
      by the cost matrix construction). It produces less-crossing pair
      couplings, generally yielding a slightly straighter learned flow
      and a smaller training-loss variance without changing the loss
      shape.
    - **Reflow** costs a full extra training run plus a synthetic ODE
      dataset build (`num_samples * num_steps` model evals). Its
      payoff is qualitatively different: it explicitly *straightens*
      the paths, so few-step Euler sampling degrades far less. This is
      exactly what Section 8b measures.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Models
    """)
    return


@app.cell
def _(mo):
    save_vanilla_filename_ui = mo.ui.text(
        value="cifar10_dit_rcfm_vanilla_v1.safetensors",
        label="Vanilla CFM filename (models/)",
        full_width=True,
    )
    save_vanilla_btn = mo.ui.run_button(label="Save Vanilla CFM Model")
    mo.vstack([save_vanilla_filename_ui, save_vanilla_btn])
    return save_vanilla_btn, save_vanilla_filename_ui


@app.cell
def _(mo, save_vanilla_btn, save_vanilla_filename_ui, trained_model_vanilla):
    if trained_model_vanilla is None:
        _out = mo.md("_Train the vanilla CFM model first (Section 5a) before saving._")
    elif not save_vanilla_btn.value:
        _out = mo.md("Click **Save Vanilla CFM Model** to write weights to `models/`.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_vanilla_filename_ui.value
        trained_model_vanilla.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Vanilla CFM weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    save_ot_filename_ui = mo.ui.text(
        value="cifar10_dit_rcfm_ot_v1.safetensors",
        label="OT-CFM filename (models/)",
        full_width=True,
    )
    save_ot_btn = mo.ui.run_button(label="Save OT-CFM Model")
    mo.vstack([save_ot_filename_ui, save_ot_btn])
    return save_ot_btn, save_ot_filename_ui


@app.cell
def _(mo, save_ot_btn, save_ot_filename_ui, trained_model_ot):
    if trained_model_ot is None:
        _out = mo.md("_Train the OT-CFM model first (Section 5b) before saving._")
    elif not save_ot_btn.value:
        _out = mo.md("Click **Save OT-CFM Model** to write weights to `models/`.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_ot_filename_ui.value
        trained_model_ot.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** OT-CFM weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    save_reflow_filename_ui = mo.ui.text(
        value="cifar10_dit_rcfm_reflow_v1.safetensors",
        label="Reflow filename (models/)",
        full_width=True,
    )
    save_reflow_btn = mo.ui.run_button(label="Save Reflow Model")
    mo.vstack([save_reflow_filename_ui, save_reflow_btn])
    return save_reflow_btn, save_reflow_filename_ui


@app.cell
def _(mo, save_reflow_btn, save_reflow_filename_ui, trained_model_reflow):
    if trained_model_reflow is None:
        _out = mo.md("_Train the reflow model first (Section 5c) before saving._")
    elif not save_reflow_btn.value:
        _out = mo.md("Click **Save Reflow Model** to write weights to `models/`.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_reflow_filename_ui.value
        trained_model_reflow.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Reflow weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 10 — Load an Existing Trained Model

    Loads any `.safetensors` file from `models/` into a fresh
    `DiffusionTransformerV1` (default architecture: `embed_dim=256`,
    `num_heads=8`, `mlp_dim=512`, `patch_size=4`). The `num_layers`
    selector below must match the layer count used at training time
    for the file being loaded.
    """)
    return


@app.function
def list_saved_model_files(models_dir: Path) -> list:
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.glob("*.safetensors")])


@app.cell
def _(mo):
    _models_dir = Path(__file__).resolve().parent.parent / "models"
    _models_dir.mkdir(parents=True, exist_ok=True)
    _files = list_saved_model_files(_models_dir)
    _fallback = ["cifar10_dit_rcfm_vanilla_v1.safetensors"]
    _options = _files if _files else _fallback
    load_filename_ui = mo.ui.dropdown(
        options=_options,
        value=_options[0],
        label="Model file to load",
    )
    load_num_layers_ui = mo.ui.slider(1, 12, value=6, step=1, label="num_layers (must match saved model)")
    load_model_btn = mo.ui.run_button(label="Load Model")
    mo.vstack(
        [
            mo.hstack([load_filename_ui, load_num_layers_ui]),
            load_model_btn,
        ]
    )
    return load_filename_ui, load_model_btn, load_num_layers_ui


@app.cell
def _(load_filename_ui, load_model_btn, load_num_layers_ui, mo):
    loaded_model = None
    if not load_model_btn.value:
        _out = mo.md("Pick a filename above and click **Load Model** to load weights from disk.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _path = _models_dir / load_filename_ui.value
        if not _path.exists():
            _out = mo.md(f"**File not found:** `{_path}`. Save a model first (Section 9) or check the filename.")
        else:
            loaded_model = build_dit_model(num_layers=load_num_layers_ui.value)
            loaded_model.load_weights(str(_path))
            mx.eval(loaded_model.parameters())
            _out = mo.md(
                f"**Loaded!** File: `{_path}` | "
                f"num_layers = `{load_num_layers_ui.value}` | "
                f"parameters = `{count_parameters(loaded_model):,}`"
            )
    _out
    return (loaded_model,)


@app.cell
def _(class_names, loaded_model, mo):
    if loaded_model is None:
        _out = mo.md("_Load a model above to sample from it._")
    else:
        _out = plot_generated_grid(
            loaded_model,
            num_steps=50,
            class_names=class_names,
            title_prefix="Samples from loaded model",
            seed=123,
        )
    _out
    return


if __name__ == "__main__":
    app.run()
