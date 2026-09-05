import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import math
    import os
    import pickle
    import tarfile
    import urllib.request
    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils

    import numpy as np
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
    # Flow Matching with a Diffusion Transformer on CIFAR-10 (MLX)

    ## Research Goal

    Train a **class-conditional flow-matching generative model** on
    **CIFAR-10** using a **Diffusion Transformer (DiT)** backbone
    implemented in **MLX**. The DiT depth (`num_layers`) is a first-class
    UI parameter. At inference time we integrate the learned velocity field
    with three ODE solvers — **Euler**, **Midpoint / Heun**, and
    **classical Runge-Kutta 4 (RK4)** — with a UI-controlled step count.

    ### Method
    - Linear interpolation path: `x_t = (1 - t) * x0 + t * x1` with
      `x0 ~ N(0, I)` and `x1` a real CIFAR-10 image
    - Target velocity is constant along the path: `v_t = x1 - x0`
    - Training loss: `MSE(model(x_t, t, y), v_t)`
    - Inference: solve `dx/dt = v_theta(x, t, y)` from `t=0` to `t=1`

    ### Notebook Outline
    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation
    4. Model definition (DiT + flow-matching loss + ODE solvers)
    5. Training loop
    6. Optional hyperparameter search
    7. Validation & cross-validation
    8. Results — loss curves, generated samples, solver comparison
    9. Save trained model
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration
    """)
    return


@app.function
def download_cifar10(root: str = "../data/cifar10") -> str:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    extracted = root_path / "cifar-10-batches-py"
    if extracted.exists():
        return str(extracted)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = root_path / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(url, str(tar_path))
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(root_path))
    return str(extracted)


@app.function
def load_cifar10_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"].astype(np.float32) / 255.0
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(d[b"labels"], dtype=np.int32)
    return data, labels


@app.function
def load_cifar10(root: str = "../data/cifar10") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    extracted = download_cifar10(root)
    train_images_list = []
    train_labels_list = []
    for i in range(1, 6):
        xi, yi = load_cifar10_batch(os.path.join(extracted, f"data_batch_{i}"))
        train_images_list.append(xi)
        train_labels_list.append(yi)
    x_train = np.concatenate(train_images_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)
    x_test, y_test = load_cifar10_batch(os.path.join(extracted, "test_batch"))
    with open(os.path.join(extracted, "batches.meta"), "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    return x_train, y_train, x_test, y_test, class_names


@app.cell
def _():
    x_train_np, y_train_np, x_test_np, y_test_np, class_names = load_cifar10("../data/cifar10")
    return class_names, x_test_np, x_train_np, y_test_np, y_train_np


@app.cell
def _(mo, x_test_np, x_train_np):
    mo.md(
        f"""
        ### Dataset overview

        CIFAR-10 is downloaded from the Toronto mirror and cached under
        `../data/cifar10/`. Each image is stored as a `float32` array shaped
        `(32, 32, 3)` in `[0, 1]`; labels are `int32` in `[0, 9]`.

        | Split | Size | Shape |
        |-------|------|-------|
        | Train (raw) | {x_train_np.shape[0]:,} | {tuple(x_train_np.shape[1:])} |
        | Test | {x_test_np.shape[0]:,} | {tuple(x_test_np.shape[1:])} |
        """
    )
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
    ## Section 4 — Model Definition
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
        return mx.concatenate([mx.sin(t[:, None] * self.freqs[None, :]), mx.cos(t[:, None] * self.freqs[None, :])], axis=-1)


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
def compute_flow_loss(model: nn.Module, x1: mx.array, y: mx.array, rng_key_unused=None) -> mx.array:
    t = mx.random.uniform(shape=(x1.shape[0],))
    x0 = mx.random.normal(shape=x1.shape)
    t_view = t.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = model(x_t, t, y)
    return mx.mean((v_pred - v_target) ** 2)


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
def midpoint_solve(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        t_mid = mx.full((x.shape[0],), i * dt + 0.5 * dt, dtype=mx.float32)
        k1 = model(x, t, y)
        x_mid = x + 0.5 * dt * k1
        k2 = model(x_mid, t_mid, y)
        x = x + dt * k2
        mx.eval(x)
    return x


@app.function
def rk4_solve(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t0 = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        t_mid = mx.full((x.shape[0],), i * dt + 0.5 * dt, dtype=mx.float32)
        t_end = mx.full((x.shape[0],), (i + 1) * dt, dtype=mx.float32)
        k1 = model(x, t0, y)
        k2 = model(x + 0.5 * dt * k1, t_mid, y)
        k3 = model(x + 0.5 * dt * k2, t_mid, y)
        k4 = model(x + dt * k3, t_end, y)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        mx.eval(x)
    return x


@app.cell
def _(mo):
    mo.md("""
    ### Model Architecture — `DiffusionTransformerV1`

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Patch embedding | `PatchifyV1` (Conv2d, stride=patch) | `(B, N, D)` where `N = (H/p)^2` |
    | Positional embedding | learnable table `(1, N, D)` | `(B, N, D)` |
    | Time embedding | `SinusoidalTimestepEmbeddingV1` + MLP | `(B, D)` |
    | Class embedding | `nn.Embedding(C+1, D)` (null token for CFG) | `(B, D)` |
    | Backbone | `DiTBlockV1 x num_layers` (AdaLN attn + AdaLN MLP) | `(B, N, D)` |
    | Head | `LayerNorm` + `UnpatchifyV1` | `(B, H, W, C)` |

    Defaults: `image_size=32, patch_size=4, embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6`.
    """)
    return


@app.cell
def _():
    default_model = DiffusionTransformerV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=6,
        dropout=0.0,
    )
    mx.eval(default_model.parameters())
    default_param_count = count_parameters(default_model)
    return (default_param_count,)


@app.cell
def _(default_param_count, mo):
    mo.md(f"**Default model parameter count**: `{default_param_count:,}`")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch Size",
    )
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    epochs_ui = mo.ui.slider(1, 100, value=30, step=1, label="Epochs")
    num_layers_ui = mo.ui.slider(1, 12, value=6, step=1, label="DiT num_layers")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Hyperparameters"),
            mo.hstack([lr_ui, bs_ui, wd_ui]),
            mo.hstack([epochs_ui, num_layers_ui]),
            train_btn,
        ]
    )
    return bs_ui, epochs_ui, lr_ui, num_layers_ui, train_btn, wd_ui


@app.function
def run_train_epoch(model: nn.Module, loss_and_grad_fn, optimizer, train_batches: list) -> float:
    epoch_loss = 0.0
    n = 0
    for xb, yb in train_batches:
        loss, grads = loss_and_grad_fn(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def run_evaluate(model: nn.Module, val_batches: list) -> float:
    total = 0.0
    n = 0
    for xb, yb in val_batches:
        loss = compute_flow_loss(model, xb, yb)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.cell
def _(
    bs_ui,
    epochs_ui,
    lr_ui,
    mo,
    num_layers_ui,
    train_btn,
    wd_ui,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses = []
    val_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training."))
    else:
        _model = DiffusionTransformerV1(
            image_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=256,
            num_heads=8,
            mlp_dim=512,
            num_layers=num_layers_ui.value,
            dropout=0.0,
        )
        mx.eval(_model.parameters())
        _optimizer = optim.AdamW(learning_rate=lr_ui.value, weight_decay=wd_ui.value)
        _loss_and_grad_fn = nn.value_and_grad(_model, compute_flow_loss)
        _n_epochs = epochs_ui.value
        _val_batches = make_batches(x_val, y_val, batch_size=bs_ui.value, shuffle=False)
        for epoch in range(_n_epochs):
            _train_batches = make_batches(x_tr, y_tr, batch_size=bs_ui.value, shuffle=True)
            tl = run_train_epoch(_model, _loss_and_grad_fn, _optimizer, _train_batches)
            vl = run_evaluate(_model, _val_batches)
            train_losses.append(tl)
            val_losses.append(vl)
            mo.output.replace(
                mo.md(f"**Epoch {epoch + 1}/{_n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model = _model
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train: {train_losses[-1]:.4f} | val: {val_losses[-1]:.4f}"
            )
        )
    return train_losses, trained_model, val_losses


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (Optional)
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_config(
    x_tr_arr: mx.array,
    y_tr_arr: mx.array,
    x_val_arr: mx.array,
    y_val_arr: mx.array,
    lr: float,
    num_layers: int,
    n_epochs: int,
    batch_size: int,
) -> float:
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
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    val_batches = make_batches(x_val_arr, y_val_arr, batch_size=batch_size, shuffle=False)
    for _ in range(n_epochs):
        train_batches = make_batches(x_tr_arr, y_tr_arr, batch_size=batch_size, shuffle=True)
        run_train_epoch(model, loss_and_grad_fn, optimizer, train_batches)
    return run_evaluate(model, val_batches)


@app.cell
def _(hp_search_cb, mo, x_tr, x_val, y_tr, y_val):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    _search_space = {"lr": [1e-4, 3e-4], "num_layers": [4, 6, 8]}
    _hp_epochs = 5
    _hp_batch_size = 128
    hp_results = []
    _sub_n = min(6000, x_tr.shape[0])
    _sub_x = x_tr[:_sub_n]
    _sub_y = y_tr[:_sub_n]
    for _lr in _search_space["lr"]:
        for _nl in _search_space["num_layers"]:
            _vl = run_hp_config(_sub_x, _sub_y, x_val, y_val, _lr, _nl, _hp_epochs, _hp_batch_size)
            hp_results.append({"lr": _lr, "num_layers": _nl, "val_loss": round(_vl, 4)})
            mo.output.replace(mo.md(f"lr={_lr}, num_layers={_nl} -> val={_vl:.4f}"))
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation
    """)
    return


@app.function
def evaluate_model(model: nn.Module, batches: list) -> float:
    return run_evaluate(model, batches)


@app.cell
def _(bs_ui, mo, trained_model, x_te, y_te):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before evaluating._")
    else:
        _test_batches = make_batches(x_te, y_te, batch_size=bs_ui.value, shuffle=False)
        test_loss = evaluate_model(trained_model, _test_batches)
        _out = mo.md(f"**Test set flow-matching loss**: `{test_loss:.4f}`")
    _out
    return


@app.function
def run_cv_fold(
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
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    val_batches = make_batches(x_va_fold, y_va_fold, batch_size=batch_size, shuffle=False)
    for _ in range(n_epochs):
        train_batches = make_batches(x_tr_fold, y_tr_fold, batch_size=batch_size, shuffle=True)
        run_train_epoch(model, loss_and_grad_fn, optimizer, train_batches)
    return run_evaluate(model, val_batches)


@app.cell
def _(mo, trained_model, x_tr, y_tr):
    if trained_model is None:
        _out = mo.md("_Train first, then k-fold cross-validation results will appear here._")
    else:
        _cv_n = min(6000, x_tr.shape[0])
        _cv_x = x_tr[:_cv_n]
        _cv_y = y_tr[:_cv_n]
        _k = 3
        _rng = np.random.default_rng(seed=0)
        _perm = _rng.permutation(_cv_n)
        _folds = np.array_split(_perm, _k)
        cv_fold_losses = []
        for _f in range(_k):
            _val_idx = _folds[_f]
            _train_idx = np.concatenate([_folds[j] for j in range(_k) if j != _f])
            _vl = run_cv_fold(_cv_x, _cv_y, _train_idx, _val_idx, 3e-4, 6, 3, 128)
            cv_fold_losses.append(_vl)
            mo.output.replace(mo.md(f"Fold {_f + 1}/{_k} — val loss: {_vl:.4f}"))
        _mean = float(np.mean(cv_fold_losses))
        _std = float(np.std(cv_fold_losses))
        cv_results = {"fold_losses": cv_fold_losses, "mean": _mean, "std": _std}
        _out = mo.md(
            f"**{len(cv_fold_losses)}-Fold CV flow loss**: `{_mean:.4f} ± {_std:.4f}` "
            f"(folds: {[round(v, 4) for v in cv_fold_losses]})"
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Results
    """)
    return


@app.function
def plot_loss_curve(train_losses: list, val_losses: list | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train")
    if val_losses:
        ax.plot(epochs, val_losses, "r-s", lw=2, ms=4, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss (MSE)")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses, trained_model, val_losses):
    if trained_model is None:
        _out = mo.md("_Train the model first to see the loss curve._")
    else:
        _out = plot_loss_curve(train_losses, val_losses)
    _out
    return


@app.function
def denormalize_cifar(x: mx.array) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    img = np.array(x) * std + mean
    return np.clip(img, 0.0, 1.0)


@app.function
def plot_generated_grid(
    model: nn.Module,
    solver_fn,
    num_steps: int,
    class_names: list,
    num_per_class: int = 1,
):
    n_classes = len(class_names)
    total = n_classes * num_per_class
    labels = np.repeat(np.arange(n_classes, dtype=np.int32), num_per_class)
    y = mx.array(labels)
    x0 = mx.random.normal(shape=(total, 32, 32, 3))
    x1 = solver_fn(model, x0, y, num_steps)
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
    fig.suptitle(f"Generated samples ({solver_fn.__name__}, {num_steps} steps)", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def plot_solver_comparison(
    model: nn.Module,
    class_idx: int,
    class_names: list,
    steps_list: list,
):
    solvers = [("Euler", euler_solve), ("Midpoint", midpoint_solve), ("RK4", rk4_solve)]
    rows = len(solvers)
    cols = len(steps_list)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.9))
    y = mx.array(np.array([class_idx], dtype=np.int32))
    seed_noise = mx.random.normal(shape=(1, 32, 32, 3))
    for r, (name, fn) in enumerate(solvers):
        for c, ns in enumerate(steps_list):
            x1 = fn(model, seed_noise, y, ns)
            mx.eval(x1)
            img = denormalize_cifar(x1)[0]
            ax = axes[r, c] if rows > 1 and cols > 1 else (axes[c] if rows == 1 else axes[r])
            ax.imshow(img)
            ax.set_title(f"{name} — {ns} steps", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Solver comparison — class `{class_names[class_idx]}`", fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, mo):
    solver_ui = mo.ui.dropdown(
        options={"Euler": "euler", "Midpoint": "midpoint", "RK4": "rk4"},
        value="Euler",
        label="ODE Solver",
    )
    steps_ui = mo.ui.slider(5, 200, value=50, step=1, label="Solver Steps")
    class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value=class_names[0],
        label="Class to generate",
    )
    mo.vstack([mo.md("### Sampling controls"), mo.hstack([solver_ui, steps_ui, class_ui])])
    return class_ui, solver_ui, steps_ui


@app.function
def resolve_solver(name: str):
    return {"euler": euler_solve, "midpoint": midpoint_solve, "rk4": rk4_solve}[name]


@app.cell
def _(class_names, mo, solver_ui, steps_ui, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to generate samples._")
    else:
        _out = plot_generated_grid(
            trained_model, resolve_solver(solver_ui.value), steps_ui.value, class_names, num_per_class=4
        )
    _out
    return


@app.cell
def _(class_names, class_ui, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to compare solvers._")
    else:
        _out = plot_solver_comparison(
            trained_model, int(class_ui.value), class_names, [10, 25, 50, 100]
        )
    _out
    return


@app.cell
def _(mo, num_layers_ui, train_losses, trained_model, val_losses):
    if trained_model is None:
        _out = mo.md("_Train the model to see a results summary._")
    else:
        _out = mo.md(
            f"""
            ### Summary

            - Backbone: **DiffusionTransformerV1** with `num_layers = {num_layers_ui.value}`,
              `embed_dim=256`, `num_heads=8`, `mlp_dim=512`, patch size `4`.
            - Trained for `{len(train_losses)}` epochs; final train loss
              `{train_losses[-1]:.4f}`, final val loss `{val_losses[-1]:.4f}`.
            - Flow-matching objective: linear-path MSE between predicted and
              target velocity `v = x1 - x0`.
            - Inference supports **Euler**, **Midpoint / Heun**, and **RK4**
              solvers over a UI-controlled number of steps. RK4 typically
              produces the best samples at fewer steps at the cost of 4x more
              network evaluations per step.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Model
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="cifar10_dit_flow_v1.safetensors",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_model_btn])
    return save_filename_ui, save_model_btn


@app.cell
def _(mo, save_filename_ui, save_model_btn, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before saving._")
    elif not save_model_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save Model** to write the trained "
            "weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_filename_ui.value
        trained_model.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Model weights written to `{_save_path}`.")
    _out
    return


if __name__ == "__main__":
    app.run()
