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
    mo.md(f"""
    **Default model parameter count**: `{default_param_count:,}`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Section 4b — Positional Encoding V2: 2D Rotary Position Embedding

    #### 1-D RoPE — Foundations

    RoPE (Su et al., 2021) encodes token position $m$ by **rotating** consecutive
    head-dimension pairs $(x_{2i},\; x_{2i+1})$ of the Q and K vectors by angle
    $m\theta_i$, where the base frequencies decay geometrically:

    $$\theta_i = 10000^{-2i/d}, \quad i \in \left[0,\; \tfrac{d}{2}\right)$$

    The 2×2 rotation applied to pair $i$ at position $m$:

    $$\begin{pmatrix}\tilde{x}_{2i}\\\tilde{x}_{2i+1}\end{pmatrix}
    = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}
    \begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}$$

    **Relative-position property.** Since $\mathbf{R}(m)^{\top}\mathbf{R}(n) = \mathbf{R}(m-n)$,
    the rotated inner product $\langle\mathbf{R}(m)\mathbf{q},\;\mathbf{R}(n)\mathbf{k}\rangle$
    depends only on the relative offset $m - n$, giving translation-equivariance for free.

    ---

    #### 2-D Extension — VisionLLaMA (arxiv:2403.13298)

    For image patches at 2-D grid position $(r, c)$ we must encode **two** coordinates
    while preserving the relative-position property in both directions.

    **Key idea**: split the head dimension $d$ into two halves and apply independent 1-D
    RoPE to each, using a different position argument:

    | Head-dim slice | Position used | Encodes |
    |---|---|---|
    | $[0,\; d/2)$ — *row half* | row index $r$ | vertical relative offset $r - r'$ |
    | $[d/2,\; d)$ — *col half* | column index $c$ | horizontal relative offset $c - c'$ |

    The attention score between patches $(r, c)$ and $(r', c')$ factorises:

    $$\langle\mathbf{q},\mathbf{k}\rangle
    = \underbrace{\langle\mathbf{q}_r^{(r)},\;\mathbf{k}_r^{(r')}\rangle}_{\text{depends on }r-r'}
    + \underbrace{\langle\mathbf{q}_c^{(c)},\;\mathbf{k}_c^{(c')}\rangle}_{\text{depends on }c-c'}$$

    Both terms individually satisfy the 1-D relative-position property, so the full 2-D
    attention score depends only on the relative displacement $(r - r',\; c - c')$.

    ---

    #### Frequency Formula for 2-D RoPE

    Each half has size $d/2$.  Substituting $d/2$ as the effective dimension into the
    base-frequency formula:

    $$\theta_i = 10000^{-2i/(d/2)} = 10000^{-4i/d}, \quad i \in \left[0,\; \tfrac{d}{4}\right)$$

    Both halves share the same frequency bank $\{\theta_i\}$ — only the position argument
    differs ($r$ vs $c$).  With `embed_dim=256, num_heads=8` → `head_dim=32` → `quarter=8`
    unique frequencies per head.

    ---

    #### Implementation: `rotate_pairs`

    ```python
    quarter   = head_dim // 4
    freqs[i]  = 10000 ** (-4*i / head_dim)         # (quarter,)

    # for position vector p ∈ {row_pos, col_pos}, shape (N,):
    angles    = p[:, None] * freqs[None, :]         # (N, quarter)
    cos_full  = repeat(cos(angles), 2, axis=-1)     # (N, half)  — each value duplicated
    sin_full  = repeat(sin(angles), 2, axis=-1)     # (N, half)

    # rotate_pairs(x, cos_full, sin_full),  x: (B, N, H, half):
    x0 = x[..., 0::2]                              # even-indexed pairs  (B,N,H,quarter)
    x1 = x[..., 1::2]                              # odd-indexed pairs   (B,N,H,quarter)
    c, s = cos_full[..., ::2], sin_full[..., ::2]  # unique values       (N,quarter)
    rot_0 = x0 * c - x1 * s
    rot_1 = x0 * s + x1 * c
    return interleave(rot_0, rot_1)                 # (B,N,H,half)
    ```

    `DiffusionTransformerV2` drops the learnable `pos_embed` table
    (≈ `num_patches × embed_dim` = `64 × 256 = 16 384` scalars) and injects
    position via fixed RoPE rotations directly into every Q and K projection.
    """)
    return


@app.class_definition
class RoPE2DAttentionV2(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        quarter = self.head_dim // 4
        # θ_i = 10000^(-4i/head_dim), shared by row-half and col-half
        self.freqs = 1.0 / (10000.0 ** (mx.arange(0, quarter, dtype=mx.float32) * 4.0 / self.head_dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _make_cos_sin(self, positions: mx.array):
        # positions: (N,) — row or col grid indices cast to float
        # angles:   (N, quarter) → cos/sin: (N, half) with each value pair-duplicated
        angles = positions[:, None] * self.freqs[None, :]
        cos = mx.repeat(mx.cos(angles), 2, axis=-1)
        sin = mx.repeat(mx.sin(angles), 2, axis=-1)
        return cos, sin

    def _apply_rotary(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x:   (B, N, H, half)   where half = head_dim // 2
        # cos/sin: (N, half) from _make_cos_sin (pair-duplicated)
        B, N, H, half = x.shape
        pairs = x.reshape(B, N, H, half // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]          # (B, N, H, half//2) each
        c = cos[:, ::2][None, :, None, :]               # (1, N, 1, half//2) unique cos
        s = sin[:, ::2][None, :, None, :]               # (1, N, 1, half//2) unique sin
        rot0 = x0 * c - x1 * s
        rot1 = x0 * s + x1 * c
        return mx.stack([rot0, rot1], axis=-1).reshape(B, N, H, half)

    def __call__(self, x: mx.array, row_pos: mx.array, col_pos: mx.array) -> mx.array:
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        half = hd // 2

        q = self.q_proj(x).reshape(B, N, H, hd)
        k = self.k_proj(x).reshape(B, N, H, hd)
        v = self.v_proj(x).reshape(B, N, H, hd)

        # split each head into row-half [0, half) and col-half [half, hd)
        q_r, q_c = q[..., :half], q[..., half:]
        k_r, k_c = k[..., :half], k[..., half:]

        # apply 1-D RoPE independently on each half with the matching position vector
        r_cos, r_sin = self._make_cos_sin(row_pos)
        c_cos, c_sin = self._make_cos_sin(col_pos)
        q_r = self._apply_rotary(q_r, r_cos, r_sin)
        k_r = self._apply_rotary(k_r, r_cos, r_sin)
        q_c = self._apply_rotary(q_c, c_cos, c_sin)
        k_c = self._apply_rotary(k_c, c_cos, c_sin)

        q = mx.concatenate([q_r, q_c], axis=-1).transpose(0, 2, 1, 3)  # (B, H, N, hd)
        k = mx.concatenate([k_r, k_c], axis=-1).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


@app.class_definition
class DiTBlockV2(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, cond_dim: int = 256):
        super().__init__()
        self.attn_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.attn = RoPE2DAttentionV2(dim, num_heads)
        self.mlp_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def __call__(self, x: mx.array, cond: mx.array, row_pos: mx.array, col_pos: mx.array) -> mx.array:
        h = self.attn_norm(x, cond)
        x = x + self.attn(h, row_pos, col_pos)
        return x + self.mlp(self.mlp_norm(x, cond))


@app.class_definition
class DiffusionTransformerV2(nn.Module):
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
        self.grid_size = image_size // patch_size
        self.patchify = PatchifyV1(patch_size, embed_dim, in_channels)
        # no pos_embed table — position injected via 2D RoPE into every Q/K projection
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.blocks = [DiTBlockV2(embed_dim, num_heads, mlp_dim, embed_dim) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatchify = UnpatchifyV1(patch_size, embed_dim, in_channels, image_size)

    def __call__(self, x: mx.array, t: mx.array, y: mx.array) -> mx.array:
        g = self.grid_size
        # row-major patch order: patch (r, c) → index r*g + c
        row_pos = mx.array([r for r in range(g) for _ in range(g)], dtype=mx.float32)
        col_pos = mx.array([c for _ in range(g) for c in range(g)], dtype=mx.float32)
        h = self.patchify(x)
        cond = self.time_embed(t) + self.class_embed(y)
        for block in self.blocks:
            h = block(h, cond, row_pos, col_pos)
        return self.unpatchify(self.final_norm(h))


@app.cell
def _(mo):
    _v1 = DiffusionTransformerV1(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6,
    )
    _v2 = DiffusionTransformerV2(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6,
    )
    mx.eval(_v1.parameters())
    mx.eval(_v2.parameters())
    _n1 = count_parameters(_v1)
    _n2 = count_parameters(_v2)
    mo.md(f"""
    ### V1 vs V2 Parameter Counts

    | Model | Parameters | Notes |
    |-------|-----------|-------|
    | `DiffusionTransformerV1` | `{_n1:,}` | includes learnable `pos_embed` table |
    | `DiffusionTransformerV2` | `{_n2:,}` | 2-D RoPE — no `pos_embed` |
    | Difference | `{_n1 - _n2:+,}` | ≈ `num_patches × embed_dim` = `{64 * 256:,}` removed |
    """)
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
    ## Section 5b — Training V2 (2D RoPE)
    """)
    return


@app.cell
def _(mo):
    lr_ui_v2 = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Learning Rate",
    )
    bs_ui_v2 = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch Size",
    )
    wd_ui_v2 = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    epochs_ui_v2 = mo.ui.slider(1, 100, value=30, step=1, label="Epochs")
    num_layers_ui_v2 = mo.ui.slider(1, 12, value=6, step=1, label="DiT num_layers")
    train_btn_v2 = mo.ui.run_button(label="Train V2")
    mo.vstack(
        [
            mo.md("### Hyperparameters (V2 — 2D RoPE)"),
            mo.hstack([lr_ui_v2, bs_ui_v2, wd_ui_v2]),
            mo.hstack([epochs_ui_v2, num_layers_ui_v2]),
            train_btn_v2,
        ]
    )
    return (
        bs_ui_v2,
        epochs_ui_v2,
        lr_ui_v2,
        num_layers_ui_v2,
        train_btn_v2,
        wd_ui_v2,
    )


@app.cell
def _(
    bs_ui_v2,
    epochs_ui_v2,
    lr_ui_v2,
    mo,
    num_layers_ui_v2,
    train_btn_v2,
    wd_ui_v2,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses_v2 = []
    val_losses_v2 = []
    trained_model_v2 = None

    if not train_btn_v2.value:
        mo.output.replace(mo.md("Click **Train V2** to begin training the 2D RoPE model."))
    else:
        _model_v2 = DiffusionTransformerV2(
            image_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=256,
            num_heads=8,
            mlp_dim=512,
            num_layers=num_layers_ui_v2.value,
            dropout=0.0,
        )
        mx.eval(_model_v2.parameters())
        _optimizer_v2 = optim.AdamW(learning_rate=lr_ui_v2.value, weight_decay=wd_ui_v2.value)
        _loss_and_grad_fn_v2 = nn.value_and_grad(_model_v2, compute_flow_loss)
        _n_epochs_v2 = epochs_ui_v2.value
        _val_batches_v2 = make_batches(x_val, y_val, batch_size=bs_ui_v2.value, shuffle=False)
        for _epoch_v2 in range(_n_epochs_v2):
            _train_batches_v2 = make_batches(x_tr, y_tr, batch_size=bs_ui_v2.value, shuffle=True)
            _tl_v2 = run_train_epoch(_model_v2, _loss_and_grad_fn_v2, _optimizer_v2, _train_batches_v2)
            _vl_v2 = run_evaluate(_model_v2, _val_batches_v2)
            train_losses_v2.append(_tl_v2)
            val_losses_v2.append(_vl_v2)
            mo.output.replace(
                mo.md(f"**Epoch {_epoch_v2 + 1}/{_n_epochs_v2}** — train: {_tl_v2:.4f} | val: {_vl_v2:.4f}")
            )
        trained_model_v2 = _model_v2
        mo.output.replace(
            mo.md(
                f"**V2 training complete!** Final train: {train_losses_v2[-1]:.4f} | val: {val_losses_v2[-1]:.4f}"
            )
        )
    return train_losses_v2, trained_model_v2, val_losses_v2


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


@app.function
def plot_euler_trajectory(
    model: nn.Module,
    class_idx: int,
    class_names: list,
    num_steps: int,
    num_frames: int = 10,
):
    y = mx.array(np.array([class_idx], dtype=np.int32))
    x0 = mx.random.normal(shape=(1, 32, 32, 3))
    trajectory = euler_solve_trajectory(model, x0, y, num_steps)
    frame_idx = sorted(set(np.linspace(0, num_steps, num_frames).round().astype(int).tolist()))
    n = len(frame_idx)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.7, 2.0))
    for col, step in enumerate(frame_idx):
        img = denormalize_cifar(trajectory[step])[0]
        ax = axes[col] if n > 1 else axes
        ax.imshow(img)
        ax.set_title(f"t={step / num_steps:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Euler solver trajectory — class `{class_names[class_idx]}` ({num_steps} steps)",
        fontsize=12,
    )
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
    sample_btn = mo.ui.run_button(label="Sample")
    compare_btn = mo.ui.run_button(label="Compare Solvers")
    mo.vstack(
        [
            mo.md("### Sampling controls"),
            mo.hstack([solver_ui, steps_ui, class_ui]),
            mo.hstack([sample_btn, compare_btn]),
        ]
    )
    return class_ui, compare_btn, sample_btn, solver_ui, steps_ui


@app.function
def resolve_solver(name: str):
    return {"euler": euler_solve, "midpoint": midpoint_solve, "rk4": rk4_solve}[name]


@app.cell
def _(class_names, mo, sample_btn, solver_ui, steps_ui, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to generate samples._")
    elif not sample_btn.value:
        _out = mo.md("Click **Sample** to generate a grid of images for every class.")
    else:
        _out = plot_generated_grid(
            trained_model, resolve_solver(solver_ui.value), steps_ui.value, class_names, num_per_class=4
        )
    _out
    return


@app.cell
def _(class_names, class_ui, compare_btn, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to compare solvers._")
    elif not compare_btn.value:
        _out = mo.md("Click **Compare Solvers** to run Euler / Midpoint / RK4 at several step counts.")
    else:
        _out = plot_solver_comparison(
            trained_model, int(class_ui.value), class_names, [10, 25, 50, 100]
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### Euler Solver Trajectory — Watching the Model Denoise

    The Euler solver integrates the learned velocity field one small step
    at a time, starting from pure Gaussian noise (`t=0`) and arriving at a
    generated image (`t=1`). The frames below sample intermediate states
    `x_t` along a single trajectory, making the model's step-by-step
    progress toward a coherent image directly visible.
    """)
    return


@app.cell
def _(class_names, mo):
    traj_class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value=class_names[0],
        label="Class to generate",
    )
    traj_steps_ui = mo.ui.slider(10, 200, value=50, step=1, label="Euler Steps")
    traj_frames_ui = mo.ui.slider(4, 20, value=10, step=1, label="Frames to display")
    traj_btn = mo.ui.run_button(label="Generate Trajectory")
    mo.vstack(
        [
            mo.hstack([traj_class_ui, traj_steps_ui, traj_frames_ui]),
            traj_btn,
        ]
    )
    return traj_btn, traj_class_ui, traj_frames_ui, traj_steps_ui


@app.cell
def _(
    class_names,
    mo,
    trained_model,
    traj_btn,
    traj_class_ui,
    traj_frames_ui,
    traj_steps_ui,
):
    if trained_model is None:
        _out = mo.md("_Train the model first to visualize the sampling trajectory._")
    elif not traj_btn.value:
        _out = mo.md("Click **Generate Trajectory** to watch the Euler solver denoise step by step.")
    else:
        _out = plot_euler_trajectory(
            trained_model,
            int(traj_class_ui.value),
            class_names,
            traj_steps_ui.value,
            num_frames=traj_frames_ui.value,
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
    ## Section 8b — V2 (2D RoPE) Results
    """)
    return


@app.function
def plot_loss_curves_v1_v2(
    train_losses_v1: list,
    val_losses_v1: list,
    train_losses_v2: list,
    val_losses_v2: list,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    e1 = range(1, len(train_losses_v1) + 1)
    e2 = range(1, len(train_losses_v2) + 1)
    ax.plot(e1, train_losses_v1, "b-o", lw=2, ms=4, label="V1 Train (learned pos)")
    ax.plot(e1, val_losses_v1,   "b--s", lw=2, ms=4, label="V1 Val")
    ax.plot(e2, train_losses_v2, "r-o", lw=2, ms=4, label="V2 Train (2D RoPE)")
    ax.plot(e2, val_losses_v2,   "r--s", lw=2, ms=4, label="V2 Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss (MSE)")
    ax.set_title("V1 (learned positional embedding) vs V2 (2D RoPE) training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    train_losses,
    train_losses_v2,
    trained_model,
    trained_model_v2,
    val_losses,
    val_losses_v2,
):
    if trained_model is None and trained_model_v2 is None:
        _out = mo.md("_Train both V1 (Section 5) and V2 (Section 5b) to compare loss curves._")
    elif trained_model is None:
        _out = plot_loss_curve(train_losses_v2, val_losses_v2)
    elif trained_model_v2 is None:
        _out = plot_loss_curve(train_losses, val_losses)
    else:
        _out = plot_loss_curves_v1_v2(train_losses, val_losses, train_losses_v2, val_losses_v2)
    _out
    return


@app.cell
def _(class_names, mo):
    solver_ui_v2 = mo.ui.dropdown(
        options={"Euler": "euler", "Midpoint": "midpoint", "RK4": "rk4"},
        value="Euler",
        label="ODE Solver",
    )
    steps_ui_v2 = mo.ui.slider(5, 200, value=50, step=1, label="Solver Steps")
    class_ui_v2 = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value=class_names[0],
        label="Class to generate",
    )
    sample_btn_v2 = mo.ui.run_button(label="Sample V2")
    compare_btn_v2 = mo.ui.run_button(label="Compare Solvers V2")
    mo.vstack(
        [
            mo.md("### V2 Sampling Controls"),
            mo.hstack([solver_ui_v2, steps_ui_v2, class_ui_v2]),
            mo.hstack([sample_btn_v2, compare_btn_v2]),
        ]
    )
    return (
        class_ui_v2,
        compare_btn_v2,
        sample_btn_v2,
        solver_ui_v2,
        steps_ui_v2,
    )


@app.cell
def _(
    class_names,
    mo,
    sample_btn_v2,
    solver_ui_v2,
    steps_ui_v2,
    trained_model_v2,
):
    if trained_model_v2 is None:
        _out = mo.md("_Train V2 (Section 5b) first to generate samples._")
    elif not sample_btn_v2.value:
        _out = mo.md("Click **Sample V2** to generate a grid of images for every class.")
    else:
        _out = plot_generated_grid(
            trained_model_v2,
            resolve_solver(solver_ui_v2.value),
            steps_ui_v2.value,
            class_names,
            num_per_class=4,
        )
    _out
    return


@app.cell
def _(class_names, class_ui_v2, compare_btn_v2, mo, trained_model_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train V2 (Section 5b) first to compare solvers._")
    elif not compare_btn_v2.value:
        _out = mo.md("Click **Compare Solvers V2** to run Euler / Midpoint / RK4 at several step counts.")
    else:
        _out = plot_solver_comparison(
            trained_model_v2, int(class_ui_v2.value), class_names, [10, 25, 50, 100]
        )
    _out
    return


@app.cell
def _(mo, num_layers_ui_v2, train_losses_v2, trained_model_v2, val_losses_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train the V2 model to see a results summary._")
    else:
        _out = mo.md(
            f"""
            ### V2 Summary

            - Backbone: **DiffusionTransformerV2** (2D RoPE) with `num_layers = {num_layers_ui_v2.value}`,
              `embed_dim=256`, `num_heads=8`, `mlp_dim=512`, patch size `4`.
            - Trained for `{len(train_losses_v2)}` epochs; final train loss
              `{train_losses_v2[-1]:.4f}`, final val loss `{val_losses_v2[-1]:.4f}`.
            - Positional encoding: **2-D RoPE** — row and column grid positions are
              injected into every Q/K projection via rotation; no learnable position
              table is present in the model.
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


@app.cell
def _(mo):
    mo.md("""
    ## Section 9b — Save V2 Trained Model
    """)
    return


@app.cell
def _(mo):
    save_filename_ui_v2 = mo.ui.text(
        value="cifar10_dit_flow_rope2d_v2.safetensors",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn_v2 = mo.ui.run_button(label="Save V2 Model")
    mo.vstack([save_filename_ui_v2, save_model_btn_v2])
    return save_filename_ui_v2, save_model_btn_v2


@app.cell
def _(mo, save_filename_ui_v2, save_model_btn_v2, trained_model_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train the V2 model first (Section 5b) before saving._")
    elif not save_model_btn_v2.value:
        _out = mo.md(
            "Enter a filename and click **Save V2 Model** to write the trained "
            "weights to `models/`."
        )
    else:
        _models_dir_v2 = Path(__file__).resolve().parent.parent / "models"
        _models_dir_v2.mkdir(parents=True, exist_ok=True)
        _save_path_v2 = _models_dir_v2 / save_filename_ui_v2.value
        trained_model_v2.save_weights(str(_save_path_v2))
        _out = mo.md(f"**Saved!** V2 model weights written to `{_save_path_v2}`.")
    _out
    return


if __name__ == "__main__":
    app.run()
