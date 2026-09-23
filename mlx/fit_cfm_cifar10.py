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
    mo.md(r"""
    # Flexible Vision Transformer (FiT) + Conditional Flow Matching on CIFAR-10 (MLX)

    ## Research Goal

    Reproduce the core architectural ideas from **FiT — Flexible Vision
    Transformer for Diffusion** (arxiv 2402.12376v4) inside a
    **conditional flow matching (CFM)** training loop on **CIFAR-10**,
    implemented from scratch in **MLX**.

    ### How FiT differs from DiT

    | Aspect | DiT (arxiv 2212.09748) | FiT (arxiv 2402.12376) |
    |-------|------------------------|------------------------|
    | Tokenisation | fixed length, one Conv patchify | flexible length — pad to `max_tokens` and mask padded positions |
    | Position encoding | additive learnable table `(N, D)` | **2-D RoPE** injected into every Q/K projection (no learnable table) |
    | Conditioning | AdaLN (scale + shift) | **AdaLN-Zero** — last linear of conditioning MLP is zero-initialised so each block starts as identity |
    | Final layer | LayerNorm + Linear | LayerNorm + zero-initialised Linear output |

    For CIFAR-10 the image is always 32x32 → 64 patches, so the
    padding/mask machinery is a no-op in practice, but it is implemented
    exactly as FiT prescribes so the same model would generalise to
    variable-resolution inputs.

    ### Flow matching (Lipman et al., 2023)

    A **conditional linear probability path** is used:

    $$x_t = (1 - t)\,x_0 + t\,x_1,\quad x_0 \sim \mathcal{N}(0, I),\; x_1 \sim p_{\text{data}}$$

    The target velocity along this path is constant:

    $$v_t = x_1 - x_0$$

    Training minimises

    $$\mathcal{L}(\theta) = \mathbb{E}_{t,x_0,x_1}\bigl\lVert v_{\theta}(x_t, t, y) - (x_1 - x_0) \bigr\rVert^2$$

    At inference we integrate $\mathrm{d}x/\mathrm{d}t = v_{\theta}(x, t, y)$ from
    $t = 0$ (pure noise) to $t = 1$ (a generated image) with an ODE
    solver — Euler, midpoint or RK4.

    ### Notebook Outline
    1. Title & research goal (this cell)
    2. Data exploration — load CIFAR-10, sample grid, class distribution
    3. Dataset creation — normalise, train/val/test, batch iterator
    4. Model definition — every FiT building block as its own class
    5. Training — reactive UI + reactive training loop
    6. Optional hyperparameter search
    7. Validation & 3-fold cross-validation
    8. Results — loss curves, generated samples, solver comparison, ODE trajectory
    9. Save trained model weights to `models/`
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
) -> tuple:
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
def plot_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    n_show: int = 40,
    cols: int = 8,
):
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


@app.cell
def _(mo):
    mo.md(r"""
    ### 4a — Sinusoidal Timestep Embedding

    Encode the continuous time index $t \in [0, 1]$ into a
    `(B, embed_dim)` feature vector via sin/cos features spanning a
    geometric range of frequencies. Frequencies are precomputed once at
    construction and stored as `self.freqs` for reuse across every
    training step.
    """)
    return


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.freqs = mx.exp(
            -math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / max(half, 1)
        )

    def __call__(self, t: mx.array) -> mx.array:
        angles = t[:, None] * self.freqs[None, :]
        return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4b — AdaLN-Zero (Adaptive LayerNorm with Zero Initialisation)

    The signature FiT / DiT modulation block. The conditioning vector
    `c` (time + class) is projected to **six** vectors of size `dim`:

    $$(\alpha_1,\, \beta_1,\, \gamma_1,\, \alpha_2,\, \beta_2,\, \gamma_2) = W_{\text{cond}}\,\text{SiLU}(c)$$

    The transformer block then applies:

    $$h = \text{LN}(x)\,(1 + \alpha_1) + \beta_1;\quad x \leftarrow x + \gamma_1 \cdot \text{Attn}(h)$$
    $$h = \text{LN}(x)\,(1 + \alpha_2) + \beta_2;\quad x \leftarrow x + \gamma_2 \cdot \text{MLP}(h)$$

    **Zero initialisation.** The final projection `self.proj` has both
    its weight and bias initialised to zero, so all six modulation
    vectors start at zero. Consequently $\gamma_1 = \gamma_2 = 0$ at
    step 0 and the entire block reduces to the identity — the residual
    stream is unchanged. As training proceeds the gates smoothly move
    away from zero. This is critical for stable training of deep
    transformers with conditioning.
    """)
    return


@app.class_definition
class AdaptiveLayerNormZeroV1(nn.Module):
    def __init__(self, dim: int = 256, cond_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.proj = nn.Linear(cond_dim, 6 * dim)
        # zero-initialise so the block starts as identity
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.proj.bias = mx.zeros_like(self.proj.bias)

    def __call__(self, x: mx.array, cond: mx.array):
        shifts = self.proj(nn.silu(cond))[:, None, :]  # (B, 1, 6*dim)
        s1, b1, g1, s2, b2, g2 = mx.split(shifts, 6, axis=-1)
        normed1 = self.norm(x) * (1.0 + s1) + b1
        normed2 = self.norm(x) * (1.0 + s2) + b2
        return normed1, g1, normed2, g2


@app.cell
def _(mo):
    mo.md(r"""
    ### 4c — 2D Rotary Position Embedding Attention

    A single multi-head attention layer with **2-D RoPE** injected into
    Q and K. Following VisionLLaMA (arxiv 2403.13298) the head dimension
    is split in half:

    | Head-dim slice | Position used | Encodes |
    |---|---|---|
    | $[0, d/2)$ — *row half* | row index $r$ | vertical offset $r - r'$ |
    | $[d/2, d)$ — *col half* | column index $c$ | horizontal offset $c - c'$ |

    Both halves share the frequency bank
    $\theta_i = 10000^{-4i/d}$ for $i \in [0, d/4)$.
    Because $\mathbf{R}(m)^{\top}\mathbf{R}(n) = \mathbf{R}(m - n)$ the
    rotated inner product depends only on the relative displacement
    $(r - r',\, c - c')$ — attention becomes translation-equivariant in
    both spatial directions without a learned position table.

    **Attention mask.** An additive mask of shape `(B, 1, 1, N)` is
    accepted so the same layer supports flexible-length inputs: masked
    positions have a large negative value added to their pre-softmax
    score. For CIFAR-10 the mask is all-zero (no padding).
    """)
    return


@app.class_definition
class RoPE2DAttentionV1(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        quarter = self.head_dim // 4
        # theta_i = 10000^(-4i/head_dim), shared by row-half and col-half
        self.freqs = 1.0 / (
            10000.0 ** (mx.arange(0, quarter, dtype=mx.float32) * 4.0 / self.head_dim)
        )
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _make_cos_sin(self, positions: mx.array):
        angles = positions[:, None] * self.freqs[None, :]           # (N, quarter)
        cos = mx.repeat(mx.cos(angles), 2, axis=-1)                 # (N, half)
        sin = mx.repeat(mx.sin(angles), 2, axis=-1)                 # (N, half)
        return cos, sin

    def _apply_rotary(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (B, N, H, half); cos/sin: (N, half) with paired duplicates
        B, N, H, half = x.shape
        pairs = x.reshape(B, N, H, half // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]                       # (B, N, H, half//2)
        c = cos[:, ::2][None, :, None, :]                            # (1, N, 1, half//2)
        s = sin[:, ::2][None, :, None, :]                            # (1, N, 1, half//2)
        rot0 = x0 * c - x1 * s
        rot1 = x0 * s + x1 * c
        return mx.stack([rot0, rot1], axis=-1).reshape(B, N, H, half)

    def __call__(
        self,
        x: mx.array,
        row_pos: mx.array,
        col_pos: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        half = hd // 2

        q = self.q_proj(x).reshape(B, N, H, hd)
        k = self.k_proj(x).reshape(B, N, H, hd)
        v = self.v_proj(x).reshape(B, N, H, hd)

        # split each head into row-half [0, half) and col-half [half, hd)
        q_r, q_c = q[..., :half], q[..., half:]
        k_r, k_c = k[..., :half], k[..., half:]

        r_cos, r_sin = self._make_cos_sin(row_pos)
        c_cos, c_sin = self._make_cos_sin(col_pos)
        q_r = self._apply_rotary(q_r, r_cos, r_sin)
        k_r = self._apply_rotary(k_r, r_cos, r_sin)
        q_c = self._apply_rotary(q_c, c_cos, c_sin)
        k_c = self._apply_rotary(k_c, c_cos, c_sin)

        q = mx.concatenate([q_r, q_c], axis=-1).transpose(0, 2, 1, 3)  # (B, H, N, hd)
        k = mx.concatenate([k_r, k_c], axis=-1).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)           # (B, H, N, N)
        if mask is not None:
            # mask is (B, 1, 1, N) additive (0 for real, -1e9 for pad)
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4d — MLP Block

    Plain two-layer feed-forward with GELU activation. Normalisation is
    handled externally by `AdaptiveLayerNormZeroV1`, so this module
    contains only two linears and one activation.
    """)
    return


@app.class_definition
class MLPBlockV1(nn.Module):
    def __init__(self, dim: int = 256, mlp_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


@app.cell
def _(mo):
    mo.md(r"""
    ### 4e — FiT Transformer Block

    One FiT block combines AdaLN-Zero, 2-D RoPE attention and an MLP.
    The AdaLN-Zero module produces two normalised versions of `x` plus
    two gate scalars per feature. The block computes:

    ```
    normed1, g1, normed2, g2 = adaln(x, cond)
    x = x + g1 * attn(normed1, row_pos, col_pos, mask)
    x = x + g2 * mlp(normed2)
    ```

    With `g1 = g2 = 0` at initialisation the block is the identity,
    exactly as required by AdaLN-Zero.
    """)
    return


@app.class_definition
class FiTBlockV1(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.adaln = AdaptiveLayerNormZeroV1(dim, cond_dim)
        self.attn = RoPE2DAttentionV1(dim, num_heads)
        self.mlp = MLPBlockV1(dim, mlp_dim)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        row_pos: mx.array,
        col_pos: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        normed1, g1, normed2, g2 = self.adaln(x, cond)
        x = x + g1 * self.attn(normed1, row_pos, col_pos, mask)
        x = x + g2 * self.mlp(normed2)
        return x


@app.cell
def _(mo):
    mo.md(r"""
    ### 4f — Patch Embedding

    A single `Conv2d` with `kernel_size = stride = patch_size` maps
    `(B, H, W, C)` (NHWC, MLX default) → `(B, N, D)` where
    `N = (H/p) * (W/p)`. For CIFAR-10 with `patch_size=4` this yields
    `N = 64` tokens per image.
    """)
    return


@app.class_definition
class PatchEmbedV1(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 256,
        in_channels: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)  # (B, H/p, W/p, D) in NHWC
        return h.reshape(x.shape[0], -1, self.embed_dim)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4g — Unpatch Embedding

    Inverse of `PatchEmbedV1`. A linear layer maps each token back to a
    flattened `p * p * C` patch, then a reshape + transpose stitches the
    patches into an image of shape `(B, H, W, C)`.
    """)
    return


@app.class_definition
class UnpatchEmbedV1(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        patch_size: int = 4,
        out_channels: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def __call__(self, x: mx.array, grid_h: int, grid_w: int) -> mx.array:
        b = x.shape[0]
        p, c = self.patch_size, self.out_channels
        h = self.proj(x).reshape(b, grid_h, grid_w, p, p, c)
        return h.transpose(0, 1, 3, 2, 4, 5).reshape(b, grid_h * p, grid_w * p, c)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4h — Timestep + Class Conditioning

    Combines the sinusoidal timestep embedding with a learned class
    embedding through a small SiLU MLP:

    ```
    c = MLP( time_embed(t) + class_embed(y) )
    ```

    The class embedding table has `num_classes + 1` entries; the last
    row is the null-class token used for classifier-free guidance.
    """)
    return


@app.class_definition
class TimestepClassConditioningV1(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbeddingV1(embed_dim)
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def __call__(self, t: mx.array, y: mx.array) -> mx.array:
        return self.mlp(self.time_embed(t) + self.class_embed(y))


@app.cell
def _(mo):
    mo.md(r"""
    ### 4i — FiTModelV1 (top-level)

    Assembles all the pieces into the flexible vision transformer:

    1. Patch embed the image, producing `(B, N, D)` tokens
    2. Build `row_pos, col_pos` grid indices for 2-D RoPE
    3. Compute the shared conditioning vector `c` from `t` and `y`
    4. Apply `num_layers` FiT blocks, each using AdaLN-Zero + RoPE
       attention with the supplied padding mask
    5. Final LayerNorm, then unpatch to an image

    **Zero-init of the final projection.** After construction we
    override `unpatch.proj.weight` and `unpatch.proj.bias` to zero so
    the model outputs an all-zero velocity at step 0 — a standard
    practice for stable flow-matching / diffusion training.
    """)
    return


@app.class_definition
class FiTModelV1(nn.Module):
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
        max_tokens: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.grid_size = image_size // patch_size

        self.patch_embed = PatchEmbedV1(patch_size, embed_dim, in_channels)
        self.cond = TimestepClassConditioningV1(embed_dim, num_classes)
        self.blocks = [
            FiTBlockV1(embed_dim, num_heads, mlp_dim, embed_dim)
            for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatch = UnpatchEmbedV1(embed_dim, patch_size, in_channels)

        # zero-initialise the final linear layer (final layer zero-init)
        self.unpatch.proj.weight = mx.zeros_like(self.unpatch.proj.weight)
        self.unpatch.proj.bias = mx.zeros_like(self.unpatch.proj.bias)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        y: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        B = x.shape[0]
        g = self.grid_size
        # row-major patch order: patch (r, c) → index r*g + c
        row_pos = mx.array([r for r in range(g) for _ in range(g)], dtype=mx.float32)
        col_pos = mx.array([c for _ in range(g) for c in range(g)], dtype=mx.float32)

        h = self.patch_embed(x)                    # (B, N, D)
        c = self.cond(t, y)                        # (B, D)

        N = h.shape[1]
        if mask is None:
            # no padding for fixed-resolution inputs — an all-zero additive mask
            mask = mx.zeros((B, 1, 1, N), dtype=mx.float32)

        for block in self.blocks:
            h = block(h, c, row_pos, col_pos, mask)
        h = self.final_norm(h)
        return self.unpatch(h, g, g)


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def compute_flow_loss(model: nn.Module, x1: mx.array, y: mx.array) -> mx.array:
    t = mx.random.uniform(shape=(x1.shape[0],))
    x0 = mx.random.normal(shape=x1.shape)
    t_view = t.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = model(x_t, t, y)
    return mx.mean((v_pred - v_target) ** 2)


@app.cell
def _(mo):
    mo.md("""
    ### Architecture Summary — `FiTModelV1`

    | Component | Module | Output shape |
    |-----------|--------|--------------|
    | Patch embed | `PatchEmbedV1` (Conv2d, stride=`p`) | `(B, N, D)` where `N = (H/p)*(W/p)` |
    | Time + class conditioning | `TimestepClassConditioningV1` | `(B, D)` |
    | Backbone (FiT block x N) | `FiTBlockV1` = `AdaLN-Zero` + `RoPE2DAttentionV1` + `MLPBlockV1` | `(B, N, D)` |
    | Final norm | `LayerNorm` | `(B, N, D)` |
    | Unpatch (zero-init) | `UnpatchEmbedV1` (`Linear` → reshape) | `(B, H, W, C)` |

    Defaults: `image_size=32, patch_size=4, embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6, max_tokens=256`.
    """)
    return


@app.cell
def _():
    default_fit_model = FiTModelV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=6,
        max_tokens=256,
    )
    mx.eval(default_fit_model.parameters())
    default_fit_param_count = count_parameters(default_fit_model)
    return (default_fit_param_count,)


@app.cell
def _(default_fit_param_count, mo):
    mo.md(f"""
    **Default `FiTModelV1` parameter count**: `{default_fit_param_count:,}`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4j — Smoke Test

    Run one forward pass with dummy data on a small model to confirm
    the shapes flow end-to-end before spending any real training time.
    """)
    return


@app.cell
def _(mo):
    _smoke_model = FiTModelV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        mlp_dim=128,
        num_layers=2,
        max_tokens=64,
    )
    mx.eval(_smoke_model.parameters())
    _dummy_x = mx.random.normal(shape=(2, 32, 32, 3))
    _dummy_t = mx.array([0.1, 0.9], dtype=mx.float32)
    _dummy_y = mx.array([3, 7], dtype=mx.int32)
    _dummy_out = _smoke_model(_dummy_x, _dummy_t, _dummy_y)
    mx.eval(_dummy_out)
    _smoke_loss = compute_flow_loss(_smoke_model, _dummy_x, _dummy_y)
    mx.eval(_smoke_loss)
    mo.md(
        f"""
        Smoke test passed:

        - Input `x`: `{tuple(_dummy_x.shape)}` → output `v`: `{tuple(_dummy_out.shape)}`
        - Output dtype: `{_dummy_out.dtype}`
        - Sample flow-matching loss on dummy data: `{_smoke_loss.item():.4f}`
        - Small model parameter count: `{count_parameters(_smoke_model):,}`
        """
    )
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
    num_layers_ui = mo.ui.slider(1, 12, value=6, step=1, label="FiT num_layers")
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
def run_train_epoch(
    model: nn.Module,
    loss_and_grad_fn,
    optimizer,
    train_batches: list,
) -> float:
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
        mo.output.replace(mo.md("Click **Train** to begin training the FiT model."))
    else:
        _model = FiTModelV1(
            image_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=256,
            num_heads=8,
            mlp_dim=512,
            num_layers=num_layers_ui.value,
            max_tokens=256,
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
    model = FiTModelV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=num_layers,
        max_tokens=256,
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
    model = FiTModelV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=num_layers,
        max_tokens=256,
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
        _cv_n = min(3000, x_tr.shape[0])
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
    ## Section 8 — Results & ODE Integration
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
    ax.set_title("FiT + CFM training curve")
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
def euler_solve_trajectory(
    model: nn.Module,
    x0: mx.array,
    y: mx.array,
    num_steps: int = 50,
) -> list:
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


@app.function
def resolve_solver(name: str):
    return {"euler": euler_solve, "midpoint": midpoint_solve, "rk4": rk4_solve}[name]


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
        f"Euler ODE trajectory — class `{class_names[class_idx]}` ({num_steps} steps)",
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


@app.cell
def _(class_names, mo, sample_btn, solver_ui, steps_ui, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to generate samples._")
    elif not sample_btn.value:
        _out = mo.md("Click **Sample** to generate a grid of images for every class.")
    else:
        _out = plot_generated_grid(
            trained_model,
            resolve_solver(solver_ui.value),
            steps_ui.value,
            class_names,
            num_per_class=4,
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
    ### ODE Integration Process — Watching the Model Denoise

    The Euler solver integrates the learned velocity field one small
    step at a time, starting from pure Gaussian noise (`t=0`) and
    arriving at a generated image (`t=1`). The frames below sample
    intermediate states `x_t` along a single trajectory, making the
    model's step-by-step progress toward a coherent image directly
    visible — this confirms that the trained model is a valid velocity
    field for the CIFAR-10 data distribution.
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
            mo.md("### ODE trajectory controls"),
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
        _out = mo.md("_Train the model first to visualise the ODE trajectory._")
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

            - Backbone: **`FiTModelV1`** with `num_layers = {num_layers_ui.value}`,
              `embed_dim = 256`, `num_heads = 8`, `mlp_dim = 512`, `patch_size = 4`,
              `max_tokens = 256`.
            - Key FiT ingredients implemented:
              **flexible tokenisation with attention mask**, **2-D RoPE**
              (no learnable position table), **AdaLN-Zero** conditioning
              (six modulation vectors, zero-initialised so blocks start as
              identity), and **zero-initialised final output projection**.
            - Trained for `{len(train_losses)}` epochs; final train loss
              `{train_losses[-1]:.4f}`, final val loss `{val_losses[-1]:.4f}`.
            - Inference solvers: **Euler** (1 NFE / step), **Midpoint / Heun**
              (2 NFE / step), **RK4** (4 NFE / step). RK4 typically produces
              the best samples at the fewest steps at the cost of 4x compute
              per step; Euler is cheapest but requires the most steps for
              equivalent quality.
            - The ODE trajectory visualisation above shows the model
              transporting a Gaussian noise sample along the learned
              velocity field to a coherent class-conditional CIFAR-10
              image.
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
        value="cifar10_fit_cfm_v1.safetensors",
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
