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
    from mlx.data.datasets import load_fashion_mnist
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
    # Class-Conditional DiT Flow Matching on Fashion-MNIST — MLX

    ## Research Goal

    Train a **Diffusion Transformer (DiT)** on Fashion-MNIST using a
    **class-conditional Flow Matching (CFM)** objective in Apple's
    **MLX** framework. Given an integer class id `y ∈ {0..9}`, the trained
    model can generate a novel 28×28 grayscale image of that garment class
    by numerically integrating a learned continuous-time velocity field
    with the **Euler ODE solver**.

    ### Method summary — rectified / OT-linear conditional flow matching

    | Symbol | Meaning |
    |--------|---------|
    | `x1`     | a real image (data sample) |
    | `x0`     | pure Gaussian noise, `~ N(0, I)`, same shape as `x1` |
    | `t`      | continuous "time" `~ U(0, 1)` sampled per example |
    | `x_t`    | interpolation `(1 - t) x0 + t x1` (linear/OT path) |
    | `v*`     | target velocity `x1 - x0` (constant along the linear path) |
    | `v_θ(x_t, t, y)` | DiT prediction |
    | Loss   | `E[ || v_θ(x_t, t, y) - (x1 - x0) ||^2 ]` |

    **Classifier-free guidance (CFG)** — during training the class label is
    replaced with a learned "null" class id (`num_classes = 10`) with
    probability `p_uncond` (default `0.1`) so the same network models
    `p(x)` and `p(x | y)` simultaneously. At sampling time the two
    velocities are combined:

    ```
    v = v_uncond + guidance_scale * (v_cond - v_uncond)
    ```

    **Euler sampler** — the *only* solver implemented here. Start at `t=0`
    with `x0 ~ N(0, I)`, march forward with fixed-step size `dt = 1 /
    num_steps`:

    ```
    x_{t+dt} = x_t + dt * v_θ(x_t, t, y)
    ```

    Terminate at `t=1` — the final `x_1` is the generated image
    (denormalized from `[-1, 1]` to `[0, 1]` for display).

    ### Data convention
    Images are normalized once at the data-pipeline stage to `[-1, 1]`
    (from `uint8` in `[0, 255]`). All model inputs, targets, and outputs
    stay in `[-1, 1]` throughout training and sampling; only when we
    **display** an image do we shift back to `[0, 1]` and clip.

    ### CRITICAL training-stability fixes applied (from an earlier session)
    - **`optim.AdamW(..., bias_correction=True)`** — mlx's default of
      `False` under-corrects Adam's early second-moment estimate and (on
      this exact machine, verified empirically on the sibling DDPM
      notebook) can leave a from-scratch DiT stuck predicting a constant
      velocity forever. This is not optional.
    - **Zero-initialized final projection layer** — the DiT's final
      `Linear(hidden → patch_pixels)` starts at `0`, so the model
      initially predicts zero velocity (a stable, well-defined prior).
    - **Gradient norm clipping** — implemented manually via
      `mlx.utils.tree_flatten` / `tree_map` (mlx has no built-in
      `clip_grad_norm`).
    - **Real held-out validation split** — carved out via
      `.perm(indices)` on disjoint index permutations; **not** two
      streams built from the same `.shuffle()` buffer (which silently
      re-reads training data — this exact bug was found and fixed this
      session).

    ### Notebook outline
    1. Title & research goal (this cell)
    2. Data exploration — Fashion-MNIST sample grid + class distribution
    3. Dataset creation — disjoint train/val/test iterators
    4. Model definition — DiT building blocks + `DiffusionTransformerV1`
    5. Training — `train_flow_matching_model(...)` with UI knobs
    6. Optional hyperparameter search
    7. Validation & 5-fold cross-validation
    8. Results — loss curves, Euler denoising progression, multi-sample gen
    9. Save trained model
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 2 — Data Exploration
    """)
    return


@app.function
def get_fashion_mnist_class_names() -> list:
    """Canonical Fashion-MNIST class-name list, indexed 0..9."""
    return [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


@app.cell
def _():
    fashion_mnist_root = str(
        (Path(__file__).resolve().parent.parent / "data" / "fashion_mnist")
    )
    train_ds = load_fashion_mnist(root=fashion_mnist_root, train=True)
    test_ds = load_fashion_mnist(root=fashion_mnist_root, train=False)
    class_names = get_fashion_mnist_class_names()
    return class_names, test_ds, train_ds


@app.cell
def _(mo, test_ds, train_ds):
    mo.md(f"""
    ### Dataset overview

    Fashion-MNIST is loaded as an `mlx.data` Buffer. Each sample is a
    dict with:

    - `image` — `uint8` array shaped `(28, 28, 1)` (channels-last)
    - `label` — scalar `uint8` in `[0, 9]`

    | Split | Size |
    |-------|------|
    | Train (raw) | {len(train_ds):,} |
    | Test | {len(test_ds):,} |
    """)
    return


@app.function
def plot_sample_grid(
    dataset,
    class_names: list,
    n_show: int = 40,
    rows: int = 5,
    cols: int = 8,
):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i in range(n_show):
        sample = dataset[i]
        img = np.array(sample["image"]).squeeze()
        label = int(np.array(sample["label"]).item())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].set_title(class_names[label], fontsize=8)
        axes[r, c].axis("off")
    fig.suptitle("Fashion-MNIST training samples", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, train_ds):
    plot_sample_grid(train_ds, class_names)
    return


@app.function
def plot_class_distribution(dataset, class_names: list):
    num_classes = len(class_names)
    labels = np.array(
        [int(np.array(dataset[i]["label"]).item()) for i in range(len(dataset))]
    )
    counts = np.bincount(labels, minlength=num_classes)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(num_classes), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Fashion-MNIST training-set class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(int(c)), ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, train_ds):
    plot_class_distribution(train_ds, class_names)
    return


@app.cell
def _(mo, train_ds):
    split_val_fraction = 0.15
    n_val_preview = int(round(len(train_ds) * split_val_fraction))
    n_train_preview = len(train_ds) - n_val_preview
    mo.md(
        f"""
    ### Planned splits (see Section 3)

    | Split | Size |
    |-------|------|
    | Train | {n_train_preview:,} |
    | Val | {n_val_preview:,} |
    | Test | 10,000 |

    Val is carved out of the 60k train set via a **seeded, disjoint index
    permutation** (`.perm(...)`) — not from a second shuffled stream.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def normalize_image_to_signed_range(x):
    """uint8 in [0, 255] -> float32 in [-1, 1]. Consistent with the flow-matching data convention."""
    return x.astype("float32") / 127.5 - 1.0


@app.function
def denormalize_for_display(x: np.ndarray) -> np.ndarray:
    """Signed [-1, 1] -> unsigned [0, 1] with clipping, for imshow."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


@app.function
def make_datasets(
    train_ds,
    test_ds,
    batch_size: int = 128,
    val_fraction: float = 0.15,
    split_seed: int = 0,
):
    """Build disjoint train / val / test streaming iterators in [-1, 1]."""
    n_total = len(train_ds)
    n_val = int(round(n_total * val_fraction))
    split_rng = np.random.default_rng(split_seed)
    shuffled_indices = split_rng.permutation(n_total)
    val_indices = shuffled_indices[:n_val].tolist()
    train_indices = shuffled_indices[n_val:].tolist()
    train_subset = train_ds.perm(train_indices)
    val_subset = train_ds.perm(val_indices)
    train_iter = (
        train_subset
        .shuffle()
        .to_stream()
        .key_transform("image", normalize_image_to_signed_range)
        .batch(batch_size)
    )
    val_iter = (
        val_subset
        .to_stream()
        .key_transform("image", normalize_image_to_signed_range)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", normalize_image_to_signed_range)
        .batch(batch_size)
    )
    return train_iter, val_iter, test_iter


@app.cell
def _(test_ds, train_ds):
    inspection_train_iter, inspection_val_iter, inspection_test_iter = make_datasets(
        train_ds, test_ds, batch_size=128
    )
    return (inspection_train_iter,)


@app.cell
def _(inspection_train_iter, mo):
    inspection_train_iter.reset()
    sample_batch = next(inspection_train_iter)
    sample_images = mx.array(sample_batch["image"])
    sample_labels = mx.array(sample_batch["label"])
    mo.md(
        f"""
    ### First batch shape check

    - `image` shape: `{tuple(sample_images.shape)}` — dtype `{sample_images.dtype}`
    - `label` shape: `{tuple(sample_labels.shape)}` — dtype `{sample_labels.dtype}`
    - image min/max: `{float(sample_images.min()):.3f} / {float(sample_images.max()):.3f}`
      (expected close to `-1.0 / 1.0`)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 4 — Model Definition

    Every module below is defined in its own `@app.class_definition` cell.
    All modules follow the strict rules: version suffix on the class name,
    typed parameters with defaults, no globals, composition.

    Building order:

    1. `SinusoidalTimeEmbeddingV1` — sinusoidal embedding of continuous `t ∈ [0, 1]` (`t` is scaled by `time_scale=1000` for frequency resolution)
    2. `TimeEmbedderV1` — small MLP on top of the sinusoidal embedding
    3. `LabelEmbedderV1` — `Embedding(num_classes + 1, hidden_size)`; index `num_classes` is the CFG null class
    4. `PatchEmbedV1` — Conv2d patchify (patch_size=4 → 7×7 = 49 tokens for 28×28)
    5. `DiTAttentionV1`, `DiTMlpV1` — standard transformer sub-layers
    6. `AdaLayerNormV1` — adaLN-Zero modulation producing 6 vectors (shift/scale/gate ×2)
    7. `DiTBlockV1` — one transformer block combining the above
    8. `DiTFinalLayerV1` — adaLN + linear back to patch pixels, **zero-initialized**
    9. `DiffusionTransformerV1` — top-level model stacking `depth` `DiTBlockV1` in a real loop; exposes `velocity_with_cfg(x, t, y, guidance_scale)` for CFG
    10. `count_parameters`, `clip_grad_norm` — training utilities

    **Design note on modularity**: only training hyperparameters (lr,
    batch_size, epochs, weight_decay, p_uncond, guidance_scale at
    sampling) are exposed as `mo.ui` controls. Architecture knobs
    (`hidden_size`, `depth`, `num_heads`, `patch_size`, `num_classes`)
    are constructor defaults — enough to swap architectures by editing
    one line, but not enough to turn the notebook UI into a knob wall.
    """)
    return


@app.class_definition
class SinusoidalTimeEmbeddingV1(nn.Module):
    """Sinusoidal embedding of a continuous timestep t in [0, 1].

    t is multiplied by `time_scale` (default 1000) *before* the sinusoidal
    frequencies so the embedding has good resolution across [0, 1] —
    otherwise sin/cos are near-constant across the tiny range 0..1.
    """

    def __init__(
        self,
        dim: int = 256,
        max_period: int = 10000,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.time_scale = time_scale

    def __call__(self, t: mx.array) -> mx.array:
        half = self.dim // 2
        freqs = mx.exp(
            -math.log(self.max_period)
            * mx.arange(0, half, dtype=mx.float32)
            / max(half, 1)
        )
        args = (t.astype(mx.float32) * self.time_scale)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb


@app.class_definition
class TimeEmbedderV1(nn.Module):
    """Sinusoidal embed -> Linear -> SiLU -> Linear."""

    def __init__(
        self,
        hidden_size: int = 192,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbeddingV1(dim=frequency_embedding_size)
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, t: mx.array) -> mx.array:
        return self.fc2(nn.silu(self.fc1(self.sinusoidal(t))))


@app.class_definition
class LabelEmbedderV1(nn.Module):
    """Embedding table with an extra slot for the CFG null class.

    Index `num_classes` is reserved as the "unconditional" null class.
    CFG dropout (randomly replacing the true label with this index during
    training) is applied *externally* in the loss function, not here.
    """

    def __init__(self, num_classes: int = 10, hidden_size: int = 192):
        super().__init__()
        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    def __call__(self, labels: mx.array) -> mx.array:
        return self.embedding(labels)


@app.class_definition
class PatchEmbedV1(nn.Module):
    """Conv2d patchify. Input NHWC (B, H, W, C) -> tokens (B, N, hidden_size)."""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_size: int = 192,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)  # (B, H/P, W/P, hidden_size)  -- MLX is NHWC
        B, Hg, Wg, C = h.shape
        return h.reshape(B, Hg * Wg, C)


@app.class_definition
class DiTAttentionV1(nn.Module):
    """Standard multi-head self-attention, manual implementation for MLX."""

    def __init__(self, hidden_size: int = 192, num_heads: int = 6):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = mx.matmul(q, mx.swapaxes(k, -2, -1)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)              # (B, H, N, D)
        out = mx.swapaxes(out, 1, 2)          # (B, N, H, D)
        out = out.reshape(B, N, C)
        return self.proj(out)


@app.class_definition
class DiTMlpV1(nn.Module):
    """Two-layer GELU MLP, ratio 4 by default."""

    def __init__(self, hidden_size: int = 192, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


@app.class_definition
class AdaLayerNormV1(nn.Module):
    """adaLN-Zero: produce 6 modulation vectors (shift/scale/gate for attn + mlp) from c.

    The output linear is zero-initialized so at t=0 the model behaves as an
    identity residual stack. Standard DiT / ADM practice.
    """

    def __init__(self, hidden_size: int = 192):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.mod = nn.Linear(hidden_size, 6 * hidden_size)
        # adaLN-Zero: zero-init so blocks start as identity.
        self.mod.weight = mx.zeros_like(self.mod.weight)
        self.mod.bias = mx.zeros_like(self.mod.bias)

    def __call__(self, x: mx.array, c: mx.array):
        params = self.mod(nn.silu(c))                      # (B, 6*H)
        parts = mx.split(params, 6, axis=-1)               # 6x (B, H)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
        x_norm = self.norm(x)
        return (
            x_norm,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )


@app.class_definition
class DiTBlockV1(nn.Module):
    """One DiT transformer block: adaLN-Zero modulated attn + adaLN-Zero modulated mlp."""

    def __init__(
        self,
        hidden_size: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.adaln = AdaLayerNormV1(hidden_size=hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.attn = DiTAttentionV1(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = DiTMlpV1(hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        (
            x_attn_norm,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln(x, c)
        x_attn_mod = x_attn_norm * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.attn(x_attn_mod)
        x_mlp_norm = self.norm_mlp(x)
        x_mlp_mod = x_mlp_norm * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(x_mlp_mod)
        return x


@app.class_definition
class DiTFinalLayerV1(nn.Module):
    """adaLN + linear back to patch pixels. Linear weights zero-initialized."""

    def __init__(
        self,
        hidden_size: int = 192,
        patch_size: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.mod = nn.Linear(hidden_size, 2 * hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        # Zero-init modulation and final linear so the model initially
        # predicts zero velocity everywhere -- the required stability trick.
        self.mod.weight = mx.zeros_like(self.mod.weight)
        self.mod.bias = mx.zeros_like(self.mod.bias)
        self.linear.weight = mx.zeros_like(self.linear.weight)
        self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        shift, scale = mx.split(self.mod(nn.silu(c)), 2, axis=-1)
        x = self.norm_final(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.linear(x)


@app.class_definition
class DiffusionTransformerV1(nn.Module):
    """Class-conditional Diffusion Transformer for flow matching (velocity prediction).

    Input images are NHWC (B, H, W, C) in [-1, 1]. `t` is a continuous scalar
    in [0, 1] per example. `y` is an integer class id in [0, num_classes] --
    index `num_classes` is the CFG null class.
    """

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        patch_size: int = 4,
        hidden_size: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        num_classes: int = 10,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = PatchEmbedV1(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        # Learned positional embedding -- mlx.nn tracks mx.array attributes as parameters.
        self.pos_embed = mx.random.normal((1, self.num_patches, hidden_size)) * 0.02
        self.t_embedder = TimeEmbedderV1(hidden_size=hidden_size)
        self.y_embedder = LabelEmbedderV1(num_classes=num_classes, hidden_size=hidden_size)
        # Real loop over `depth` blocks -- not `depth` hand-copied blocks.
        self.blocks = [
            DiTBlockV1(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(depth)
        ]
        self.final_layer = DiTFinalLayerV1(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
        )

    def unpatchify(self, tokens: mx.array) -> mx.array:
        """(B, N, P*P*C) -> (B, H, W, C) in NHWC."""
        B = tokens.shape[0]
        p = self.patch_size
        c = self.out_channels
        h = w = self.grid_size
        x = tokens.reshape(B, h, w, p, p, c)
        # (B, h, w, p, p, c) -> (B, h, p, w, p, c)
        x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
        return x.reshape(B, h * p, w * p, c)

    def __call__(self, x: mx.array, t: mx.array, y: mx.array) -> mx.array:
        h = self.patch_embed(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y)
        for block in self.blocks:
            h = block(h, c)
        h = self.final_layer(h, c)
        return self.unpatchify(h)

    def velocity_with_cfg(
        self,
        x: mx.array,
        t: mx.array,
        y: mx.array,
        guidance_scale: float = 1.0,
    ) -> mx.array:
        """Classifier-free-guided velocity: v_uncond + w * (v_cond - v_uncond)."""
        combined_x = mx.concatenate([x, x], axis=0)
        combined_t = mx.concatenate([t, t], axis=0)
        null_labels = mx.full((y.shape[0],), self.null_class_index, dtype=y.dtype)
        combined_y = mx.concatenate([y, null_labels], axis=0)
        v = self(combined_x, combined_t, combined_y)
        v_cond, v_uncond = mx.split(v, 2, axis=0)
        return v_uncond + guidance_scale * (v_cond - v_uncond)


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def clip_grad_norm(grads, max_norm: float = 1.0):
    """Clip a gradient pytree by its global L2 norm (mlx has no built-in equivalent)."""
    leaves = mlx.utils.tree_flatten(grads)
    total_norm_sq = sum(mx.sum(v.astype(mx.float32) ** 2) for _, v in leaves)
    total_norm = mx.sqrt(total_norm_sq)
    scale = mx.minimum(1.0, max_norm / (total_norm + 1e-6))
    clipped = mlx.utils.tree_map(lambda g: g * scale, grads)
    return clipped, total_norm


@app.function
def compute_flow_matching_loss(
    model: nn.Module,
    x1: mx.array,
    labels: mx.array,
    p_uncond: float = 0.1,
) -> mx.array:
    """CFM loss with CFG label dropout.

    x1: real images already normalized to [-1, 1].
    Samples t ~ U(0, 1), x0 ~ N(0, I); constructs x_t = (1-t)*x0 + t*x1;
    target velocity v* = x1 - x0; loss = mean((v_theta - v*)^2). Class
    labels are replaced by the null class with probability p_uncond.
    """
    B = x1.shape[0]
    t = mx.random.uniform(shape=(B,))
    x0 = mx.random.normal(x1.shape)
    t_expanded = t.reshape(B, 1, 1, 1)
    x_t = (1.0 - t_expanded) * x0 + t_expanded * x1
    v_target = x1 - x0
    drop_mask = mx.random.uniform(shape=(B,)) < p_uncond
    null_label = mx.full((B,), model.null_class_index, dtype=labels.dtype)
    effective_labels = mx.where(drop_mask, null_label, labels)
    v_pred = model(x_t, t, effective_labels)
    return mx.mean((v_pred - v_target) ** 2)


@app.function
def train_flow_matching_model(
    model: nn.Module,
    train_iter,
    val_iter,
    num_epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    p_uncond: float = 0.1,
    max_grad_norm: float = 1.0,
    progress_callback=None,
):
    """Full training loop for the DiT + CFM model.

    CRITICAL: passes `bias_correction=True` to AdamW -- mlx defaults to
    False, which was empirically shown to leave the model stuck.
    """
    optimizer = optim.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        bias_correction=True,
    )

    def _loss_fn(m, xb, yb):
        return compute_flow_matching_loss(m, xb, yb, p_uncond)

    loss_and_grad = nn.value_and_grad(model, _loss_fn)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_iter.reset()
        total = 0.0
        n = 0
        for batch in train_iter:
            xb = mx.array(batch["image"], dtype=mx.float32)
            yb = mx.array(batch["label"])
            loss, grads = loss_and_grad(model, xb, yb)
            grads, _gn = clip_grad_norm(grads, max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            total += float(loss.item())
            n += 1
        tl = total / max(n, 1)
        train_losses.append(tl)

        val_iter.reset()
        vt = 0.0
        vn = 0
        for batch in val_iter:
            xb = mx.array(batch["image"], dtype=mx.float32)
            yb = mx.array(batch["label"])
            vloss = compute_flow_matching_loss(model, xb, yb, p_uncond)
            mx.eval(vloss)
            vt += float(vloss.item())
            vn += 1
        vl = vt / max(vn, 1)
        val_losses.append(vl)

        if progress_callback is not None:
            progress_callback(epoch, num_epochs, tl, vl)

    return model, train_losses, val_losses


@app.function
def euler_sample(
    model: nn.Module,
    labels: mx.array,
    image_shape: tuple = (28, 28, 1),
    num_steps: int = 50,
    guidance_scale: float = 3.0,
    return_trajectory: bool = False,
    num_snapshots: int = 8,
):
    """Euler ODE sampler for flow matching (the only solver implemented).

    Starts at t=0 with x0 ~ N(0, I) and marches to t=1 with fixed dt =
    1/num_steps: `x_{t+dt} = x_t + dt * v_theta(x_t, t, y)`.
    Classifier-free-guided velocity via `model.velocity_with_cfg`.

    When `return_trajectory=True` this also records `num_snapshots`
    intermediate states (as numpy arrays) at evenly spaced integration
    step indices, including t=0 (pure noise) and t=1 (final image),
    used by the "denoising progression" panel in Section 8.
    """
    B = int(labels.shape[0])
    x = mx.random.normal((B,) + image_shape)
    dt = 1.0 / num_steps
    trajectory = {}
    t_values = {}
    if return_trajectory:
        snap_ks = {
            int(round(i * num_steps / max(num_snapshots - 1, 1)))
            for i in range(num_snapshots)
        }
        snap_ks.add(0)
        snap_ks.add(num_steps)
        mx.eval(x)
        if 0 in snap_ks:
            trajectory[0] = np.array(x)
            t_values[0] = 0.0
    for step in range(num_steps):
        t_val = step * dt
        t_batch = mx.full((B,), t_val, dtype=mx.float32)
        v = model.velocity_with_cfg(x, t_batch, labels, guidance_scale)
        x = x + dt * v
        mx.eval(x)
        k = step + 1
        if return_trajectory and k in snap_ks:
            trajectory[k] = np.array(x)
            t_values[k] = k * dt
    if return_trajectory:
        ks = sorted(trajectory.keys())
        traj_list = [trajectory[k] for k in ks]
        tval_list = [t_values[k] for k in ks]
        return np.array(x), traj_list, tval_list
    return np.array(x)


@app.function
def evaluate_model(
    model: nn.Module,
    data_iter,
    p_uncond: float = 0.1,
) -> float:
    """Mean flow-matching MSE loss over a data iterator (single pass)."""
    data_iter.reset()
    total = 0.0
    n = 0
    for batch in data_iter:
        xb = mx.array(batch["image"], dtype=mx.float32)
        yb = mx.array(batch["label"])
        loss = compute_flow_matching_loss(model, xb, yb, p_uncond)
        mx.eval(loss)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Architecture — `DiffusionTransformerV1`

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Patch embed | `PatchEmbedV1` (Conv2d 1→192, k=4, s=4) | `(B, 49, 192)` |
    | + Positional embed | learned `pos_embed` param | `(B, 49, 192)` |
    | Time embed | `TimeEmbedderV1` (sinusoidal → MLP) | `(B, 192)` |
    | Label embed | `LabelEmbedderV1` (`Embedding(11, 192)`) | `(B, 192)` |
    | Conditioning `c` | `t_emb + y_emb` | `(B, 192)` |
    | DiT blocks | `DiTBlockV1` × 6 (adaLN-Zero + attn + mlp) | `(B, 49, 192)` |
    | Final layer | `DiTFinalLayerV1` (adaLN + Linear 192→16) | `(B, 49, 16)` |
    | Unpatchify | reshape/transpose | `(B, 28, 28, 1)` |

    Patch size **4** divides 28 evenly, giving `7 × 7 = 49` tokens per
    image. The final `Linear(192 → 16)` (16 = 4·4·1 pixels per patch) is
    zero-initialized so `v_θ(x_0, t=0, y) ≈ 0` at the start of training.

    **Loss**: `E[|| v_θ(x_t, t, y) - (x_1 - x_0) ||^2]` with `y` dropped
    to the null class (index 10) with prob. `p_uncond = 0.1`.

    **Sampler**: Euler ODE from `t=0` (pure noise) to `t=1` (image) with
    `dt = 1 / num_steps`, CFG applied at every step.
    """)
    return


@app.cell
def _(mo):
    reference_model = DiffusionTransformerV1()
    mx.eval(reference_model.parameters())
    reference_param_count = count_parameters(reference_model)
    mo.md(
        f"**Reference `DiffusionTransformerV1` parameter count**: "
        f"`{reference_param_count:,}`"
    )
    return (reference_param_count,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 5 — Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "2e-4": 2e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="5e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(
        options=[32, 64, 128, 256], value=128, label="Batch Size"
    )
    epochs_ui = mo.ui.slider(1, 50, value=10, step=1, label="Epochs")
    p_uncond_ui = mo.ui.dropdown(
        options={"0.05": 0.05, "0.1": 0.1, "0.2": 0.2},
        value="0.1",
        label="p_uncond (CFG dropout)",
    )
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="0",
        label="Weight Decay",
    )
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Training hyperparameters"),
            mo.hstack([lr_ui, bs_ui, epochs_ui]),
            mo.hstack([p_uncond_ui, wd_ui]),
            train_btn,
        ]
    )
    return bs_ui, epochs_ui, lr_ui, p_uncond_ui, train_btn, wd_ui


@app.cell
def _(bs_ui, test_ds, train_ds):
    train_iter, val_iter, test_iter = make_datasets(
        train_ds, test_ds, batch_size=int(bs_ui.value)
    )
    return test_iter, train_iter, val_iter


@app.cell
def _(
    epochs_ui,
    lr_ui,
    mo,
    p_uncond_ui,
    train_btn,
    train_iter,
    val_iter,
    wd_ui,
):
    train_losses = []
    val_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(
            mo.md("Click **Train** to begin DiT + CFM training on Fashion-MNIST.")
        )
    else:
        run_model = DiffusionTransformerV1()
        mx.eval(run_model.parameters())

        def _progress(epoch, total, tl, vl):
            mo.output.replace(
                mo.md(
                    f"**Epoch {epoch + 1}/{total}** — "
                    f"train CFM MSE: `{tl:.4f}`  |  val CFM MSE: `{vl:.4f}`"
                )
            )

        run_model, train_losses, val_losses = train_flow_matching_model(
            run_model,
            train_iter,
            val_iter,
            num_epochs=int(epochs_ui.value),
            lr=float(lr_ui.value),
            weight_decay=float(wd_ui.value),
            p_uncond=float(p_uncond_ui.value),
            max_grad_norm=1.0,
            progress_callback=_progress,
        )
        trained_model = run_model
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train: `{train_losses[-1]:.4f}` "
                f"| final val: `{val_losses[-1]:.4f}`."
            )
        )
    return train_losses, trained_model, val_losses


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 6 — Hyperparameter Search (Optional)
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(
        label="Enable Hyperparameter Search", value=False
    )
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_search(
    train_ds,
    test_ds,
    lr_grid: list,
    wd_grid: list,
    n_epochs: int = 2,
    subset_size: int = 3000,
    batch_size: int = 128,
    p_uncond: float = 0.1,
    seed: int = 0,
    progress_callback=None,
):
    """Small grid search over lr x weight_decay on a train subset for speed."""
    rng = np.random.default_rng(seed)
    subset_indices = rng.permutation(len(train_ds))[:subset_size].tolist()
    subset_ds = train_ds.perm(subset_indices)
    results = []
    total_configs = len(lr_grid) * len(wd_grid)
    idx = 0
    for lr in lr_grid:
        for wd in wd_grid:
            idx += 1
            train_iter, val_iter, _ = make_datasets(
                subset_ds, test_ds, batch_size=batch_size, val_fraction=0.2, split_seed=seed
            )
            model = DiffusionTransformerV1()
            mx.eval(model.parameters())
            model, _tl, _vl = train_flow_matching_model(
                model,
                train_iter,
                val_iter,
                num_epochs=n_epochs,
                lr=lr,
                weight_decay=wd,
                p_uncond=p_uncond,
                max_grad_norm=1.0,
            )
            final_val = float(_vl[-1]) if _vl else float("nan")
            results.append(
                {
                    "lr": lr,
                    "weight_decay": wd,
                    "val_loss": round(final_val, 4),
                    "epochs": n_epochs,
                }
            )
            if progress_callback is not None:
                progress_callback(idx, total_configs, lr, wd, final_val)
    results.sort(key=lambda r: r["val_loss"])
    return results


@app.cell
def _(hp_search_cb, mo, test_ds, train_ds):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    lr_grid = [2e-4, 5e-4]
    wd_grid = [0.0, 1e-4]

    def _hp_progress(idx, total, lr, wd, vl):
        mo.output.replace(
            mo.md(
                f"HP config {idx}/{total} — lr={lr}, wd={wd} → val_loss={vl:.4f}"
            )
        )

    hp_results = run_hp_search(
        train_ds,
        test_ds,
        lr_grid=lr_grid,
        wd_grid=wd_grid,
        n_epochs=2,
        subset_size=3000,
        batch_size=128,
        p_uncond=0.1,
        seed=0,
        progress_callback=_hp_progress,
    )
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 7 — Validation & Cross-Validation
    """)
    return


@app.cell
def _(mo, test_iter, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to evaluate on the test set._")
    else:
        test_cfm_loss = evaluate_model(trained_model, test_iter, p_uncond=0.1)
        _out = mo.md(
            f"""
    ### Test-set CFM loss

    | Metric | Value |
    |--------|-------|
    | Test CFM MSE (1 pass, `p_uncond=0.1`) | `{test_cfm_loss:.4f}` |

    Flow-matching loss on a held-out set correlates weakly with sample
    quality (as it does for diffusion). Treat it as a sanity check that
    training reached a reasonable level, then use the Euler-sampler
    denoising progression (Section 8) as the *qualitative* diagnostic.
    """
        )
    _out
    return


@app.function
def cross_validate_flow_matching(
    train_ds,
    k: int = 5,
    batch_size: int = 128,
    num_epochs: int = 2,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    subset_size: int = 5000,
    p_uncond: float = 0.1,
    seed: int = 42,
    progress_callback=None,
):
    """k-fold CV on a small train subset for CPU feasibility.

    Each fold builds a fresh model, trains for `num_epochs`, and reports
    the CFM val loss on that fold's held-out indices.
    """
    rng = np.random.default_rng(seed)
    subset = rng.permutation(len(train_ds))[:subset_size].tolist()
    fold_size = subset_size // k
    fold_losses = []
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_idx = subset[val_start:val_end]
        train_idx = subset[:val_start] + subset[val_end:]
        train_sub = train_ds.perm(train_idx)
        val_sub = train_ds.perm(val_idx)
        train_iter = (
            train_sub
            .shuffle()
            .to_stream()
            .key_transform("image", normalize_image_to_signed_range)
            .batch(batch_size)
        )
        val_iter = (
            val_sub
            .to_stream()
            .key_transform("image", normalize_image_to_signed_range)
            .batch(batch_size)
        )
        model = DiffusionTransformerV1()
        mx.eval(model.parameters())
        model, _tl, _vl = train_flow_matching_model(
            model,
            train_iter,
            val_iter,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            p_uncond=p_uncond,
            max_grad_norm=1.0,
        )
        vl = evaluate_model(model, val_iter, p_uncond=p_uncond)
        fold_losses.append(vl)
        if progress_callback is not None:
            progress_callback(fold, k, vl)
    return fold_losses


@app.cell
def _(mo):
    cv_run_cb = mo.ui.checkbox(
        label="Run 5-fold Cross-Validation (small subset, few epochs each)",
        value=False,
    )
    cv_run_cb
    return (cv_run_cb,)


@app.cell
def _(cv_run_cb, mo, train_ds):
    mo.stop(
        not cv_run_cb.value,
        mo.md("_Tick the checkbox above to run 5-fold CV. It trains 5 tiny models._"),
    )

    def _cv_progress(fold, total, vl):
        mo.output.replace(
            mo.md(f"CV fold {fold + 1}/{total} — val CFM MSE: `{vl:.4f}`")
        )

    cv_fold_losses = cross_validate_flow_matching(
        train_ds,
        k=5,
        batch_size=128,
        num_epochs=2,
        lr=5e-4,
        weight_decay=0.0,
        subset_size=5000,
        p_uncond=0.1,
        seed=42,
        progress_callback=_cv_progress,
    )
    cv_mean = float(np.mean(cv_fold_losses))
    cv_std = float(np.std(cv_fold_losses))
    cv_results = {
        "fold_losses": cv_fold_losses,
        "mean": cv_mean,
        "std": cv_std,
    }
    _rows = [
        {"fold": i + 1, "val_cfm_mse": round(v, 4)}
        for i, v in enumerate(cv_fold_losses)
    ]
    _rows.append(
        {
            "fold": "mean ± std",
            "val_cfm_mse": f"{cv_mean:.4f} ± {cv_std:.4f}",
        }
    )
    mo.output.replace(mo.ui.table(_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 8 — Results
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8a — Loss curves
    """)
    return


@app.function
def plot_loss_curve(train_losses: list, val_losses: list | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train CFM MSE")
    if val_losses:
        ax.plot(epochs, val_losses, "r-s", lw=2, ms=4, label="Val CFM MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CFM MSE loss")
    ax.set_title("DiT Flow-Matching Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses, val_losses):
    if not train_losses:
        _out = mo.md("_Train first to see the loss curve._")
    else:
        _out = plot_loss_curve(train_losses, val_losses)
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8b — Denoising progression (Euler sampler)

    This panel visually confirms the model is actually removing noise.
    We call `euler_sample(..., return_trajectory=True)` for one chosen
    class, and plot the state at evenly spaced integration steps from
    `t=0` (pure Gaussian noise) to `t=1` (final generated image). A
    working model shows the image sharpen into a recognizable garment
    of the requested class as `t → 1`. A broken model just returns
    static all the way through.
    """)
    return


@app.cell
def _(class_names, mo):
    traj_class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value="Ankle boot",
        label="Class",
    )
    traj_guidance_ui = mo.ui.slider(
        1.0, 8.0, value=3.0, step=0.5, label="CFG guidance scale"
    )
    traj_steps_ui = mo.ui.slider(
        10, 200, value=50, step=10, label="Euler steps"
    )
    traj_snapshots_ui = mo.ui.slider(
        4, 12, value=8, step=1, label="Number of snapshots"
    )
    traj_btn = mo.ui.run_button(label="Show Denoising Progression")
    mo.vstack(
        [
            mo.md("#### Denoising-progression controls"),
            mo.hstack([traj_class_ui, traj_guidance_ui]),
            mo.hstack([traj_steps_ui, traj_snapshots_ui]),
            traj_btn,
        ]
    )
    return (
        traj_btn,
        traj_class_ui,
        traj_guidance_ui,
        traj_snapshots_ui,
        traj_steps_ui,
    )


@app.function
def plot_euler_trajectory(
    trajectory: list,
    t_values: list,
    class_name: str,
    sample_index: int = 0,
):
    n = len(trajectory)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.4))
    if n == 1:
        axes = np.array([axes])
    for col, (img_arr, t_val) in enumerate(zip(trajectory, t_values)):
        img = denormalize_for_display(img_arr[sample_index]).squeeze()
        axes[col].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[col].set_title(f"t = {t_val:.2f}", fontsize=9)
        axes[col].axis("off")
    fig.suptitle(
        f"Euler ODE denoising progression — class '{class_name}'  (t=0: noise → t=1: image)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(
    class_names,
    mo,
    trained_model,
    traj_btn,
    traj_class_ui,
    traj_guidance_ui,
    traj_snapshots_ui,
    traj_steps_ui,
):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to enable the denoising progression view._")
    elif not traj_btn.value:
        _out = mo.md("Choose a class and click **Show Denoising Progression** to run the Euler sampler.")
    else:
        _class_idx = int(traj_class_ui.value)
        _labels = mx.full((1,), _class_idx, dtype=mx.int32)
        _final, _traj, _tvals = euler_sample(
            trained_model,
            _labels,
            image_shape=(28, 28, 1),
            num_steps=int(traj_steps_ui.value),
            guidance_scale=float(traj_guidance_ui.value),
            return_trajectory=True,
            num_snapshots=int(traj_snapshots_ui.value),
        )
        _out = plot_euler_trajectory(_traj, _tvals, class_names[_class_idx])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8c — Multiple sample generation

    Sample a full batch of images of the chosen class in one Euler-sampler
    call. Each generation is independent (fresh noise per sample).
    """)
    return


@app.cell
def _(class_names, mo):
    gen_class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value="Sneaker",
        label="Class",
    )
    gen_guidance_ui = mo.ui.slider(
        1.0, 8.0, value=3.0, step=0.5, label="CFG guidance scale"
    )
    gen_n_ui = mo.ui.slider(
        1, 16, value=8, step=1, label="Number of samples"
    )
    gen_steps_ui = mo.ui.slider(
        10, 200, value=50, step=10, label="Euler steps"
    )
    gen_btn = mo.ui.run_button(label="Generate")
    mo.vstack(
        [
            mo.md("#### Multi-sample generation controls"),
            mo.hstack([gen_class_ui, gen_guidance_ui]),
            mo.hstack([gen_n_ui, gen_steps_ui]),
            gen_btn,
        ]
    )
    return gen_btn, gen_class_ui, gen_guidance_ui, gen_n_ui, gen_steps_ui


@app.function
def plot_generated_grid(
    images: np.ndarray,
    class_name: str,
    guidance_scale: float,
):
    n = images.shape[0]
    cols = min(n, 8)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes_flat = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n:
            img = denormalize_for_display(images[i]).squeeze()
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    fig.suptitle(
        f"Generated '{class_name}' — CFG={guidance_scale:.1f}, n={n}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(
    class_names,
    gen_btn,
    gen_class_ui,
    gen_guidance_ui,
    gen_n_ui,
    gen_steps_ui,
    mo,
    trained_model,
):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to generate samples._")
    elif not gen_btn.value:
        _out = mo.md("Choose a class and click **Generate** to sample a batch of images.")
    else:
        _class_idx = int(gen_class_ui.value)
        _n = int(gen_n_ui.value)
        _labels = mx.full((_n,), _class_idx, dtype=mx.int32)
        _images = euler_sample(
            trained_model,
            _labels,
            image_shape=(28, 28, 1),
            num_steps=int(gen_steps_ui.value),
            guidance_scale=float(gen_guidance_ui.value),
            return_trajectory=False,
        )
        _out = plot_generated_grid(
            _images, class_names[_class_idx], float(gen_guidance_ui.value)
        )
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8d — Summary
    """)
    return


@app.cell
def _(
    epochs_ui,
    lr_ui,
    mo,
    p_uncond_ui,
    reference_param_count,
    train_losses,
    val_losses,
):
    if not train_losses:
        _summary = mo.md(
            f"""
    - **Framework**: MLX (Apple Silicon)
    - **Dataset**: Fashion-MNIST 28×28×1, normalized to `[-1, 1]`
      (60k train → 85% train / 15% val, 10k test)
    - **Model**: `DiffusionTransformerV1` — 6-layer DiT, hidden 192,
      6 heads, patch 4 (49 tokens), adaLN-Zero conditioning
      (`{reference_param_count:,}` params)
    - **Objective**: OT-linear conditional flow matching
      (`x_t = (1-t) x0 + t x1`, target `v* = x1 - x0`)
    - **Sampler**: Euler ODE, CFG guidance combining
      `v = v_uncond + w * (v_cond - v_uncond)`
    - **Not yet trained** — click **Train** in Section 5 to populate this
      summary with actual final losses.
    """
        )
    else:
        _summary = mo.md(
            f"""
    - **Framework**: MLX (Apple Silicon)
    - **Dataset**: Fashion-MNIST 28×28×1, normalized to `[-1, 1]`
    - **Model**: `DiffusionTransformerV1` — 6-layer DiT, hidden 192,
      6 heads, patch 4 (49 tokens), adaLN-Zero (`{reference_param_count:,}` params)
    - **Objective**: OT-linear conditional flow matching, MSE loss on velocity
    - **Training config used**: {int(epochs_ui.value)} epoch(s),
      lr={lr_ui.value}, p_uncond={p_uncond_ui.value}
    - **Final train CFM MSE**: `{train_losses[-1]:.4f}`
    - **Final val CFM MSE**: `{val_losses[-1]:.4f}`
    - **Sampler**: Euler ODE — the only solver implemented; CFG applied at every step

    ### Notes
    - `optim.AdamW` is constructed with **`bias_correction=True`** on
      every call (in `train_flow_matching_model`, in `run_hp_search`, and
      in `cross_validate_flow_matching` via `train_flow_matching_model`).
    - The final projection layer of the DiT and every adaLN modulation
      output are **zero-initialized**, so the model starts by predicting
      zero velocity everywhere.
    - Gradient global-norm clipping (`max_grad_norm=1.0`) is applied
      every optimizer step.
    - Validation uses **disjoint indices** via `.perm(...)` — never two
      `.shuffle()` streams built from the same buffer.
    """
        )
    _summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 9 — Save Trained Model

    Persist the trained `DiffusionTransformerV1` weights to the project's
    `models/` directory. The `.safetensors` extension is chosen by
    default (natively supported by MLX; portable).
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="fashion_mnist_dit_cfm_v1.safetensors",
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
            "Enter a filename and click **Save Model** to write the "
            "trained weights to `models/`."
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
