import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import math
    from pathlib import Path
    import time
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
    # Classifier-Free Guided DDPM & DiT on CIFAR-10 — MLX

    ## Research Goal

    Train **two** class-conditional Denoising Diffusion Probabilistic Models
    on CIFAR-10 using Apple's **MLX** framework, sharing the same
    epsilon-parameterization diffusion machinery:

    - **Backbone A: Convolutional UNet** (`ConvUNetDenoiserV1`) — the standard
      DDPM backbone with encoder/decoder + skip connections, GroupNorm + SiLU
      residual blocks, sinusoidal time embedding and additive class
      conditioning (a la Ho et al. 2020).
    - **Backbone B: Diffusion Transformer** (`DiffusionTransformerV1`) — a
      DiT-style patch transformer with **adaLN-Zero** blocks (Peebles & Xie
      2023), operating directly on pixel patches (no VAE latent) since
      CIFAR-10 is only 32x32.

    Both networks are trained with **classifier-free guidance (CFG)**: with
    probability `p_uncond=0.1` the class label is replaced by a null token so
    the same network learns `epsilon_theta(x_t,t,y)` and
    `epsilon_theta(x_t,t,null)`. At sampling time we combine them via
    `eps = eps_uncond + w * (eps_cond - eps_uncond)`.

    Two interchangeable **samplers** are implemented and apply to either
    backbone unmodified:

    1. **DDPM ancestral sampler** — full `T=1000`-step reverse process.
    2. **DDIM sampler** — deterministic (`eta=0`) or near-deterministic
       accelerated sampler with configurable step count.

    ### Method summary
    - Linear beta schedule from `1e-4` to `2e-2` over `T=1000` steps
    - Images normalized to `[-1, 1]` float32 (standard for diffusion training)
    - Null class index = `10` (10 classes + 1 null token = 11 entries)
    - Batched CFG forward: conditional + unconditional concatenated
      along batch dim, split, then combined

    ### Notebook Outline
    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation
    4. Model definition (schedule, UNet, DiT, samplers)
    5. Training (choose which backbone(s) to train)
    6. Optional hyperparameter search
    7. Held-out test evaluation + k-fold cross-validation
    8. Interactive conditional sampling + results
    9. Save trained model weights
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


@app.cell
def _():
    train_ds = load_cifar10(root="../data/cifar10", train=True)
    test_ds = load_cifar10(root="../data/cifar10", train=False)
    return test_ds, train_ds


@app.cell
def _(mo, test_ds, train_ds):
    mo.md(f"""
    ### Dataset overview

    CIFAR-10 is loaded as an `mlx.data` Buffer. Each sample is a dict with:

    - `image` — `uint8` array shaped `(32, 32, 3)` (channels-last, RGB)
    - `label` — scalar `uint8` in `[0, 9]`

    Class names (label id -> name): {", ".join(f"{i}={n}" for i, n in enumerate(cifar10_class_names()))}.

    | Split | Size |
    |-------|------|
    | Train | {len(train_ds):,} |
    | Test  | {len(test_ds):,} |
    """)
    return


@app.function
def plot_sample_grid(dataset, n_show: int = 40, rows: int = 5, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    names = cifar10_class_names()
    for i in range(n_show):
        sample = dataset[i]
        img = np.array(sample["image"])
        label = int(np.array(sample["label"]).item())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(names[label], fontsize=8)
        axes[r, c].axis("off")
    fig.suptitle("CIFAR-10 training samples (class name as title)", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_sample_grid(train_ds)
    return


@app.function
def plot_class_distribution(dataset, num_classes: int = 10):
    labels = np.array(
        [int(np.array(dataset[i]["label"]).item()) for i in range(len(dataset))]
    )
    counts = np.bincount(labels, minlength=num_classes)
    names = cifar10_class_names()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(num_classes), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 training-set class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(int(c)), ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_class_distribution(train_ds)
    return


@app.cell
def _(mo, test_ds, train_ds):
    mo.md(f"""
    ### Split plan (see Section 3)

    | Split | Size | Notes |
    |-------|------|-------|
    | Train | {len(train_ds):,} | Used for both training and k-fold CV (Section 7) |
    | Test  | {len(test_ds):,} | Held-out set for final MSE denoising evaluation |

    No separate validation split is needed — DDPM training is unsupervised in
    the loss sense (class labels are conditioning inputs, not targets).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def normalize_to_signed_unit(x):
    # DDPM/DiT convention: predict epsilon on images in [-1, 1] rather than
    # [0, 1]. This differs from the MNIST sibling notebook (which used [0, 1])
    # because CIFAR-10 pixel-space diffusion trains more stably in [-1, 1].
    return x.astype("float32") / 127.5 - 1.0


@app.function
def make_datasets(train_ds, test_ds, batch_size: int = 128):
    train_iter = (
        train_ds
        .shuffle()
        .to_stream()
        .key_transform("image", normalize_to_signed_unit)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", normalize_to_signed_unit)
        .batch(batch_size)
    )
    return train_iter, test_iter


@app.cell
def _(test_ds, train_ds):
    inspection_train_iter, inspection_test_iter = make_datasets(
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
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition

    In order below:

    1. `DiffusionScheduleV1` — precomputed linear beta schedule
    2. `sinusoidal_timestep_embedding` — shared sinusoidal `t` embedding
    3. `classifier_free_guidance` — CFG combination formula (shared)
    4. `q_sample` — forward diffusion `x_t | x_0`
    5. UNet backbone building blocks + `ConvUNetDenoiserV1`
    6. DiT backbone building blocks + `DiffusionTransformerV1`
    7. Loss + train-epoch helpers
    8. `ddpm_sample` and `ddim_sample` — two interchangeable samplers
    """)
    return


@app.class_definition
class DiffusionScheduleV1:
    """Precomputed diffusion coefficients for a linear beta schedule."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = mx.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = mx.cumprod(alphas)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas = mx.sqrt(alphas)
        self.sqrt_alphas_cumprod = mx.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - alphas_cumprod)
        self.sqrt_betas = mx.sqrt(betas)


@app.function
def sinusoidal_timestep_embedding(t: mx.array, emb_dim: int = 256) -> mx.array:
    """Vaswani-style sinusoidal embedding of shape (B, emb_dim) from int t (B,)."""
    half_dim = emb_dim // 2
    freq_scale = math.log(10000.0) / max(half_dim - 1, 1)
    freqs = mx.exp(-mx.arange(half_dim, dtype=mx.float32) * freq_scale)
    t_float = t.astype(mx.float32)
    args = t_float[:, None] * freqs[None, :]
    return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


@app.function
def classifier_free_guidance(
    eps_cond: mx.array,
    eps_uncond: mx.array,
    guidance_scale: float,
) -> mx.array:
    """CFG combination: `eps = eps_uncond + w * (eps_cond - eps_uncond)`.

    Equivalent to the paper form `(1+w)*eps_cond - w*eps_uncond`. `w=0`
    recovers unguided sampling. Shared by DDPM and DDIM samplers and by
    both backbones.
    """
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@app.function
def gather_schedule_value(schedule_array: mx.array, t: mx.array) -> mx.array:
    """Index a schedule tensor of shape (T,) by an integer tensor of shape (B,)."""
    return mx.take(schedule_array, t, axis=0)


@app.function
def q_sample(
    x0: mx.array,
    t: mx.array,
    noise: mx.array,
    schedule: "DiffusionScheduleV1",
) -> mx.array:
    """Forward process: `x_t = sqrt(alphabar_t) * x0 + sqrt(1 - alphabar_t) * noise`."""
    sqrt_ab = gather_schedule_value(schedule.sqrt_alphas_cumprod, t)
    sqrt_one_minus_ab = gather_schedule_value(schedule.sqrt_one_minus_alphas_cumprod, t)
    sqrt_ab = sqrt_ab[:, None, None, None]
    sqrt_one_minus_ab = sqrt_one_minus_ab[:, None, None, None]
    return sqrt_ab * x0 + sqrt_one_minus_ab * noise


@app.class_definition
class ResidualConvBlockV1(nn.Module):
    """Conv 3x3 -> GN -> SiLU -> Conv 3x3 -> GN -> SiLU with additive cond + skip."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        cond_dim: int = 256,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels, pytorch_compatible=True)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        h = self.conv1(x)
        h = self.norm1(h)
        cond_bias = self.cond_proj(nn.silu(cond))
        h = h + cond_bias[:, None, None, :]
        h = nn.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = nn.silu(h)
        residual = x if self.skip is None else self.skip(x)
        return h + residual


@app.class_definition
class DownBlockV1(nn.Module):
    """Encoder level: two ResidualConvBlocks then a stride-2 conv downsample."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
        cond_dim: int = 256,
        num_groups: int = 8,
        downsample: bool = True,
    ):
        super().__init__()
        self.res1 = ResidualConvBlockV1(in_channels, out_channels, cond_dim, num_groups)
        self.res2 = ResidualConvBlockV1(out_channels, out_channels, cond_dim, num_groups)
        if downsample:
            self.down = nn.Conv2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1,
            )
        else:
            self.down = None

    def __call__(self, x: mx.array, cond: mx.array):
        h = self.res1(x, cond)
        h = self.res2(h, cond)
        skip = h
        if self.down is not None:
            h = self.down(h)
        return h, skip


@app.class_definition
class UpBlockV1(nn.Module):
    """Decoder level: ConvTranspose2d upsample -> concat skip -> two ResidualConvBlocks."""

    def __init__(
        self,
        in_channels: int = 256,
        skip_channels: int = 256,
        out_channels: int = 128,
        cond_dim: int = 256,
        num_groups: int = 8,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1,
        )
        self.res1 = ResidualConvBlockV1(
            in_channels + skip_channels, out_channels, cond_dim, num_groups
        )
        self.res2 = ResidualConvBlockV1(out_channels, out_channels, cond_dim, num_groups)

    def __call__(self, x: mx.array, skip: mx.array, cond: mx.array) -> mx.array:
        h = self.up(x)
        h = mx.concatenate([h, skip], axis=-1)
        h = self.res1(h, cond)
        h = self.res2(h, cond)
        return h


@app.class_definition
class ConvUNetDenoiserV1(nn.Module):
    """Convolutional UNet noise-prediction backbone for CIFAR-10 DDPM.

    Spatial trajectory (H=W): 32 -> 16 -> 8 -> 4 (bottleneck) -> 8 -> 16 -> 32.
    Channels: base -> 2*base -> 4*base -> 4*base at bottleneck.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 4),
        time_emb_dim: int = 256,
        num_classes: int = 10,
        num_groups: int = 8,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.time_emb_dim = time_emb_dim

        ch = [base_channels * m for m in channel_mults]

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        self.stem = nn.Conv2d(image_channels, ch[0], kernel_size=3, padding=1)

        self.down1 = DownBlockV1(ch[0], ch[0], time_emb_dim, num_groups, downsample=True)
        self.down2 = DownBlockV1(ch[0], ch[1], time_emb_dim, num_groups, downsample=True)
        self.down3 = DownBlockV1(ch[1], ch[2], time_emb_dim, num_groups, downsample=True)

        self.mid_res1 = ResidualConvBlockV1(ch[2], ch[3], time_emb_dim, num_groups)
        self.mid_res2 = ResidualConvBlockV1(ch[3], ch[3], time_emb_dim, num_groups)

        self.up1 = UpBlockV1(ch[3], ch[2], ch[2], time_emb_dim, num_groups)
        self.up2 = UpBlockV1(ch[2], ch[1], ch[1], time_emb_dim, num_groups)
        self.up3 = UpBlockV1(ch[1], ch[0], ch[0], time_emb_dim, num_groups)

        self.out_norm = nn.GroupNorm(num_groups, ch[0], pytorch_compatible=True)
        self.out_conv = nn.Conv2d(ch[0], image_channels, kernel_size=1)

    def encode_condition(self, t: mx.array, labels: mx.array) -> mx.array:
        t_emb = sinusoidal_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(labels)
        return t_emb + c_emb

    def __call__(self, x: mx.array, t: mx.array, labels: mx.array) -> mx.array:
        cond = self.encode_condition(t, labels)
        h = self.stem(x)
        h, skip1 = self.down1(h, cond)
        h, skip2 = self.down2(h, cond)
        h, skip3 = self.down3(h, cond)
        h = self.mid_res1(h, cond)
        h = self.mid_res2(h, cond)
        h = self.up1(h, skip3, cond)
        h = self.up2(h, skip2, cond)
        h = self.up3(h, skip1, cond)
        h = self.out_norm(h)
        h = nn.silu(h)
        return self.out_conv(h)


@app.class_definition
class TimestepEmbedderV1(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP (DiT-style)."""

    def __init__(self, hidden_size: int = 256, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def __call__(self, t: mx.array) -> mx.array:
        freq_emb = sinusoidal_timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq_emb)


@app.class_definition
class MultiHeadSelfAttentionV1(nn.Module):
    """Standard multi-head self-attention (no masking, seq2seq over patches)."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 8):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.proj(out)


@app.class_definition
class DiTBlockV1(nn.Module):
    """DiT block with adaLN-Zero conditioning (Peebles & Xie 2023).

    Two sub-layers (MHSA and MLP) each with:
      - LayerNorm (elementwise_affine=False)
      - scale/shift modulation from cond `c`: (1+gamma)*LN(x) + beta
      - residual gate `alpha` (zero-initialized) on the sub-layer output
    Final adaLN linear is zero-initialized so each block starts as identity.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, affine=False)
        self.attn = MultiHeadSelfAttentionV1(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # 6 modulation params per block: (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.ada_ln_mod = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # adaLN-Zero: initialize the final linear to zero so alpha=gamma=beta=0
        # and each block starts as identity.
        self.ada_ln_mod.weight = mx.zeros_like(self.ada_ln_mod.weight)
        self.ada_ln_mod.bias = mx.zeros_like(self.ada_ln_mod.bias)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.ada_ln_mod(nn.silu(c))
        g1, b1, a1, g2, b2, a2 = mx.split(mod, 6, axis=-1)
        g1 = g1[:, None, :]
        b1 = b1[:, None, :]
        a1 = a1[:, None, :]
        g2 = g2[:, None, :]
        b2 = b2[:, None, :]
        a2 = a2[:, None, :]
        h = self.norm1(x)
        h = (1.0 + g1) * h + b1
        h = x + a1 * self.attn(h)
        h2 = self.norm2(h)
        h2 = (1.0 + g2) * h2 + b2
        return h + a2 * self.mlp(h2)


@app.class_definition
class DiTFinalLayerV1(nn.Module):
    """AdaLN-modulated LayerNorm + Linear to (patch_size^2 * out_channels)."""

    def __init__(self, hidden_size: int = 256, patch_size: int = 4, out_channels: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size, affine=False)
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.ada_ln_mod = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        # adaLN-Zero on final layer as well (gamma=beta=0 at init).
        self.ada_ln_mod.weight = mx.zeros_like(self.ada_ln_mod.weight)
        self.ada_ln_mod.bias = mx.zeros_like(self.ada_ln_mod.bias)
        # Zero the output projection so initial epsilon prediction is zero
        # (paper recommendation for stability).
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.proj.bias = mx.zeros_like(self.proj.bias)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.ada_ln_mod(nn.silu(c))
        g, b = mx.split(mod, 2, axis=-1)
        g = g[:, None, :]
        b = b[:, None, :]
        h = self.norm(x)
        h = (1.0 + g) * h + b
        return self.proj(h)


@app.class_definition
class DiffusionTransformerV1(nn.Module):
    """Diffusion Transformer (DiT) backbone with adaLN-Zero conditioning.

    Operates directly on 32x32x3 pixel patches (no VAE). Sequence length is
    (32/patch_size)^2. Class conditioning uses a learned Embedding table
    over `num_classes + 1` entries where index `num_classes` is the null
    token for classifier-free guidance — matches the UNet convention so both
    backbones share the same CFG plumbing.
    """

    def __init__(
        self,
        image_size: int = 32,
        image_channels: int = 3,
        patch_size: int = 4,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must divide by patch_size"
        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        patch_dim = patch_size * patch_size * image_channels
        self.patch_embed = nn.Linear(patch_dim, hidden_size, bias=True)
        # Learned positional embedding (simple, works well at this scale).
        self.pos_embed = mx.random.normal((1, self.num_patches, hidden_size)) * 0.02

        self.time_embedder = TimestepEmbedderV1(hidden_size, hidden_size)
        self.class_emb = nn.Embedding(num_classes + 1, hidden_size)

        self.blocks = [
            DiTBlockV1(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ]
        self.final_layer = DiTFinalLayerV1(hidden_size, patch_size, image_channels)

    def patchify(self, x: mx.array) -> mx.array:
        """(B, H, W, C) -> (B, N, patch_size*patch_size*C) with N = (H/p)^2."""
        b, h, w, c = x.shape
        p = self.patch_size
        gh, gw = h // p, w // p
        x = x.reshape(b, gh, p, gw, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, gh * gw, p * p * c)

    def unpatchify(self, x: mx.array) -> mx.array:
        """(B, N, p*p*C) -> (B, H, W, C)."""
        b, n, _ = x.shape
        p = self.patch_size
        c = self.image_channels
        gh = gw = self.grid_size
        x = x.reshape(b, gh, gw, p, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, gh * p, gw * p, c)

    def __call__(self, x: mx.array, t: mx.array, labels: mx.array) -> mx.array:
        h = self.patchify(x)
        h = self.patch_embed(h) + self.pos_embed
        t_emb = self.time_embedder(t)
        y_emb = self.class_emb(labels)
        c = t_emb + y_emb
        for block in self.blocks:
            h = block(h, c)
        h = self.final_layer(h, c)
        return self.unpatchify(h)


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def compute_ddpm_loss(
    model: nn.Module,
    x0: mx.array,
    labels: mx.array,
    schedule: "DiffusionScheduleV1",
    p_uncond: float = 0.1,
) -> mx.array:
    """CFG training loss (MSE on predicted noise). Works with either backbone."""
    batch_size = x0.shape[0]
    t = mx.random.randint(0, schedule.num_timesteps, shape=(batch_size,))
    noise = mx.random.normal(x0.shape)
    x_t = q_sample(x0, t, noise, schedule)
    drop_mask = mx.random.uniform(shape=(batch_size,)) < p_uncond
    null_label = mx.full((batch_size,), model.null_class_index, dtype=labels.dtype)
    effective_labels = mx.where(drop_mask, null_label, labels)
    pred_noise = model(x_t, t, effective_labels)
    return mx.mean((pred_noise - noise) ** 2)


@app.function
def run_train_epoch(
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    optimizer,
    train_iter,
    p_uncond: float = 0.1,
) -> float:
    def loss_fn(model_, x0, labels):
        return compute_ddpm_loss(model_, x0, labels, schedule, p_uncond)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    epoch_loss = 0.0
    n_batches = 0
    train_iter.reset()
    for batch in train_iter:
        x0 = mx.array(batch["image"], dtype=mx.float32)
        labels = mx.array(batch["label"])
        loss, grads = loss_and_grad(model, x0, labels)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += float(loss.item())
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


@app.function
def evaluate_model(
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    data_iter,
    p_uncond: float = 0.0,
) -> float:
    """Mean epsilon-prediction MSE over `data_iter`. `p_uncond=0` = conditional-only."""
    total = 0.0
    n = 0
    data_iter.reset()
    for batch in data_iter:
        x0 = mx.array(batch["image"], dtype=mx.float32)
        labels = mx.array(batch["label"])
        loss = compute_ddpm_loss(model, x0, labels, schedule, p_uncond)
        mx.eval(loss)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@app.function
def batched_cfg_forward(
    model: nn.Module,
    x: mx.array,
    t_batch: mx.array,
    labels: mx.array,
    null_labels: mx.array,
    guidance_scale: float,
) -> mx.array:
    """Run cond+uncond in ONE forward pass by concatenating along batch dim.

    Cheaper than two separate forward passes. Both DDPM and DDIM samplers
    call this helper so CFG is centralized.
    """
    x_pair = mx.concatenate([x, x], axis=0)
    t_pair = mx.concatenate([t_batch, t_batch], axis=0)
    y_pair = mx.concatenate([labels, null_labels], axis=0)
    eps_pair = model(x_pair, t_pair, y_pair)
    eps_cond, eps_uncond = mx.split(eps_pair, 2, axis=0)
    return classifier_free_guidance(eps_cond, eps_uncond, guidance_scale)


@app.function
def ddpm_sample(
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    labels: mx.array,
    image_shape: tuple = (32, 32, 3),
    guidance_scale: float = 3.0,
    progress_callback=None,
) -> mx.array:
    """DDPM ancestral sampling with CFG — full `T=1000` reverse process.

    Signature matches `ddim_sample` so the two are interchangeable; both
    work with either `ConvUNetDenoiserV1` or `DiffusionTransformerV1`.
    """
    batch_size = int(labels.shape[0])
    x = mx.random.normal((batch_size,) + image_shape)
    null_labels = mx.full((batch_size,), model.null_class_index, dtype=labels.dtype)
    total_steps = schedule.num_timesteps
    for step in range(total_steps - 1, -1, -1):
        t_batch = mx.full((batch_size,), step, dtype=mx.int32)
        eps = batched_cfg_forward(
            model, x, t_batch, labels, null_labels, guidance_scale
        )
        beta_t = schedule.betas[step]
        sqrt_alpha_t = schedule.sqrt_alphas[step]
        sqrt_one_minus_ab_t = schedule.sqrt_one_minus_alphas_cumprod[step]
        mean = (x - (beta_t / sqrt_one_minus_ab_t) * eps) / sqrt_alpha_t
        if step > 0:
            noise = mx.random.normal(x.shape)
            x = mean + schedule.sqrt_betas[step] * noise
        else:
            x = mean
        mx.eval(x)
        if progress_callback is not None and (step % 100 == 0 or step == 0):
            progress_callback(step, total_steps)
    # Images live in [-1, 1] during diffusion; rescale to [0, 1] for display.
    return mx.clip((x + 1.0) / 2.0, 0.0, 1.0)


@app.function
def ddim_sample(
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    labels: mx.array,
    image_shape: tuple = (32, 32, 3),
    guidance_scale: float = 3.0,
    num_steps: int = 50,
    eta: float = 0.0,
    progress_callback=None,
) -> mx.array:
    """DDIM accelerated sampling with CFG.

    `eta=0.0` = fully deterministic; `eta=1.0` recovers DDPM-like stochasticity.
    Uses `num_steps` uniformly spaced timesteps. Same trained model as DDPM;
    only the reverse procedure differs.
    """
    total_steps = schedule.num_timesteps
    batch_size = int(labels.shape[0])

    step_size = total_steps // num_steps
    sampling_ts = mx.array(
        [min(i * step_size, total_steps - 1) for i in range(1, num_steps + 1)],
        dtype=mx.int32,
    )
    alpha_bar_s = mx.take(schedule.alphas_cumprod, sampling_ts)
    alpha_bar_s_prev = mx.concatenate(
        [schedule.alphas_cumprod[0:1], alpha_bar_s[:-1]]
    )
    sqrt_alpha_bar_s = mx.sqrt(alpha_bar_s)
    sqrt_alpha_bar_s_prev = mx.sqrt(alpha_bar_s_prev)
    sqrt_one_minus_alpha_bar_s = mx.sqrt(1.0 - alpha_bar_s)
    sigma = eta * mx.sqrt(
        (1.0 - alpha_bar_s_prev) / (1.0 - alpha_bar_s)
        * (1.0 - alpha_bar_s / alpha_bar_s_prev)
    )
    dir_coef = mx.sqrt(mx.maximum(1.0 - alpha_bar_s_prev - sigma ** 2, 0.0))
    mx.eval(
        alpha_bar_s, alpha_bar_s_prev, sqrt_alpha_bar_s, sqrt_alpha_bar_s_prev,
        sqrt_one_minus_alpha_bar_s, sigma, dir_coef,
    )

    null_labels = mx.full((batch_size,), model.null_class_index, dtype=labels.dtype)
    x = mx.random.normal((batch_size,) + image_shape)
    mx.eval(x)

    log_interval = max(num_steps // 10, 1)
    for tau in range(num_steps - 1, -1, -1):
        t_actual = int(sampling_ts[tau].item())
        t_batch = mx.full((batch_size,), t_actual, dtype=mx.int32)
        eps = batched_cfg_forward(
            model, x, t_batch, labels, null_labels, guidance_scale
        )
        sab = float(sqrt_alpha_bar_s[tau].item())
        sab_prev = float(sqrt_alpha_bar_s_prev[tau].item())
        s1mab = float(sqrt_one_minus_alpha_bar_s[tau].item())
        dc = float(dir_coef[tau].item())
        sig = float(sigma[tau].item())

        x0_pred = (x - s1mab * eps) / sab
        direction = dc * eps
        noise = mx.random.normal(x.shape) if (eta > 0.0 and tau > 0) else mx.zeros(x.shape)
        x = sab_prev * x0_pred + direction + sig * noise
        mx.eval(x)

        if progress_callback is not None and (tau % log_interval == 0 or tau == 0):
            progress_callback(tau, num_steps)
    return mx.clip((x + 1.0) / 2.0, 0.0, 1.0)


@app.cell
def _(mo):
    mo.md(r"""
    ### Architecture comparison — UNet vs DiT

    | | `ConvUNetDenoiserV1` | `DiffusionTransformerV1` |
    |---|---|---|
    | Type | Convolutional UNet | Vision Transformer |
    | Input | 32x32x3 image (channels-last NHWC) | 32x32x3 patchified to 8x8=64 tokens |
    | Levels | 3 encoder + bottleneck + 3 decoder | 6 sequential DiT blocks |
    | Base width | 64 (mults 1,2,4,4 -> [64,128,256,256]) | hidden_size = 256 |
    | Norm | GroupNorm (pytorch_compatible) | LayerNorm (affine=False) |
    | Activation | SiLU | GELU (MLP) / SiLU (cond) |
    | Conditioning | Additive (t+class) bias inside each residual block | adaLN-Zero: 6-param modulation per block from `SiLU(c) -> zero-init Linear` |
    | Class embedding | `Embedding(11, 256)` | `Embedding(11, 256)` |
    | Skip connections | Yes (per encoder level) | No (residual within transformer blocks) |
    | Init trick | Standard | adaLN-Zero: last modulation Linear and final proj initialized to zero so each block/final projection starts as identity/zero |

    Both backbones share:

    - Signature `model(x_t, t, labels) -> predicted_epsilon`
    - Null class index = `num_classes` = 10 (embedding table row 10)
    - Compatibility with `ddpm_sample`, `ddim_sample`, `compute_ddpm_loss`,
      `run_train_epoch`, `evaluate_model` — no code changes needed to swap.
    """)
    return


@app.cell
def _(mo):
    reference_unet = ConvUNetDenoiserV1()
    mx.eval(reference_unet.parameters())
    reference_unet_params = count_parameters(reference_unet)
    reference_dit = DiffusionTransformerV1()
    mx.eval(reference_dit.parameters())
    reference_dit_params = count_parameters(reference_dit)
    mo.md(
        f"""
    ### Parameter counts

    | Backbone | Params |
    |----------|-------:|
    | `ConvUNetDenoiserV1` (default: base=64, mults=(1,2,4,4)) | `{reference_unet_params:,}` |
    | `DiffusionTransformerV1` (default: hidden=256, depth=6, patch=4) | `{reference_dit_params:,}` |
    """
    )
    return reference_dit_params, reference_unet_params


@app.cell
def _(mo):
    reference_schedule = DiffusionScheduleV1(num_timesteps=1000)
    mo.md(
        f"""
    ### Diffusion schedule — `DiffusionScheduleV1`

    - Steps `T`: `{reference_schedule.num_timesteps}`
    - `beta_start`: `{reference_schedule.beta_start:.0e}`
    - `beta_end`: `{reference_schedule.beta_end:.0e}`
    - Terminal `alpha_bar_T`: `{float(reference_schedule.alphas_cumprod[-1]):.6f}` (close to 0 -> x_T is nearly pure noise)
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
        options={"1e-4": 1e-4, "2e-4": 2e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="2e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(options=[64, 128, 256], value=128, label="Batch Size")
    epochs_ui = mo.ui.slider(1, 50, value=5, step=1, label="Epochs")
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
    backbone_ui = mo.ui.radio(
        options=["UNet", "DiT", "Both"],
        value="UNet",
        label="Backbone(s) to train",
    )
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Hyperparameters"),
            mo.hstack([lr_ui, bs_ui, epochs_ui]),
            mo.hstack([p_uncond_ui, wd_ui, backbone_ui]),
            train_btn,
        ]
    )
    return backbone_ui, bs_ui, epochs_ui, lr_ui, p_uncond_ui, train_btn, wd_ui


@app.cell
def _(bs_ui, test_ds, train_ds):
    train_iter, test_iter = make_datasets(
        train_ds, test_ds, batch_size=int(bs_ui.value)
    )
    return test_iter, train_iter


@app.function
def train_backbone(
    backbone_name: str,
    schedule: "DiffusionScheduleV1",
    train_iter,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    p_uncond: float,
    log_fn,
) -> tuple:
    """Instantiate the requested backbone, run `n_epochs` of DDPM training, return (losses, model)."""
    if backbone_name == "UNet":
        model = ConvUNetDenoiserV1()
    elif backbone_name == "DiT":
        model = DiffusionTransformerV1()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    losses = []
    for epoch in range(n_epochs):
        t0 = time.time()
        loss = run_train_epoch(model, schedule, optimizer, train_iter, p_uncond)
        dt = time.time() - t0
        losses.append(loss)
        log_fn(backbone_name, epoch + 1, n_epochs, loss, dt)
    return losses, model


@app.cell
def _(
    backbone_ui,
    epochs_ui,
    lr_ui,
    mo,
    p_uncond_ui,
    train_btn,
    train_iter,
    wd_ui,
):
    train_losses_unet = []
    train_losses_dit = []
    trained_unet = None
    trained_dit = None
    trained_schedule = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training the selected backbone(s)."))
    else:
        trained_schedule = DiffusionScheduleV1(num_timesteps=1000)
        selected = backbone_ui.value
        to_train = ["UNet", "DiT"] if selected == "Both" else [selected]

        def log_progress(name, epoch, total, loss, dt):
            mo.output.replace(
                mo.md(
                    f"**[{name}] Epoch {epoch}/{total}** — train MSE loss: "
                    f"{loss:.4f} — {dt:.1f}s"
                )
            )

        for backbone_name in to_train:
            losses, model = train_backbone(
                backbone_name,
                trained_schedule,
                train_iter,
                float(lr_ui.value),
                float(wd_ui.value),
                int(epochs_ui.value),
                float(p_uncond_ui.value),
                log_progress,
            )
            if backbone_name == "UNet":
                train_losses_unet = losses
                trained_unet = model
            else:
                train_losses_dit = losses
                trained_dit = model

        parts = []
        if trained_unet is not None:
            parts.append(
                f"UNet final MSE: `{train_losses_unet[-1]:.4f}` "
                f"({len(train_losses_unet)} epoch(s))"
            )
        if trained_dit is not None:
            parts.append(
                f"DiT final MSE: `{train_losses_dit[-1]:.4f}` "
                f"({len(train_losses_dit)} epoch(s))"
            )
        mo.output.replace(mo.md("**Training complete!** " + " · ".join(parts)))
    return (
        train_losses_dit,
        train_losses_unet,
        trained_dit,
        trained_schedule,
        trained_unet,
    )


@app.function
def plot_loss_curves(train_losses_unet: list, train_losses_dit: list):
    fig, ax = plt.subplots(figsize=(9, 4))
    if train_losses_unet:
        ax.plot(
            range(1, len(train_losses_unet) + 1),
            train_losses_unet,
            "b-o", lw=2, ms=4, label="UNet train MSE",
        )
    if train_losses_dit:
        ax.plot(
            range(1, len(train_losses_dit) + 1),
            train_losses_dit,
            "r-s", lw=2, ms=4, label="DiT train MSE",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE noise-prediction loss")
    ax.set_title("DDPM Training Loss — UNet vs DiT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses_dit, train_losses_unet):
    if not train_losses_unet and not train_losses_dit:
        _out = mo.md("_Train at least one backbone to see the loss curve._")
    else:
        _out = plot_loss_curves(train_losses_unet, train_losses_dit)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Optional Hyperparameter Search

    Enable the checkbox to run a small `lr x guidance_scale` grid on the UNet
    backbone (fewer epochs), reporting final train MSE per config. Kept small
    to remain tractable on Apple Silicon.
    """)
    return


@app.cell
def _(mo):
    enable_hp_search_ui = mo.ui.checkbox(
        label="Enable Hyperparameter Search", value=False
    )
    hp_epochs_ui = mo.ui.slider(1, 5, value=2, step=1, label="Epochs per config")
    hp_run_btn = mo.ui.run_button(label="Run Grid Search")
    mo.vstack([enable_hp_search_ui, hp_epochs_ui, hp_run_btn])
    return enable_hp_search_ui, hp_epochs_ui, hp_run_btn


@app.function
def run_hp_search(
    train_iter,
    lrs: list,
    guidance_scales: list,
    n_epochs: int,
) -> list:
    """Small grid over (lr, guidance_scale) on a UNet. Trains a fresh model per config."""
    results = []
    for lr in lrs:
        for guidance_scale in guidance_scales:
            schedule = DiffusionScheduleV1(num_timesteps=1000)
            model = ConvUNetDenoiserV1()
            mx.eval(model.parameters())
            optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0)
            final_loss = None
            for _ in range(n_epochs):
                final_loss = run_train_epoch(model, schedule, optimizer, train_iter, 0.1)
            results.append(
                {
                    "lr": lr,
                    "guidance_scale": guidance_scale,
                    "n_epochs": n_epochs,
                    "final_train_mse": float(final_loss),
                }
            )
    return results


@app.cell
def _(enable_hp_search_ui, hp_epochs_ui, hp_run_btn, mo, train_iter):
    mo.stop(
        not enable_hp_search_ui.value,
        mo.md("_Enable the checkbox and click **Run Grid Search** to explore hyperparameters._"),
    )
    mo.stop(
        not hp_run_btn.value,
        mo.md("_Click **Run Grid Search** to start._"),
    )
    hp_results = run_hp_search(
        train_iter,
        lrs=[1e-4, 3e-4],
        guidance_scales=[1.0, 3.0, 5.0],
        n_epochs=int(hp_epochs_ui.value),
    )
    hp_results_sorted = sorted(hp_results, key=lambda r: r["final_train_mse"])
    mo.ui.table(hp_results_sorted, label="HP search results (sorted by final MSE)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation

    Two evaluations:

    - **Test-set denoising MSE**: mean of `|| eps - eps_theta(x_t, t, y) ||^2`
      over the 10k CIFAR-10 test set at `p_uncond=0` (conditional-only).
    - **k=5 fold cross-validation** on a small subset of the train set to
      show variance of the training loss across folds.
    """)
    return


@app.cell
def _(mo):
    enable_cv_ui = mo.ui.checkbox(
        label="Enable Test-Set Evaluation + k-Fold CV", value=False
    )
    cv_epochs_ui = mo.ui.slider(1, 5, value=1, step=1, label="Epochs per fold")
    cv_subset_ui = mo.ui.slider(
        1000, 10000, value=2000, step=1000, label="CV subset size (train samples)"
    )
    cv_run_btn = mo.ui.run_button(label="Run Evaluation")
    mo.vstack([enable_cv_ui, cv_epochs_ui, cv_subset_ui, cv_run_btn])
    return cv_epochs_ui, cv_run_btn, cv_subset_ui, enable_cv_ui


@app.function
def run_kfold_cv(
    train_ds,
    k: int,
    subset_size: int,
    n_epochs: int,
    batch_size: int,
) -> list:
    """k-fold CV on `subset_size` random train indices; report final train MSE per fold."""
    n_total = len(train_ds)
    rng = np.random.default_rng(0)
    subset_indices = rng.choice(n_total, size=min(subset_size, n_total), replace=False)
    fold_size = len(subset_indices) // k
    fold_losses = []
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else len(subset_indices)
        val_ix = subset_indices[val_start:val_end]
        train_ix = np.concatenate(
            [subset_indices[:val_start], subset_indices[val_end:]]
        )
        train_records = [
            {"image": np.array(train_ds[int(i)]["image"]),
             "label": np.array(train_ds[int(i)]["label"])}
            for i in train_ix
        ]
        val_records = [
            {"image": np.array(train_ds[int(i)]["image"]),
             "label": np.array(train_ds[int(i)]["label"])}
            for i in val_ix
        ]
        import mlx.data as dx
        fold_train_iter = (
            dx.buffer_from_vector(train_records)
            .shuffle()
            .to_stream()
            .key_transform("image", normalize_to_signed_unit)
            .batch(batch_size)
        )
        fold_val_iter = (
            dx.buffer_from_vector(val_records)
            .to_stream()
            .key_transform("image", normalize_to_signed_unit)
            .batch(batch_size)
        )
        schedule = DiffusionScheduleV1(num_timesteps=1000)
        model = ConvUNetDenoiserV1()
        mx.eval(model.parameters())
        optimizer = optim.AdamW(learning_rate=2e-4, weight_decay=0.0)
        for _ in range(n_epochs):
            run_train_epoch(model, schedule, optimizer, fold_train_iter, 0.1)
        val_mse = evaluate_model(model, schedule, fold_val_iter, p_uncond=0.0)
        fold_losses.append(val_mse)
    return fold_losses


@app.cell
def _(
    cv_epochs_ui,
    cv_run_btn,
    cv_subset_ui,
    enable_cv_ui,
    mo,
    test_iter,
    train_ds,
    trained_schedule,
    trained_unet,
):
    mo.stop(
        not enable_cv_ui.value,
        mo.md("_Enable the checkbox and click **Run Evaluation** to compute test MSE and k-fold CV._"),
    )
    mo.stop(
        not cv_run_btn.value,
        mo.md("_Click **Run Evaluation** to start._"),
    )
    if trained_unet is None or trained_schedule is None:
        _out = mo.md("_Train the UNet backbone first (Section 5) to run test-set evaluation._")
    else:
        test_mse = evaluate_model(trained_unet, trained_schedule, test_iter, p_uncond=0.0)
        cv_losses = run_kfold_cv(
            train_ds,
            k=5,
            subset_size=int(cv_subset_ui.value),
            n_epochs=int(cv_epochs_ui.value),
            batch_size=128,
        )
        cv_mean = float(np.mean(cv_losses))
        cv_std = float(np.std(cv_losses))
        fold_rows = "\n".join(
            f"| Fold {i + 1} | `{v:.4f}` |" for i, v in enumerate(cv_losses)
        )
        _out = mo.md(
            f"""
    ### Test-set denoising MSE (UNet)

    | Split | MSE |
    |-------|-----|
    | Test (10k) | `{test_mse:.4f}` |

    ### k=5 fold CV on {int(cv_subset_ui.value):,} train samples ({int(cv_epochs_ui.value)} epoch(s)/fold)

    | Fold | Final val MSE |
    |------|---------------|
    {fold_rows}
    | **Mean +/- Std** | **`{cv_mean:.4f} +/- {cv_std:.4f}`** |
    """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Interactive Class-Conditional Sampling

    Pick a CIFAR-10 class, a backbone (UNet or DiT), a sampler (DDPM 1000 steps
    or DDIM with configurable step count), a CFG guidance scale, and click
    **Sample**. Both samplers apply CFG at every step and work with either
    backbone unmodified.
    """)
    return


@app.cell
def _(mo):
    class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(cifar10_class_names())},
        value="cat",
        label="Class",
    )
    backbone_choice_ui = mo.ui.dropdown(
        options=["UNet", "DiT"], value="UNet", label="Backbone"
    )
    sampler_ui = mo.ui.dropdown(
        options=["DDPM (1000 steps)", "DDIM"],
        value="DDIM",
        label="Sampler",
    )
    guidance_ui = mo.ui.slider(0.0, 8.0, value=3.0, step=0.5, label="Guidance scale w")
    num_samples_ui = mo.ui.slider(1, 16, value=8, step=1, label="Number of samples")
    ddim_steps_ui = mo.ui.slider(10, 200, value=50, step=10, label="DDIM steps")
    ddim_eta_ui = mo.ui.slider(0.0, 1.0, value=0.0, step=0.1, label="DDIM eta")
    sample_btn = mo.ui.run_button(label="Sample")
    mo.vstack(
        [
            mo.md("### Sampling controls"),
            mo.hstack([class_ui, backbone_choice_ui, sampler_ui]),
            mo.hstack([guidance_ui, num_samples_ui]),
            mo.hstack([ddim_steps_ui, ddim_eta_ui]),
            sample_btn,
        ]
    )
    return (
        backbone_choice_ui,
        class_ui,
        ddim_eta_ui,
        ddim_steps_ui,
        guidance_ui,
        num_samples_ui,
        sample_btn,
        sampler_ui,
    )


@app.function
def plot_generated_grid(images: np.ndarray, class_name: str, guidance_scale: float):
    n = images.shape[0]
    cols = min(n, 8)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes_flat = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n:
            ax.imshow(np.clip(images[i], 0.0, 1.0))
        ax.axis("off")
    fig.suptitle(
        f"Generated '{class_name}' — guidance w={guidance_scale:.1f}, n={n}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


@app.function
def pick_sampler_and_run(
    sampler_name: str,
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    labels: mx.array,
    guidance_scale: float,
    ddim_steps: int,
    ddim_eta: float,
    progress_callback,
) -> mx.array:
    """Dispatch to `ddpm_sample` or `ddim_sample`. Both accept the same model interface."""
    if sampler_name == "DDIM":
        return ddim_sample(
            model, schedule, labels,
            image_shape=(32, 32, 3),
            guidance_scale=guidance_scale,
            num_steps=ddim_steps,
            eta=ddim_eta,
            progress_callback=progress_callback,
        )
    return ddpm_sample(
        model, schedule, labels,
        image_shape=(32, 32, 3),
        guidance_scale=guidance_scale,
        progress_callback=progress_callback,
    )


@app.cell
def _(
    backbone_choice_ui,
    class_ui,
    ddim_eta_ui,
    ddim_steps_ui,
    guidance_ui,
    mo,
    num_samples_ui,
    sample_btn,
    sampler_ui,
    trained_dit,
    trained_schedule,
    trained_unet,
):
    if trained_schedule is None:
        _out = mo.md("_Train at least one backbone first (Section 5) to enable sampling._")
    elif not sample_btn.value:
        _out = mo.md("Configure options and click **Sample** to generate images.")
    else:
        _selected_backbone = backbone_choice_ui.value
        _selected_model = trained_unet if _selected_backbone == "UNet" else trained_dit
        if _selected_model is None:
            _out = mo.md(
                f"_The **{_selected_backbone}** backbone has not been trained yet. "
                f"Go to Section 5, select it (or 'Both'), and re-train._"
            )
        else:
            _class_id = int(class_ui.value)
            _class_name = cifar10_class_names()[_class_id]
            _guidance = float(guidance_ui.value)
            _n = int(num_samples_ui.value)
            _sampler_name = "DDIM" if sampler_ui.value == "DDIM" else "DDPM"
            _labels = mx.full((_n,), _class_id, dtype=mx.int32)

            def _progress(step, total):
                mo.output.replace(
                    mo.md(
                        f"**{_selected_backbone} / {_sampler_name}** — "
                        f"class '{_class_name}' — step {total - step}/{total}"
                    )
                )

            _samples = pick_sampler_and_run(
                _sampler_name,
                _selected_model,
                trained_schedule,
                _labels,
                _guidance,
                int(ddim_steps_ui.value),
                float(ddim_eta_ui.value),
                _progress,
            )
            mx.eval(_samples)
            _images_np = np.array(_samples)
            _out = plot_generated_grid(_images_np, _class_name, _guidance)
    _out
    return


@app.cell
def _(
    mo,
    reference_dit_params,
    reference_unet_params,
    train_losses_dit,
    train_losses_unet,
):
    unet_row = (
        f"| UNet | `{reference_unet_params:,}` | `{train_losses_unet[-1]:.4f}` | {len(train_losses_unet)} |"
        if train_losses_unet
        else f"| UNet | `{reference_unet_params:,}` | not trained | 0 |"
    )
    dit_row = (
        f"| DiT | `{reference_dit_params:,}` | `{train_losses_dit[-1]:.4f}` | {len(train_losses_dit)} |"
        if train_losses_dit
        else f"| DiT | `{reference_dit_params:,}` | not trained | 0 |"
    )
    mo.md(
        f"""
    ### Model comparison

    | Backbone | Params | Final train MSE | Epochs |
    |----------|-------:|-----------------|-------:|
    {unet_row}
    {dit_row}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Summary

    - **Framework**: MLX (Apple Silicon), channels-last NHWC layout
    - **Dataset**: CIFAR-10 (32x32x3, 10 classes), normalized to `[-1, 1]`
    - **Diffusion**: linear beta `1e-4 -> 2e-2` over `T=1000` steps,
      epsilon-parameterization (predict noise)
    - **CFG**: single network trained with `p_uncond=0.1` label dropout to
      null token (index 10); sampling combines `eps_uncond + w*(eps_cond - eps_uncond)`
      via a **batched** forward pass over cond+uncond
    - **Backbones**: `ConvUNetDenoiserV1` (3-level UNet with residual blocks,
      GroupNorm, additive class+time conditioning) and
      `DiffusionTransformerV1` (6-block DiT with **adaLN-Zero** identity init,
      patch_size=4 pixel patches, learned positional embeddings, `nn.Embedding`
      class table)
    - **Samplers**: `ddpm_sample` (1000-step ancestral) and `ddim_sample`
      (configurable steps, `eta=0` deterministic default). Both accept the
      same `model(x_t, t, y)` signature so either sampler works with either
      backbone unmodified.

    ### Notes
    - MSE noise-prediction loss should drop into the low `1e-2`s within a
      few epochs. Visual sample quality is more diagnostic than the exact
      value; CIFAR-10 pixel-space diffusion needs many more epochs (~100+)
      for high quality — this notebook demonstrates plumbing correctness.
    - Setting `eta=1.0` in DDIM recovers DDPM-like stochastic behavior, which
      is why DDIM is described as *an option alongside DDPM* rather than a
      strict replacement.
    - The DiT block's adaLN-Zero initialization is essential: without zeroing
      the modulation `Linear` and the final projection, DiT training on
      CIFAR-10 diverges in the first hundred steps.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Model Weights

    Save whichever backbone(s) were trained to the repo-level `models/`
    directory. The file extension chosen determines the on-disk format:
    `.safetensors` or `.npz` (both natively supported by MLX).
    """)
    return


@app.cell
def _(mo):
    save_unet_filename_ui = mo.ui.text(
        value="cifar10_unet_ddpm_v1.safetensors",
        label="UNet filename (saved into models/)",
        full_width=True,
    )
    save_dit_filename_ui = mo.ui.text(
        value="cifar10_dit_ddpm_v1.safetensors",
        label="DiT filename (saved into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model(s)")
    mo.vstack([save_unet_filename_ui, save_dit_filename_ui, save_model_btn])
    return save_dit_filename_ui, save_model_btn, save_unet_filename_ui


@app.cell
def _(
    mo,
    save_dit_filename_ui,
    save_model_btn,
    save_unet_filename_ui,
    trained_dit,
    trained_unet,
):
    if trained_unet is None and trained_dit is None:
        _out = mo.md("_Train at least one backbone (Section 5) before saving._")
    elif not save_model_btn.value:
        _out = mo.md(
            "Set filenames and click **Save Model(s)** to write trained weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _lines = []
        if trained_unet is not None:
            _unet_path = _models_dir / save_unet_filename_ui.value
            trained_unet.save_weights(str(_unet_path))
            _lines.append(f"- UNet -> `{_unet_path}`")
        if trained_dit is not None:
            _dit_path = _models_dir / save_dit_filename_ui.value
            trained_dit.save_weights(str(_dit_path))
            _lines.append(f"- DiT -> `{_dit_path}`")
        _out = mo.md("**Saved!**\n" + "\n".join(_lines))
    _out
    return


if __name__ == "__main__":
    app.run()
