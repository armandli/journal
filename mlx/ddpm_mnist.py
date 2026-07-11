import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import math
    from pathlib import Path
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx.data.datasets import load_mnist
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
    # Classifier-Free Guided DDPM on MNIST — MLX

    ## Research Goal

    Train a **class-conditional Denoising Diffusion Probabilistic Model
    (DDPM)** on MNIST using Apple's **MLX** framework. Digit labels
    (0–9) are one-hot encoded and injected into a **convolutional UNet**
    (3 encoder + 3 decoder levels) as an additive conditioning signal
    alongside the sinusoidal timestep embedding.

    Training uses **classifier-free guidance (CFG)**: with probability
    `p_uncond=0.1` the class label is replaced by a special *null* class
    (index 10) so the same network learns both the conditional
    distribution `p(x | y)` and the unconditional distribution `p(x)`.
    At sample time we combine both scores via
    `eps = eps_uncond + w * (eps_cond - eps_uncond)`.

    ### Method
    - Linear beta schedule from `1e-4 → 0.02` over `T=1000` steps
    - UNet with 3 encoder levels `[64, 128, 256]` + bottleneck + 3 decoder levels
    - Sinusoidal time embedding + one-hot class embedding fused as `cond`
    - Loss: MSE between predicted and true noise
    - Sampling: **1000-step DDPM reverse process** with CFG guidance scale

    ### Notebook Outline
    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation
    4. Model definition (schedule, UNet, loss, sampler)
    5. Training loop
    6. Interactive sampling widget
    7. Results summary
    8. Save trained model
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration
    """)
    return


@app.cell
def _():
    train_ds = load_mnist(root="../data/mnist", train=True)
    test_ds = load_mnist(root="../data/mnist", train=False)
    return test_ds, train_ds


@app.cell
def _(mo, test_ds, train_ds):
    mo.md(f"""
    ### Dataset overview

    MNIST is loaded as an `mlx.data` Buffer. Each sample is a dict with:

    - `image` — `uint8` array shaped `(28, 28, 1)` (channels-last)
    - `label` — scalar `uint8` in `[0, 9]`

    | Split | Size |
    |-------|------|
    | Train (raw) | {len(train_ds):,} |
    | Test | {len(test_ds):,} |
    """)
    return


@app.function
def plot_sample_grid(dataset, n_show: int = 40, rows: int = 5, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    for i in range(n_show):
        sample = dataset[i]
        img = np.array(sample["image"]).squeeze()
        label = int(np.array(sample["label"]).item())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].set_title(str(label), fontsize=9)
        axes[r, c].axis("off")
    fig.suptitle("MNIST training samples (digit labels as titles)", fontsize=13)
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
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(num_classes), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(num_classes))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Count")
    ax.set_title("MNIST training-set class distribution")
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
def _(mo, train_ds):
    split_val_fraction = 0.15
    n_val_preview = int(round(len(train_ds) * split_val_fraction))
    n_train_preview = len(train_ds) - n_val_preview
    mo.md(f"""
    ### Planned splits (see Section 3)

    | Split | Size |
    |-------|------|
    | Train | {n_train_preview:,} |
    | Val | {n_val_preview:,} |
    | Test | 10,000 |

    Val split is carved out of the 60k MNIST train set with fraction `{split_val_fraction:.2f}`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def normalize_to_unit_interval(x):
    return x.astype("float32") / 255.0


@app.function
def make_ddpm_datasets(
    train_ds,
    test_ds,
    batch_size: int = 128,
    val_fraction: float = 0.15,
):
    n_total = len(train_ds)
    n_val = int(round(n_total * val_fraction))
    shuffled = train_ds.shuffle()
    train_iter = (
        shuffled
        .to_stream()
        .key_transform("image", normalize_to_unit_interval)
        .batch(batch_size)
    )
    val_iter = (
        shuffled
        .to_stream()
        .key_transform("image", normalize_to_unit_interval)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", normalize_to_unit_interval)
        .batch(batch_size)
    )
    return train_iter, val_iter, test_iter


@app.cell
def _(test_ds, train_ds):
    inspection_train_iter, inspection_val_iter, inspection_test_iter = make_ddpm_datasets(
        train_ds, test_ds, batch_size=128
    )
    return (inspection_train_iter,)


@app.cell
def _(inspection_train_iter, mo):
    inspection_train_iter.reset()
    sample_batch = next(inspection_train_iter)
    sample_images = mx.array(sample_batch["image"])
    sample_labels = mx.array(sample_batch["label"])
    mo.md(f"""
    ### First batch shape check

    - `image` shape: `{tuple(sample_images.shape)}` — dtype `{sample_images.dtype}`
    - `label` shape: `{tuple(sample_labels.shape)}` — dtype `{sample_labels.dtype}`
    - image min/max: `{float(sample_images.min()):.3f} / {float(sample_images.max()):.3f}`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition

    We define, in order:

    1. `DiffusionScheduleV1` — precomputed linear beta schedule
    2. `SinusoidalPositionEmbeddingV1` — sinusoidal timestep embedding
    3. `ResidualBlockV1` — GroupNorm-conditioned conv residual block
    4. `DownBlockV1` — 2 residual blocks + stride-2 conv downsample
    5. `UpBlockV1` — ConvTranspose upsample + skip concat + 2 residual blocks
    6. `UNetV1` — full model with time + class conditioning
    7. `compute_ddpm_loss` — MSE noise-prediction loss with CFG dropout
    8. `ddpm_sample` — 1000-step reverse process with classifier-free guidance
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


@app.class_definition
class SinusoidalPositionEmbeddingV1(nn.Module):
    """Standard sinusoidal position embedding (Vaswani et al. 2017)."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim

    def __call__(self, t: mx.array) -> mx.array:
        half_dim = self.emb_dim // 2
        freq_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = mx.exp(-mx.arange(half_dim, dtype=mx.float32) * freq_scale)
        t_float = t.astype(mx.float32)
        args = t_float[:, None] * freqs[None, :]
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


@app.class_definition
class ResidualBlockV1(nn.Module):
    """Conv 3x3 -> GN -> SiLU -> Conv 3x3 -> GN -> SiLU with additive conditioning + skip."""

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
    """Encoder level: two ResidualBlocks then a stride-2 conv downsample."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
        cond_dim: int = 256,
        num_groups: int = 8,
        downsample: bool = True,
        downsample_kernel: int = 3,
        downsample_padding: int = 1,
    ):
        super().__init__()
        self.res1 = ResidualBlockV1(in_channels, out_channels, cond_dim, num_groups)
        self.res2 = ResidualBlockV1(out_channels, out_channels, cond_dim, num_groups)
        if downsample:
            self.down = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=downsample_kernel,
                stride=2,
                padding=downsample_padding,
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
    """Decoder level: ConvTranspose2d upsample -> concat skip -> two ResidualBlocks."""

    def __init__(
        self,
        in_channels: int = 256,
        skip_channels: int = 256,
        out_channels: int = 128,
        cond_dim: int = 256,
        num_groups: int = 8,
        upsample_kernel: int = 4,
        upsample_stride: int = 2,
        upsample_padding: int = 1,
        upsample_output_padding: int = 0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=upsample_kernel,
            stride=upsample_stride,
            padding=upsample_padding,
            output_padding=upsample_output_padding,
        )
        self.res1 = ResidualBlockV1(
            in_channels + skip_channels, out_channels, cond_dim, num_groups
        )
        self.res2 = ResidualBlockV1(out_channels, out_channels, cond_dim, num_groups)

    def __call__(self, x: mx.array, skip: mx.array, cond: mx.array) -> mx.array:
        h = self.up(x)
        h = mx.concatenate([h, skip], axis=-1)
        h = self.res1(h, cond)
        h = self.res2(h, cond)
        return h


@app.class_definition
class UNetV1(nn.Module):
    """3-level UNet with time + one-hot class conditioning for MNIST DDPM.

    Spatial trajectory (H=W): 28 -> 14 -> 7 -> 4 (bottleneck) -> 7 -> 14 -> 28.
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        time_emb_dim: int = 256,
        class_emb_dim: int = 256,
        num_classes: int = 10,
        num_groups_by_channels: tuple = ((64, 8), (128, 16), (256, 32)),
    ):
        super().__init__()
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.time_emb_dim = time_emb_dim
        self.class_emb_dim = class_emb_dim
        assert time_emb_dim == class_emb_dim, "time and class embeddings are summed"

        ch = [base_channels * m for m in channel_mults]
        gn_lookup = {c: g for c, g in num_groups_by_channels}

        self.time_pos = SinusoidalPositionEmbeddingV1(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)

        self.stem = nn.Conv2d(image_channels, ch[0], kernel_size=3, padding=1)

        self.down1 = DownBlockV1(
            ch[0], ch[0], cond_dim=time_emb_dim, num_groups=gn_lookup[ch[0]],
            downsample=True, downsample_kernel=4, downsample_padding=1,
        )
        self.down2 = DownBlockV1(
            ch[0], ch[1], cond_dim=time_emb_dim, num_groups=gn_lookup[ch[1]],
            downsample=True, downsample_kernel=4, downsample_padding=1,
        )
        self.down3 = DownBlockV1(
            ch[1], ch[2], cond_dim=time_emb_dim, num_groups=gn_lookup[ch[2]],
            downsample=True, downsample_kernel=3, downsample_padding=1,
        )

        self.mid_res1 = ResidualBlockV1(ch[2], ch[2], time_emb_dim, gn_lookup[ch[2]])
        self.mid_res2 = ResidualBlockV1(ch[2], ch[2], time_emb_dim, gn_lookup[ch[2]])

        self.up1 = UpBlockV1(
            in_channels=ch[2], skip_channels=ch[2], out_channels=ch[1],
            cond_dim=time_emb_dim, num_groups=gn_lookup[ch[1]],
            upsample_kernel=3, upsample_stride=2, upsample_padding=1,
            upsample_output_padding=0,
        )
        self.up2 = UpBlockV1(
            in_channels=ch[1], skip_channels=ch[1], out_channels=ch[0],
            cond_dim=time_emb_dim, num_groups=gn_lookup[ch[0]],
            upsample_kernel=4, upsample_stride=2, upsample_padding=1,
            upsample_output_padding=0,
        )
        self.up3 = UpBlockV1(
            in_channels=ch[0], skip_channels=ch[0], out_channels=ch[0],
            cond_dim=time_emb_dim, num_groups=gn_lookup[ch[0]],
            upsample_kernel=4, upsample_stride=2, upsample_padding=1,
            upsample_output_padding=0,
        )

        self.out_norm = nn.GroupNorm(gn_lookup[ch[0]], ch[0], pytorch_compatible=True)
        self.out_conv = nn.Conv2d(ch[0], image_channels, kernel_size=1)

    def encode_condition(self, t: mx.array, labels: mx.array) -> mx.array:
        t_emb = self.time_pos(t)
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


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


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
    """Forward process: x_t = sqrt(alphabar_t) * x0 + sqrt(1 - alphabar_t) * noise."""
    sqrt_ab = gather_schedule_value(schedule.sqrt_alphas_cumprod, t)
    sqrt_one_minus_ab = gather_schedule_value(schedule.sqrt_one_minus_alphas_cumprod, t)
    sqrt_ab = sqrt_ab[:, None, None, None]
    sqrt_one_minus_ab = sqrt_one_minus_ab[:, None, None, None]
    return sqrt_ab * x0 + sqrt_one_minus_ab * noise


@app.function
def compute_ddpm_loss(
    model: nn.Module,
    x0: mx.array,
    labels: mx.array,
    schedule: "DiffusionScheduleV1",
    p_uncond: float = 0.1,
) -> mx.array:
    """Classifier-free-guidance training loss (MSE on predicted noise)."""
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
def run_ddpm_train_epoch(
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
def ddpm_sample(
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    labels: mx.array,
    image_shape: tuple = (28, 28, 1),
    guidance_scale: float = 3.0,
    progress_callback=None,
) -> mx.array:
    """DDPM ancestral sampling with classifier-free guidance."""
    batch_size = int(labels.shape[0])
    x = mx.random.normal((batch_size,) + image_shape)
    null_labels = mx.full((batch_size,), model.null_class_index, dtype=labels.dtype)
    T = schedule.num_timesteps
    for step in range(T - 1, -1, -1):
        t_batch = mx.full((batch_size,), step, dtype=mx.int32)
        eps_cond = model(x, t_batch, labels)
        eps_uncond = model(x, t_batch, null_labels)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
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
            progress_callback(step, T)
    x = mx.clip(x, 0.0, 1.0)
    return x


@app.function
def ddim_sample(
    model: nn.Module,
    schedule: "DiffusionScheduleV1",
    labels: mx.array,
    image_shape: tuple = (28, 28, 1),
    guidance_scale: float = 3.0,
    num_steps: int = 50,
    eta: float = 0.0,
    progress_callback=None,
) -> mx.array:
    """DDIM accelerated sampling with classifier-free guidance.

    Uses num_steps uniformly-spaced timesteps instead of all T=1000 steps.
    eta=0 is fully deterministic; eta=1 recovers DDPM-style stochasticity.
    Same trained model as DDPM — only the sampling procedure differs.
    """
    T = schedule.num_timesteps
    batch_size = int(labels.shape[0])

    # Uniformly spaced sampling timesteps (1-indexed to match PyTorch DDIMScheduler convention)
    step_size = T // num_steps
    sampling_ts = mx.array(
        [min(i * step_size, T - 1) for i in range(1, num_steps + 1)],
        dtype=mx.int32,
    )  # ascending, shape (num_steps,)

    # alpha_bar at each sampling timestep and its predecessor
    alpha_bar_s = mx.take(schedule.alphas_cumprod, sampling_ts)
    alpha_bar_s_prev = mx.concatenate([
        schedule.alphas_cumprod[0:1],
        alpha_bar_s[:-1],
    ])

    sqrt_alpha_bar_s = mx.sqrt(alpha_bar_s)
    sqrt_alpha_bar_s_prev = mx.sqrt(alpha_bar_s_prev)
    sqrt_one_minus_alpha_bar_s = mx.sqrt(1.0 - alpha_bar_s)

    # sigma: stochasticity (0 = deterministic DDIM, 1 = DDPM-like)
    sigma = eta * mx.sqrt(
        (1.0 - alpha_bar_s_prev) / (1.0 - alpha_bar_s) *
        (1.0 - alpha_bar_s / alpha_bar_s_prev)
    )
    # direction-to-x_t coefficient: sqrt(1 - alpha_bar_s_prev - sigma^2), clamped >= 0
    dir_coef = mx.sqrt(mx.maximum(1.0 - alpha_bar_s_prev - sigma ** 2, 0.0))

    mx.eval(alpha_bar_s, alpha_bar_s_prev, sqrt_alpha_bar_s, sqrt_alpha_bar_s_prev,
            sqrt_one_minus_alpha_bar_s, sigma, dir_coef)

    null_labels = mx.full((batch_size,), model.null_class_index, dtype=labels.dtype)
    x = mx.random.normal((batch_size,) + image_shape)
    mx.eval(x)

    log_interval = max(num_steps // 10, 1)
    for tau in range(num_steps - 1, -1, -1):
        t_actual = int(sampling_ts[tau].item())
        t_batch = mx.full((batch_size,), t_actual, dtype=mx.int32)

        eps_cond = model(x, t_batch, labels)
        eps_uncond = model(x, t_batch, null_labels)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

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

    return mx.clip(x, 0.0, 1.0)


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Architecture — `UNetV1`

    | Stage | Module | Spatial (H×W) | Channels |
    |-------|--------|---------------|----------|
    | Stem | `Conv2d(1→64, k=3, p=1)` | 28×28 | 64 |
    | Down 1 | `DownBlockV1` (2×`ResidualBlockV1` + stride-2 conv k=4) | 28→14 | 64 |
    | Down 2 | `DownBlockV1` (2×`ResidualBlockV1` + stride-2 conv k=4) | 14→7 | 64→128 |
    | Down 3 | `DownBlockV1` (2×`ResidualBlockV1` + stride-2 conv k=3) | 7→4 | 128→256 |
    | Bottleneck | 2×`ResidualBlockV1` | 4×4 | 256 |
    | Up 1 | `UpBlockV1` (ConvT k=3 s=2 p=1) + skip3 + 2×`ResidualBlockV1` | 4→7 | 256→128 |
    | Up 2 | `UpBlockV1` (ConvT k=4 s=2 p=1) + skip2 + 2×`ResidualBlockV1` | 7→14 | 128→64 |
    | Up 3 | `UpBlockV1` (ConvT k=4 s=2 p=1) + skip1 + 2×`ResidualBlockV1` | 14→28 | 64→64 |
    | Head | `GroupNorm` → `SiLU` → `Conv2d(64→1, k=1)` | 28×28 | 1 |

    **Conditioning**: sinusoidal time embedding (dim=256) → 2-layer MLP,
    added to `Embedding(11, 256)(class_label)` where index `10` is the null
    class for classifier-free guidance. The resulting `cond` vector is
    projected inside every `ResidualBlockV1` and added as a channel-wise bias.

    **Loss**: `E[|| eps - eps_theta(x_t, t, y) ||^2]` with `y` dropped to the
    null class with probability `p_uncond = 0.1`.

    **Sampling**: 1000-step DDPM reverse process with CFG
    `eps = eps_uncond + w * (eps_cond - eps_uncond)`.
    """)
    return


@app.cell
def _(mo):
    reference_model = UNetV1()
    mx.eval(reference_model.parameters())
    reference_param_count = count_parameters(reference_model)
    mo.md(
        f"**Reference `UNetV1` parameter count**: `{reference_param_count:,}`"
    )
    return (reference_param_count,)


@app.cell
def _(mo):
    reference_schedule = DiffusionScheduleV1(num_timesteps=1000)
    mo.md(f"""
    ### Diffusion schedule — `DiffusionScheduleV1`

    - Steps `T`: `{reference_schedule.num_timesteps}`
    - `beta_start`: `{reference_schedule.beta_start:.0e}`
    - `beta_end`: `{reference_schedule.beta_end:.0e}`
    - Terminal `alpha_bar_T`: `{float(reference_schedule.alphas_cumprod[-1]):.6f}` (close to 0 → x_T ≈ pure noise)
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
        options={"1e-4": 1e-4, "2e-4": 2e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="2e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(
        options=[64, 128, 256], value=128, label="Batch Size"
    )
    epochs_ui = mo.ui.slider(1, 50, value=20, step=1, label="Epochs")
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
            mo.md("### Hyperparameters"),
            mo.hstack([lr_ui, bs_ui, epochs_ui]),
            mo.hstack([p_uncond_ui, wd_ui]),
            train_btn,
        ]
    )
    return bs_ui, epochs_ui, lr_ui, p_uncond_ui, train_btn, wd_ui


@app.cell
def _(bs_ui, test_ds, train_ds):
    train_iter, val_iter, test_iter = make_ddpm_datasets(
        train_ds, test_ds, batch_size=int(bs_ui.value)
    )
    return (train_iter,)


@app.cell
def _(epochs_ui, lr_ui, mo, p_uncond_ui, train_btn, train_iter, wd_ui):
    train_losses = []
    trained_model = None
    trained_schedule = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin DDPM training."))
    else:
        run_schedule = DiffusionScheduleV1(num_timesteps=1000)
        run_model = UNetV1()
        mx.eval(run_model.parameters())
        run_optimizer = optim.AdamW(
            learning_rate=float(lr_ui.value),
            weight_decay=float(wd_ui.value),
        )
        n_epochs = int(epochs_ui.value)
        for epoch in range(n_epochs):
            epoch_loss = run_ddpm_train_epoch(
                run_model,
                run_schedule,
                run_optimizer,
                train_iter,
                p_uncond=float(p_uncond_ui.value),
            )
            train_losses.append(epoch_loss)
            mo.output.replace(
                mo.md(
                    f"**Epoch {epoch + 1}/{n_epochs}** — "
                    f"train MSE loss: {epoch_loss:.4f}"
                )
            )
        trained_model = run_model
        trained_schedule = run_schedule
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train MSE loss: "
                f"{train_losses[-1]:.4f} over {n_epochs} epoch(s)."
            )
        )
    return train_losses, trained_model, trained_schedule


@app.function
def plot_loss_curve(train_losses: list):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE noise-prediction loss")
    ax.set_title("DDPM Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses):
    if not train_losses:
        _out = mo.md("_Train first to see the loss curve._")
    else:
        _out = plot_loss_curve(train_losses)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Interactive Sampling Widget

    Pick a digit class, a classifier-free-guidance scale, and how many samples
    to generate; then press **Generate**. Each click runs the **1000-step
    DDPM reverse process** with CFG applied at every step.
    """)
    return


@app.cell
def _(mo):
    digit_ui = mo.ui.dropdown(
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        value=7,
        label="Digit class",
    )
    guidance_ui = mo.ui.slider(
        1.0, 10.0, value=3.0, step=0.5, label="Guidance scale"
    )
    num_samples_ui = mo.ui.slider(
        1, 16, value=4, step=1, label="Number of samples"
    )
    sampler_ui = mo.ui.dropdown(
        options=["DDPM (1000 steps)", "DDIM"],
        value="DDPM (1000 steps)",
        label="Sampler",
    )
    ddim_steps_ui = mo.ui.slider(
        10, 200, value=50, step=10, label="DDIM steps"
    )
    sample_btn = mo.ui.run_button(label="Generate")
    mo.vstack(
        [
            mo.md("### Sampling controls"),
            mo.hstack([digit_ui, guidance_ui, num_samples_ui]),
            mo.hstack([sampler_ui, ddim_steps_ui]),
            sample_btn,
        ]
    )
    return (
        ddim_steps_ui,
        digit_ui,
        guidance_ui,
        num_samples_ui,
        sample_btn,
        sampler_ui,
    )


@app.function
def plot_generated_grid(
    images: np.ndarray, digit: int, guidance_scale: float
):
    n = images.shape[0]
    cols = min(n, 8)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes_flat = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n:
            ax.imshow(images[i].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    fig.suptitle(
        f"Generated digit {digit} — guidance={guidance_scale:.1f}, n={n}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(
    ddim_steps_ui,
    digit_ui,
    guidance_ui,
    mo,
    num_samples_ui,
    sample_btn,
    sampler_ui,
    trained_model,
    trained_schedule,
):
    if trained_model is None or trained_schedule is None:
        _out = mo.md("_Train the model first (Section 5) to enable sampling._")
    elif not sample_btn.value:
        _out = mo.md("Choose a digit and click **Generate** to sample images.")
    else:
        _digit = int(digit_ui.value)
        _guidance = float(guidance_ui.value)
        _n = int(num_samples_ui.value)
        _labels = mx.full((_n,), _digit, dtype=mx.int32)
        _use_ddim = sampler_ui.value == "DDIM"
        _ddim_steps = int(ddim_steps_ui.value)
        _sampler_name = f"DDIM ({_ddim_steps} steps)" if _use_ddim else "DDPM (1000 steps)"

        def _progress(step, total):
            mo.output.replace(
                mo.md(
                    f"{_sampler_name} — digit {_digit} — "
                    f"step {total - step}/{total}"
                )
            )

        if _use_ddim:
            _samples = ddim_sample(
                trained_model,
                trained_schedule,
                _labels,
                image_shape=(28, 28, 1),
                guidance_scale=_guidance,
                num_steps=_ddim_steps,
                eta=0.0,
                progress_callback=_progress,
            )
        else:
            _samples = ddpm_sample(
                trained_model,
                trained_schedule,
                _labels,
                image_shape=(28, 28, 1),
                guidance_scale=_guidance,
                progress_callback=_progress,
            )
        mx.eval(_samples)
        _images_np = np.array(_samples)
        _out = plot_generated_grid(_images_np, _digit, _guidance)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Results Summary
    """)
    return


@app.cell
def _(
    epochs_ui,
    guidance_ui,
    lr_ui,
    mo,
    p_uncond_ui,
    reference_param_count,
    train_losses,
):
    if not train_losses:
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX (Apple Silicon)
    - **Dataset**: MNIST (28×28×1, [0, 1] normalized, val split 15%)
    - **Model**: `UNetV1` — 3-level UNet with residual blocks,
      GroupNorm, SiLU, sinusoidal time embedding, and one-hot class
      conditioning fused via addition.
    - **Parameters**: `{reference_param_count:,}`
    - **Noise schedule**: linear β from `1e-4` to `2e-2` over
      `T=1000` steps.
    - **Sampler**: 1000-step DDPM reverse process with
      classifier-free guidance (null class index = 10).

    Train the model in Section 5, then use the sampling widget in
    Section 6 to generate digits.
    """
        )
    else:
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX (Apple Silicon)
    - **Dataset**: MNIST (60k train split → 85% train / 15% val, 10k test)
    - **Model**: `UNetV1` — 3-level UNet (`{reference_param_count:,}` params)
    - **Training**: {epochs_ui.value} epoch(s) at lr={lr_ui.value},
      p_uncond={p_uncond_ui.value}
    - **Final MSE loss**: `{train_losses[-1]:.4f}`
    - **Sampling**: 1000 DDPM steps, current guidance scale `{guidance_ui.value:.1f}`

    ### Notes
    - Loss is the standard DDPM ε-prediction MSE. It should quickly drop to
      the `1e-2` range and then slowly improve; visual sample quality is
      more diagnostic than the exact loss value.
    - Classifier-free guidance is enabled by training with a probability
      `p_uncond` of replacing the class label with the null class (index
      10). At sampling time, higher guidance scales sharpen adherence to
      the requested digit at the cost of diversity.
    - The convolutional UNet operates in MLX's channels-last (NHWC)
      layout, which is why `nn.GroupNorm` uses `pytorch_compatible=True`
      for standard PyTorch-style behavior.
    - No cross-validation section is included: DDPM training is
      unsupervised in the usual sense (the "labels" are only conditioning
      inputs), and evaluating diffusion models by loss on a held-out set
      does not correlate closely with sample quality. The interactive
      sampling widget provides the qualitative evaluation instead.
    """
        )
    _summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Save Trained Model

    Persist the trained `UNetV1` weights to the project's `models/`
    directory. The file extension chosen determines the on-disk format:
    `.safetensors` or `.npz` (both natively supported by MLX).
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="mnist_ddpm_cfd_v1.safetensors",
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
