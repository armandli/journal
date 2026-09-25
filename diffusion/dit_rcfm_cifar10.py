import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import json
    import math
    import random
    import time
    from dataclasses import dataclass, asdict, field
    from pathlib import Path
    from typing import Callable, Dict, List, Optional, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch import nn


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DiT + 2D RoPE + Rectified Conditional Flow Matching on CIFAR-10

    ## Research Goal

    Train a **Diffusion Transformer** (DiT with adaLN-Zero blocks) on CIFAR-10 using
    **rectified conditional flow matching** with **2D axial rotary position embedding**
    (RoPE) as the *only* positional signal. The model is class-conditional with
    classifier-free-guidance support, runs on both CPU and CUDA (mixed-precision by
    default), is fully configurable, and can be *reflowed* (Liu et al. 2022) to obtain a
    2-rectified flow that samples correctly in only a handful of Euler steps.

    ## Flow Convention

    Let $x_0 \sim \mathcal{N}(0, I)$ be the *source* (noise) and $x_1$ be a data sample.
    We use straight, linear conditional paths (rectified flow, $\sigma_\min = 0$):

    $$x_t \;=\; (1 - t)\, x_0 \;+\; t\, x_1, \qquad t \in [0, 1].$$

    The pointwise time derivative gives a natural regression target for the velocity
    field:

    $$v^{*}(x_t, t) \;=\; \frac{d x_t}{d t} \;=\; x_1 - x_0.$$

    The **conditional flow-matching** objective (Lipman et al. 2023, Liu et al. 2022,
    Albergo & Vanden-Eijnden 2023) is then

    $$\mathcal{L}_{\text{CFM}}(\theta) \;=\; \mathbb{E}_{t,\, x_0,\, (x_1, y)}
        \Big\lVert\, v_{\theta}(x_t,\, t,\, y) \,-\, (x_1 - x_0) \,\Big\rVert^{2}.$$

    Generation solves the initial-value problem forward from $t = 0$ to $t = 1$:

    $$\frac{d x}{d t} \;=\; v_{\theta}(x, t, y), \qquad x(0) = x_0 \sim \mathcal{N}(0, I).$$

    Inversion (used to build reflow pairs) solves the **same** ODE **backward** from
    $t = 1$ to $t = 0$ with Euler steps.

    ## "Conditional" — the two meanings we implement

    1. **Conditional flow matching (per-sample conditional path)** — the loss above is
       an *unbiased* estimator of the *marginal* flow-matching loss because we sample a
       fresh $x_0$ for every training example, defining a per-sample conditional path.
       This is the "conditional" in *conditional flow matching*.
    2. **Class-conditioning + classifier-free guidance** — the model also conditions on
       the CIFAR-10 class label $y$; during training we drop $y$ to a learned null
       class with a configurable probability, and at sampling time we mix conditional
       and unconditional velocity with a guidance scale $w$:
       $\tilde{v} = v_{\text{uncond}} + w \, (v_{\text{cond}} - v_{\text{uncond}})$
       (Ho & Salimans 2022). Guidance scale $w = 1$ recovers standard conditional
       sampling.

    ## Outline (11 sections)

    1. **Title & Research Goal** — this cell.
    2. **Data Exploration** — CIFAR-10 samples, class distribution, channel stats.
    3. **Dataset Creation** — 45,000 / 5,000 / 10,000 train/val/test tensors in $[-1, 1]$.
    4. **Model Definition** — versioned DiT modules, configurable, with a builder
       function and flow-matching helpers (loss, Euler ODE, CFG, straightness).
    5. **Training (1-Rectified Flow)** — AdamW + linear warmup + AMP + EMA, gated by
       a run button.
    6. **Hyperparameter Search** — small grid (optional).
    7. **Validation & Cross-Validation** — mean loss, per-timestep-bin loss,
       straightness, 5-fold CV.
    8. **Reflow (2-Rectified Flow)** — invert the 1st-gen model over the whole
       train + val split to build $(x_0, x_1, y)$ pairs; retrain on those pairs.
    9. **Denoising Process Demonstration** — snapshots of $x_t$ and $\hat{x}_1$ at
       evenly spaced $t$ for each generation.
    10. **Results** — loss curves, class-conditional samples, few-step comparison
        (1, 2, 4, 8, 16, 64 Euler steps), model comparison table.
    11. **Save Trained Model** — state dict + JSON config sidecar under `models/`.
    """)
    return


@app.function
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.function
def build_device_options() -> Dict[str, torch.device]:
    opts: Dict[str, torch.device] = {}
    if torch.cuda.is_available():
        opts["cuda"] = torch.device("cuda")
    opts["cpu"] = torch.device("cpu")
    return opts


@app.function
def select_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


@app.function
def make_grad_scaler(device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> "torch.amp.GradScaler":
    return torch.amp.GradScaler(device.type, enabled=(use_amp and amp_dtype == torch.float16))


@app.function
def amp_backward_supported(device: torch.device, amp_dtype: torch.dtype) -> Tuple[bool, str]:
    probe = nn.ModuleDict(
        {
            "patch": nn.Conv2d(3, 8, kernel_size=2, stride=2),
            "norm": nn.LayerNorm(8, elementwise_affine=False),
            "qkv": nn.Linear(8, 24),
            "out": nn.Linear(8, 8),
        }
    ).to(device)
    x = torch.randn(2, 3, 4, 4, device=device)
    try:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            h = probe["patch"](x).flatten(2).transpose(1, 2)
            q, k, v = probe["qkv"](probe["norm"](h)).view(2, 4, 3, 2, 4).permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(2, 4, 8)
            out = probe["out"](F.gelu(attn, approximate="tanh") * F.silu(h))
        out.float().pow(2).mean().backward()
    except RuntimeError as err:
        return False, str(err).splitlines()[0]
    return True, ""


@app.function
def format_amp_status(
    device: torch.device,
    amp_dtype: torch.dtype,
    requested: bool,
    supported: bool,
    probe_error: str,
) -> str:
    dtype_name = str(amp_dtype).replace("torch.", "")
    if not requested:
        return f"`disabled` (unchecked) — training runs in float32 on `{device.type}`"
    if supported:
        return f"`enabled`, dtype = `{dtype_name}`"
    return (
        f"`disabled automatically` — a {dtype_name} autocast backward probe failed on "
        f"`{device.type}` (`{probe_error}`), so training runs in float32 on this device. "
        "This is a backend/hardware limitation (e.g. oneDNN has no bf16 convolution "
        "backward on AVX-VNNI-2 CPUs), not a model error."
    )


@app.function
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@app.cell
def _(mo):
    seed_ui = mo.ui.number(value=1337, label="Seed", start=0, stop=2**31 - 1)
    device_options = build_device_options()
    device_ui = mo.ui.dropdown(
        options=device_options,
        value="cuda" if "cuda" in device_options else "cpu",
        label="Device",
    )
    amp_ui = mo.ui.checkbox(value=True, label="Use Mixed Precision (AMP)")
    mo.hstack([seed_ui, device_ui, amp_ui])
    return amp_ui, device_ui, seed_ui


@app.cell
def _(amp_ui, device_ui, mo, seed_ui):
    device = device_ui.value
    amp_dtype = select_amp_dtype(device)
    amp_supported, amp_probe_error = amp_backward_supported(device, amp_dtype)
    use_amp = bool(amp_ui.value) and amp_supported
    set_seed(int(seed_ui.value))
    mo.md(
        f"""**Active device**: `{device}`

    **AMP**: {format_amp_status(device, amp_dtype, bool(amp_ui.value), amp_supported, amp_probe_error)}

    **Seed**: `{int(seed_ui.value)}`
    """
    )
    return amp_dtype, device, use_amp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data Exploration

    CIFAR-10 is downloaded (if missing) into the repo-root `data/cifar10` directory.
    All 60,000 images (50k train + 10k test) are loaded once as a `uint8`
    NCHW tensor on CPU. This bulk-tensor layout lets us do fast slicing +
    on-device normalization + optional horizontal-flip augmentation each batch,
    which is critical for CPU training and sidesteps DataLoader multiprocessing
    pitfalls inside marimo (we do not use `torch.utils.data.DataLoader` for the
    hot loop).
    """)
    return


@app.function
def load_cifar10_raw(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    train_images = torch.from_numpy(train_ds.data).permute(0, 3, 1, 2).contiguous()
    train_labels = torch.tensor(train_ds.targets, dtype=torch.int64)
    test_images = torch.from_numpy(test_ds.data).permute(0, 3, 1, 2).contiguous()
    test_labels = torch.tensor(test_ds.targets, dtype=torch.int64)
    class_names = list(train_ds.classes)
    return train_images, train_labels, test_images, test_labels, class_names


@app.cell
def _():
    cifar_dir = Path("~") / "data" / "cifar10"
    train_images_all, train_labels_all, test_images, test_labels, class_names = load_cifar10_raw(cifar_dir)
    return (
        class_names,
        test_images,
        test_labels,
        train_images_all,
        train_labels_all,
    )


@app.function
def plot_sample_grid(
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    n_show: int = 40,
    cols: int = 10,
    title: str = "CIFAR-10 samples",
):
    rows = int(math.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.35))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n_show:
            img = images_uint8[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title(class_names[int(labels[i].item())], fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, train_images_all, train_labels_all):
    plot_sample_grid(train_images_all, train_labels_all, class_names, n_show=40, cols=10)
    return


@app.function
def plot_class_distribution(labels: torch.Tensor, class_names: List[str], title: str = "Class distribution"):
    counts = torch.bincount(labels, minlength=len(class_names)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(class_names, counts, color="steelblue")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, train_labels_all):
    plot_class_distribution(train_labels_all, class_names, title="Train class distribution")
    return


@app.function
def plot_channel_histograms(images_uint8: torch.Tensor, title: str = "Per-channel intensity in [-1, 1]"):
    normalized = images_uint8.float().div(127.5).sub(1.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    ch_names = ["R", "G", "B"]
    colors = ["red", "green", "blue"]
    for i in range(3):
        axes[i].hist(normalized[:, i].flatten().cpu().numpy(), bins=60, color=colors[i], alpha=0.7)
        axes[i].set_title(f"{ch_names[i]} channel")
        axes[i].set_xlim(-1.05, 1.05)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.cell
def _(train_images_all):
    plot_channel_histograms(train_images_all[:5000])
    return


@app.function
def channel_stats(images_uint8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = images_uint8.float().div(127.5).sub(1.0)
    return x.mean(dim=[0, 2, 3]), x.std(dim=[0, 2, 3])


@app.cell
def _(mo, test_images, train_images_all):
    mean_stats, std_stats = channel_stats(train_images_all)
    mo.md(
        f"""
    | Split | Images | Notes |
    |---|---|---|
    | Full train (before split) | {train_images_all.shape[0]:,} | Section 3 splits this into 45,000 train + 5,000 val. |
    | Test | {test_images.shape[0]:,} | Held out for Section 7 evaluation. |

    **Per-channel mean** (normalized to $[-1, 1]$): R = {mean_stats[0]:.4f}, G = {mean_stats[1]:.4f}, B = {mean_stats[2]:.4f}

    **Per-channel std**: R = {std_stats[0]:.4f}, G = {std_stats[1]:.4f}, B = {std_stats[2]:.4f}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Dataset Creation

    - **Splits** — 50,000 raw train images are split *deterministically* (seeded
      `torch.Generator`) into **45,000 train** + **5,000 validation**. The 10,000
      test images are held out for Section 7.
    - **Normalization** — we scale $[0, 255] \to [-1, 1]$ float32. This deliberately
      overrides the template's $[0, 1]$ default because the *source* distribution is
      $\mathcal{N}(0, I)$: matching the data support to the noise support makes flow
      matching numerically much better behaved.
    - **Augmentation** — optional random horizontal flip is applied on-device per
      batch during training only (toggled from the Section 5 controls).
    """)
    return


@app.function
def make_datasets(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_fraction: float = 0.1,
    seed: int = 1234,
) -> Dict[str, torch.Tensor]:
    n = train_images.shape[0]
    n_val = int(round(n * val_fraction))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return {
        "train_images": train_images[tr_idx].contiguous(),
        "train_labels": train_labels[tr_idx].contiguous(),
        "val_images": train_images[val_idx].contiguous(),
        "val_labels": train_labels[val_idx].contiguous(),
        "train_indices": tr_idx,
        "val_indices": val_idx,
    }


@app.function
def normalize_to_neg1_1(images_uint8: torch.Tensor) -> torch.Tensor:
    return images_uint8.float().div(127.5).sub(1.0)


@app.function
def denormalize_from_neg1_1(images_norm: torch.Tensor) -> torch.Tensor:
    return images_norm.add(1.0).mul(127.5).clamp(0.0, 255.0)


@app.cell
def _(train_images_all, train_labels_all):
    splits = make_datasets(train_images_all, train_labels_all, val_fraction=0.1, seed=1234)
    train_images_uint8 = splits["train_images"]
    train_labels = splits["train_labels"]
    val_images_uint8 = splits["val_images"]
    val_labels = splits["val_labels"]
    train_indices = splits["train_indices"]
    val_indices = splits["val_indices"]
    return train_images_uint8, train_labels, val_images_uint8, val_labels


@app.cell
def _(mo, test_images, train_images_uint8, val_images_uint8):
    _train_norm = normalize_to_neg1_1(train_images_uint8[:8])
    mo.md(
        f"""
    | Split | Shape | dtype |
    |---|---|---|
    | Train images (uint8) | `{tuple(train_images_uint8.shape)}` | `{train_images_uint8.dtype}` |
    | Train labels | `{tuple(train_images_uint8.shape[:1])}` | `int64` |
    | Val images (uint8) | `{tuple(val_images_uint8.shape)}` | `{val_images_uint8.dtype}` |
    | Test images (uint8) | `{tuple(test_images.shape)}` | `{test_images.dtype}` |
    | Sample normalized batch (float32, first 8) | `{tuple(_train_norm.shape)}` | `{_train_norm.dtype}`, range = [{_train_norm.min().item():.3f}, {_train_norm.max().item():.3f}] |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Model Definition — Diffusion Transformer with 2D Axial RoPE

    ### Architecture summary

    | Component | Class | Notes |
    |---|---|---|
    | Patch embed | `PatchEmbedV1` | `Conv2d(kernel=stride=patch_size)` |
    | Time embed | `TimestepEmbedV1` | sinusoidal @ $t \times 1000$ + 2-layer MLP |
    | Label embed | `LabelEmbedV1` | Embedding with an extra null class; training-time dropout |
    | 2D axial RoPE | `RotaryEmbedding2DV1` | first $D/2$ dims rotate with row index, last $D/2$ with column index |
    | Attention | `MultiHeadSelfAttentionRoPEV1` | RoPE on $q, k$; `F.scaled_dot_product_attention` |
    | Feed-forward | `FeedForwardV1` | GELU MLP, `mlp_ratio` × hidden dim |
    | DiT block | `DiTBlockV1` | pre-norm + adaLN-Zero (shift/scale/gate) |
    | Final layer | `FinalLayerV1` | adaLN + zero-init linear + unpatchify |
    | Top-level | `DiffusionTransformerV1` | composes all of the above, predicts velocity |

    RoPE is the *only* positional signal — there is no absolute position embedding.
    All adaLN-Zero modulation and the final projection are zero-initialized so the
    residual stream starts as the identity function.
    """)
    return


@app.class_definition
@dataclass
class DiTConfigV1:
    image_size: int = 32
    in_channels: int = 3
    patch_size: int = 4
    hidden_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 10
    class_dropout_prob: float = 0.1
    rope_base: float = 10000.0
    time_max_period: float = 10000.0
    time_scale: float = 1000.0

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")
        head = self.hidden_dim // self.num_heads
        if head % 4 != 0:
            raise ValueError(f"head_dim ({head}) must be divisible by 4 for 2D axial RoPE.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size}).")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size


@app.function
def dit_presets() -> Dict[str, Dict]:
    return {
        "cpu-tiny": {
            "patch_size": 4,
            "hidden_dim": 256,
            "depth": 6,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        },
        "gpu-s2": {
            "patch_size": 2,
            "hidden_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
    }


@app.function
def make_dit_config(**fields) -> Tuple[Optional[DiTConfigV1], str]:
    try:
        return DiTConfigV1(**fields), ""
    except (TypeError, ValueError) as err:
        return None, str(err)


@app.function
def build_2d_rope_cache(
    grid_size: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2D axial RoPE.")
    d_axis = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, d_axis, 2, dtype=torch.float32) / d_axis))
    row_positions = (
        torch.arange(grid_size, dtype=torch.float32)
        .view(-1, 1)
        .expand(-1, grid_size)
        .reshape(-1)
    )
    col_positions = (
        torch.arange(grid_size, dtype=torch.float32)
        .view(1, -1)
        .expand(grid_size, -1)
        .reshape(-1)
    )
    angles_row = torch.outer(row_positions, freqs)
    angles_col = torch.outer(col_positions, freqs)
    cos_row = torch.cat([angles_row.cos(), angles_row.cos()], dim=-1)
    sin_row = torch.cat([angles_row.sin(), angles_row.sin()], dim=-1)
    cos_col = torch.cat([angles_col.cos(), angles_col.cos()], dim=-1)
    sin_col = torch.cat([angles_col.sin(), angles_col.sin()], dim=-1)
    return cos_row, sin_row, cos_col, sin_col


@app.function
def rotate_half_last_dim(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


@app.function
def apply_rotary_embedding(
    x: torch.Tensor,
    cos_row: torch.Tensor,
    sin_row: torch.Tensor,
    cos_col: torch.Tensor,
    sin_col: torch.Tensor,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    d = x_f.shape[-1]
    d_axis = d // 2
    x_row = x_f[..., :d_axis]
    x_col = x_f[..., d_axis:]
    x_row_rot = x_row * cos_row + rotate_half_last_dim(x_row) * sin_row
    x_col_rot = x_col * cos_col + rotate_half_last_dim(x_col) * sin_col
    return torch.cat([x_row_rot, x_col_rot], dim=-1).to(orig_dtype)


@app.class_definition
class RotaryEmbedding2DV1(nn.Module):
    def __init__(self, grid_size: int = 8, head_dim: int = 64, base: float = 10000.0):
        super().__init__()
        cos_row, sin_row, cos_col, sin_col = build_2d_rope_cache(grid_size, head_dim, base)
        self.register_buffer("cos_row", cos_row, persistent=False)
        self.register_buffer("sin_row", sin_row, persistent=False)
        self.register_buffer("cos_col", cos_col, persistent=False)
        self.register_buffer("sin_col", sin_col, persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cos_row, self.sin_row, self.cos_col, self.sin_col


@app.class_definition
class PatchEmbedV1(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return h.flatten(2).transpose(1, 2)


@app.class_definition
class TimestepEmbedV1(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        frequency_dim: int = 256,
        max_period: float = 10000.0,
        scale: float = 1000.0,
    ):
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.max_period = float(max_period)
        self.scale = float(scale)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t.to(torch.float32) * self.scale
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = t_scaled[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.frequency_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


@app.class_definition
class LabelEmbedV1(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_dim: int = 256, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout_prob = float(dropout_prob)
        self.embedding = nn.Embedding(self.num_classes + 1, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, y: torch.Tensor, train_dropout: bool = False) -> torch.Tensor:
        if train_dropout and self.dropout_prob > 0.0:
            mask = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            null_y = torch.full_like(y, self.num_classes)
            y = torch.where(mask, null_y, y)
        return self.embedding(y)


@app.class_definition
class MultiHeadSelfAttentionRoPEV1(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // self.num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos_row: torch.Tensor,
        sin_row: torch.Tensor,
        cos_col: torch.Tensor,
        sin_col: torch.Tensor,
    ) -> torch.Tensor:
        b, l, h = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_embedding(q, cos_row, sin_row, cos_col, sin_col)
        k = apply_rotary_embedding(k, cos_row, sin_row, cos_col, sin_col)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b, l, h)
        return self.proj(out)


@app.class_definition
class FeedForwardV1(nn.Module):
    def __init__(self, hidden_dim: int = 256, mlp_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@app.class_definition
class DiTBlockV1(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttentionRoPEV1(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForwardV1(hidden_dim, int(hidden_dim * mlp_ratio))
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos_row: torch.Tensor,
        sin_row: torch.Tensor,
        cos_col: torch.Tensor,
        sin_col: torch.Tensor,
    ) -> torch.Tensor:
        mods = self.ada_mod(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        h1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(h1, cos_row, sin_row, cos_col, sin_col)
        h2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h2)
        return x


@app.class_definition
class FinalLayerV1(nn.Module):
    def __init__(self, hidden_dim: int = 256, patch_size: int = 4, out_channels: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mods = self.ada_mod(c)
        shift, scale = mods.chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(h)


@app.class_definition
class DiffusionTransformerV1(nn.Module):
    def __init__(self, config: DiTConfigV1):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedV1(
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
        )
        self.time_embed = TimestepEmbedV1(
            hidden_dim=config.hidden_dim,
            frequency_dim=config.hidden_dim,
            max_period=config.time_max_period,
            scale=config.time_scale,
        )
        self.label_embed = LabelEmbedV1(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            dropout_prob=config.class_dropout_prob,
        )
        self.rope = RotaryEmbedding2DV1(
            grid_size=config.grid_size,
            head_dim=config.head_dim,
            base=config.rope_base,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlockV1(config.hidden_dim, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final = FinalLayerV1(
            hidden_dim=config.hidden_dim,
            patch_size=config.patch_size,
            out_channels=config.in_channels,
        )

    @property
    def null_index(self) -> int:
        return self.config.num_classes

    def unpatchify(self, h: torch.Tensor) -> torch.Tensor:
        b, l, d = h.shape
        p = self.config.patch_size
        c = self.config.in_channels
        g = self.config.grid_size
        h = h.reshape(b, g, g, p, p, c)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
        return h.reshape(b, c, g * p, g * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x)
        t_emb = self.time_embed(t)
        y_emb = self.label_embed(y, train_dropout=self.training)
        cond = t_emb + y_emb
        cos_row, sin_row, cos_col, sin_col = self.rope()
        for block in self.blocks:
            h = block(h, cond, cos_row, sin_row, cos_col, sin_col)
        h = self.final(h, cond)
        return self.unpatchify(h)


@app.function
def build_model_from_config(config: DiTConfigV1, device: Optional[torch.device] = None) -> DiffusionTransformerV1:
    model = DiffusionTransformerV1(config)
    if device is not None:
        model = model.to(device)
    return model


@app.function
def sample_timesteps(
    batch_size: int,
    scheme: str = "uniform",
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if scheme == "uniform":
        if generator is not None:
            return torch.rand(batch_size, generator=generator).to(device) if device is not None else torch.rand(batch_size, generator=generator)
        return torch.rand(batch_size, device=device)
    if scheme == "logit_normal":
        if generator is not None:
            u = torch.randn(batch_size, generator=generator)
            u = u.to(device) if device is not None else u
        else:
            u = torch.randn(batch_size, device=device)
        return torch.sigmoid(u)
    raise ValueError(f"Unknown timestep scheme: {scheme!r}")


@app.function
def interpolate_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_view = t.view(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - t_view) * x0 + t_view * x1


@app.function
def flow_matching_loss(
    model: nn.Module,
    x1: torch.Tensor,
    y: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    t_scheme: str = "uniform",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if x0 is None:
        x0 = torch.randn_like(x1)
    t = sample_timesteps(x1.shape[0], t_scheme, device=x1.device, generator=generator)
    xt = interpolate_path(x0, x1, t)
    target_v = x1 - x0
    pred_v = model(xt, t, y)
    return F.mse_loss(pred_v.to(torch.float32), target_v.to(torch.float32))


@app.function
def classifier_free_velocity(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    null_index: int,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return model(x_t, t, y)
    y_null = torch.full_like(y, null_index)
    v_cond = model(x_t, t, y)
    v_uncond = model(x_t, t, y_null)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


@app.function
def euler_time_grid(
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    data_end_power: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if data_end_power <= 0.0:
        raise ValueError(f"data_end_power must be > 0, got {data_end_power}.")
    if data_end_power == 1.0:
        return torch.linspace(t_start, t_end, num_steps + 1, device=device, dtype=torch.float32)
    s_start = (1.0 - t_start) ** (1.0 / data_end_power)
    s_end = (1.0 - t_end) ** (1.0 / data_end_power)
    s = torch.linspace(s_start, s_end, num_steps + 1, device=device, dtype=torch.float32)
    return 1.0 - s.pow(data_end_power)


@app.function
def euler_integrate(
    model: nn.Module,
    x_start: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    null_index: int,
    guidance_scale: float = 1.0,
    t_start: float = 0.0,
    t_end: float = 1.0,
    capture_indices: Optional[set] = None,
    data_end_power: float = 1.0,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    x = x_start.to(torch.float32)
    ts = euler_time_grid(num_steps, t_start, t_end, data_end_power, device=x.device)
    snapshots: Dict[int, torch.Tensor] = {}
    if capture_indices is not None and 0 in capture_indices:
        snapshots[0] = x.detach().clone()
    for i in range(num_steps):
        t_i = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = t_i.expand(x.shape[0])
        v = classifier_free_velocity(model, x, t_batch, y, null_index, guidance_scale=guidance_scale)
        x = x + v.to(torch.float32) * dt
        if capture_indices is not None and (i + 1) in capture_indices:
            snapshots[i + 1] = x.detach().clone()
    return x, snapshots


@app.function
def sample_images(
    model: nn.Module,
    y: torch.Tensor,
    image_shape: Tuple[int, int, int],
    num_steps: int,
    null_index: int,
    guidance_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    capture_indices: Optional[set] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    b = y.shape[0]
    full_shape = (b, *image_shape)
    if generator is not None:
        x0 = torch.randn(full_shape, generator=generator)
        if device is not None:
            x0 = x0.to(device)
    else:
        x0 = torch.randn(full_shape, device=device if device is not None else y.device)
    return euler_integrate(
        model=model,
        x_start=x0,
        y=y,
        num_steps=num_steps,
        null_index=null_index,
        guidance_scale=guidance_scale,
        t_start=0.0,
        t_end=1.0,
        capture_indices=capture_indices,
    )


@app.function
def invert_images(
    model: nn.Module,
    x1: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    null_index: int,
    guidance_scale: float = 1.0,
    capture_indices: Optional[set] = None,
    data_end_power: float = 1.0,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    return euler_integrate(
        model=model,
        x_start=x1,
        y=y,
        num_steps=num_steps,
        null_index=null_index,
        guidance_scale=guidance_scale,
        t_start=1.0,
        t_end=0.0,
        capture_indices=capture_indices,
        data_end_power=data_end_power,
    )


@app.function
def straightness_metric(
    model: nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    null_index: int,
    guidance_scale: float = 1.0,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        capture = set(range(num_steps + 1))
        x_final, snapshots = euler_integrate(
            model=model,
            x_start=x0,
            y=y,
            num_steps=num_steps,
            null_index=null_index,
            guidance_scale=guidance_scale,
            t_start=0.0,
            t_end=1.0,
            capture_indices=capture,
        )
        target = x_final - x0
        ts = euler_time_grid(num_steps, 0.0, 1.0, device=x0.device)
        for i in range(num_steps + 1):
            t_batch = ts[i].expand(x0.shape[0])
            v_pred = classifier_free_velocity(
                model,
                snapshots[i],
                t_batch,
                y,
                null_index,
                guidance_scale=guidance_scale,
            ).to(torch.float32)
            total += float(((target - v_pred) ** 2).mean().item())
            n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


@app.function
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float, num_updates: Optional[int] = None) -> None:
    if num_updates is not None:
        decay = min(decay, (1.0 + num_updates) / (10.0 + num_updates))
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        for name, buf in model.named_buffers():
            ema_buffers[name].data.copy_(buf.data)


@app.function
def run_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    train_images_uint8: torch.Tensor,
    train_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_clip: float = 1.0,
    t_scheme: str = "uniform",
    augment_flip: bool = True,
    generator: Optional[torch.Generator] = None,
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.0,
    x0_tensor: Optional[torch.Tensor] = None,
    start_step: int = 0,
) -> float:
    model.train()
    n = train_images_uint8.shape[0]
    idx = torch.randperm(n, generator=generator)
    losses: List[float] = []
    for step, start in enumerate(range(0, n, batch_size), start=start_step):
        batch_idx = idx[start : start + batch_size]
        x1 = train_images_uint8[batch_idx].to(device, non_blocking=True).float().div(127.5).sub(1.0)
        y = train_labels[batch_idx].to(device, non_blocking=True)
        if augment_flip and x0_tensor is None:
            flip = torch.rand(x1.shape[0], device=device) < 0.5
            x1 = torch.where(flip[:, None, None, None], x1.flip(dims=[-1]), x1)
        x0 = None
        if x0_tensor is not None:
            x0 = x0_tensor[batch_idx].to(device, non_blocking=True).to(torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = flow_matching_loss(model, x1, y, x0=x0, t_scheme=t_scheme)
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema(ema_model, model, ema_decay, num_updates=step)
        losses.append(float(loss.detach().item()))
    return sum(losses) / max(len(losses), 1)


@app.function
def run_evaluate(
    model: nn.Module,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    t_scheme: str = "uniform",
    seed: int = 12345,
    x0_tensor: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    n = images_uint8.shape[0]
    losses: List[float] = []
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = torch.arange(start, end)
            x1 = images_uint8[batch_idx].to(device).float().div(127.5).sub(1.0)
            y = labels[batch_idx].to(device)
            if x0_tensor is None:
                x0 = torch.randn(x1.shape, generator=cpu_gen).to(device)
            else:
                x0 = x0_tensor[batch_idx].to(device).to(torch.float32)
            t = sample_timesteps(x1.shape[0], t_scheme, device=None, generator=cpu_gen).to(device)
            xt = interpolate_path(x0, x1, t)
            target_v = x1 - x0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_v = model(xt, t, y)
            loss = F.mse_loss(pred_v.to(torch.float32), target_v)
            losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


@app.function
def evaluate_model(
    model: nn.Module,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    null_index: int,
    num_bins: int = 10,
    straightness_batch: int = 32,
    straightness_steps: int = 50,
    seed: int = 2026,
) -> Dict[str, object]:
    model.eval()
    n = images_uint8.shape[0]
    sum_loss = 0.0
    count = 0
    bin_sums = [0.0] * num_bins
    bin_counts = [0] * num_bins
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = torch.arange(start, end)
            x1 = images_uint8[batch_idx].to(device).float().div(127.5).sub(1.0)
            y = labels[batch_idx].to(device)
            x0 = torch.randn(x1.shape, generator=cpu_gen).to(device)
            t = torch.rand(x1.shape[0], generator=cpu_gen).to(device)
            xt = interpolate_path(x0, x1, t)
            target_v = x1 - x0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_v = model(xt, t, y)
            per_sample = ((pred_v.to(torch.float32) - target_v) ** 2).mean(dim=[1, 2, 3])
            sum_loss += float(per_sample.sum().item())
            count += per_sample.shape[0]
            bins = (t * num_bins).long().clamp(0, num_bins - 1)
            for b in range(num_bins):
                mask = bins == b
                if mask.any():
                    bin_sums[b] += float(per_sample[mask].sum().item())
                    bin_counts[b] += int(mask.sum().item())
        x0_s = torch.randn(straightness_batch, images_uint8.shape[1], images_uint8.shape[2], images_uint8.shape[3], generator=cpu_gen).to(device)
        y_s = torch.randint(0, max(int(labels.max().item()) + 1, 1), (straightness_batch,), generator=cpu_gen).to(device)
    straightness = straightness_metric(model, x0_s, y_s, straightness_steps, null_index, guidance_scale=1.0)
    loss_per_bin = [bin_sums[b] / max(bin_counts[b], 1) for b in range(num_bins)]
    return {
        "mean_loss": sum_loss / max(count, 1),
        "loss_per_bin": loss_per_bin,
        "bin_counts": bin_counts,
        "straightness": straightness,
    }


@app.function
def build_reflow_pairs(
    model: nn.Module,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    batch_size: int,
    device: torch.device,
    null_index: int,
    guidance_scale: float = 1.0,
    storage_dtype: torch.dtype = torch.float32,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    data_end_power: float = 1.0,
) -> torch.Tensor:
    model.eval()
    n = images_uint8.shape[0]
    c, h, w = images_uint8.shape[1], images_uint8.shape[2], images_uint8.shape[3]
    storage = torch.empty((n, c, h, w), dtype=storage_dtype)
    processed = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x1 = images_uint8[start:end].to(device).float().div(127.5).sub(1.0)
            y = labels[start:end].to(device)
            x0, _ = invert_images(
                model,
                x1,
                y,
                num_steps,
                null_index,
                guidance_scale=guidance_scale,
                data_end_power=data_end_power,
            )
            storage[start:end] = x0.to(storage_dtype).cpu()
            processed += end - start
            if progress_cb is not None:
                progress_cb(processed, n)
    return storage


@app.function
def roundtrip_reconstruction(
    model: nn.Module,
    images_uint8: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    null_index: int,
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    data_end_power: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x1 = images_uint8.to(device).float().div(127.5).sub(1.0)
        y = labels.to(device)
        x0, _ = invert_images(
            model,
            x1,
            y,
            num_steps,
            null_index,
            guidance_scale=guidance_scale,
            data_end_power=data_end_power,
        )
        x1_hat, _ = euler_integrate(
            model,
            x0,
            y,
            num_steps,
            null_index,
            guidance_scale,
            0.0,
            1.0,
            data_end_power=data_end_power,
        )
        mse = float(((x1_hat - x1) ** 2).mean().item())
        return {
            "roundtrip_mse": mse,
            "x0_mean": float(x0.mean().item()),
            "x0_std": float(x0.std().item()),
        }


@app.class_definition
class ReflowPairDatasetV1(torch.utils.data.Dataset):
    def __init__(self, x0_tensor: torch.Tensor, x1_images_uint8: torch.Tensor, labels: torch.Tensor):
        if not (len(x0_tensor) == len(x1_images_uint8) == len(labels)):
            raise ValueError("x0, x1 and labels must have the same length.")
        self.x0 = x0_tensor
        self.x1 = x1_images_uint8
        self.labels = labels

    def __len__(self) -> int:
        return int(self.x0.shape[0])

    def __getitem__(self, idx: int):
        x0 = self.x0[idx]
        x1 = self.x1[idx].float().div(127.5).sub(1.0)
        y = self.labels[idx]
        return x0, x1, y


@app.cell
def _(mo):
    preset_ui = mo.ui.dropdown(
        options=list(dit_presets().keys()),
        value="cpu-tiny",
        label="Model Preset",
    )
    mo.vstack([mo.md("### Model Config"), preset_ui])
    return (preset_ui,)


@app.cell
def _(mo, preset_ui):
    preset_values = dit_presets()[preset_ui.value]
    patch_size_ui = mo.ui.dropdown(options=[2, 4, 8], value=preset_values["patch_size"], label="Patch Size")
    hidden_dim_ui = mo.ui.dropdown(
        options=[128, 192, 256, 384, 512],
        value=preset_values["hidden_dim"],
        label="Hidden Dim",
    )
    depth_ui = mo.ui.slider(2, 16, value=preset_values["depth"], step=1, label="Depth")
    num_heads_ui = mo.ui.dropdown(options=[2, 4, 6, 8], value=preset_values["num_heads"], label="Num Heads")
    mlp_ratio_ui = mo.ui.dropdown(
        options={"2.0": 2.0, "3.0": 3.0, "4.0": 4.0},
        value=f"{float(preset_values['mlp_ratio']):.1f}",
        label="MLP Ratio",
    )
    class_dropout_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "0.1": 0.1, "0.15": 0.15, "0.2": 0.2},
        value="0.1",
        label="Class Dropout",
    )
    mo.vstack(
        [
            mo.md(f"_Fields start from the `{preset_ui.value}` preset; edit any of them to override it._"),
            mo.hstack([patch_size_ui, hidden_dim_ui, num_heads_ui]),
            mo.hstack([depth_ui, mlp_ratio_ui, class_dropout_ui]),
        ]
    )
    return (
        class_dropout_ui,
        depth_ui,
        hidden_dim_ui,
        mlp_ratio_ui,
        num_heads_ui,
        patch_size_ui,
    )


@app.cell
def _(
    class_dropout_ui,
    depth_ui,
    hidden_dim_ui,
    mlp_ratio_ui,
    mo,
    num_heads_ui,
    patch_size_ui,
):
    model_cfg, model_cfg_error = make_dit_config(
        image_size=32,
        in_channels=3,
        patch_size=int(patch_size_ui.value),
        hidden_dim=int(hidden_dim_ui.value),
        depth=int(depth_ui.value),
        num_heads=int(num_heads_ui.value),
        mlp_ratio=float(mlp_ratio_ui.value),
        num_classes=10,
        class_dropout_prob=float(class_dropout_ui.value),
    )
    mo.stop(model_cfg is None, mo.md(f"**Invalid model configuration** — {model_cfg_error}"))
    return (model_cfg,)


@app.cell
def _(mo, model_cfg):
    mo.md(f"""
    ### Instantiated `DiffusionTransformerV1`

    | Field | Value |
    |---|---|
    | image_size | {model_cfg.image_size} |
    | in_channels | {model_cfg.in_channels} |
    | patch_size | {model_cfg.patch_size} |
    | grid_size (`image_size / patch_size`) | {model_cfg.grid_size} |
    | num_patches | {model_cfg.num_patches} |
    | hidden_dim | {model_cfg.hidden_dim} |
    | depth | {model_cfg.depth} |
    | num_heads | {model_cfg.num_heads} |
    | head_dim | {model_cfg.head_dim} |
    | mlp_ratio | {model_cfg.mlp_ratio} |
    | class_dropout_prob | {model_cfg.class_dropout_prob} |

    **Total parameters**: `{count_parameters(build_model_from_config(model_cfg)):,}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Training — 1-Rectified Flow

    We train the DiT to regress the linear-path velocity $v^{*} = x_1 - x_0$ with
    fresh $x_0 \sim \mathcal{N}(0, I)$ per training example (this is the standard
    1-rectified-flow / conditional-flow-matching objective). Optimiser: **AdamW**
    with linear warmup, gradient clipping (`none` disables it), optional **EMA** of
    the weights, mixed precision (bf16 on CPU and Ampere+ CUDA, fp16 on older CUDA —
    chosen automatically, and switched off automatically where the backend cannot
    run the autocast backward pass), and configurable label dropout (the model
    config's *Class Dropout*) for classifier-free guidance.

    The EMA uses the standard warmup $\beta_n = \min\!\big(\beta, \tfrac{1 + n}{10 + n}\big)$
    at update $n$, so the averaged weights track training on short runs as well as
    long ones (a fixed $\beta = 0.9999$ would otherwise keep ~75% of the
    initialization after ~2,800 steps). The same `fit_flow_model` rite trains the
    reflow generation in Section 8.
    """)
    return


@app.class_definition
@dataclass
class TrainConfigV1:
    lr: float = 3e-4
    batch_size: int = 128
    weight_decay: float = 0.01
    epochs: int = 8
    warmup_steps: int = 200
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    t_scheme: str = "uniform"
    augment_flip: bool = True
    seed: int = 1337


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"3e-5": 3e-5, "1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(options=[32, 64, 128, 256, 512], value=128, label="Batch Size")
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-2": 1e-2, "5e-2": 5e-2},
        value="1e-2",
        label="Weight Decay",
    )
    epochs_ui = mo.ui.slider(1, 100, value=8, step=1, label="Epochs")
    warmup_ui = mo.ui.dropdown(options=[0, 100, 200, 500, 1000], value=200, label="Warmup Steps")
    grad_clip_ui = mo.ui.dropdown(
        options={"none": 0.0, "0.5": 0.5, "1.0": 1.0, "2.0": 2.0, "5.0": 5.0},
        value="1.0",
        label="Grad Clip",
    )
    ema_ui = mo.ui.dropdown(
        options={"off": 0.0, "0.999": 0.999, "0.9995": 0.9995, "0.9999": 0.9999},
        value="0.9999",
        label="EMA Decay",
    )
    t_scheme_ui = mo.ui.dropdown(options=["uniform", "logit_normal"], value="uniform", label="Timestep Scheme")
    flip_ui = mo.ui.checkbox(value=True, label="Horizontal Flip Aug")
    train_btn = mo.ui.run_button(label="Train (1-Rectified Flow)")
    mo.vstack(
        [
            mo.md("### Training Hyperparameters"),
            mo.hstack([lr_ui, bs_ui, wd_ui, epochs_ui]),
            mo.hstack([warmup_ui, grad_clip_ui, ema_ui, t_scheme_ui]),
            mo.hstack([flip_ui, train_btn]),
        ]
    )
    return (
        bs_ui,
        ema_ui,
        epochs_ui,
        flip_ui,
        grad_clip_ui,
        lr_ui,
        t_scheme_ui,
        train_btn,
        warmup_ui,
        wd_ui,
    )


@app.function
def make_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    if warmup_steps <= 0:
        return None

    def _lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


@app.function
def fit_flow_model(
    model: nn.Module,
    train_cfg: TrainConfigV1,
    train_images_uint8: torch.Tensor,
    train_labels: torch.Tensor,
    val_images_uint8: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    train_x0: Optional[torch.Tensor] = None,
    val_x0: Optional[torch.Tensor] = None,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, object]:
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.99),
    )
    scaler = make_grad_scaler(device, use_amp, amp_dtype)
    warmup = make_warmup_scheduler(optimizer, train_cfg.warmup_steps)
    generator = torch.Generator().manual_seed(train_cfg.seed)
    ema_model: Optional[nn.Module] = None
    if train_cfg.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.requires_grad_(False)
        ema_model.eval()
    steps_per_epoch = math.ceil(train_images_uint8.shape[0] / train_cfg.batch_size)
    train_losses: List[float] = []
    val_losses: List[float] = []
    start_time = time.perf_counter()
    for epoch in range(train_cfg.epochs):
        train_loss = run_train_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_images_uint8=train_images_uint8,
            train_labels=train_labels,
            batch_size=train_cfg.batch_size,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_clip=train_cfg.grad_clip,
            t_scheme=train_cfg.t_scheme,
            augment_flip=train_cfg.augment_flip,
            generator=generator,
            warmup_scheduler=warmup,
            ema_model=ema_model,
            ema_decay=train_cfg.ema_decay,
            x0_tensor=train_x0,
            start_step=epoch * steps_per_epoch,
        )
        val_loss = run_evaluate(
            model=ema_model if ema_model is not None else model,
            images_uint8=val_images_uint8,
            labels=val_labels,
            batch_size=train_cfg.batch_size,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            t_scheme=train_cfg.t_scheme,
            seed=train_cfg.seed + 1000,
            x0_tensor=val_x0,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if progress_cb is not None:
            progress_cb(epoch + 1, train_cfg.epochs, train_loss, val_loss)
    final_model = ema_model if ema_model is not None else model
    final_model.eval()
    return {
        "model": final_model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "wall_time": time.perf_counter() - start_time,
        "ema_used": ema_model is not None,
    }


@app.cell
def _(
    amp_dtype,
    bs_ui,
    device,
    ema_ui,
    epochs_ui,
    flip_ui,
    grad_clip_ui,
    lr_ui,
    mo,
    model_cfg,
    seed_ui,
    t_scheme_ui,
    train_btn,
    train_images_uint8,
    train_labels,
    use_amp,
    val_images_uint8,
    val_labels,
    warmup_ui,
    wd_ui,
):
    gen1_run: Optional[Dict[str, object]] = None
    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train (1-Rectified Flow)** to begin training."))
    else:
        gen1_train_cfg = TrainConfigV1(
            lr=float(lr_ui.value),
            batch_size=int(bs_ui.value),
            weight_decay=float(wd_ui.value),
            epochs=int(epochs_ui.value),
            warmup_steps=int(warmup_ui.value),
            grad_clip=float(grad_clip_ui.value),
            ema_decay=float(ema_ui.value),
            t_scheme=str(t_scheme_ui.value),
            augment_flip=bool(flip_ui.value),
            seed=int(seed_ui.value),
        )
        set_seed(gen1_train_cfg.seed)
        gen1_run = fit_flow_model(
            model=build_model_from_config(model_cfg, device=device),
            train_cfg=gen1_train_cfg,
            train_images_uint8=train_images_uint8,
            train_labels=train_labels,
            val_images_uint8=val_images_uint8,
            val_labels=val_labels,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            progress_cb=lambda epoch, total, tl, vl: mo.output.replace(
                mo.md(f"**Epoch {epoch}/{total}** — train loss: {tl:.4f} | val loss: {vl:.4f}")
            ),
        )
        mo.output.replace(
            mo.md(
                f"**Training complete** in {gen1_run['wall_time']:.1f}s — final train "
                f"{gen1_run['train_losses'][-1]:.4f} | final val {gen1_run['val_losses'][-1]:.4f}"
                f" | using EMA weights: {gen1_run['ema_used']}"
            )
        )
    train_losses: List[float] = gen1_run["train_losses"] if gen1_run else []
    val_losses: List[float] = gen1_run["val_losses"] if gen1_run else []
    trained_model: Optional[nn.Module] = gen1_run["model"] if gen1_run else None
    train_wall_time: float = gen1_run["wall_time"] if gen1_run else 0.0
    trained_ema_used: bool = bool(gen1_run["ema_used"]) if gen1_run else False
    return (
        train_losses,
        train_wall_time,
        trained_ema_used,
        trained_model,
        val_losses,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Hyperparameter Search (optional)

    Small grid over learning rate and hidden dim. Runs a few epochs on a subset
    of the training data per configuration to stay tractable, then reports the
    per-config validation loss.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_epochs_ui = mo.ui.slider(1, 10, value=3, step=1, label="HP-Search Epochs")
    hp_subset_ui = mo.ui.dropdown(
        options=[2000, 5000, 10000, 20000, 45000],
        value=5000,
        label="Train Subset Size",
    )
    hp_run_btn = mo.ui.run_button(label="Run HP Search")
    mo.vstack(
        [
            hp_search_cb,
            mo.hstack([hp_epochs_ui, hp_subset_ui, hp_run_btn]),
        ]
    )
    return hp_epochs_ui, hp_run_btn, hp_search_cb, hp_subset_ui


@app.function
def hp_search_grid() -> List[Dict]:
    grid: List[Dict] = []
    for lr in [1e-4, 3e-4, 1e-3]:
        for hidden_dim in [128, 256]:
            grid.append({"lr": lr, "hidden_dim": hidden_dim})
    return grid


@app.cell
def _(
    amp_dtype,
    device,
    hp_epochs_ui,
    hp_run_btn,
    hp_search_cb,
    hp_subset_ui,
    mo,
    train_images_uint8,
    train_labels,
    use_amp,
    val_images_uint8,
    val_labels,
):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable **Hyperparameter Search** above to run this section._"),
    )
    mo.stop(
        not hp_run_btn.value,
        mo.md("Toggle enabled — now click **Run HP Search** to launch."),
    )
    hp_results: List[Dict] = []
    _subset = min(int(hp_subset_ui.value), train_images_uint8.shape[0])
    _sub_images = train_images_uint8[:_subset]
    _sub_labels = train_labels[:_subset]
    _grid = hp_search_grid()
    for _cfg_idx, _cfg in enumerate(_grid):
        _mcfg = DiTConfigV1(hidden_dim=int(_cfg["hidden_dim"]), depth=4, num_heads=4, patch_size=4)
        _model = build_model_from_config(_mcfg, device=device)
        _opt = torch.optim.AdamW(_model.parameters(), lr=float(_cfg["lr"]), weight_decay=1e-2, betas=(0.9, 0.99))
        _scaler = make_grad_scaler(device, use_amp, amp_dtype)
        _gen = torch.Generator().manual_seed(999 + _cfg_idx)
        for _ep in range(int(hp_epochs_ui.value)):
            run_train_epoch(
                model=_model,
                optimizer=_opt,
                scaler=_scaler,
                train_images_uint8=_sub_images,
                train_labels=_sub_labels,
                batch_size=128,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_clip=1.0,
                t_scheme="uniform",
                augment_flip=True,
                generator=_gen,
            )
        _vl = run_evaluate(
            model=_model,
            images_uint8=val_images_uint8,
            labels=val_labels,
            batch_size=256,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=777,
        )
        hp_results.append(
            {
                "lr": _cfg["lr"],
                "hidden_dim": int(_cfg["hidden_dim"]),
                "params": count_parameters(_model),
                "val_loss": round(_vl, 4),
            }
        )
        mo.output.replace(
            mo.md(
                f"[{_cfg_idx + 1}/{len(_grid)}] lr={_cfg['lr']}, hidden_dim={_cfg['hidden_dim']} -> val {_vl:.4f}"
            )
        )
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.ui.table(hp_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Validation & Cross-Validation

    We report three metrics on the test set for the 1st-generation model:

    - **Mean flow-matching loss** (deterministic, fixed CPU generator seed).
    - **Loss per timestep bin** — 10 bins of $t \in [0, 1]$, highlighting where
      the model struggles.
    - **Straightness** — Liu et al. 2022's straightness estimator,
      $\mathbb{E}\!\int_{0}^{1} \lVert (X_1 - X_0) - v_\theta(X_t, t) \rVert^2\, dt$,
      approximated along the Euler trajectory. Lower = straighter (reflow should
      lower it).

    5-fold cross-validation on the training split is gated separately (trains 5
    models).
    """)
    return


@app.cell
def _(
    amp_dtype,
    device,
    mo,
    test_images,
    test_labels,
    trained_model: Optional[nn.Module],
    use_amp,
):
    gen1_test_metrics: Optional[Dict[str, object]] = None
    if trained_model is None:
        _out = mo.md("_Train the 1st-generation model first (Section 5) to see test metrics._")
    else:
        gen1_test_metrics = evaluate_model(
            model=trained_model,
            images_uint8=test_images,
            labels=test_labels,
            batch_size=256,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            null_index=trained_model.null_index,
            num_bins=10,
            straightness_batch=32,
            straightness_steps=32,
            seed=2026,
        )
        _out = mo.md(
            f"""
    **Test mean loss**: {gen1_test_metrics['mean_loss']:.4f}

    **Straightness (32-step, 32 samples)**: {gen1_test_metrics['straightness']:.4f}

    {format_loss_bin_table(gen1_test_metrics['loss_per_bin'], gen1_test_metrics['bin_counts'])}
            """
        )
    _out
    return (gen1_test_metrics,)


@app.function
def format_loss_bin_table(loss_per_bin: List[float], bin_counts: List[int]) -> str:
    num_bins = len(loss_per_bin)
    rows = [
        f"| bin {b} (t in [{b / num_bins:.2f}, {(b + 1) / num_bins:.2f}]) | {loss_per_bin[b]:.4f} | {bin_counts[b]} |"
        for b in range(num_bins)
    ]
    return "\n".join(["| Timestep bin | Mean loss | Count |", "|---|---|---|", *rows])


@app.function
def plot_per_timestep_bins(loss_per_bin: List[float], title: str = "Loss per timestep bin"):
    xs = [(b + 0.5) / len(loss_per_bin) for b in range(len(loss_per_bin))]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(xs, loss_per_bin, width=1.0 / len(loss_per_bin) * 0.9, color="darkorange", alpha=0.8)
    ax.set_xlabel("t (mid of bin)")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(gen1_test_metrics: Optional[Dict[str, object]], mo):
    if gen1_test_metrics is None:
        _out = mo.md("_Train first to see the per-timestep loss plot._")
    else:
        _out = plot_per_timestep_bins(gen1_test_metrics["loss_per_bin"], title="1st-gen test loss per t bin")
    _out
    return


@app.cell
def _(mo):
    cv_cb = mo.ui.checkbox(label="Enable 5-Fold Cross-Validation", value=False)
    cv_epochs_ui = mo.ui.slider(1, 10, value=2, step=1, label="Epochs per Fold")
    cv_subset_ui = mo.ui.dropdown(
        options=[2000, 5000, 10000, 20000, 45000],
        value=5000,
        label="CV Subset Size",
    )
    cv_run_btn = mo.ui.run_button(label="Run 5-Fold CV")
    mo.vstack(
        [
            cv_cb,
            mo.hstack([cv_epochs_ui, cv_subset_ui, cv_run_btn]),
        ]
    )
    return cv_cb, cv_epochs_ui, cv_run_btn, cv_subset_ui


@app.function
def run_fold(
    fold_train_images: torch.Tensor,
    fold_train_labels: torch.Tensor,
    fold_val_images: torch.Tensor,
    fold_val_labels: torch.Tensor,
    model_cfg: DiTConfigV1,
    lr: float,
    n_epochs: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    seed: int,
) -> float:
    set_seed(seed)
    model = build_model_from_config(model_cfg, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.99))
    scaler = make_grad_scaler(device, use_amp, amp_dtype)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_epochs):
        run_train_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_images_uint8=fold_train_images,
            train_labels=fold_train_labels,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_clip=1.0,
            t_scheme="uniform",
            augment_flip=True,
            generator=gen,
        )
    val_loss = run_evaluate(
        model=model,
        images_uint8=fold_val_images,
        labels=fold_val_labels,
        batch_size=256,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        seed=seed + 500,
    )
    return val_loss


@app.cell
def _(
    amp_dtype,
    cv_cb,
    cv_epochs_ui,
    cv_run_btn,
    cv_subset_ui,
    device,
    lr_ui,
    mo,
    model_cfg,
    train_images_uint8,
    train_labels,
    use_amp,
):
    mo.stop(
        not cv_cb.value,
        mo.md("_Enable **5-Fold Cross-Validation** above to run this section._"),
    )
    mo.stop(
        not cv_run_btn.value,
        mo.md("Toggle enabled — now click **Run 5-Fold CV** to launch."),
    )
    cv_results: Dict = {"fold_losses": [], "mean": 0.0, "std": 0.0}
    _subset = min(int(cv_subset_ui.value), train_images_uint8.shape[0])
    _images = train_images_uint8[:_subset]
    _labels = train_labels[:_subset]
    _k = 5
    _perm = torch.randperm(_subset, generator=torch.Generator().manual_seed(42))
    _fold_size = _subset // _k
    _fold_losses: List[float] = []
    for _f in range(_k):
        _val_idx = _perm[_f * _fold_size : (_f + 1) * _fold_size]
        _tr_mask = torch.ones(_subset, dtype=torch.bool)
        _tr_mask[_val_idx] = False
        _tr_idx = torch.arange(_subset)[_tr_mask]
        _vl = run_fold(
            fold_train_images=_images[_tr_idx],
            fold_train_labels=_labels[_tr_idx],
            fold_val_images=_images[_val_idx],
            fold_val_labels=_labels[_val_idx],
            model_cfg=model_cfg,
            lr=float(lr_ui.value),
            n_epochs=int(cv_epochs_ui.value),
            batch_size=128,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=1000 + _f,
        )
        _fold_losses.append(_vl)
        mo.output.replace(mo.md(f"Fold {_f + 1}/{_k} — val loss: {_vl:.4f}"))
    _mean = float(np.mean(_fold_losses))
    _std = float(np.std(_fold_losses))
    cv_results = {"fold_losses": _fold_losses, "mean": _mean, "std": _std}
    mo.output.replace(
        mo.md(
            f"**5-Fold CV mean loss**: {_mean:.4f} +/- {_std:.4f}\n\n"
            "| Fold | Val loss |\n|---|---|\n"
            + "\n".join(f"| {i + 1} | {v:.4f} |" for i, v in enumerate(_fold_losses))
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Reflow — 2-Rectified Flow

    Liu et al. 2022 show that repeatedly *reflowing* an ODE with itself
    straightens its trajectories, dramatically reducing the number of Euler
    steps needed for accurate sampling.

    **Pairing procedure used here** — for every image $x_1$ in the split used for
    1st-generation training (all 45,000 train images and all 5,000 validation
    images; **no subsetting**), with its class label $y$:

    1. Integrate the 1st-generation velocity field *backward* from $t = 1$ to
       $t = 0$ with $K$ Euler steps to obtain $x_0 = \Phi^{-1}(x_1, y)$.
    2. Store the pair $(x_0, x_1, y)$ with the same index as $x_1$ in the
       original split.

    **Euler time grid.** Near $t = 1$ the backward ODE is stiff: along a data
    direction with standard deviation $s \ll 1$ the exact flow must expand $x$ by
    a factor of roughly $1/s$ within $t \in [1 - s, 1]$, and more than half of
    CIFAR-10's 3,072 pixel-space principal directions have $s < 0.05$. A uniform
    grid's first step ($\Delta t = 1/K$) cannot resolve that, so the recovered
    $x_0$ comes out too narrow along those directions — a mismatch with the
    $\mathcal{N}(0, I)$ source used at sampling time. The Euler nodes are
    therefore placed at $t_i = 1 - (i/K)^{\rho}$: $\rho = 1$ is the uniform grid,
    and $\rho = 2$ (the default) concentrates steps at the data end. The sanity
    check below reports the recovered $x_0$ statistics so the choice of $K$ and
    $\rho$ can be verified.

    The 2-rectified flow model is trained on those **fixed** couples with the
    same `fit_flow_model` rite as Section 5 and its own configurable
    hyperparameters (label dropout still applies for CFG at inference).
    Horizontal-flip augmentation is disabled for coupled training, because
    flipping $x_1$ without its $x_0$ would break the coupling. By default the
    new generation starts from the previous generation's weights (as in the
    paper); *from-scratch* re-initializes it with the previous generation's
    config. `init_next_generation_model` and `build_reflow_pairs` accept any
    generation, so the procedure can be repeated for $k$-rectified flows.
    """)
    return


@app.cell
def _(mo):
    reflow_steps_ui = mo.ui.slider(4, 200, value=50, step=1, label="Reflow Euler Steps")
    reflow_grid_power_ui = mo.ui.dropdown(
        options={"1 (uniform)": 1.0, "2": 2.0, "3": 3.0},
        value="2",
        label="Inversion Grid ρ",
    )
    reflow_batch_ui = mo.ui.dropdown(options=[64, 128, 256, 512], value=256, label="Reflow Gen Batch")
    reflow_guidance_ui = mo.ui.dropdown(
        options={"1.0": 1.0, "1.5": 1.5, "2.0": 2.0},
        value="1.0",
        label="Reflow Gen Guidance",
    )
    reflow_dtype_ui = mo.ui.dropdown(
        options={"float32": "float32", "float16": "float16"},
        value="float32",
        label="x0 Storage Dtype",
    )
    reflow_pairs_btn = mo.ui.run_button(label="Generate Reflow Pairs")
    mo.vstack(
        [
            mo.md("### Reflow pair-generation controls"),
            mo.hstack([reflow_steps_ui, reflow_grid_power_ui, reflow_batch_ui]),
            mo.hstack([reflow_guidance_ui, reflow_dtype_ui, reflow_pairs_btn]),
        ]
    )
    return (
        reflow_batch_ui,
        reflow_dtype_ui,
        reflow_grid_power_ui,
        reflow_guidance_ui,
        reflow_pairs_btn,
        reflow_steps_ui,
    )


@app.function
def dtype_from_string(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


@app.cell
def _(
    device,
    mo,
    reflow_batch_ui,
    reflow_dtype_ui,
    reflow_grid_power_ui,
    reflow_guidance_ui,
    reflow_pairs_btn,
    reflow_steps_ui,
    train_images_uint8,
    train_labels,
    trained_model: Optional[nn.Module],
    val_images_uint8,
    val_labels,
):
    reflow_train_x0: Optional[torch.Tensor] = None
    reflow_val_x0: Optional[torch.Tensor] = None
    reflow_gen_time = 0.0
    if trained_model is None:
        mo.output.replace(mo.md("_Train the 1st-generation model first (Section 5) before generating reflow pairs._"))
    elif not reflow_pairs_btn.value:
        mo.output.replace(mo.md("Click **Generate Reflow Pairs** to build $(x_0, x_1, y)$ couples for train + val splits."))
    else:
        _storage_dtype = dtype_from_string(str(reflow_dtype_ui.value))
        _steps = int(reflow_steps_ui.value)
        _bs = int(reflow_batch_ui.value)
        _guide = float(reflow_guidance_ui.value)

        def _progress_tr(done, total):
            mo.output.replace(mo.md(f"Train pairs: {done}/{total}"))

        def _progress_va(done, total):
            mo.output.replace(mo.md(f"Val pairs: {done}/{total}"))

        _t0 = time.perf_counter()
        reflow_train_x0 = build_reflow_pairs(
            model=trained_model,
            images_uint8=train_images_uint8,
            labels=train_labels,
            num_steps=_steps,
            batch_size=_bs,
            device=device,
            null_index=trained_model.null_index,
            guidance_scale=_guide,
            storage_dtype=_storage_dtype,
            progress_cb=_progress_tr,
            data_end_power=float(reflow_grid_power_ui.value),
        )
        reflow_val_x0 = build_reflow_pairs(
            model=trained_model,
            images_uint8=val_images_uint8,
            labels=val_labels,
            num_steps=_steps,
            batch_size=_bs,
            device=device,
            null_index=trained_model.null_index,
            guidance_scale=_guide,
            storage_dtype=_storage_dtype,
            progress_cb=_progress_va,
            data_end_power=float(reflow_grid_power_ui.value),
        )
        reflow_gen_time = time.perf_counter() - _t0
        assert reflow_train_x0.shape[0] == train_images_uint8.shape[0], (
            f"Reflow train pairs {reflow_train_x0.shape[0]} != train split {train_images_uint8.shape[0]}"
        )
        assert reflow_val_x0.shape[0] == val_images_uint8.shape[0], (
            f"Reflow val pairs {reflow_val_x0.shape[0]} != val split {val_images_uint8.shape[0]}"
        )
        mo.output.replace(
            mo.md(
                f"**Reflow pairs generated** in {reflow_gen_time:.1f}s\n\n"
                f"| Split | x0 count | x1 count | Match? |\n|---|---|---|---|\n"
                f"| train | {reflow_train_x0.shape[0]:,} | {train_images_uint8.shape[0]:,} | "
                f"{reflow_train_x0.shape[0] == train_images_uint8.shape[0]} |\n"
                f"| val | {reflow_val_x0.shape[0]:,} | {val_images_uint8.shape[0]:,} | "
                f"{reflow_val_x0.shape[0] == val_images_uint8.shape[0]} |"
            )
        )
    return reflow_train_x0, reflow_val_x0


@app.cell
def _(
    device,
    mo,
    reflow_grid_power_ui,
    reflow_guidance_ui,
    reflow_steps_ui,
    reflow_train_x0: Optional[torch.Tensor],
    train_images_uint8,
    train_labels,
    trained_model: Optional[nn.Module],
):
    if trained_model is None or reflow_train_x0 is None:
        _out = mo.md("_Generate reflow pairs first (button above) to see the round-trip sanity check._")
    else:
        _stats = roundtrip_reconstruction(
            model=trained_model,
            images_uint8=train_images_uint8[:16],
            labels=train_labels[:16],
            device=device,
            null_index=trained_model.null_index,
            num_steps=int(reflow_steps_ui.value),
            guidance_scale=float(reflow_guidance_ui.value),
            data_end_power=float(reflow_grid_power_ui.value),
        )
        _x0_sub = reflow_train_x0[:2048].to(torch.float32)
        _out = mo.md(
            f"""
    ### Sanity check (16-image round-trip, same steps / ρ / guidance as the pairs)

    - **Round-trip MSE** ($x_1 \\to x_0 \\to \\hat{{x}}_1$): {_stats['roundtrip_mse']:.4f}
    - **x0 mean** (over first 2048 recovered): {_x0_sub.mean().item():.4f} (target ~0)
    - **x0 std** (over first 2048 recovered): {_x0_sub.std().item():.4f} (target ~1)

    _An x0 std well below 1 means the backward Euler pass under-expanded the
    low-variance directions of the data; the reflow model would then be trained
    on narrower "noise" than the $\\mathcal{{N}}(0, I)$ it receives at sampling
    time, which shows up as high-frequency speckle. Raise the Euler steps or ρ._
            """
        )
    _out
    return


@app.cell
def _(
    mo,
    reflow_train_x0: Optional[torch.Tensor],
    train_images_uint8,
    train_labels,
    trained_model: Optional[nn.Module],
):
    if trained_model is None or reflow_train_x0 is None:
        _out = mo.md("_Generate reflow pairs first to inspect the ReflowPairDataset._")
    else:
        _ds = ReflowPairDatasetV1(reflow_train_x0[:128], train_images_uint8[:128], train_labels[:128])
        _x0_s, _x1_s, _y_s = _ds[0]
        _out = mo.md(
            f"""
    **`ReflowPairDatasetV1` preview** — length: {len(_ds)}

    - x0 shape/dtype: `{tuple(_x0_s.shape)}` `{_x0_s.dtype}`
    - x1 shape/dtype: `{tuple(_x1_s.shape)}` `{_x1_s.dtype}` (already normalized to [-1, 1])
    - y: `{int(_y_s.item())}`
            """
        )
    _out
    return


@app.cell
def _(mo):
    reflow_lr_ui = mo.ui.dropdown(
        options={"3e-5": 3e-5, "1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Reflow LR",
    )
    reflow_bs_ui = mo.ui.dropdown(options=[64, 128, 256], value=128, label="Reflow Batch Size")
    reflow_wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-2": 1e-2, "5e-2": 5e-2},
        value="1e-2",
        label="Reflow Weight Decay",
    )
    reflow_epochs_ui = mo.ui.slider(1, 100, value=5, step=1, label="Reflow Epochs")
    reflow_warmup_ui = mo.ui.dropdown(options=[0, 100, 200, 500], value=100, label="Reflow Warmup Steps")
    reflow_grad_clip_ui = mo.ui.dropdown(
        options={"none": 0.0, "0.5": 0.5, "1.0": 1.0, "2.0": 2.0, "5.0": 5.0},
        value="1.0",
        label="Reflow Grad Clip",
    )
    reflow_ema_ui = mo.ui.dropdown(
        options={"off": 0.0, "0.999": 0.999, "0.9995": 0.9995, "0.9999": 0.9999},
        value="0.9999",
        label="Reflow EMA Decay",
    )
    reflow_t_scheme_ui = mo.ui.dropdown(
        options=["uniform", "logit_normal"],
        value="uniform",
        label="Reflow Timestep Scheme",
    )
    reflow_init_ui = mo.ui.dropdown(
        options=["from-previous", "from-scratch"],
        value="from-previous",
        label="Reflow Init",
    )
    reflow_train_btn = mo.ui.run_button(label="Train Reflow Model")
    mo.vstack(
        [
            mo.md("### Reflow training hyperparameters"),
            mo.hstack([reflow_lr_ui, reflow_bs_ui, reflow_wd_ui, reflow_epochs_ui]),
            mo.hstack([reflow_warmup_ui, reflow_grad_clip_ui, reflow_ema_ui, reflow_t_scheme_ui]),
            mo.hstack([reflow_init_ui, reflow_train_btn]),
        ]
    )
    return (
        reflow_bs_ui,
        reflow_ema_ui,
        reflow_epochs_ui,
        reflow_grad_clip_ui,
        reflow_init_ui,
        reflow_lr_ui,
        reflow_t_scheme_ui,
        reflow_train_btn,
        reflow_warmup_ui,
        reflow_wd_ui,
    )


@app.function
def init_next_generation_model(previous_model: nn.Module, mode: str, device: torch.device) -> nn.Module:
    if mode == "from-previous":
        return copy.deepcopy(previous_model).to(device)
    if mode == "from-scratch":
        return build_model_from_config(previous_model.config, device=device)
    raise ValueError(f"Unknown reflow init mode: {mode!r}")


@app.cell
def _(
    amp_dtype,
    device,
    mo,
    reflow_bs_ui,
    reflow_ema_ui,
    reflow_epochs_ui,
    reflow_grad_clip_ui,
    reflow_init_ui,
    reflow_lr_ui,
    reflow_t_scheme_ui,
    reflow_train_btn,
    reflow_train_x0: Optional[torch.Tensor],
    reflow_val_x0: Optional[torch.Tensor],
    reflow_warmup_ui,
    reflow_wd_ui,
    seed_ui,
    train_images_uint8,
    train_labels,
    trained_model: Optional[nn.Module],
    use_amp,
    val_images_uint8,
    val_labels,
):
    reflow_run: Optional[Dict[str, object]] = None
    if trained_model is None or reflow_train_x0 is None or reflow_val_x0 is None:
        mo.output.replace(mo.md("_Generate reflow pairs first (Section 8 button) before training the reflow model._"))
    elif not reflow_train_btn.value:
        mo.output.replace(mo.md("Click **Train Reflow Model** to fit a 2-rectified flow on the reflow pairs."))
    else:
        reflow_train_cfg = TrainConfigV1(
            lr=float(reflow_lr_ui.value),
            batch_size=int(reflow_bs_ui.value),
            weight_decay=float(reflow_wd_ui.value),
            epochs=int(reflow_epochs_ui.value),
            warmup_steps=int(reflow_warmup_ui.value),
            grad_clip=float(reflow_grad_clip_ui.value),
            ema_decay=float(reflow_ema_ui.value),
            t_scheme=str(reflow_t_scheme_ui.value),
            augment_flip=False,
            seed=int(seed_ui.value) + 7,
        )
        set_seed(reflow_train_cfg.seed)
        reflow_run = fit_flow_model(
            model=init_next_generation_model(trained_model, str(reflow_init_ui.value), device),
            train_cfg=reflow_train_cfg,
            train_images_uint8=train_images_uint8,
            train_labels=train_labels,
            val_images_uint8=val_images_uint8,
            val_labels=val_labels,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            train_x0=reflow_train_x0,
            val_x0=reflow_val_x0,
            progress_cb=lambda epoch, total, tl, vl: mo.output.replace(
                mo.md(f"**Reflow epoch {epoch}/{total}** — train {tl:.4f} | val {vl:.4f}")
            ),
        )
        mo.output.replace(
            mo.md(
                f"**Reflow training complete** in {reflow_run['wall_time']:.1f}s — "
                f"final train {reflow_run['train_losses'][-1]:.4f} | final val {reflow_run['val_losses'][-1]:.4f}"
                f" | using EMA weights: {reflow_run['ema_used']}"
            )
        )
    reflow_train_losses: List[float] = reflow_run["train_losses"] if reflow_run else []
    reflow_val_losses: List[float] = reflow_run["val_losses"] if reflow_run else []
    reflow_model: Optional[nn.Module] = reflow_run["model"] if reflow_run else None
    reflow_wall_time: float = reflow_run["wall_time"] if reflow_run else 0.0
    return (
        reflow_model,
        reflow_train_losses,
        reflow_val_losses,
        reflow_wall_time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Denoising Process Demonstration

    Sample a batch, integrate the ODE with a chosen number of Euler steps, take
    snapshots of $x_t$ at evenly spaced timesteps, and display two rows per
    sample:

    - **Top**: the current $x_t$.
    - **Bottom**: the model's implied clean estimate
      $\hat{x}_1 = x_t + (1 - t)\, v_\theta(x_t, t, y)$.

    When both generations are trained, the same seeds are rendered through both
    models side by side so the *straighter* reflow trajectories are visible.
    """)
    return


@app.cell
def _(
    class_names,
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    _options: Dict[str, str] = {}
    if trained_model is not None:
        _options["1st gen"] = "gen1"
    if reflow_model is not None:
        _options["reflow"] = "reflow"
    if trained_model is not None and reflow_model is not None:
        _options["both"] = "both"
    if not _options:
        _options = {"(train first)": "none"}
    denoise_model_ui = mo.ui.dropdown(options=_options, value=list(_options.keys())[0], label="Model")
    denoise_class_ui = mo.ui.dropdown(
        options={"all classes": -1, **{c: i for i, c in enumerate(class_names)}},
        value="all classes",
        label="Class",
    )
    denoise_steps_ui = mo.ui.slider(2, 200, value=32, step=1, label="Euler Steps")
    denoise_snapshots_ui = mo.ui.slider(3, 12, value=8, step=1, label="Snapshots")
    denoise_guidance_ui = mo.ui.dropdown(
        options={"1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "3.0": 3.0, "5.0": 5.0},
        value="1.0",
        label="Guidance",
    )
    denoise_seed_ui = mo.ui.number(value=7, label="Seed", start=0, stop=2**31 - 1)
    denoise_btn = mo.ui.run_button(label="Generate")
    mo.vstack(
        [
            mo.md("### Denoising demo controls"),
            mo.hstack([denoise_model_ui, denoise_class_ui, denoise_guidance_ui]),
            mo.hstack([denoise_steps_ui, denoise_snapshots_ui, denoise_seed_ui, denoise_btn]),
        ]
    )
    return (
        denoise_btn,
        denoise_class_ui,
        denoise_guidance_ui,
        denoise_model_ui,
        denoise_seed_ui,
        denoise_snapshots_ui,
        denoise_steps_ui,
    )


@app.function
def evenly_spaced_snapshot_steps(num_steps: int, num_snapshots: int) -> List[int]:
    if num_snapshots > num_steps + 1:
        num_snapshots = num_steps + 1
    if num_snapshots <= 1:
        return [num_steps]
    return list({int(round(x)) for x in np.linspace(0, num_steps, num_snapshots)})


@app.function
def build_denoise_trajectory(
    model: nn.Module,
    y: torch.Tensor,
    image_shape: Tuple[int, int, int],
    num_steps: int,
    snapshot_indices: List[int],
    guidance_scale: float,
    seed: int,
    device: torch.device,
    null_index: int,
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    b = y.shape[0]
    x0 = torch.randn((b, *image_shape), generator=gen).to(device)
    capture = set(snapshot_indices)
    with torch.no_grad():
        _, snapshots = euler_integrate(
            model=model,
            x_start=x0,
            y=y,
            num_steps=num_steps,
            null_index=null_index,
            guidance_scale=guidance_scale,
            t_start=0.0,
            t_end=1.0,
            capture_indices=capture,
        )
    ts = euler_time_grid(num_steps, 0.0, 1.0, device=device)
    xt_list: List[torch.Tensor] = []
    x1_hat_list: List[torch.Tensor] = []
    tvals: List[float] = []
    with torch.no_grad():
        for idx in sorted(snapshots.keys()):
            xt = snapshots[idx]
            t_val = float(ts[idx].item())
            tvals.append(t_val)
            xt_list.append(xt.cpu())
            t_batch = ts[idx].expand(b)
            v = classifier_free_velocity(model, xt, t_batch, y, null_index, guidance_scale=guidance_scale).to(torch.float32)
            x1_hat = xt + (1.0 - t_val) * v
            x1_hat_list.append(x1_hat.cpu())
    return {
        "xt": torch.stack(xt_list, dim=1),
        "x1_hat": torch.stack(x1_hat_list, dim=1),
        "tvals": torch.tensor(tvals),
    }


@app.function
def plot_denoising_process(
    traj: Dict[str, torch.Tensor],
    title: str = "Denoising process",
):
    xt = traj["xt"]
    x1_hat = traj["x1_hat"]
    tvals = traj["tvals"].tolist()
    n_samples, n_snap = xt.shape[0], xt.shape[1]
    fig, axes = plt.subplots(2 * n_samples, n_snap, figsize=(1.5 * n_snap, 1.5 * 2 * n_samples))
    axes = np.array(axes).reshape(2 * n_samples, n_snap)
    for i in range(n_samples):
        for j in range(n_snap):
            _im = denormalize_from_neg1_1(xt[i, j]).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            axes[2 * i, j].imshow(_im)
            axes[2 * i, j].axis("off")
            if i == 0:
                axes[2 * i, j].set_title(f"t={tvals[j]:.2f}", fontsize=8)
            if j == 0:
                axes[2 * i, j].set_ylabel(f"x_t #{i}", fontsize=8, rotation=0, ha="right", va="center")
            _im2 = denormalize_from_neg1_1(x1_hat[i, j]).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            axes[2 * i + 1, j].imshow(_im2)
            axes[2 * i + 1, j].axis("off")
            if j == 0:
                axes[2 * i + 1, j].set_ylabel(f"x1_hat #{i}", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.cell
def _(
    denoise_btn,
    denoise_class_ui,
    denoise_guidance_ui,
    denoise_model_ui,
    denoise_seed_ui,
    denoise_snapshots_ui,
    denoise_steps_ui,
    device,
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    if not denoise_btn.value:
        _out = mo.md("Click **Generate** to run the denoising demo.")
    elif str(denoise_model_ui.value) == "none":
        _out = mo.md("_Train first (Section 5) — no model available._")
    else:
        _cls = int(denoise_class_ui.value)
        _steps = int(denoise_steps_ui.value)
        _n_snap = int(denoise_snapshots_ui.value)
        _guide = float(denoise_guidance_ui.value)
        _seed = int(denoise_seed_ui.value)
        _snap_idxs = evenly_spaced_snapshot_steps(_steps, _n_snap)
        if _cls < 0:
            _y = torch.arange(10, device=device)
        else:
            _y = torch.full((4,), _cls, dtype=torch.int64, device=device)
        _figs = []
        if str(denoise_model_ui.value) in {"gen1", "both"} and trained_model is not None:
            _cfg_gen1 = trained_model.config
            _traj = build_denoise_trajectory(
                model=trained_model,
                y=_y,
                image_shape=(_cfg_gen1.in_channels, _cfg_gen1.image_size, _cfg_gen1.image_size),
                num_steps=_steps,
                snapshot_indices=_snap_idxs,
                guidance_scale=_guide,
                seed=_seed,
                device=device,
                null_index=trained_model.null_index,
            )
            _figs.append(plot_denoising_process(_traj, title=f"1st gen denoising (steps={_steps}, guide={_guide})"))
        if str(denoise_model_ui.value) in {"reflow", "both"} and reflow_model is not None:
            _cfg_rf = reflow_model.config
            _traj2 = build_denoise_trajectory(
                model=reflow_model,
                y=_y,
                image_shape=(_cfg_rf.in_channels, _cfg_rf.image_size, _cfg_rf.image_size),
                num_steps=_steps,
                snapshot_indices=_snap_idxs,
                guidance_scale=_guide,
                seed=_seed,
                device=device,
                null_index=reflow_model.null_index,
            )
            _figs.append(plot_denoising_process(_traj2, title=f"Reflow denoising (steps={_steps}, guide={_guide})"))
        _out = mo.vstack(_figs) if _figs else mo.md("_Nothing to plot._")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Results

    Loss curves for the two generations, class-conditional sample grids, a
    few-step comparison (1, 2, 4, 8, 16, 64 Euler steps with the same fixed
    noise per generation — the key payoff of reflow), and a model comparison
    table.
    """)
    return


@app.function
def plot_loss_curves(
    gen1_train: List[float],
    gen1_val: List[float],
    reflow_train: List[float],
    reflow_val: List[float],
):
    fig, ax = plt.subplots(figsize=(9, 4))
    if gen1_train:
        ax.plot(range(1, len(gen1_train) + 1), gen1_train, "b-o", ms=4, label="1st gen train")
    if gen1_val:
        ax.plot(range(1, len(gen1_val) + 1), gen1_val, "b--s", ms=4, label="1st gen val")
    if reflow_train:
        ax.plot(range(1, len(reflow_train) + 1), reflow_train, "r-o", ms=4, label="reflow train")
    if reflow_val:
        ax.plot(range(1, len(reflow_val) + 1), reflow_val, "r--s", ms=4, label="reflow val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training / validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    reflow_train_losses: List[float],
    reflow_val_losses: List[float],
    train_losses: List[float],
    val_losses: List[float],
):
    if not train_losses and not reflow_train_losses:
        _out = mo.md("_Train something to see the loss curves._")
    else:
        _out = plot_loss_curves(train_losses, val_losses, reflow_train_losses, reflow_val_losses)
    _out
    return


@app.function
def plot_class_conditional_grid(
    model: nn.Module,
    class_names: List[str],
    num_per_class: int,
    num_steps: int,
    guidance_scale: float,
    device: torch.device,
    seed: int,
    title: str = "Class-conditional samples",
):
    y = torch.arange(len(class_names), device=device).repeat_interleave(num_per_class)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    cfg = model.config
    x_final, _ = sample_images(
        model=model,
        y=y,
        image_shape=(cfg.in_channels, cfg.image_size, cfg.image_size),
        num_steps=num_steps,
        null_index=model.null_index,
        guidance_scale=guidance_scale,
        generator=gen,
        device=device,
    )
    imgs = denormalize_from_neg1_1(x_final.cpu()).clamp(0, 255).byte()
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, num_per_class, figsize=(num_per_class * 1.4, n_classes * 1.4))
    axes = np.array(axes).reshape(n_classes, num_per_class)
    for i in range(n_classes):
        for j in range(num_per_class):
            axes[i, j].imshow(imgs[i * num_per_class + j].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(class_names[i], fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, device, mo, trained_model: Optional[nn.Module]):
    if trained_model is None:
        _out = mo.md("_Train first to see the 1st-gen class-conditional sample grid._")
    else:
        _out = plot_class_conditional_grid(
            model=trained_model,
            class_names=class_names,
            num_per_class=4,
            num_steps=32,
            guidance_scale=1.5,
            device=device,
            seed=101,
            title="1st gen — 32-step samples (guidance=1.5)",
        )
    _out
    return


@app.cell
def _(class_names, device, mo, reflow_model: Optional[nn.Module]):
    if reflow_model is None:
        _out = mo.md("_Train the reflow model (Section 8) to see the reflow class-conditional grid._")
    else:
        _out = plot_class_conditional_grid(
            model=reflow_model,
            class_names=class_names,
            num_per_class=4,
            num_steps=8,
            guidance_scale=1.5,
            device=device,
            seed=101,
            title="Reflow — 8-step samples (guidance=1.5)",
        )
    _out
    return


@app.function
def plot_few_step_comparison(
    models_by_name: Dict[str, nn.Module],
    class_names: List[str],
    device: torch.device,
    seed: int = 202,
    step_counts: Tuple[int, ...] = (1, 2, 4, 8, 16, 64),
    guidance_scale: float = 1.0,
    title: str = "Few-step sampling comparison",
):
    n_models = len(models_by_name)
    n_classes = len(class_names)
    n_steps = len(step_counts)
    fig, axes = plt.subplots(n_models * n_classes, n_steps, figsize=(n_steps * 1.35, n_models * n_classes * 1.35))
    axes = np.array(axes).reshape(n_models * n_classes, n_steps)
    for m_idx, (name, model) in enumerate(models_by_name.items()):
        cfg = model.config
        img_shape = (cfg.in_channels, cfg.image_size, cfg.image_size)
        for c in range(n_classes):
            row = m_idx * n_classes + c
            for s_idx, s in enumerate(step_counts):
                gen = torch.Generator(device="cpu").manual_seed(int(seed) + c * 17 + m_idx * 991)
                y = torch.full((1,), c, dtype=torch.int64, device=device)
                x_final, _ = sample_images(
                    model=model,
                    y=y,
                    image_shape=img_shape,
                    num_steps=int(s),
                    null_index=model.null_index,
                    guidance_scale=guidance_scale,
                    generator=gen,
                    device=device,
                )
                img = denormalize_from_neg1_1(x_final[0].cpu()).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                axes[row, s_idx].imshow(img)
                axes[row, s_idx].axis("off")
                if row == 0:
                    axes[row, s_idx].set_title(f"steps={s}", fontsize=8)
                if s_idx == 0:
                    axes[row, s_idx].set_ylabel(f"{name}\n{class_names[c]}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.cell
def _(
    class_names,
    device,
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    _models: Dict[str, nn.Module] = {}
    if trained_model is not None:
        _models["1st gen"] = trained_model
    if reflow_model is not None:
        _models["reflow"] = reflow_model
    if not _models:
        _out = mo.md("_Train at least one model to see the few-step comparison._")
    else:
        _sub_classes = class_names[:3]
        _out = plot_few_step_comparison(
            models_by_name=_models,
            class_names=_sub_classes,
            device=device,
            seed=707,
            step_counts=(1, 2, 4, 8, 16, 64),
            guidance_scale=1.0,
            title="Few-step comparison — same noise, different Euler step counts",
        )
    _out
    return


@app.function
def few_step_deviation(
    model: nn.Module,
    device: torch.device,
    seed: int,
    step_counts: Tuple[int, ...] = (1, 2, 4, 8, 16),
    reference_steps: int = 128,
    num_samples: int = 16,
    guidance_scale: float = 1.0,
) -> Dict[int, float]:
    cfg = model.config
    img_shape = (cfg.in_channels, cfg.image_size, cfg.image_size)
    y = torch.randint(0, max(cfg.num_classes, 1), (num_samples,), generator=torch.Generator().manual_seed(int(seed))).to(device)
    ref_gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    x_ref, _ = sample_images(
        model=model,
        y=y,
        image_shape=img_shape,
        num_steps=reference_steps,
        null_index=model.null_index,
        guidance_scale=guidance_scale,
        generator=ref_gen,
        device=device,
    )
    out: Dict[int, float] = {}
    for s in step_counts:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        x_k, _ = sample_images(
            model=model,
            y=y,
            image_shape=img_shape,
            num_steps=int(s),
            null_index=model.null_index,
            guidance_scale=guidance_scale,
            generator=gen,
            device=device,
        )
        out[int(s)] = float(((x_k - x_ref) ** 2).mean().item())
    return out


@app.function
def summarize_generation(
    name: str,
    model: nn.Module,
    val_losses: List[float],
    wall_time: float,
    device: torch.device,
    seed: int = 808,
    num_samples: int = 16,
    straightness_steps: int = 32,
    step_counts: Tuple[int, ...] = (1, 2, 4, 8),
    reference_steps: int = 64,
) -> Dict[str, object]:
    cfg = model.config
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x0 = torch.randn(num_samples, cfg.in_channels, cfg.image_size, cfg.image_size, generator=gen).to(device)
    y = torch.randint(0, max(cfg.num_classes, 1), (num_samples,), generator=gen).to(device)
    straightness = straightness_metric(model, x0, y, straightness_steps, model.null_index)
    deviations = few_step_deviation(
        model,
        device,
        seed=seed,
        step_counts=step_counts,
        reference_steps=reference_steps,
        num_samples=num_samples,
    )
    return {
        "model": name,
        "params": count_parameters(model),
        "straightness": round(straightness, 4),
        **{f"dev@{k}": round(v, 4) for k, v in deviations.items()},
        "final_val_loss": round(val_losses[-1], 4) if val_losses else None,
        "train_time_s": round(wall_time, 1),
    }


@app.cell
def _(
    device,
    mo,
    reflow_model: Optional[nn.Module],
    reflow_val_losses: List[float],
    reflow_wall_time: float,
    train_wall_time: float,
    trained_model: Optional[nn.Module],
    val_losses: List[float],
):
    if trained_model is None and reflow_model is None:
        _out = mo.md("_Train something to see the comparison table._")
    else:
        comparison_rows: List[Dict[str, object]] = []
        if trained_model is not None:
            comparison_rows.append(summarize_generation("1st gen", trained_model, val_losses, train_wall_time, device))
        if reflow_model is not None:
            comparison_rows.append(summarize_generation("reflow", reflow_model, reflow_val_losses, reflow_wall_time, device))
        _out = mo.vstack(
            [
                mo.md(
                    "**Model comparison** — `final_val_loss` is on each model's own "
                    "objective (1st gen: Gaussian x0; reflow: coupled x0). "
                    "Lower `straightness` and lower `dev@k` mean straighter, "
                    "few-step-accurate flows. Both rows use the same fixed-seed noise "
                    "and labels."
                ),
                mo.ui.table(comparison_rows),
            ]
        )
    _out
    return


@app.cell
def _(
    mo,
    reflow_model: Optional[nn.Module],
    reflow_train_losses: List[float],
    reflow_val_losses: List[float],
    train_losses: List[float],
    trained_ema_used: bool,
    trained_model: Optional[nn.Module],
    val_losses: List[float],
):
    _bits = []
    if trained_model is not None and train_losses:
        _bits.append(
            f"- **1st-generation model** trained for {len(train_losses)} epochs; "
            f"final train {train_losses[-1]:.4f}, val {val_losses[-1]:.4f}. "
            f"Using EMA weights downstream: **{trained_ema_used}**."
        )
    if reflow_model is not None and reflow_train_losses:
        _bits.append(
            f"- **Reflow model** trained for {len(reflow_train_losses)} epochs "
            f"on the same-size $(x_0, x_1, y)$ pair dataset generated by the "
            f"1st-generation model; final train {reflow_train_losses[-1]:.4f}, "
            f"val {reflow_val_losses[-1]:.4f}."
        )
    _bits.append(
        "- If reflow is behaving correctly, its `dev@1`/`dev@2` in the "
        "comparison table should be substantially lower than the 1st gen's, "
        "reflecting the straighter trajectories Liu et al. 2022 predict."
    )
    if not _bits:
        _bits = ["_No trained models yet — run Sections 5 and/or 8._"]
    mo.md("### Summary\n\n" + "\n\n".join(_bits))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Save Trained Model

    The chosen generation's state dict is written to the repo-root `models/`
    directory (created if missing), together with a `.json` sidecar carrying
    the exact `DiTConfigV1` used, so `build_model_from_config` can reconstruct
    the model later.
    """)
    return


@app.cell
def _(
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    _options: Dict[str, str] = {}
    if trained_model is not None:
        _options["1st gen"] = "gen1"
    if reflow_model is not None:
        _options["reflow"] = "reflow"
    if not _options:
        _options = {"(none — train first)": "none"}
    save_which_ui = mo.ui.dropdown(options=_options, value=list(_options.keys())[0], label="Which Model")
    save_which_ui
    return (save_which_ui,)


@app.function
def default_checkpoint_name(generation: str) -> str:
    return {
        "gen1": "cifar10_dit_rcfm_v1.pt",
        "reflow": "cifar10_dit_rcfm_reflow_v1.pt",
    }.get(generation, "cifar10_dit_rcfm_v1.pt")


@app.cell
def _(mo, save_which_ui):
    save_filename_ui = mo.ui.text(
        value=default_checkpoint_name(str(save_which_ui.value)),
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_btn])
    return save_btn, save_filename_ui


@app.function
def save_model_and_config(model: nn.Module, models_dir: Path, filename: str) -> Tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / filename
    config_path = weights_path.with_suffix(".json")
    torch.save(model.state_dict(), weights_path)
    with open(config_path, "w") as f:
        json.dump(asdict(model.config), f, indent=2, sort_keys=True)
    return weights_path, config_path


@app.cell
def _(
    mo,
    reflow_model: Optional[nn.Module],
    save_btn,
    save_filename_ui,
    save_which_ui,
    trained_model: Optional[nn.Module],
):
    if str(save_which_ui.value) == "none":
        _out = mo.md("_Train the 1st-generation model (Section 5) and/or the reflow model (Section 8) first._")
    elif not save_btn.value:
        _out = mo.md("Choose which model to save and click **Save Model**.")
    else:
        _target = trained_model if str(save_which_ui.value) == "gen1" else reflow_model
        if _target is None:
            _out = mo.md("_The chosen model is not trained yet._")
        else:
            _fname = str(save_filename_ui.value).strip() or default_checkpoint_name(str(save_which_ui.value))
            _models_dir = Path(__file__).resolve().parent.parent / "models"
            _weights_path, _config_path = save_model_and_config(_target, _models_dir, _fname)
            _out = mo.md(
                f"""
    **Saved.**

    - weights: `{_weights_path}`
    - config: `{_config_path}`
                """
            )
    _out
    return


if __name__ == "__main__":
    app.run()
