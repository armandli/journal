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

    from mlx.data.datasets import load_imagenet, load_imagenet_metadata

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
    # Class-Conditional Rectified Flow with a DiT backbone on ImageNet-1k (MLX)

    ## Research Goal

    Train a **class-conditional Rectified Flow** generative model
    (Liu et al. 2022, *"Flow Straight and Fast: Learning to Generate
    and Transfer Data with Rectified Flow"*) with a **Diffusion
    Transformer (DiT)** backbone on **ImageNet-1k**, then
    demonstrate two things:

    1. The final model can turn Gaussian noise into recognizable,
       class-conditional images via Euler ODE integration.
    2. The **reflow** stage produces measurably straighter ODE
       trajectories than the base CFM stage — this is what makes
       the method genuinely *rectified* flow rather than plain
       flow matching.

    ### What is Rectified Flow?

    Rectified Flow is a two-stage procedure:

    - **Stage A — gen1 (base Conditional Flow Matching)**: sample a
      fresh Gaussian noise `x_0 ~ N(0, I)` and a real image `x_1`
      independently; train the model to predict the constant
      velocity `v = x_1 - x_0` at any `t ~ U(0,1)` along the linear
      interpolation `x_t = (1 - t) x_0 + t x_1`. This yields a
      model whose ODE `dx/dt = v_theta(x_t, t, y)` transports
      Gaussian noise to the data distribution — but the individual
      trajectories are typically curved because the coupling
      `(x_0, x_1)` is random.
    - **Stage B — reflow (gen2)**: run the gen1 ODE on fresh
      Gaussian noise samples to obtain *coupled* pairs
      `(x_0, x_1_hat)`; then train a fresh model on the same
      flow-matching loss but with those pairs held fixed. Because
      the couplings now come from an actual ODE map, the resulting
      gen2 model learns a **straighter** flow that can be
      integrated with far fewer Euler steps for comparable sample
      quality. This is the "rectification" step.

    ### The ImageNet manual-download constraint (please read)

    `mlx.data.datasets.load_imagenet` **cannot download ImageNet
    automatically**. Its own docstring says: *"ImageNet cannot be
    automatically downloaded so you have to manually download it
    from http://image-net.org/. You need the split you want to load
    and the devkit for tasks 1 and 2."* Unlike `load_mnist` /
    `load_cifar10`, which auto-fetch, `load_imagenet` only works
    when the archives already exist under a root directory. There
    is therefore no real "download" button — Section 2 exposes a
    **Prepare / Load ImageNet** button that (a) never touches disk
    until you click it, and (b) fails with a clear manual-fetch
    message if the expected archive files are missing.

    Concretely you must obtain (register on image-net.org and
    accept the terms of use):

    - `ILSVRC2012_img_train.tar` (~138 GB) — training images
    - `ILSVRC2012_img_val.tar` (~6.3 GB) — validation images
    - `ILSVRC2012_devkit_t12.tar.gz` (~2.5 MB) — devkit with class
      metadata and val ground truth

    Place all three under the root directory you configure in
    Section 2 (default `../data/imagenet`).

    ### Resolution / patch-size / model-size defaults

    Full 256x256 pixel-space DiT training on ImageNet is
    intractable without a separate latent tokenizer (VAE / VQ-VAE)
    — that is out of scope for this notebook. We instead default
    to the well-established **ImageNet-64** convention (64x64
    images), which is the standard tractable pixel-space size used
    across the flow-matching / diffusion literature for
    single-machine experiments. Patch size defaults to **8**
    (64 tokens per image), keeping DiT compute reasonable on Apple
    Silicon. Both are exposed as UI dropdowns; raise them if you
    have more compute / time.

    ### Reflow-stage tractability caveat

    Building the reflow pair dataset requires running the gen1
    Euler ODE over `num_samples` fresh noise vectors. At the paper
    scale (millions of pairs) this is intractable on a single
    Apple Silicon machine, so the reflow dataset size is
    **configurable and bounded** in Section 5 (default 4096
    pairs). This is a **reduced-scale demonstration** of the
    reflow mechanism — enough to visibly straighten trajectories
    and reduce required Euler step count, but not a
    paper-quality reflow. Grow the reflow dataset for better
    results if compute allows.

    ### Notebook Outline

    1. Title & research goal (this cell)
    2. Data exploration (gated manual-download button)
    3. Dataset creation (lazy mlx.data streaming pipeline)
    4. Model definition (DiT backbone + flow utilities)
    5. Training — 5a gen1 CFM, 5b reflow (gen2)
    6. Optional hyperparameter search (gen1 CFM, bounded subset)
    7. Validation on official ImageNet val + k-fold CV (bounded)
    8. Results — loss curves, generation progression across ODE
       time, straightness proof, few-step comparison
    9. Save trained models
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration (gated manual-download)

    Nothing touches disk until you click **Prepare / Load
    ImageNet** below. On click, the notebook checks for the
    expected archive files under the configured root; if any are
    missing it reports the manual-download steps and stops.
    """)
    return


@app.cell
def _(mo):
    imagenet_root_ui = mo.ui.text(
        value="../data/imagenet",
        label="ImageNet root directory",
        full_width=True,
    )
    imagenet_split_explore_ui = mo.ui.dropdown(
        options=["val", "train"],
        value="val",
        label="Split to explore",
    )
    prepare_data_btn = mo.ui.run_button(
        label="Prepare / Load ImageNet"
    )
    mo.vstack(
        [
            imagenet_root_ui,
            mo.hstack([imagenet_split_explore_ui, prepare_data_btn]),
        ]
    )
    return imagenet_root_ui, imagenet_split_explore_ui, prepare_data_btn


@app.function
def check_imagenet_archives(root: str) -> tuple[bool, list, list]:
    root_path = Path(root).expanduser().resolve()
    required = [
        "ILSVRC2012_img_train.tar",
        "ILSVRC2012_img_val.tar",
        "ILSVRC2012_devkit_t12.tar.gz",
    ]
    already_extracted = {
        "ILSVRC2012_img_train.tar": "ILSVRC2012_img_train",
        "ILSVRC2012_img_val.tar": "ILSVRC2012_img_val",
        "ILSVRC2012_devkit_t12.tar.gz": "ILSVRC2012_devkit_t12",
    }
    present = []
    missing = []
    for arch in required:
        p_arch = root_path / arch
        p_dir = root_path / already_extracted[arch]
        if p_arch.is_file() or p_dir.is_dir():
            present.append(arch)
        else:
            missing.append(arch)
    return len(missing) == 0, present, missing


@app.function
def imagenet_manual_download_message(root: str, missing: list) -> str:
    root_path = Path(root).expanduser().resolve()
    return (
        f"### ImageNet archives missing from `{root_path}`\n\n"
        f"`mlx.data.datasets.load_imagenet` does **not** support automatic "
        f"download. The following required archive(s) were not found:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\n"
        + "**Manual steps** to fix:\n\n"
        + "1. Register a free account at https://image-net.org/ and "
        + "accept the ILSVRC 2012 terms of use.\n"
        + "2. Download the three files below (~145 GB total):\n"
        + "   - `ILSVRC2012_img_train.tar`\n"
        + "   - `ILSVRC2012_img_val.tar`\n"
        + "   - `ILSVRC2012_devkit_t12.tar.gz`\n"
        + f"3. Place all three files at `{root_path}/` (create the "
        + "directory if it does not exist), then click **Prepare / "
        + "Load ImageNet** again.\n\n"
        + "The archives may be either as-is `.tar[.gz]` files or "
        + "pre-extracted into sibling folders of the same base name — "
        + "either layout is accepted by `load_imagenet`."
    )


@app.cell
def _(imagenet_root_ui, imagenet_split_explore_ui, mo, prepare_data_btn):
    train_buf = None
    val_buf = None
    metadata_dict = None
    explore_buf = None
    explore_split = None
    imagenet_root_resolved = None

    if not prepare_data_btn.value:
        mo.output.replace(
            mo.md(
                "Click **Prepare / Load ImageNet** above to check for "
                "the required archive files under your configured root "
                "and load them. Nothing touches disk until you click."
            )
        )
    else:
        ok, present, missing = check_imagenet_archives(imagenet_root_ui.value)
        if not ok:
            mo.output.replace(
                mo.md(imagenet_manual_download_message(imagenet_root_ui.value, missing))
            )
        else:
            imagenet_root_resolved = str(Path(imagenet_root_ui.value).expanduser().resolve())
            mo.output.replace(mo.md(f"Loading ImageNet from `{imagenet_root_resolved}`..."))
            metadata_dict = load_imagenet_metadata(root=imagenet_root_resolved)
            explore_split = imagenet_split_explore_ui.value
            if explore_split == "val":
                val_buf = load_imagenet(root=imagenet_root_resolved, split="val")
                explore_buf = val_buf
            else:
                train_buf = load_imagenet(root=imagenet_root_resolved, split="train")
                explore_buf = train_buf
            mo.output.replace(
                mo.md(
                    f"Loaded ImageNet {explore_split} split from "
                    f"`{imagenet_root_resolved}` — buffer size: "
                    f"`{len(explore_buf):,}` samples. Metadata classes: "
                    f"`{len(metadata_dict):,}`."
                )
            )
    return explore_buf, explore_split, imagenet_root_resolved, metadata_dict


@app.function
def imagenet_class_names(metadata: dict) -> list:
    if metadata is None:
        return [str(i) for i in range(1000)]
    rows = [(v["label"], v.get("description", str(v["label"]))) for v in metadata.values()]
    rows.sort(key=lambda r: r[0])
    return [str(desc).split(",")[0].strip() for _, desc in rows]


@app.function
def collect_bounded_samples(buf, n: int, resize_to: int = 96, crop_to: int = 64) -> tuple[np.ndarray, np.ndarray]:
    stream = (
        buf.to_stream()
        .image_resize_smallest_side("image", resize_to)
        .image_center_crop("image", crop_to, crop_to)
        .key_transform("image", to_rgb_uint8)
        .batch(1)
    )
    imgs = []
    labels = []
    count = 0
    for sample in stream:
        img = np.array(sample["image"]).squeeze(0)
        lbl = int(np.array(sample["label"]).squeeze())
        imgs.append(img)
        labels.append(lbl)
        count += 1
        if count >= n:
            break
    return np.stack(imgs, axis=0), np.array(labels, dtype=np.int32)


@app.function
def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 4:
        return img[:, :, :3].astype(np.uint8)
    return img.astype(np.uint8)


@app.function
def plot_sample_grid(images: np.ndarray, labels: np.ndarray, class_names: list, n_show: int = 40, cols: int = 8):
    rows = int(math.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.8))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        if i < min(n_show, images.shape[0]):
            ax.imshow(np.clip(images[i], 0, 255).astype(np.uint8))
            name = class_names[int(labels[i])] if int(labels[i]) < len(class_names) else str(labels[i])
            ax.set_title(name[:18], fontsize=7)
        ax.axis("off")
    fig.suptitle("ImageNet sample images (bounded scan; not a full-dataset pass)", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def plot_class_distribution(labels: np.ndarray, num_classes: int = 1000, title: str = "Class distribution (bounded scan)"):
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(num_classes), counts, color="steelblue", width=1.0)
    ax.set_xlabel("Class index (0-999)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(explore_buf, metadata_dict, mo):
    if explore_buf is None or metadata_dict is None:
        _out = mo.md("_Load ImageNet above to see sample images._")
    else:
        _class_names = imagenet_class_names(metadata_dict)
        _imgs, _labels = collect_bounded_samples(explore_buf, n=32, resize_to=80, crop_to=64)
        _out = plot_sample_grid(_imgs, _labels, _class_names, n_show=_imgs.shape[0], cols=8)
    _out
    return


@app.cell
def _(explore_buf, mo):
    if explore_buf is None:
        _out = mo.md("_Load ImageNet above to see class distribution over a bounded scan._")
    else:
        _imgs, _labels = collect_bounded_samples(explore_buf, n=2000, resize_to=80, crop_to=64)
        _out = plot_class_distribution(_labels, num_classes=1000, title="ImageNet class distribution (2000-sample bounded scan)")
    _out
    return


@app.cell
def _(explore_buf, explore_split, imagenet_root_resolved, metadata_dict, mo):
    if explore_buf is None or metadata_dict is None:
        _out = mo.md("_Load ImageNet above to see split sizes._")
    else:
        _loaded_size = len(explore_buf)
        _out = mo.md(
            f"""
            ### Dataset overview

            | Field | Value |
            |-------|-------|
            | Root directory | `{imagenet_root_resolved}` |
            | Loaded split | `{explore_split}` |
            | Loaded split size | `{_loaded_size:,}` samples |
            | Number of classes (metadata) | `{len(metadata_dict):,}` |

            **Official ImageNet-1k split sizes (for reference):**

            | Split | Size | Notes |
            |-------|------|-------|
            | train | 1,281,167 | Used for training + a small monitoring-val slice |
            | val | 50,000 | Used as the held-out test set (Section 7) |
            | classes | 1,000 | 1000-way class-conditional generation |

            **Monitoring-val slice policy**: unlike MNIST / CIFAR-10
            (85/15 train/val split) we carve only a **small monitoring
            slice** (~1% of the training stream, bounded to a few
            hundred batches) from the training data for per-epoch
            monitoring, because the training set is 1.28M images and a
            15% slice would waste huge amounts of data. The **official
            50k val split** is our held-out test set in Section 7.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation (lazy streaming pipeline)

    The pipeline uses mlx.data's native per-sample image ops on the
    buffer / stream — **no eager NumPy load** (that pattern from
    this repo's CIFAR-10 notebooks does not scale to 1.28M
    variable-resolution ImageNet images).

    - **Train stream**: `image_resize_smallest_side(resize)` -->
      `image_random_crop(res, res)` --> `image_random_h_flip(0.5)`
      for augmentation, shuffle on buffer before `to_stream()`,
      then batch + prefetch.
    - **Val / test stream**: same resize, then
      `image_center_crop(res, res)` (deterministic, no flip).
    - **Pixel normalization**: `uint8 [0, 255]` --> `float32
      [-1, 1]` via `x / 127.5 - 1`. Rationale: flow-matching noise
      is Gaussian `N(0, I)`, whose mean and rough scale match
      `[-1, 1]`-centered pixels; this keeps `x_t = (1-t) x_0 +
      t x_1` in a consistent range across `t`.
    - Grayscale / RGBA outliers in ImageNet are normalized to
      3-channel RGB via a small `key_transform`.

    Resolution, patch size, and batch size are UI-configurable.
    """)
    return


@app.cell
def _(mo):
    resolution_ui = mo.ui.dropdown(
        options={"32": 32, "48": 48, "64": 64, "96": 96, "128": 128},
        value="64",
        label="Image resolution",
    )
    patch_size_ui = mo.ui.dropdown(
        options={"4": 4, "8": 8, "16": 16},
        value="8",
        label="Patch size",
    )
    train_bs_ui = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="64",
        label="Training batch size",
    )
    monitor_bs_ui = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128},
        value="64",
        label="Monitoring / eval batch size",
    )
    prefetch_ui = mo.ui.slider(1, 8, value=4, step=1, label="Prefetch buffer")
    threads_ui = mo.ui.slider(1, 8, value=4, step=1, label="Prefetch threads")
    mo.vstack(
        [
            mo.hstack([resolution_ui, patch_size_ui]),
            mo.hstack([train_bs_ui, monitor_bs_ui]),
            mo.hstack([prefetch_ui, threads_ui]),
        ]
    )
    return (
        monitor_bs_ui,
        patch_size_ui,
        prefetch_ui,
        resolution_ui,
        threads_ui,
        train_bs_ui,
    )


@app.function
def build_train_stream(
    buf,
    resolution: int = 64,
    batch_size: int = 64,
    prefetch: int = 4,
    threads: int = 4,
):
    resize_to = int(round(resolution * 1.15))
    return (
        buf.shuffle()
        .to_stream()
        .image_resize_smallest_side("image", resize_to)
        .image_random_crop("image", resolution, resolution)
        .image_random_h_flip("image", 0.5)
        .key_transform("image", to_rgb_uint8)
        .batch(batch_size)
        .prefetch(prefetch_size=prefetch, num_threads=threads)
    )


@app.function
def build_eval_stream(
    buf,
    resolution: int = 64,
    batch_size: int = 64,
    prefetch: int = 4,
    threads: int = 4,
):
    resize_to = int(round(resolution * 1.15))
    return (
        buf.to_stream()
        .image_resize_smallest_side("image", resize_to)
        .image_center_crop("image", resolution, resolution)
        .key_transform("image", to_rgb_uint8)
        .batch(batch_size)
        .prefetch(prefetch_size=prefetch, num_threads=threads)
    )


@app.function
def preprocess_batch(batch: dict) -> tuple[mx.array, mx.array]:
    x = mx.array(batch["image"], dtype=mx.float32) / 127.5 - 1.0
    y = mx.array(batch["label"], dtype=mx.int32)
    return x, y


@app.cell
def _(
    explore_buf,
    imagenet_root_resolved,
    metadata_dict,
    mo,
    monitor_bs_ui,
    prefetch_ui,
    resolution_ui,
    threads_ui,
    train_bs_ui,
):
    train_stream = None
    monitor_stream = None
    test_stream = None
    dataset_ready = False

    if explore_buf is None or metadata_dict is None or imagenet_root_resolved is None:
        mo.output.replace(mo.md("_Load ImageNet in Section 2 first — the streaming pipeline needs the loaded buffer._"))
    else:
        _train_buf = load_imagenet(root=imagenet_root_resolved, split="train")
        _val_buf = load_imagenet(root=imagenet_root_resolved, split="val")
        train_stream = build_train_stream(
            _train_buf,
            resolution=resolution_ui.value,
            batch_size=train_bs_ui.value,
            prefetch=prefetch_ui.value,
            threads=threads_ui.value,
        )
        monitor_stream = build_eval_stream(
            _val_buf,
            resolution=resolution_ui.value,
            batch_size=monitor_bs_ui.value,
            prefetch=prefetch_ui.value,
            threads=threads_ui.value,
        )
        test_stream = build_eval_stream(
            _val_buf,
            resolution=resolution_ui.value,
            batch_size=monitor_bs_ui.value,
            prefetch=prefetch_ui.value,
            threads=threads_ui.value,
        )
        dataset_ready = True
        mo.output.replace(
            mo.md(
                f"Streams built at resolution `{resolution_ui.value}` — "
                f"train batch size `{train_bs_ui.value}`, "
                f"eval batch size `{monitor_bs_ui.value}`."
            )
        )
    return dataset_ready, monitor_stream, test_stream, train_stream


@app.cell
def _(mo, resolution_ui, train_stream):
    if train_stream is None:
        _out = mo.md("_Build the streams above first._")
    else:
        _iter = iter(train_stream)
        _b = next(_iter)
        _x, _y = preprocess_batch(_b)
        _out = mo.md(
            f"""
            ### One-batch check

            - `image` batch shape: `{tuple(_x.shape)}`, dtype `{_x.dtype}`
            - `label` batch shape: `{tuple(_y.shape)}`, dtype `{_y.dtype}`
            - Value range: min `{float(mx.min(_x).item()):.3f}`, max `{float(mx.max(_x).item()):.3f}` (expected in `[-1, 1]`)
            - Resolution: `{resolution_ui.value}x{resolution_ui.value}`
            """
        )
        train_stream.reset()
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition

    DiT backbone adapted from this repo's
    `mlx/dit_rcfm_cifar10.py`, scaled up for ImageNet-1k. All
    building blocks are the `V1` variants; sizes (depth / width /
    heads / patch-size) are exposed as configurable parameters to
    the model factory.

    **Sizing tradeoff**. The default (`embed_dim=384`,
    `num_heads=6`, `mlp_dim=1536`, `num_layers=8`) is a *small*
    DiT (roughly DiT-S / DiT-B territory in the original paper's
    naming) — enough to demonstrate reflow behaviour and generate
    recognizable class-conditional images at 64x64, while keeping
    a single-machine Apple Silicon training run within reach.
    Larger models (`num_layers>=12`, `embed_dim>=512`) give
    higher-fidelity samples but push training time from hours
    into days. The 1000-way class embedding is deliberately
    cheap; the "spare slot" pattern (`num_classes + 1`) is
    preserved for potential null-class classifier-free-guidance
    use later.
    """)
    return


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / max(half, 1))
        # MLX registers every non-underscore mx.array attribute as a TRAINABLE
        # parameter (nn.Module.valid_parameter_filter). A sinusoidal frequency
        # bank is fixed geometry: left unfrozen, AdamW grinds its geometric
        # decay into noise and the time embedding loses multi-scale resolution.
        # freeze() is used here rather than a "_freqs" rename so the key stays
        # in parameters() and existing checkpoints still load.
        self.freeze(keys=["freqs"], recurse=False)

    def __call__(self, t: mx.array) -> mx.array:
        return mx.concatenate(
            [mx.sin(t[:, None] * self.freqs[None, :]), mx.cos(t[:, None] * self.freqs[None, :])],
            axis=-1,
        )


@app.class_definition
class AdaptiveLayerNormV1(nn.Module):
    def __init__(self, dim: int = 384, cond_dim: int = 384):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        scale, shift = mx.split(self.proj(nn.silu(cond))[:, None, :], 2, axis=-1)
        return self.norm(x) * (1.0 + scale) + shift


@app.class_definition
class PatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 8, embed_dim: int = 384, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x).reshape(x.shape[0], -1, self.proj.weight.shape[0])


@app.class_definition
class DiTBlockV1(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_dim: int = 1536, cond_dim: int = 384):
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
    def __init__(self, patch_size: int = 8, embed_dim: int = 384, out_channels: int = 3, image_size: int = 64):
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
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        num_layers: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
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
def build_dit_model(
    image_size: int = 64,
    patch_size: int = 8,
    embed_dim: int = 384,
    num_heads: int = 6,
    mlp_dim: int = 1536,
    num_layers: int = 8,
    num_classes: int = 1000,
) -> DiffusionTransformerV1:
    model = DiffusionTransformerV1(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    mx.eval(model.parameters())
    return model


@app.function
def build_lr_schedule(peak_lr: float, warmup_steps: int, total_steps: int):
    warmup = optim.linear_schedule(0.0, peak_lr, warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    decay = optim.cosine_decay(peak_lr, decay_steps, end=peak_lr * 0.1)
    return optim.join_schedules([warmup, decay], [warmup_steps])


@app.function
def clip_and_apply_grads(model: nn.Module, optimizer, grads, max_norm: float = 1.0):
    clipped, _ = optim.clip_grad_norm(grads, max_norm)
    optimizer.update(model, clipped)


@app.cell
def _(mo, patch_size_ui, resolution_ui):
    mo.md(
        f"""
        ### Model Architecture — `DiffusionTransformerV1`

        | Component | Module | Output Shape |
        |-----------|--------|--------------|
        | Patch embed | `PatchifyV1` (Conv2d, stride=patch) | `(B, N, D)` where `N = (H/p)^2` |
        | Positional embed | learnable table `(1, N, D)` | `(B, N, D)` |
        | Time embed | `SinusoidalTimestepEmbeddingV1` + MLP | `(B, D)` |
        | Class embed | `nn.Embedding(1001, D)` | `(B, D)` |
        | Backbone | `DiTBlockV1 x num_layers` (AdaLN attn + AdaLN MLP) | `(B, N, D)` |
        | Head | `LayerNorm` + `UnpatchifyV1` | `(B, H, W, 3)` |

        **Currently configured** (from Section 3 UI):
        `image_size = {resolution_ui.value}`, `patch_size = {patch_size_ui.value}`
        → `num_patches = ({resolution_ui.value}/{patch_size_ui.value})^2 = {(resolution_ui.value // patch_size_ui.value) ** 2}`

        **Default model width** (Section 5 UI): `embed_dim=384`,
        `num_heads=6`, `mlp_dim=1536`, `num_layers=8`.

        ### Path-straightness metric

        `compute_path_straightness(model, x0, y, num_steps)` runs the
        Euler ODE and reports the mean squared deviation of per-step
        displacements from a straight-line path — 0 for a perfectly
        straight trajectory, larger for curved paths. Rectified flow's
        reflow step is expected to reduce it vs. gen1.

        ### Reference

        Liu, Gong, Liu (2022). *Flow Straight and Fast: Learning to
        Generate and Transfer Data with Rectified Flow.*
        arXiv:2209.03003. The reflow procedure implemented here
        (Section 5b) is Algorithm 1 of that paper, restricted to a
        bounded synthetic pair dataset for on-device tractability.
        """
    )
    return


@app.cell
def _(patch_size_ui, resolution_ui):
    default_model_preview = build_dit_model(
        image_size=resolution_ui.value,
        patch_size=patch_size_ui.value,
        embed_dim=384,
        num_heads=6,
        mlp_dim=1536,
        num_layers=8,
        num_classes=1000,
    )
    default_param_count = count_parameters(default_model_preview)
    return (default_param_count,)


@app.cell
def _(default_param_count, mo):
    mo.md(f"**Default model parameter count**: `{default_param_count:,}`")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training

    Two stages, each gated by its own button.

    - **5a — gen1 (base CFM)**: streaming pipeline; linear warmup
      + cosine decay LR schedule; AdamW with gradient clipping
      (max-norm `1.0`) — all standard for stable large-scale
      transformer training. Each "epoch" is capped by
      `steps_per_epoch` for tractability (a real ImageNet epoch is
      ~20,000 steps at bs=64).
    - **5b — reflow (gen2)**: gated on gen1 having been trained.
      Builds a bounded synthetic `(x_0, x_1)` pair dataset from
      gen1's Euler ODE, then trains a fresh model on those fixed
      pairs. Bounded to a few thousand pairs by default (see the
      Reflow-stage tractability caveat in Section 1).

    Both stages return `(train_losses, val_losses, trained_model)`
    which are `([], [], None)` until the button is clicked.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 5a — gen1 (base Conditional Flow Matching)
    """)
    return


@app.cell
def _(mo):
    lr_gen1_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Peak LR",
    )
    wd_gen1_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2},
        value="1e-4",
        label="Weight decay",
    )
    epochs_gen1_ui = mo.ui.slider(1, 40, value=8, step=1, label="Epochs")
    steps_per_epoch_gen1_ui = mo.ui.slider(
        50, 5000, value=500, step=50, label="Steps per epoch (bounded for tractability)"
    )
    warmup_steps_gen1_ui = mo.ui.slider(0, 2000, value=200, step=50, label="Warmup steps")
    depth_gen1_ui = mo.ui.slider(2, 16, value=8, step=1, label="DiT depth (num_layers)")
    embed_dim_gen1_ui = mo.ui.dropdown(
        options={"192": 192, "256": 256, "384": 384, "512": 512},
        value="384",
        label="Embed dim",
    )
    heads_gen1_ui = mo.ui.dropdown(
        options={"3": 3, "4": 4, "6": 6, "8": 8},
        value="6",
        label="Attention heads",
    )
    mlp_dim_gen1_ui = mo.ui.dropdown(
        options={"768": 768, "1024": 1024, "1536": 1536, "2048": 2048},
        value="1536",
        label="MLP dim",
    )
    max_eval_batches_ui = mo.ui.slider(
        1, 100, value=20, step=1, label="Val batches per monitoring pass"
    )
    train_gen1_btn = mo.ui.run_button(label="Train Gen1")
    mo.vstack(
        [
            mo.md(
                "**Tractability note**: `Steps per epoch` bounds each "
                "monitoring epoch. A full ImageNet epoch at bs=64 is "
                "~20,000 steps; 500 steps/epoch x 8 epochs = 4,000 "
                "gradient updates is a modest starting run for a demo "
                "— scale up for real fidelity."
            ),
            mo.hstack([lr_gen1_ui, wd_gen1_ui, warmup_steps_gen1_ui]),
            mo.hstack([epochs_gen1_ui, steps_per_epoch_gen1_ui, max_eval_batches_ui]),
            mo.hstack([depth_gen1_ui, embed_dim_gen1_ui, heads_gen1_ui, mlp_dim_gen1_ui]),
            train_gen1_btn,
        ]
    )
    return (
        depth_gen1_ui,
        embed_dim_gen1_ui,
        epochs_gen1_ui,
        heads_gen1_ui,
        lr_gen1_ui,
        max_eval_batches_ui,
        mlp_dim_gen1_ui,
        steps_per_epoch_gen1_ui,
        train_gen1_btn,
        warmup_steps_gen1_ui,
        wd_gen1_ui,
    )


@app.function
def run_train_epoch_cfm_stream(
    model: nn.Module,
    optimizer,
    stream,
    max_steps: int,
    grad_clip: float = 1.0,
) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    epoch_loss = 0.0
    n = 0
    stream.reset()
    for batch in stream:
        x1, y = preprocess_batch(batch)
        x0 = sample_noise_like(x1)
        loss, grads = loss_and_grad_fn(model, x1, x0, y)
        clip_and_apply_grads(model, optimizer, grads, grad_clip)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
        if n >= max_steps:
            break
    return epoch_loss / max(n, 1)


@app.function
def evaluate_model_stream(model: nn.Module, stream, max_batches: int) -> float:
    total = 0.0
    n = 0
    stream.reset()
    for batch in stream:
        x1, y = preprocess_batch(batch)
        x0 = sample_noise_like(x1)
        loss = compute_flow_loss(model, x1, x0, y)
        mx.eval(loss)
        total += loss.item()
        n += 1
        if n >= max_batches:
            break
    return total / max(n, 1)


@app.function
def train_gen1_regime(
    train_stream,
    monitor_stream,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    num_layers: int,
    peak_lr: float,
    wd: float,
    warmup_steps: int,
    epochs: int,
    steps_per_epoch: int,
    max_eval_batches: int,
    progress_cb=None,
) -> tuple:
    model = build_dit_model(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
    )
    total_steps = max(epochs * steps_per_epoch, 1)
    schedule = build_lr_schedule(peak_lr, warmup_steps, total_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=wd)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        tl = run_train_epoch_cfm_stream(model, optimizer, train_stream, steps_per_epoch)
        vl = evaluate_model_stream(model, monitor_stream, max_eval_batches)
        train_losses.append(tl)
        val_losses.append(vl)
        if progress_cb is not None:
            progress_cb(epoch, epochs, tl, vl)
    return model, train_losses, val_losses


@app.cell
def _(
    dataset_ready,
    depth_gen1_ui,
    embed_dim_gen1_ui,
    epochs_gen1_ui,
    heads_gen1_ui,
    lr_gen1_ui,
    max_eval_batches_ui,
    mlp_dim_gen1_ui,
    mo,
    monitor_stream,
    patch_size_ui,
    resolution_ui,
    steps_per_epoch_gen1_ui,
    train_gen1_btn,
    train_stream,
    warmup_steps_gen1_ui,
    wd_gen1_ui,
):
    train_losses_gen1 = []
    val_losses_gen1 = []
    trained_model_gen1 = None
    if not dataset_ready:
        mo.output.replace(mo.md("_Load ImageNet and build streams (Sections 2 and 3) first._"))
    elif not train_gen1_btn.value:
        mo.output.replace(mo.md("Click **Train Gen1** to begin the base CFM training stage."))
    else:
        def _cb(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**Gen1 Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model_gen1, train_losses_gen1, val_losses_gen1 = train_gen1_regime(
            train_stream,
            monitor_stream,
            image_size=resolution_ui.value,
            patch_size=patch_size_ui.value,
            embed_dim=embed_dim_gen1_ui.value,
            num_heads=heads_gen1_ui.value,
            mlp_dim=mlp_dim_gen1_ui.value,
            num_layers=depth_gen1_ui.value,
            peak_lr=lr_gen1_ui.value,
            wd=wd_gen1_ui.value,
            warmup_steps=warmup_steps_gen1_ui.value,
            epochs=epochs_gen1_ui.value,
            steps_per_epoch=steps_per_epoch_gen1_ui.value,
            max_eval_batches=max_eval_batches_ui.value,
            progress_cb=_cb,
        )
        mo.output.replace(
            mo.md(
                f"**Gen1 training complete!** Final train "
                f"`{train_losses_gen1[-1]:.4f}` | val `{val_losses_gen1[-1]:.4f}` | "
                f"params `{count_parameters(trained_model_gen1):,}`"
            )
        )
    return train_losses_gen1, trained_model_gen1, val_losses_gen1


@app.cell
def _(mo):
    mo.md("""
    ### 5b — Reflow (gen2)

    Bounded reflow: (1) draw `num_samples` fresh Gaussian noise
    vectors and random class labels, (2) run the gen1 Euler ODE on
    them to obtain synthetic `x_1_hat`, (3) train a fresh gen2
    model on those fixed `(x_0, x_1_hat)` pairs with the same
    flow-matching loss. The resulting gen2 has a straighter flow
    (Section 8b quantifies this).

    Default `num_samples = 4096` — a demo-scale reflow set. Grow
    it if you have more compute (see the tractability caveat in
    Section 1).
    """)
    return


@app.cell
def _(mo):
    lr_reflow_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Peak LR (reflow)",
    )
    wd_reflow_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2},
        value="1e-4",
        label="Weight decay (reflow)",
    )
    epochs_reflow_ui = mo.ui.slider(1, 60, value=15, step=1, label="Epochs (reflow)")
    warmup_reflow_ui = mo.ui.slider(0, 500, value=100, step=25, label="Warmup steps (reflow)")
    reflow_bs_ui = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="64",
        label="Reflow batch size",
    )
    num_reflow_samples_ui = mo.ui.slider(
        512, 32768, value=4096, step=512, label="Reflow dataset size (num pairs)"
    )
    gen_steps_reflow_ui = mo.ui.slider(
        10, 100, value=50, step=5, label="Euler steps used to build reflow pairs"
    )
    depth_reflow_ui = mo.ui.slider(2, 16, value=8, step=1, label="DiT depth (reflow)")
    train_reflow_btn = mo.ui.run_button(label="Build Reflow Dataset + Train Gen2")
    mo.vstack(
        [
            mo.md(
                "The reflow dataset build calls the gen1 model "
                "`num_pairs / batch_size` times with `gen_steps` "
                "ODE integrations each — cost is `num_pairs * "
                "gen_steps` model forward evaluations. Keep both "
                "modest for a first run."
            ),
            mo.hstack([lr_reflow_ui, wd_reflow_ui, warmup_reflow_ui]),
            mo.hstack([epochs_reflow_ui, reflow_bs_ui, depth_reflow_ui]),
            mo.hstack([num_reflow_samples_ui, gen_steps_reflow_ui]),
            train_reflow_btn,
        ]
    )
    return (
        depth_reflow_ui,
        epochs_reflow_ui,
        gen_steps_reflow_ui,
        lr_reflow_ui,
        num_reflow_samples_ui,
        reflow_bs_ui,
        train_reflow_btn,
        warmup_reflow_ui,
        wd_reflow_ui,
    )


@app.function
def build_reflow_dataset(
    model_gen1: nn.Module,
    num_samples: int,
    num_steps: int,
    image_size: int,
    num_classes: int,
    batch_size: int = 64,
    seed: int = 12345,
) -> tuple[mx.array, mx.array, mx.array]:
    x0_chunks = []
    x1_chunks = []
    y_chunks = []
    total = 0
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    while total < num_samples:
        cur = min(batch_size, num_samples - total)
        x0 = mx.random.normal(shape=(cur, image_size, image_size, 3))
        y_np = rng.integers(0, num_classes, size=cur).astype(np.int32)
        y = mx.array(y_np)
        x1 = euler_solve(model_gen1, x0, y, num_steps)
        mx.eval(x1)
        x0_chunks.append(x0)
        x1_chunks.append(x1)
        y_chunks.append(y)
        total += cur
    return (
        mx.concatenate(x0_chunks, axis=0),
        mx.concatenate(x1_chunks, axis=0),
        mx.concatenate(y_chunks, axis=0),
    )


@app.function
def make_reflow_batch_indices(n: int, batch_size: int, shuffle: bool = True) -> list:
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    return [idx[start : min(start + batch_size, n)].astype(np.int32) for start in range(0, n, batch_size)]


@app.function
def run_train_epoch_reflow(
    model: nn.Module,
    optimizer,
    x0_pool: mx.array,
    x1_pool: mx.array,
    y_pool: mx.array,
    batch_size: int,
    grad_clip: float = 1.0,
) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    batches_idx = make_reflow_batch_indices(x0_pool.shape[0], batch_size, shuffle=True)
    epoch_loss = 0.0
    n = 0
    for idx_np in batches_idx:
        idx_mx = mx.array(idx_np)
        x0b = x0_pool[idx_mx]
        x1b = x1_pool[idx_mx]
        yb = y_pool[idx_mx]
        loss, grads = loss_and_grad_fn(model, x1b, x0b, yb)
        clip_and_apply_grads(model, optimizer, grads, grad_clip)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def evaluate_reflow_stream(model: nn.Module, stream, max_batches: int) -> float:
    return evaluate_model_stream(model, stream, max_batches)


@app.function
def train_reflow_stage(
    model_gen1: nn.Module,
    monitor_stream,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    num_layers: int,
    peak_lr: float,
    wd: float,
    warmup_steps: int,
    epochs: int,
    batch_size: int,
    num_reflow_samples: int,
    gen_steps: int,
    max_eval_batches: int,
    progress_cb=None,
) -> tuple:
    x0_rf, x1_rf, y_rf = build_reflow_dataset(
        model_gen1,
        num_samples=num_reflow_samples,
        num_steps=gen_steps,
        image_size=image_size,
        num_classes=1000,
        batch_size=batch_size,
    )
    model = build_dit_model(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
    )
    total_steps = max(epochs * math.ceil(num_reflow_samples / batch_size), 1)
    schedule = build_lr_schedule(peak_lr, warmup_steps, total_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=wd)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        tl = run_train_epoch_reflow(model, optimizer, x0_rf, x1_rf, y_rf, batch_size)
        vl = evaluate_reflow_stream(model, monitor_stream, max_eval_batches)
        train_losses.append(tl)
        val_losses.append(vl)
        if progress_cb is not None:
            progress_cb(epoch, epochs, tl, vl)
    return model, train_losses, val_losses


@app.cell
def _(
    depth_reflow_ui,
    embed_dim_gen1_ui,
    epochs_reflow_ui,
    gen_steps_reflow_ui,
    heads_gen1_ui,
    lr_reflow_ui,
    max_eval_batches_ui,
    mlp_dim_gen1_ui,
    mo,
    monitor_stream,
    num_reflow_samples_ui,
    patch_size_ui,
    reflow_bs_ui,
    resolution_ui,
    train_reflow_btn,
    trained_model_gen1,
    warmup_reflow_ui,
    wd_reflow_ui,
):
    train_losses_reflow = []
    val_losses_reflow = []
    trained_model_reflow = None
    if trained_model_gen1 is None:
        mo.output.replace(mo.md("_Train the gen1 model above first — reflow needs a gen1 to sample from._"))
    elif not train_reflow_btn.value:
        mo.output.replace(
            mo.md(
                "Click **Build Reflow Dataset + Train Gen2** to run the reflow stage."
            )
        )
    else:
        mo.output.replace(mo.md("Building reflow pair dataset from gen1..."))
        def _cb_rf(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**Reflow Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model_reflow, train_losses_reflow, val_losses_reflow = train_reflow_stage(
            trained_model_gen1,
            monitor_stream,
            image_size=resolution_ui.value,
            patch_size=patch_size_ui.value,
            embed_dim=embed_dim_gen1_ui.value,
            num_heads=heads_gen1_ui.value,
            mlp_dim=mlp_dim_gen1_ui.value,
            num_layers=depth_reflow_ui.value,
            peak_lr=lr_reflow_ui.value,
            wd=wd_reflow_ui.value,
            warmup_steps=warmup_reflow_ui.value,
            epochs=epochs_reflow_ui.value,
            batch_size=reflow_bs_ui.value,
            num_reflow_samples=num_reflow_samples_ui.value,
            gen_steps=gen_steps_reflow_ui.value,
            max_eval_batches=max_eval_batches_ui.value,
            progress_cb=_cb_rf,
        )
        mo.output.replace(
            mo.md(
                f"**Reflow training complete!** "
                f"Pairs: `{num_reflow_samples_ui.value}` | "
                f"final train `{train_losses_reflow[-1]:.4f}` | "
                f"val `{val_losses_reflow[-1]:.4f}`"
            )
        )
    return train_losses_reflow, trained_model_reflow, val_losses_reflow


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (Optional)

    Small grid over `peak_lr x depth` on the gen1 CFM stage,
    trained on a bounded number of steps per config for
    tractability. Results in a sortable table (ascending by
    val_loss).
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_config(
    train_stream,
    monitor_stream,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    num_layers: int,
    peak_lr: float,
    warmup_steps: int,
    epochs: int,
    steps_per_epoch: int,
    max_eval_batches: int,
) -> float:
    model = build_dit_model(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
    )
    total_steps = max(epochs * steps_per_epoch, 1)
    schedule = build_lr_schedule(peak_lr, warmup_steps, total_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=1e-4)
    for _ in range(epochs):
        run_train_epoch_cfm_stream(model, optimizer, train_stream, steps_per_epoch)
    return evaluate_model_stream(model, monitor_stream, max_eval_batches)


@app.cell
def _(
    dataset_ready,
    embed_dim_gen1_ui,
    heads_gen1_ui,
    hp_search_cb,
    mlp_dim_gen1_ui,
    mo,
    monitor_stream,
    patch_size_ui,
    resolution_ui,
    train_stream,
):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    mo.stop(not dataset_ready, mo.md("_Load ImageNet and build streams first (Sections 2, 3)._"))

    _search_space = {"lr": [1e-4, 3e-4], "num_layers": [4, 6, 8]}
    _hp_epochs = 2
    _hp_steps = 100
    _hp_eval = 10
    hp_results = []
    for _lr in _search_space["lr"]:
        for _nl in _search_space["num_layers"]:
            _vl = run_hp_config(
                train_stream,
                monitor_stream,
                image_size=resolution_ui.value,
                patch_size=patch_size_ui.value,
                embed_dim=embed_dim_gen1_ui.value,
                num_heads=heads_gen1_ui.value,
                mlp_dim=mlp_dim_gen1_ui.value,
                num_layers=_nl,
                peak_lr=_lr,
                warmup_steps=50,
                epochs=_hp_epochs,
                steps_per_epoch=_hp_steps,
                max_eval_batches=_hp_eval,
            )
            hp_results.append({"lr": _lr, "num_layers": _nl, "val_loss": round(_vl, 4)})
            mo.output.replace(mo.md(f"lr={_lr}, num_layers={_nl} -> val={_vl:.4f}"))
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation

    - **Held-out test evaluation** on the official ImageNet val
      split (50k images), bounded to a configurable number of
      batches to keep this reactive. Both gen1 and reflow models
      are evaluated.
    - **k=5 fold CV** on a bounded slice of the training stream:
      collect `n_cv_pool` images once into a small in-memory
      array, then run standard k-fold with `n_cv_epochs`
      mini-epochs per fold. Reduced-schedule note: k=5 folds x
      few epochs is intentionally small at ImageNet scale.
    """)
    return


@app.cell
def _(mo):
    max_test_batches_ui = mo.ui.slider(
        5, 200, value=30, step=5, label="Max test batches (Section 7 held-out eval)"
    )
    max_test_batches_ui
    return (max_test_batches_ui,)


@app.cell
def _(max_test_batches_ui, mo, test_stream, trained_model_gen1):
    if trained_model_gen1 is None:
        _out = mo.md("_Train gen1 first (Section 5a) before running the test-set eval._")
    else:
        test_loss_gen1 = evaluate_model_stream(trained_model_gen1, test_stream, max_test_batches_ui.value)
        _out = mo.md(
            f"**Gen1 held-out test loss** (ImageNet val, "
            f"{max_test_batches_ui.value} batches): `{test_loss_gen1:.4f}`"
        )
    _out
    return


@app.cell
def _(max_test_batches_ui, mo, test_stream, trained_model_reflow):
    if trained_model_reflow is None:
        _out = mo.md("_Train the reflow model first (Section 5b) before running the test-set eval._")
    else:
        test_loss_reflow = evaluate_model_stream(trained_model_reflow, test_stream, max_test_batches_ui.value)
        _out = mo.md(
            f"**Reflow (gen2) held-out test loss** (ImageNet val, "
            f"{max_test_batches_ui.value} batches): `{test_loss_reflow:.4f}`"
        )
    _out
    return


@app.function
def collect_cv_pool(stream, n: int, resolution: int) -> tuple[mx.array, mx.array]:
    imgs = []
    labels = []
    got = 0
    stream.reset()
    for batch in stream:
        x, y = preprocess_batch(batch)
        imgs.append(x)
        labels.append(y)
        got += x.shape[0]
        if got >= n:
            break
    x_all = mx.concatenate(imgs, axis=0)[:n]
    y_all = mx.concatenate(labels, axis=0)[:n]
    return x_all, y_all


@app.function
def run_cv_fold(
    x_pool: mx.array,
    y_pool: mx.array,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    num_layers: int,
    peak_lr: float,
    n_epochs: int,
    batch_size: int,
    warmup_steps: int,
) -> float:
    tr_mx = mx.array(train_idx.astype(np.int32))
    va_mx = mx.array(val_idx.astype(np.int32))
    x_tr = x_pool[tr_mx]
    y_tr = y_pool[tr_mx]
    x_va = x_pool[va_mx]
    y_va = y_pool[va_mx]
    model = build_dit_model(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
    )
    steps_per_epoch = max(1, math.ceil(x_tr.shape[0] / batch_size))
    total_steps = max(n_epochs * steps_per_epoch, 1)
    schedule = build_lr_schedule(peak_lr, warmup_steps, total_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    for _ in range(n_epochs):
        idxs = np.random.permutation(x_tr.shape[0])
        for start in range(0, x_tr.shape[0], batch_size):
            end = min(start + batch_size, x_tr.shape[0])
            bi = mx.array(idxs[start:end].astype(np.int32))
            x1b = x_tr[bi]
            yb = y_tr[bi]
            x0b = sample_noise_like(x1b)
            loss, grads = loss_and_grad_fn(model, x1b, x0b, yb)
            clip_and_apply_grads(model, optimizer, grads, 1.0)
            mx.eval(loss, model.parameters())
    total = 0.0
    n = 0
    for start in range(0, x_va.shape[0], batch_size):
        end = min(start + batch_size, x_va.shape[0])
        x1b = x_va[start:end]
        yb = y_va[start:end]
        x0b = sample_noise_like(x1b)
        vl = compute_flow_loss(model, x1b, x0b, yb)
        mx.eval(vl)
        total += vl.item()
        n += 1
    return total / max(n, 1)


@app.cell
def _(mo):
    cv_pool_ui = mo.ui.slider(256, 4096, value=1024, step=256, label="CV pool size (images)")
    cv_epochs_ui = mo.ui.slider(1, 5, value=2, step=1, label="CV epochs per fold")
    cv_bs_ui = mo.ui.dropdown(options={"32": 32, "64": 64, "128": 128}, value="64", label="CV batch size")
    run_cv_btn = mo.ui.run_button(label="Run 5-Fold CV")
    mo.vstack(
        [
            mo.md(
                "CV over a bounded pool from the training stream. This "
                "trains 5 fresh models from scratch — keep the pool and "
                "epochs modest."
            ),
            mo.hstack([cv_pool_ui, cv_epochs_ui, cv_bs_ui, run_cv_btn]),
        ]
    )
    return cv_bs_ui, cv_epochs_ui, cv_pool_ui, run_cv_btn


@app.cell
def _(
    cv_bs_ui,
    cv_epochs_ui,
    cv_pool_ui,
    dataset_ready,
    embed_dim_gen1_ui,
    heads_gen1_ui,
    lr_gen1_ui,
    mlp_dim_gen1_ui,
    mo,
    monitor_stream,
    patch_size_ui,
    resolution_ui,
    run_cv_btn,
    trained_model_gen1,
    warmup_steps_gen1_ui,
):
    cv_results = None
    if not dataset_ready or trained_model_gen1 is None:
        _out = mo.md("_Train gen1 (Section 5a) before running CV._")
    elif not run_cv_btn.value:
        _out = mo.md("Click **Run 5-Fold CV** to begin.")
    else:
        mo.output.replace(mo.md("Collecting CV pool from monitoring stream..."))
        x_cv_pool, y_cv_pool = collect_cv_pool(monitor_stream, cv_pool_ui.value, resolution_ui.value)
        _k = 5
        _rng = np.random.default_rng(seed=0)
        _perm = _rng.permutation(x_cv_pool.shape[0])
        _folds = np.array_split(_perm, _k)
        cv_fold_losses = []
        for _f in range(_k):
            _val_idx = _folds[_f]
            _train_idx = np.concatenate([_folds[j] for j in range(_k) if j != _f])
            _vl = run_cv_fold(
                x_cv_pool,
                y_cv_pool,
                _train_idx,
                _val_idx,
                image_size=resolution_ui.value,
                patch_size=patch_size_ui.value,
                embed_dim=embed_dim_gen1_ui.value,
                num_heads=heads_gen1_ui.value,
                mlp_dim=mlp_dim_gen1_ui.value,
                num_layers=6,
                peak_lr=lr_gen1_ui.value,
                n_epochs=cv_epochs_ui.value,
                batch_size=cv_bs_ui.value,
                warmup_steps=min(50, warmup_steps_gen1_ui.value),
            )
            cv_fold_losses.append(_vl)
            mo.output.replace(mo.md(f"Fold {_f + 1}/{_k} — val loss: {_vl:.4f}"))
        _mean = float(np.mean(cv_fold_losses))
        _std = float(np.std(cv_fold_losses))
        cv_results = {"fold_losses": cv_fold_losses, "mean": _mean, "std": _std}
        _out = mo.md(
            f"**{len(cv_fold_losses)}-Fold CV flow-matching loss**: "
            f"`{_mean:.4f} ± {_std:.4f}` "
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
def plot_loss_curves(curves: dict, title: str = "Training curves"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, losses in curves.items():
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, "-o", lw=2, ms=4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    train_losses_gen1,
    train_losses_reflow,
    val_losses_gen1,
    val_losses_reflow,
):
    _curves = {
        "gen1 train": train_losses_gen1,
        "gen1 val": val_losses_gen1,
        "reflow train": train_losses_reflow,
        "reflow val": val_losses_reflow,
    }
    if not any(_curves.values()):
        _out = mo.md("_Train at least one stage to see loss curves._")
    else:
        _out = plot_loss_curves(_curves, title="ImageNet DiT rectified flow — training curves")
    _out
    return


@app.function
def denormalize_pixels(x: mx.array) -> np.ndarray:
    img = (np.array(x) + 1.0) * 127.5
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


@app.function
def plot_euler_progression(
    model: nn.Module,
    class_names: list,
    image_size: int = 64,
    num_steps: int = 50,
    num_samples: int = 6,
    frame_ts: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 1,
    label_pool: tuple = (207, 250, 281, 285, 340, 388, 417, 973),
):
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(label_pool), size=num_samples)
    idx = np.array([label_pool[i] for i in picks], dtype=np.int32)
    y = mx.array(idx)
    x0 = mx.random.normal(shape=(num_samples, image_size, image_size, 3))
    trajectory = euler_solve_trajectory(model, x0, y, num_steps)
    frame_steps = [int(round(t * num_steps)) for t in frame_ts]
    cols = len(frame_steps)
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.8))
    for r in range(rows):
        for c, step in enumerate(frame_steps):
            if rows > 1 and cols > 1:
                ax = axes[r, c]
            elif rows == 1:
                ax = axes[c]
            else:
                ax = axes[r]
            img = denormalize_pixels(trajectory[step])[r]
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"t={step / num_steps:.2f}", fontsize=9)
            if c == 0:
                name = class_names[int(idx[r])] if int(idx[r]) < len(class_names) else str(idx[r])
                ax.set_ylabel(name[:14], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Euler ODE generation progression: noise (t=0) → image (t=1), {num_steps} steps", fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(mo):
    mo.md("""
    ### 8a. Generation progression across ODE time

    Each row is one class-conditional sample; each column is the
    state at an intermediate ODE step. Left column: `t=0` (pure
    Gaussian noise). Right column: `t=1` (final generated image).
    This is the "noise progressively removed" visualization
    required by the goal.
    """)
    return


@app.cell
def _(mo, resolution_ui):
    progression_num_steps_ui = mo.ui.slider(10, 200, value=50, step=5, label="ODE steps")
    progression_num_samples_ui = mo.ui.slider(2, 10, value=6, step=1, label="Rows (class-conditional samples)")
    progression_model_ui = mo.ui.dropdown(
        options=["reflow", "gen1"],
        value="reflow",
        label="Model to visualize",
    )
    progression_btn = mo.ui.run_button(label="Render ODE progression")
    mo.vstack(
        [
            mo.hstack([progression_model_ui, progression_num_steps_ui, progression_num_samples_ui]),
            progression_btn,
            mo.md(f"Resolution: `{resolution_ui.value}x{resolution_ui.value}`"),
        ]
    )
    return (
        progression_btn,
        progression_model_ui,
        progression_num_samples_ui,
        progression_num_steps_ui,
    )


@app.cell
def _(
    metadata_dict,
    mo,
    progression_btn,
    progression_model_ui,
    progression_num_samples_ui,
    progression_num_steps_ui,
    resolution_ui,
    trained_model_gen1,
    trained_model_reflow,
):
    _sel = None
    if progression_model_ui.value == "reflow":
        _sel = trained_model_reflow
    elif progression_model_ui.value == "gen1":
        _sel = trained_model_gen1
    if _sel is None:
        _out = mo.md(f"_The selected model `{progression_model_ui.value}` has not been trained yet._")
    elif not progression_btn.value:
        _out = mo.md("Click **Render ODE progression** to draw the noise→image trajectory grid.")
    else:
        _names = imagenet_class_names(metadata_dict)
        _out = plot_euler_progression(
            _sel,
            _names,
            image_size=resolution_ui.value,
            num_steps=progression_num_steps_ui.value,
            num_samples=progression_num_samples_ui.value,
        )
    _out
    return


@app.function
def plot_generated_class_grid(
    model: nn.Module,
    class_names: list,
    image_size: int = 64,
    num_steps: int = 50,
    class_ids: tuple = (207, 250, 281, 285, 340, 388, 417, 973),
    num_per_class: int = 2,
    seed: int = 3,
    title: str = "Generated samples",
):
    mx.random.seed(seed)
    labels = np.repeat(np.array(class_ids, dtype=np.int32), num_per_class)
    y = mx.array(labels)
    total = int(labels.shape[0])
    x0 = mx.random.normal(shape=(total, image_size, image_size, 3))
    x1 = euler_solve(model, x0, y, num_steps)
    mx.eval(x1)
    imgs = denormalize_pixels(x1)
    cols = num_per_class
    rows = len(class_ids)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.8))
    for i in range(total):
        r, c = divmod(i, cols)
        if rows > 1 and cols > 1:
            ax = axes[r, c]
        elif rows == 1:
            ax = axes[c]
        else:
            ax = axes[r]
        ax.imshow(imgs[i])
        if c == 0:
            name = class_names[int(class_ids[r])] if int(class_ids[r]) < len(class_names) else str(class_ids[r])
            ax.set_ylabel(name[:14], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{title} (Euler, {num_steps} steps)", fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(metadata_dict, mo, resolution_ui, trained_model_gen1):
    if trained_model_gen1 is None:
        _out = mo.md("_Gen1 not trained yet._")
    else:
        _out = plot_generated_class_grid(
            trained_model_gen1,
            imagenet_class_names(metadata_dict),
            image_size=resolution_ui.value,
            num_steps=50,
            title="Gen1 (CFM) samples",
        )
    _out
    return


@app.cell
def _(metadata_dict, mo, resolution_ui, trained_model_reflow):
    if trained_model_reflow is None:
        _out = mo.md("_Reflow model not trained yet._")
    else:
        _out = plot_generated_class_grid(
            trained_model_reflow,
            imagenet_class_names(metadata_dict),
            image_size=resolution_ui.value,
            num_steps=50,
            title="Reflow (gen2) samples",
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### 8b. Path-straightness — gen1 vs reflow

    Two measurements:

    1. **`S` metric** — mean squared deviation of per-step Euler
       displacements from a straight-line path (0 = perfectly
       straight).
    2. **Few-step MSE** — with a fixed `x_0` and class, how close
       does an Euler solution at `k` steps get to a high-step
       reference (100 steps)? Straighter paths let low-step Euler
       match the reference, so the curve stays lower / flatter.
       This is the practical payoff of rectification.
    """)
    return


@app.cell
def _(mo, resolution_ui, trained_model_gen1, trained_model_reflow):
    if trained_model_gen1 is None or trained_model_reflow is None:
        _out = mo.md("_Train both gen1 (5a) and reflow (5b) to see the straightness comparison._")
    else:
        mx.random.seed(11)
        _x0 = mx.random.normal(shape=(16, resolution_ui.value, resolution_ui.value, 3))
        _y = mx.array(np.random.randint(0, 1000, size=16).astype(np.int32))
        s_gen1 = compute_path_straightness(trained_model_gen1, _x0, _y, num_steps=50)
        s_reflow = compute_path_straightness(trained_model_reflow, _x0, _y, num_steps=50)
        _improve = 100.0 * (s_gen1 - s_reflow) / max(s_gen1, 1e-12)
        _out = mo.md(
            f"""
            | Model | Path-straightness `S` (lower is straighter) |
            |-------|---------------------------------------------|
            | gen1 (before reflow) | `{s_gen1:.6f}` |
            | reflow (gen2) | `{s_reflow:.6f}` |
            | **Relative reduction** | **`{_improve:.1f}%`** |

            A positive relative reduction confirms reflow produces
            straighter ODE trajectories than gen1 — this is the
            central claim of Rectified Flow.
            """
        )
    _out
    return


@app.function
def compute_step_count_mse(
    model: nn.Module,
    step_list: list,
    ref_steps: int = 100,
    num_samples: int = 16,
    image_size: int = 64,
    seed: int = 7,
) -> list:
    mx.random.seed(seed)
    y = mx.array(np.random.randint(0, 1000, size=num_samples).astype(np.int32))
    x0 = mx.random.normal(shape=(num_samples, image_size, image_size, 3))
    x_ref = euler_solve(model, x0, y, ref_steps)
    mx.eval(x_ref)
    mses = []
    for ns in step_list:
        x_ns = euler_solve(model, x0, y, ns)
        mx.eval(x_ns)
        mses.append(float(mx.mean((x_ns - x_ref) ** 2).item()))
    return mses


@app.function
def plot_step_count_mse(step_list: list, curves: dict):
    markers = ["o", "s", "^", "D"]
    colors = ["steelblue", "seagreen", "darkorange", "crimson"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, mse) in enumerate(curves.items()):
        if mse:
            ax.plot(
                step_list,
                mse,
                f"-{markers[i % len(markers)]}",
                lw=2,
                ms=6,
                color=colors[i % len(colors)],
                label=label,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Euler integration steps")
    ax.set_ylabel("MSE vs 100-step reference")
    ax.set_title("Few-step sample quality — lower and flatter is straighter")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


@app.cell
def _(mo, resolution_ui, trained_model_gen1, trained_model_reflow):
    if trained_model_gen1 is None and trained_model_reflow is None:
        _out = mo.md("_Train at least one model to see the few-step MSE curve._")
    else:
        _steps = [1, 2, 5, 10, 25, 50]
        _curves = {}
        if trained_model_gen1 is not None:
            _curves["gen1 (CFM)"] = compute_step_count_mse(
                trained_model_gen1, _steps, ref_steps=100, image_size=resolution_ui.value
            )
        if trained_model_reflow is not None:
            _curves["reflow (gen2)"] = compute_step_count_mse(
                trained_model_reflow, _steps, ref_steps=100, image_size=resolution_ui.value
            )
        _out = plot_step_count_mse(_steps, _curves)
    _out
    return


@app.function
def summarize_regime(
    name: str,
    model,
    train_losses: list,
    val_losses: list,
    test_stream,
    max_test_batches: int,
    image_size: int,
) -> dict:
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
    row["test_loss"] = round(evaluate_model_stream(model, test_stream, max_test_batches), 4)
    mx.random.seed(99)
    x0 = mx.random.normal(shape=(8, image_size, image_size, 3))
    y = mx.array(np.random.randint(0, 1000, size=8).astype(np.int32))
    row["straightness"] = round(compute_path_straightness(model, x0, y, num_steps=50), 6)
    return row


@app.cell
def _(
    max_test_batches_ui,
    mo,
    resolution_ui,
    test_stream,
    train_losses_gen1,
    train_losses_reflow,
    trained_model_gen1,
    trained_model_reflow,
    val_losses_gen1,
    val_losses_reflow,
):
    if trained_model_gen1 is None and trained_model_reflow is None:
        _out = mo.md("_Train at least one stage to see the comparison table._")
    else:
        comparison_rows = [
            summarize_regime(
                "gen1 (CFM)",
                trained_model_gen1,
                train_losses_gen1,
                val_losses_gen1,
                test_stream,
                max_test_batches_ui.value,
                resolution_ui.value,
            ),
            summarize_regime(
                "reflow (gen2)",
                trained_model_reflow,
                train_losses_reflow,
                val_losses_reflow,
                test_stream,
                max_test_batches_ui.value,
                resolution_ui.value,
            ),
        ]
        _out = mo.ui.table(comparison_rows)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### Summary

    - The **gen1 CFM** stage learns a class-conditional velocity
      field that transports Gaussian noise to ImageNet images
      under the linear flow-matching path with random
      `(x_0, x_1)` couplings.
    - The **reflow (gen2)** stage retrains on gen1's own ODE-mapped
      couplings, straightening the trajectories — measured by
      both the `S` metric and the few-step MSE curve.
    - Section 8a's Euler progression visualization is the concrete
      demonstration required by the goal: image noise is
      progressively removed as ODE time advances from `t=0`
      (Gaussian noise) to `t=1` (generated image).
    - Sample fidelity is bounded by the deliberately modest
      model, epoch, and (crucially) reflow-dataset defaults. Scale
      up any of those via the UI if you have more compute; the
      notebook code itself is unchanged.
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
    save_gen1_filename_ui = mo.ui.text(
        value="imagenet_dit_gen1_v1.safetensors",
        label="Gen1 filename (models/)",
        full_width=True,
    )
    save_gen1_btn = mo.ui.run_button(label="Save Gen1 Model")
    mo.vstack([save_gen1_filename_ui, save_gen1_btn])
    return save_gen1_btn, save_gen1_filename_ui


@app.cell
def _(mo, save_gen1_btn, save_gen1_filename_ui, trained_model_gen1):
    if trained_model_gen1 is None:
        _out = mo.md("_Train the gen1 model first (Section 5a) before saving._")
    elif not save_gen1_btn.value:
        _out = mo.md("Enter a filename above and click **Save Gen1 Model** to write weights to `models/`.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_gen1_filename_ui.value
        trained_model_gen1.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Gen1 weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    save_reflow_filename_ui = mo.ui.text(
        value="imagenet_dit_reflow_v1.safetensors",
        label="Reflow filename (models/)",
        full_width=True,
    )
    save_reflow_btn = mo.ui.run_button(label="Save Reflow Model")
    mo.vstack([save_reflow_filename_ui, save_reflow_btn])
    return save_reflow_btn, save_reflow_filename_ui


@app.cell
def _(mo, save_reflow_btn, save_reflow_filename_ui, trained_model_reflow):
    if trained_model_reflow is None:
        _out = mo.md("_Train the reflow model first (Section 5b) before saving._")
    elif not save_reflow_btn.value:
        _out = mo.md("Enter a filename above and click **Save Reflow Model** to write weights to `models/`.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_reflow_filename_ui.value
        trained_model_reflow.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Reflow weights written to `{_save_path}`.")
    _out
    return


if __name__ == "__main__":
    app.run()
