import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
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
    # Vision Transformer (ViT) on CIFAR-10 — MLX

    ## Research Goal

    Train a **Vision Transformer (ViT)** on the **CIFAR-10** image classification
    dataset using Apple's **MLX** framework. The model splits each 32x32 RGB image
    into 4x4 patches, projects them into an embedding space, prepends a learnable
    classification token, adds positional embeddings, and processes the sequence
    with a stack of transformer encoder blocks. The final classification token is
    fed to a linear head to predict one of the 10 classes.

    ### Notebook outline

    1. **Title & Research Goal** — this cell
    2. **Data Exploration** — load CIFAR-10 via `mlx.data.datasets.load_cifar10`
       (auto-downloads to `../data/cifar10` on first use), visualize samples and
       class distribution
    3. **Dataset Creation** — `mlx.data` Buffer/Stream pipeline: train / val / test
       splits, per-channel standardization, reshuffled batch streams
    4. **Model Definition** — `PatchEmbeddingV1`, `MultiHeadSelfAttentionV1`,
       `TransformerEncoderBlockV1`, `VisionTransformerV1`
    5. **Training** — interactive hyperparameter controls with live progress
    6. **Hyperparameter Search** — optional grid search over LR x embed_dim
    7. **Validation & Cross-Validation** — test accuracy and k=5 fold CV
    8. **Results** — loss curves, confusion matrix, sample predictions
    9. **Save Trained Model** — write weights to `models/`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration

    CIFAR-10 is loaded as an `mlx.data` `Buffer` via
    `mlx.data.datasets.load_cifar10`. The loader downloads and preprocesses the
    dataset into `../data/cifar10/` on first use and reads the cached pickle
    files thereafter.
    """)
    return


@app.function
def cifar10_class_names() -> list[str]:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


@app.function
def collect_labels(buffer) -> np.ndarray:
    return np.array(
        [int(np.asarray(buffer[i]["label"])) for i in range(len(buffer))],
        dtype=np.int64,
    )


@app.cell
def _():
    data_dir = Path("../data/cifar10")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_buffer = load_cifar10(root=str(data_dir), train=True, quiet=True)
    test_buffer = load_cifar10(root=str(data_dir), train=False, quiet=True)
    class_names = cifar10_class_names()
    return class_names, test_buffer, train_buffer


@app.cell
def _(class_names, mo, test_buffer, train_buffer):
    _image_shape = tuple(np.asarray(train_buffer[0]["image"]).shape)
    mo.md(f"""
    ### CIFAR-10 dataset overview

    CIFAR-10 consists of 60,000 color images in 10 classes, with 6,000 images per class.
    There are 50,000 training images and 10,000 test images.

    The data is delivered as an `mlx.data` Buffer cached under `../data/cifar10/`.
    Each sample is a dict with `image` (`uint8`, H x W x C) and `label`
    (scalar int in `[0, 9]`).

    - Image shape: `{_image_shape}`
    - Number of classes: **{len(class_names)}**
    - Class names: {", ".join(f"`{c}`" for c in class_names)}

    | Split | Size |
    |-------|------|
    | Train | {len(train_buffer):,} |
    | Test | {len(test_buffer):,} |
    """)
    return


@app.function
def plot_sample_grid(buffer, class_names: list[str], n_show: int = 40, rows: int = 5, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i in range(n_show):
        sample = buffer[i]
        img = np.asarray(sample["image"])
        label = int(np.asarray(sample["label"]))
        r, c = divmod(i, cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(class_names[label], fontsize=9)
        axes[r, c].axis("off")
    fig.suptitle("CIFAR-10 training samples", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, train_buffer):
    plot_sample_grid(train_buffer, class_names)
    return


@app.function
def plot_class_distribution(labels: np.ndarray, class_names: list[str]):
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(class_names)), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 training-set class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(int(c)), ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, train_buffer):
    plot_class_distribution(collect_labels(train_buffer), class_names)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def normalize_image(images: np.ndarray) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    scaled = images.astype(np.float32) / 255.0
    return (scaled - mean) / std


@app.function
def split_train_val(train_buffer, val_fraction: float = 0.15, seed: int = 0):
    n_total = len(train_buffer)
    n_val = int(round(n_total * val_fraction))
    perm = np.random.default_rng(seed).permutation(n_total)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_buffer.perm(train_idx), train_buffer.perm(val_idx)


@app.function
def make_stream(buffer, batch_size: int, shuffle: bool = False, seed: int = 0):
    buf = buffer
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(buffer)).tolist()
        buf = buffer.perm(order)
    return (
        buf
        .batch(batch_size)
        .key_transform("image", normalize_image)
        .ordered_prefetch(4, 2)
    )


@app.cell
def _(test_buffer, train_buffer):
    train_split, val_split = split_train_val(train_buffer, val_fraction=0.15, seed=42)
    datasets = {
        "train": train_split,
        "val": val_split,
        "test": test_buffer,
        "sizes": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_buffer),
        },
    }
    return (datasets,)


@app.cell
def _(datasets, mo):
    _peek = make_stream(datasets["train"], 128, shuffle=False)
    _peek.reset()
    _batch = next(_peek)
    _x, _y = _batch["image"], _batch["label"]
    mo.md(f"""
    ### Split sizes

    | Split | Size |
    |-------|------|
    | Train | {datasets["sizes"]["train"]:,} |
    | Val   | {datasets["sizes"]["val"]:,}   |
    | Test  | {datasets["sizes"]["test"]:,}  |

    ### Example batch (from an `mlx.data` Stream)

    - `image` shape: `{tuple(_x.shape)}` — dtype `{_x.dtype}` — mean `{_x.mean():.3f}`
    - `label` shape: `{tuple(_y.shape)}` — dtype `{_y.dtype}`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition
    """)
    return


@app.class_definition
class PatchEmbeddingV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)
        b, hp, wp, c = h.shape
        return h.reshape(b, hp * wp, c)


@app.class_definition
class MultiHeadSelfAttentionV1(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.proj_dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn_scores, axis=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, self.embed_dim)
        return self.proj_dropout(self.proj(out))


@app.class_definition
class TransformerEncoderBlockV1(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttentionV1(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.mlp_dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        h = nn.gelu(self.fc1(h))
        h = self.mlp_dropout(h)
        h = self.fc2(h)
        h = self.mlp_dropout(h)
        return x + h


@app.class_definition
class VisionTransformerV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbeddingV1(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.blocks = [
            TransformerEncoderBlockV1(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        b = x.shape[0]
        h = self.patch_embed(x)
        cls_tokens = mx.broadcast_to(self.cls_token, (b, 1, h.shape[-1]))
        h = mx.concatenate([cls_tokens, h], axis=1)
        h = h + self.pos_embed
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h[:, 0])


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def compute_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()


@app.cell
def _():
    model_config = {
        "image_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "mlp_dim": 512,
        "num_classes": 10,
        "dropout": 0.1,
    }
    demo_model = VisionTransformerV1(**model_config)
    mx.eval(demo_model.parameters())
    demo_param_count = count_parameters(demo_model)
    return demo_param_count, model_config


@app.cell
def _(demo_param_count, mo, model_config):
    mo.md(f"""
    ### Model Architecture — `VisionTransformerV1`

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Patch embedding | `PatchEmbeddingV1` (Conv2d {model_config["patch_size"]}x{model_config["patch_size"]}) | `(B, {(model_config["image_size"] // model_config["patch_size"]) ** 2}, {model_config["embed_dim"]})` |
    | CLS token + positional embedding | learnable parameters | `(B, {(model_config["image_size"] // model_config["patch_size"]) ** 2 + 1}, {model_config["embed_dim"]})` |
    | Transformer encoder | `TransformerEncoderBlockV1` x {model_config["num_layers"]} | `(B, {(model_config["image_size"] // model_config["patch_size"]) ** 2 + 1}, {model_config["embed_dim"]})` |
    | LayerNorm + Head | `LayerNorm` + `Linear` | `(B, {model_config["num_classes"]})` |

    **Configuration**

    - `image_size` = {model_config["image_size"]}
    - `patch_size` = {model_config["patch_size"]}
    - `embed_dim` = {model_config["embed_dim"]}
    - `num_heads` = {model_config["num_heads"]}
    - `num_layers` = {model_config["num_layers"]}
    - `mlp_dim` = {model_config["mlp_dim"]}
    - `dropout` = {model_config["dropout"]}

    **Total parameters**: `{demo_param_count:,}`
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
        options=[64, 128, 256],
        value=128,
        label="Batch Size",
    )
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    epochs_ui = mo.ui.slider(1, 50, value=20, step=1, label="Epochs")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack([
        mo.md("### Hyperparameters"),
        mo.hstack([lr_ui, bs_ui]),
        mo.hstack([wd_ui, epochs_ui]),
        train_btn,
    ])
    return bs_ui, epochs_ui, lr_ui, train_btn, wd_ui


@app.function
def run_train_epoch(model: nn.Module, loss_and_grad_fn, optimizer, stream) -> float:
    stream.reset()
    epoch_loss = 0.0
    n_batches = 0
    for batch in stream:
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


@app.function
def run_evaluate(model: nn.Module, stream) -> float:
    stream.reset()
    total = 0.0
    n = 0
    for batch in stream:
        loss = compute_loss(model, mx.array(batch["image"]), mx.array(batch["label"]))
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.cell
def _(bs_ui, datasets, epochs_ui, lr_ui, mo, model_config, train_btn, wd_ui):
    train_losses = []
    val_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training."))
    else:
        _model = VisionTransformerV1(**model_config)
        mx.eval(_model.parameters())
        _optimizer = optim.AdamW(learning_rate=lr_ui.value, weight_decay=wd_ui.value)
        _loss_and_grad_fn = nn.value_and_grad(_model, compute_loss)
        _n_epochs = epochs_ui.value
        _bs = bs_ui.value

        for _epoch in range(_n_epochs):
            _train_stream = make_stream(datasets["train"], _bs, shuffle=True, seed=_epoch)
            _val_stream = make_stream(datasets["val"], _bs, shuffle=False)
            _tl = run_train_epoch(_model, _loss_and_grad_fn, _optimizer, _train_stream)
            _vl = run_evaluate(_model, _val_stream)
            train_losses.append(_tl)
            val_losses.append(_vl)
            mo.output.replace(
                mo.md(f"**Epoch {_epoch + 1}/{_n_epochs}** — train loss: {_tl:.4f} | val loss: {_vl:.4f}")
            )

        trained_model = _model
        mo.output.replace(
            mo.md(
                f"**Training complete!** "
                f"Final train loss: {train_losses[-1]:.4f} | "
                f"Final val loss: {val_losses[-1]:.4f}"
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
def train_hp_config(
    datasets: dict,
    model_config: dict,
    lr: float,
    embed_dim: int,
    n_epochs: int = 5,
    weight_decay: float = 1e-4,
    seed: int = 0,
    batch_size: int = 128,
) -> dict:
    config = dict(model_config)
    config["embed_dim"] = embed_dim
    model = VisionTransformerV1(**config)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    final_tl = 0.0
    for epoch in range(n_epochs):
        stream = make_stream(datasets["train"], batch_size, shuffle=True, seed=seed + epoch)
        final_tl = run_train_epoch(model, loss_and_grad_fn, optimizer, stream)
    vl = run_evaluate(model, make_stream(datasets["val"], batch_size, shuffle=False))
    return {"lr": lr, "embed_dim": embed_dim, "train_loss": round(final_tl, 4), "val_loss": round(vl, 4)}


@app.cell
def _(datasets, hp_search_cb, mo, model_config):
    mo.stop(not hp_search_cb.value, mo.md("_Enable hyperparameter search above to run this section._"))

    _search_space = {"lr": [1e-4, 3e-4, 1e-3], "embed_dim": [128, 256]}
    hp_results = []
    _idx = 0
    _total = len(_search_space["lr"]) * len(_search_space["embed_dim"])
    for _lr in _search_space["lr"]:
        for _ed in _search_space["embed_dim"]:
            _idx += 1
            mo.output.replace(mo.md(f"Running config {_idx}/{_total}: lr={_lr}, embed_dim={_ed}"))
            _result = train_hp_config(datasets, model_config, lr=_lr, embed_dim=_ed, n_epochs=5)
            hp_results.append(_result)

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
def evaluate_model(model: nn.Module, stream) -> tuple[float, float]:
    stream.reset()
    total_loss = 0.0
    correct = 0
    total = 0
    n = 0
    for batch in stream:
        x = mx.array(batch["image"])
        y = mx.array(batch["label"])
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y).mean()
        preds = mx.argmax(logits, axis=-1)
        mx.eval(logits, loss, preds)
        total_loss += loss.item()
        correct += int(mx.sum(preds == y).item())
        total += y.shape[0]
        n += 1
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(n, 1)
    return accuracy, avg_loss


@app.cell
def _(datasets, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to compute test metrics._")
    else:
        _acc, _loss = evaluate_model(trained_model, make_stream(datasets["test"], 128, shuffle=False))
        _out = mo.md(f"""
        ### Test Set Evaluation

        | Metric | Value |
        |--------|-------|
        | Test accuracy | **{_acc:.4f}** |
        | Test loss     | **{_loss:.4f}** |
        | Test samples  | {datasets["sizes"]["test"]:,} |
        """)
    _out
    return


@app.function
def run_fold(
    cv_buffer,
    train_idx,
    val_idx,
    model_config: dict,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
) -> tuple[float, float]:
    train_fold = cv_buffer.perm(list(train_idx))
    val_fold = cv_buffer.perm(list(val_idx))
    model = VisionTransformerV1(**model_config)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    for _epoch in range(n_epochs):
        run_train_epoch(
            model, loss_and_grad_fn, optimizer,
            make_stream(train_fold, batch_size, shuffle=True, seed=_epoch),
        )
    acc, loss = evaluate_model(model, make_stream(val_fold, batch_size, shuffle=False))
    return acc, loss


@app.cell
def _(mo):
    cv_enable_cb = mo.ui.checkbox(label="Enable 5-Fold Cross-Validation (slow)", value=False)
    cv_enable_cb
    return (cv_enable_cb,)


@app.cell
def _(cv_enable_cb, mo, model_config, train_buffer):
    mo.stop(not cv_enable_cb.value, mo.md("_Enable 5-fold cross-validation above to run this section._"))

    _k = 5
    _n_epochs = 3
    _batch_size = 128
    _cv_subset_size = 5000
    _sub_idx = np.random.default_rng(0).permutation(len(train_buffer))[:_cv_subset_size].tolist()
    _cv_buffer = train_buffer.perm(_sub_idx)

    _fold_size = _cv_subset_size // _k
    fold_metrics = []
    for _fold in range(_k):
        _val_start = _fold * _fold_size
        _val_end = _val_start + _fold_size
        _val_indices = list(range(_val_start, _val_end))
        _train_indices = list(range(0, _val_start)) + list(range(_val_end, _cv_subset_size))
        mo.output.replace(mo.md(f"Running fold {_fold + 1}/{_k}..."))
        _acc, _loss = run_fold(
            _cv_buffer, _train_indices, _val_indices, model_config,
            lr=3e-4, weight_decay=1e-4, n_epochs=_n_epochs, batch_size=_batch_size,
        )
        fold_metrics.append({"fold": _fold + 1, "accuracy": round(_acc, 4), "loss": round(_loss, 4)})

    _accs = np.array([m["accuracy"] for m in fold_metrics])
    _losses = np.array([m["loss"] for m in fold_metrics])
    cv_results = {
        "fold_metrics": fold_metrics,
        "acc_mean": float(_accs.mean()),
        "acc_std": float(_accs.std()),
        "loss_mean": float(_losses.mean()),
        "loss_std": float(_losses.std()),
    }
    mo.output.replace(mo.vstack([
        mo.md(f"""
        ### {_k}-Fold Cross-Validation Results

        Trained on a {_cv_subset_size}-sample subset of the training data for {_n_epochs} epochs per fold.

        **Mean accuracy**: {cv_results["acc_mean"]:.4f} +/- {cv_results["acc_std"]:.4f}
        **Mean loss**: {cv_results["loss_mean"]:.4f} +/- {cv_results["loss_std"]:.4f}
        """),
        mo.ui.table(fold_metrics),
    ]))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Results
    """)
    return


@app.function
def plot_loss_curve(train_losses: list, val_losses: list | None = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train loss")
    if val_losses:
        ax.plot(epochs, val_losses, "r-s", lw=2, ms=4, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses, trained_model, val_losses):
    if trained_model is None or not train_losses:
        _out = mo.md("_Train the model first (Section 5) to see the loss curve._")
    else:
        _out = plot_loss_curve(train_losses, val_losses)
    _out
    return


@app.function
def plot_confusion_matrix(model: nn.Module, stream, class_names: list[str]):
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    stream.reset()
    for batch in stream:
        preds = mx.argmax(model(mx.array(batch["image"])), axis=-1)
        mx.eval(preds)
        for t, p in zip(np.asarray(batch["label"]), np.asarray(preds)):
            cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, datasets, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to see the confusion matrix._")
    else:
        _out = plot_confusion_matrix(
            trained_model, make_stream(datasets["test"], 128, shuffle=False), class_names
        )
    _out
    return


@app.function
def plot_sample_predictions(
    model: nn.Module,
    test_buffer,
    class_names: list[str],
    n: int = 16,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_buffer), size=n, replace=False)
    x_sel_raw = np.stack([np.asarray(test_buffer[int(i)]["image"]) for i in idx])
    y_sel = np.array([int(np.asarray(test_buffer[int(i)]["label"])) for i in idx])
    x_norm = normalize_image(x_sel_raw)
    logits = model(mx.array(x_norm))
    preds = mx.argmax(logits, axis=-1)
    mx.eval(preds)
    preds_np = np.array(preds)
    rows = 4
    cols = n // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    for i in range(n):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(x_sel_raw[i])
        true_lbl = class_names[int(y_sel[i])]
        pred_lbl = class_names[int(preds_np[i])]
        color = "green" if int(y_sel[i]) == int(preds_np[i]) else "red"
        ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", fontsize=8, color=color)
        ax.axis("off")
    fig.suptitle("Sample predictions (green=correct, red=wrong)", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, mo, test_buffer, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to see sample predictions._")
    else:
        _out = plot_sample_predictions(trained_model, test_buffer, class_names, n=16)
    _out
    return


@app.cell
def _(datasets, mo, train_losses, trained_model, val_losses):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to see the final summary._")
    else:
        _test_acc, _test_loss = evaluate_model(
            trained_model, make_stream(datasets["test"], 128, shuffle=False)
        )
        _out = mo.md(f"""
        ### Summary

        The Vision Transformer (`VisionTransformerV1`) was trained on CIFAR-10 with
        {len(train_losses)} epochs.

        | Metric | Value |
        |--------|-------|
        | Final train loss | {train_losses[-1]:.4f} |
        | Final val loss   | {val_losses[-1]:.4f} |
        | Test accuracy    | **{_test_acc:.4f}** |
        | Test loss        | {_test_loss:.4f} |
        | Trainable params | {count_parameters(trained_model):,} |

        The classification head reads out the CLS token after `LayerNorm` and produces
        logits over the 10 CIFAR-10 classes. Training accuracy typically converges within
        20–40 epochs at moderate learning rates (3e-4) with AdamW.
        """)
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
        value="cifar10_vit_v1.safetensors",
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
