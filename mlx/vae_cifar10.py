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
    # Variational Autoencoders on CIFAR-10 — MLX

    ## Research Goal

    Train **Variational Autoencoders (VAEs)** on the CIFAR-10 dataset using
    Apple's **MLX** framework and compare four different architectures:

    1. **`ConvolutionalVariationalAutoEncoderV1`** — 2 stride-2 conv blocks
       (32 to 64 channels, spatial 32 to 16 to 8)
    2. **`ConvolutionalVariationalAutoEncoderV2`** — 3 stride-2 conv blocks
       (32 to 64 to 128 channels, spatial 32 to 16 to 8 to 4)
    3. **`ConvolutionalVariationalAutoEncoderV3`** — 4 stride-2 conv blocks
       (32 to 64 to 128 to 256 channels, spatial 32 to 16 to 8 to 4 to 2)
    4. **`VisionAttentionVariationalAutoEncoderV1`** — a ViT-style
       attention-only VAE with `PatchEmbeddingV1`, learnable `[CLS]` token,
       learnable 1-D position embeddings, and `TransformerEncoderBlockV1`
       (pre-norm self-attention + GELU MLP) stacks in both the encoder and
       the decoder. Contains **zero convolutions**.

    ### Sections
    - Data exploration
    - Dataset creation (train / test iterators, no validation split)
    - Model definition (four variants + a shared architecture table)
    - Training all four models under identical hyperparameters
    - Optional hyperparameter search (on the representative Conv V1 model)
    - Sampling and generation from each trained model (replaces validation
      and cross-validation for this notebook)
    - Results: overlaid training-loss curves + comparison table
    - Save trained model weights to `models/`
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
    train_ds = load_cifar10(root="../data/cifar10", train=True)
    test_ds = load_cifar10(root="../data/cifar10", train=False)
    return test_ds, train_ds


@app.cell
def _(mo, test_ds, train_ds):
    mo.md(f"""
    ### Dataset overview

    CIFAR-10 is loaded as an `mlx.data` Buffer. Each sample is a dict with:

    - `image` — `uint8` array shaped `(32, 32, 3)`
    - `label` — scalar `uint8` in `[0, 9]`

    | Split | Size |
    |-------|------|
    | Train | {len(train_ds):,} |
    | Test | {len(test_ds):,} |

    The 10 classes are:
    `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`,
    `ship`, `truck`.
    """)
    return


@app.function
def cifar10_class_names() -> list[str]:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


@app.function
def plot_cifar10_samples(train_ds, n_show: int = 40, rows: int = 5, cols: int = 8):
    class_names = cifar10_class_names()
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9))
    for i in range(n_show):
        sample = train_ds[i]
        img = np.array(sample["image"])
        label = int(np.array(sample["label"]).item())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(class_names[label], fontsize=9)
        axes[r, c].axis("off")
    fig.suptitle("CIFAR-10 training samples (class shown as title)", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_cifar10_samples(train_ds)
    return


@app.function
def plot_cifar10_class_distribution(train_ds):
    class_names = cifar10_class_names()
    labels = np.array(
        [int(np.array(train_ds[i]["label"]).item()) for i in range(len(train_ds))]
    )
    counts = np.bincount(labels, minlength=10)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(10), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 training-set class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 30, str(int(c)), ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_cifar10_class_distribution(train_ds)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation

    No validation split is used in this notebook — the research question is
    about comparing training convergence and generation quality across
    architectures rather than measuring generalization. We build only
    `train_iter` and `test_iter`.
    """)
    return


@app.function
def make_datasets(train_ds, test_ds, batch_size: int):
    def normalize(x):
        return x.astype("float32") / 255.0

    train_iter = (
        train_ds
        .shuffle()
        .to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    return train_iter, test_iter


@app.cell
def _(mo, test_ds, train_ds):
    _peek_train, _peek_test = make_datasets(train_ds, test_ds, 64)
    _peek_train.reset()
    _peek_batch = next(_peek_train)
    _peek_img = np.array(_peek_batch["image"])
    mo.md(f"""
    ### Batch preview

    - **Batch image shape**: `{tuple(_peek_img.shape)}`
    - **Dtype**: `{_peek_img.dtype}`
    - **Pixel range**: `[{_peek_img.min():.3f}, {_peek_img.max():.3f}]` (normalized to `[0, 1]`)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition
    """)
    return


@app.class_definition
class ConvolutionBlockV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.conv(x))


@app.class_definition
class ConvolutionTransposeBlockV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        final: bool = False,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.final = final

    def __call__(self, x: mx.array) -> mx.array:
        h = self.deconv(x)
        return mx.sigmoid(h) if self.final else nn.relu(h)


@app.class_definition
class ConvolutionalEncoderV1(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.block1 = ConvolutionBlockV1(in_channels, 32, stride=2)
        self.block2 = ConvolutionBlockV1(32, 64, stride=2)
        flat_dim = 8 * 8 * 64
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def __call__(self, x: mx.array):
        h = self.block1(x)
        h = self.block2(h)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_logvar(h)


@app.class_definition
class ConvolutionalDecoderV1(nn.Module):
    def __init__(self, latent_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.spatial = 8
        self.channels = 64
        self.fc = nn.Linear(latent_dim, self.spatial * self.spatial * self.channels)
        self.deconv1 = ConvolutionTransposeBlockV1(64, 32)
        self.deconv2 = ConvolutionTransposeBlockV1(32, out_channels, final=True)

    def __call__(self, z: mx.array) -> mx.array:
        h = nn.relu(self.fc(z))
        h = h.reshape(h.shape[0], self.spatial, self.spatial, self.channels)
        h = self.deconv1(h)
        return self.deconv2(h)


@app.class_definition
class ConvolutionalVariationalAutoEncoderV1(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.encoder = ConvolutionalEncoderV1(in_channels, latent_dim)
        self.decoder = ConvolutionalDecoderV1(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: mx.array, logvar: mx.array) -> mx.array:
        std = mx.exp(0.5 * logvar)
        return mu + mx.random.normal(std.shape) * std

    def __call__(self, x: mx.array):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)


@app.class_definition
class ConvolutionalEncoderV2(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.block1 = ConvolutionBlockV1(in_channels, 32, stride=2)
        self.block2 = ConvolutionBlockV1(32, 64, stride=2)
        self.block3 = ConvolutionBlockV1(64, 128, stride=2)
        flat_dim = 4 * 4 * 128
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def __call__(self, x: mx.array):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_logvar(h)


@app.class_definition
class ConvolutionalDecoderV2(nn.Module):
    def __init__(self, latent_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.spatial = 4
        self.channels = 128
        self.fc = nn.Linear(latent_dim, self.spatial * self.spatial * self.channels)
        self.deconv1 = ConvolutionTransposeBlockV1(128, 64)
        self.deconv2 = ConvolutionTransposeBlockV1(64, 32)
        self.deconv3 = ConvolutionTransposeBlockV1(32, out_channels, final=True)

    def __call__(self, z: mx.array) -> mx.array:
        h = nn.relu(self.fc(z))
        h = h.reshape(h.shape[0], self.spatial, self.spatial, self.channels)
        h = self.deconv1(h)
        h = self.deconv2(h)
        return self.deconv3(h)


@app.class_definition
class ConvolutionalVariationalAutoEncoderV2(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.encoder = ConvolutionalEncoderV2(in_channels, latent_dim)
        self.decoder = ConvolutionalDecoderV2(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: mx.array, logvar: mx.array) -> mx.array:
        std = mx.exp(0.5 * logvar)
        return mu + mx.random.normal(std.shape) * std

    def __call__(self, x: mx.array):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)


@app.class_definition
class ConvolutionalEncoderV3(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.block1 = ConvolutionBlockV1(in_channels, 32, stride=2)
        self.block2 = ConvolutionBlockV1(32, 64, stride=2)
        self.block3 = ConvolutionBlockV1(64, 128, stride=2)
        self.block4 = ConvolutionBlockV1(128, 256, stride=2)
        flat_dim = 2 * 2 * 256
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def __call__(self, x: mx.array):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_logvar(h)


@app.class_definition
class ConvolutionalDecoderV3(nn.Module):
    def __init__(self, latent_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.spatial = 2
        self.channels = 256
        self.fc = nn.Linear(latent_dim, self.spatial * self.spatial * self.channels)
        self.deconv1 = ConvolutionTransposeBlockV1(256, 128)
        self.deconv2 = ConvolutionTransposeBlockV1(128, 64)
        self.deconv3 = ConvolutionTransposeBlockV1(64, 32)
        self.deconv4 = ConvolutionTransposeBlockV1(32, out_channels, final=True)

    def __call__(self, z: mx.array) -> mx.array:
        h = nn.relu(self.fc(z))
        h = h.reshape(h.shape[0], self.spatial, self.spatial, self.channels)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        return self.deconv4(h)


@app.class_definition
class ConvolutionalVariationalAutoEncoderV3(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.encoder = ConvolutionalEncoderV3(in_channels, latent_dim)
        self.decoder = ConvolutionalDecoderV3(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: mx.array, logvar: mx.array) -> mx.array:
        std = mx.exp(0.5 * logvar)
        return mu + mx.random.normal(std.shape) * std

    def __call__(self, x: mx.array):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)


@app.class_definition
class PatchEmbeddingV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        p = self.patch_size
        gh = h // p
        gw = w // p
        x = x.reshape(b, gh, p, gw, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, gh * gw, p * p * c)
        return self.proj(x)


@app.class_definition
class MultiLayerPerceptronBlockV1(nn.Module):
    def __init__(self, dims: int = 128, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dims, dims * expansion)
        self.fc2 = nn.Linear(dims * expansion, dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


@app.class_definition
class TransformerEncoderBlockV1(nn.Module):
    def __init__(self, dims: int = 128, num_heads: int = 4, mlp_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.norm2 = nn.LayerNorm(dims)
        self.mlp = MultiLayerPerceptronBlockV1(dims, mlp_expansion)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)
        h = self.norm2(x)
        return x + self.mlp(h)


@app.class_definition
class VisionAttentionEncoderV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.patch_embed = PatchEmbeddingV1(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))
        self.blocks = [
            TransformerEncoderBlockV1(embed_dim, num_heads, mlp_expansion)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def __call__(self, x: mx.array):
        b = x.shape[0]
        tokens = self.patch_embed(x)
        cls = mx.broadcast_to(self.cls_token, (b, 1, self.cls_token.shape[-1]))
        h = mx.concatenate([cls, tokens], axis=1)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h)
        cls_out = self.norm(h[:, 0])
        return self.fc_mu(cls_out), self.fc_logvar(cls_out)


@app.class_definition
class VisionAttentionDecoderV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        out_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        latent_dim: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size * out_channels
        self.latent_proj = nn.Linear(latent_dim, self.num_patches * embed_dim)
        self.pos_embed = mx.zeros((1, self.num_patches, embed_dim))
        self.blocks = [
            TransformerEncoderBlockV1(embed_dim, num_heads, mlp_expansion)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.patch_dim)

    def __call__(self, z: mx.array) -> mx.array:
        b = z.shape[0]
        tokens = self.latent_proj(z).reshape(b, self.num_patches, self.embed_dim)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        patches = self.head(tokens)
        gh = self.grid_size
        gw = self.grid_size
        p = self.patch_size
        c = self.out_channels
        x = patches.reshape(b, gh, gw, p, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, gh * p, gw * p, c)
        return mx.sigmoid(x)


@app.class_definition
class VisionAttentionVariationalAutoEncoderV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.encoder = VisionAttentionEncoderV1(
            image_size, patch_size, in_channels, embed_dim,
            depth, num_heads, mlp_expansion, latent_dim,
        )
        self.decoder = VisionAttentionDecoderV1(
            image_size, patch_size, in_channels, embed_dim,
            depth, num_heads, mlp_expansion, latent_dim,
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu: mx.array, logvar: mx.array) -> mx.array:
        std = mx.exp(0.5 * logvar)
        return mu + mx.random.normal(std.shape) * std

    def __call__(self, x: mx.array):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def compute_vae_loss(model: nn.Module, x: mx.array) -> mx.array:
    recon, mu, logvar = model(x)
    x_flat = x.reshape(x.shape[0], -1)
    recon_flat = recon.reshape(recon.shape[0], -1)
    recon_loss = mx.mean(mx.sum((recon_flat - x_flat) ** 2, axis=-1))
    kl_loss = -0.5 * mx.mean(
        mx.sum(1 + logvar - mu ** 2 - mx.exp(logvar), axis=-1)
    )
    return recon_loss + kl_loss


@app.function
def run_train_epoch(model: nn.Module, loss_fn, optimizer, train_iter, preprocess_fn) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    epoch_loss = 0.0
    n_batches = 0
    train_iter.reset()
    for batch in train_iter:
        x = preprocess_fn(batch)
        loss, grads = loss_and_grad_fn(model, x)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


@app.function
def run_evaluate(model: nn.Module, loss_fn, data_iter, preprocess_fn) -> float:
    total = 0.0
    n = 0
    data_iter.reset()
    for batch in data_iter:
        x = preprocess_fn(batch)
        loss = loss_fn(model, x)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.function
def preprocess_image_batch(batch) -> mx.array:
    return mx.array(batch["image"], dtype=mx.float32)


@app.function
def train_model_epochs(
    model: nn.Module,
    train_iter,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    preprocess_fn,
    on_epoch_end=None,
) -> list:
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    losses = []
    for epoch in range(n_epochs):
        tl = run_train_epoch(model, compute_vae_loss, optimizer, train_iter, preprocess_fn)
        losses.append(tl)
        if on_epoch_end is not None:
            on_epoch_end(epoch, n_epochs, tl)
    return losses


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Architectures — comparison table

    All four VAEs share the same interface:
    `__call__(x) -> (recon, mu, logvar)` and `decode(z) -> recon`, and use
    the same `compute_vae_loss` (squared-error reconstruction + KL to
    `N(0, I)`). All decoders end in a `sigmoid` producing `(B, 32, 32, 3)`.

    | Model | Encoder path | Bottleneck flat dim | Decoder path |
    |-------|--------------|--------------------|--------------|
    | `ConvolutionalVariationalAutoEncoderV1` | 32→16→8, ch 3→32→64 | 8·8·64 = 4096 | 8→16→32, ch 64→32→3 |
    | `ConvolutionalVariationalAutoEncoderV2` | 32→16→8→4, ch 3→32→64→128 | 4·4·128 = 2048 | 4→8→16→32, ch 128→64→32→3 |
    | `ConvolutionalVariationalAutoEncoderV3` | 32→16→8→4→2, ch 3→32→64→128→256 | 2·2·256 = 1024 | 2→4→8→16→32, ch 256→128→64→32→3 |
    | `VisionAttentionVariationalAutoEncoderV1` | 4×4 patches (N=64 tokens) + `[CLS]`, `TransformerEncoderBlockV1` × 4 | `embed_dim` = 128 (from CLS) | latent → N tokens, `TransformerEncoderBlockV1` × 4, per-patch linear, fold to 32×32×3 |

    The ViT-style VAE follows the equations from the ViT note exactly:

    - $\mathbf{z}_0 = [\mathbf{x}_\text{class};\ \mathbf{x}_p^1 \mathbf{E};\ \cdots;\ \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$
    - $\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$
    - $\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$
    - $\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$
    """)
    return


@app.cell
def _():
    default_latent_dim = 64
    _conv_v1 = ConvolutionalVariationalAutoEncoderV1(latent_dim=default_latent_dim)
    _conv_v2 = ConvolutionalVariationalAutoEncoderV2(latent_dim=default_latent_dim)
    _conv_v3 = ConvolutionalVariationalAutoEncoderV3(latent_dim=default_latent_dim)
    _vit = VisionAttentionVariationalAutoEncoderV1(latent_dim=default_latent_dim)
    mx.eval(_conv_v1.parameters())
    mx.eval(_conv_v2.parameters())
    mx.eval(_conv_v3.parameters())
    mx.eval(_vit.parameters())
    param_count_conv_v1 = count_parameters(_conv_v1)
    param_count_conv_v2 = count_parameters(_conv_v2)
    param_count_conv_v3 = count_parameters(_conv_v3)
    param_count_vit = count_parameters(_vit)
    return (
        param_count_conv_v1,
        param_count_conv_v2,
        param_count_conv_v3,
        param_count_vit,
    )


@app.cell
def _(
    mo,
    param_count_conv_v1,
    param_count_conv_v2,
    param_count_conv_v3,
    param_count_vit,
):
    mo.md(f"""
    ### Reference parameter counts (`latent_dim = 64`)

    | Model | Parameters |
    |-------|-----------|
    | `ConvolutionalVariationalAutoEncoderV1` (2 conv blocks) | `{param_count_conv_v1:,}` |
    | `ConvolutionalVariationalAutoEncoderV2` (3 conv blocks) | `{param_count_conv_v2:,}` |
    | `ConvolutionalVariationalAutoEncoderV3` (4 conv blocks) | `{param_count_conv_v3:,}` |
    | `VisionAttentionVariationalAutoEncoderV1` (ViT, depth 4, heads 4, embed 128) | `{param_count_vit:,}` |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training all four models
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="1e-3",
        label="Learning Rate",
    )
    epochs_ui = mo.ui.slider(1, 50, value=10, step=1, label="Epochs")
    bs_ui = mo.ui.dropdown(
        options=[32, 64, 128, 256], value=128, label="Batch Size"
    )
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    latent_dim_ui = mo.ui.dropdown(
        options=[16, 32, 64, 128], value=64, label="Latent Dim"
    )
    train_btn = mo.ui.run_button(label="Train All Models")
    mo.vstack(
        [
            mo.md("### Hyperparameters (shared across all four models)"),
            mo.hstack([lr_ui, epochs_ui, bs_ui]),
            mo.hstack([wd_ui, latent_dim_ui]),
            train_btn,
        ]
    )
    return bs_ui, epochs_ui, latent_dim_ui, lr_ui, train_btn, wd_ui


@app.cell
def _(bs_ui, test_ds, train_ds):
    train_iter, test_iter = make_datasets(train_ds, test_ds, int(bs_ui.value))
    return (train_iter,)


@app.cell
def _(epochs_ui, latent_dim_ui, lr_ui, mo, train_btn, train_iter, wd_ui):
    train_losses_v1: list[float] = []
    train_losses_v2: list[float] = []
    train_losses_v3: list[float] = []
    train_losses_attn: list[float] = []
    trained_v1 = None
    trained_v2 = None
    trained_v3 = None
    trained_attn = None

    if not train_btn.value:
        mo.output.replace(
            mo.md("Click **Train All Models** to train all four VAE variants under identical hyperparameters.")
        )
    else:
        _ld = int(latent_dim_ui.value)
        _n_epochs = int(epochs_ui.value)
        _lr = float(lr_ui.value)
        _wd = float(wd_ui.value)

        _v1 = ConvolutionalVariationalAutoEncoderV1(latent_dim=_ld)
        _v2 = ConvolutionalVariationalAutoEncoderV2(latent_dim=_ld)
        _v3 = ConvolutionalVariationalAutoEncoderV3(latent_dim=_ld)
        _vit = VisionAttentionVariationalAutoEncoderV1(latent_dim=_ld)
        mx.eval(_v1.parameters())
        mx.eval(_v2.parameters())
        mx.eval(_v3.parameters())
        mx.eval(_vit.parameters())

        def _make_progress(name):
            def _cb(epoch, n_epochs, tl):
                mo.output.replace(
                    mo.md(f"**{name}** — Epoch {epoch + 1}/{n_epochs}  train: {tl:.4f}")
                )
            return _cb

        train_losses_v1 = train_model_epochs(
            _v1, train_iter, _n_epochs, _lr, _wd,
            preprocess_image_batch, _make_progress("ConvVAE V1"),
        )
        train_losses_v2 = train_model_epochs(
            _v2, train_iter, _n_epochs, _lr, _wd,
            preprocess_image_batch, _make_progress("ConvVAE V2"),
        )
        train_losses_v3 = train_model_epochs(
            _v3, train_iter, _n_epochs, _lr, _wd,
            preprocess_image_batch, _make_progress("ConvVAE V3"),
        )
        train_losses_attn = train_model_epochs(
            _vit, train_iter, _n_epochs, _lr, _wd,
            preprocess_image_batch, _make_progress("ViT VAE"),
        )
        trained_v1 = _v1
        trained_v2 = _v2
        trained_v3 = _v3
        trained_attn = _vit

        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train losses:\n\n"
                f"- ConvVAE V1: {train_losses_v1[-1]:.4f}\n"
                f"- ConvVAE V2: {train_losses_v2[-1]:.4f}\n"
                f"- ConvVAE V3: {train_losses_v3[-1]:.4f}\n"
                f"- ViT VAE: {train_losses_attn[-1]:.4f}"
            )
        )
    return (
        train_losses_attn,
        train_losses_v1,
        train_losses_v2,
        train_losses_v3,
        trained_attn,
        trained_v1,
        trained_v2,
        trained_v3,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (Optional)

    Since there is no validation split, the sweep ranks configurations by
    **final training loss**. A representative
    `ConvolutionalVariationalAutoEncoderV1` is used to keep runtime tractable.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(
        label="Enable Hyperparameter Search", value=False
    )
    hp_search_cb
    return (hp_search_cb,)


@app.cell
def _(hp_search_cb, mo, train_iter):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    _space = {"lr": [3e-4, 1e-3, 3e-3], "latent_dim": [32, 64, 128]}
    _results = []
    for _lr in _space["lr"]:
        for _ld in _space["latent_dim"]:
            _m = ConvolutionalVariationalAutoEncoderV1(latent_dim=_ld)
            mx.eval(_m.parameters())
            _losses = train_model_epochs(
                _m, train_iter, 3, _lr, 0.0, preprocess_image_batch, None,
            )
            _final = _losses[-1]
            _results.append(
                {"lr": _lr, "latent_dim": _ld, "final_train_loss": round(_final, 4)}
            )
            mo.output.replace(
                mo.md(f"lr={_lr}, latent_dim={_ld} → final train: {_final:.4f}")
            )
    _results.sort(key=lambda r: r["final_train_loss"])
    mo.ui.table(_results)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 7 — Sampling & Generation

    Replaces validation/cross-validation for this notebook. For each trained
    model, we draw fresh samples $z \sim \mathcal{N}(0, I)$ of shape
    `(n_samples, latent_dim)` and decode them into images.
    """)
    return


@app.function
def sample_and_generate(model, latent_dim: int, n_samples: int = 24) -> np.ndarray:
    z = mx.random.normal(shape=(n_samples, latent_dim))
    images = model.decode(z)
    mx.eval(images)
    return np.array(images)


@app.function
def plot_generated_grid(images: np.ndarray, title: str, rows: int = 3, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    n = min(rows * cols, images.shape[0])
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        if i < n:
            img = np.clip(images[i], 0.0, 1.0)
            ax.imshow(img)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


@app.cell
def _(latent_dim_ui, mo, trained_v1):
    if trained_v1 is None:
        _out = mo.md("_Train the models first (Section 5)._")
    else:
        _imgs = sample_and_generate(trained_v1, int(latent_dim_ui.value), 24)
        _out = plot_generated_grid(_imgs, "Samples from ConvVAE V1 (2 conv blocks)")
    _out
    return


@app.cell
def _(latent_dim_ui, mo, trained_v2):
    if trained_v2 is None:
        _out = mo.md("_Train the models first (Section 5)._")
    else:
        _imgs = sample_and_generate(trained_v2, int(latent_dim_ui.value), 24)
        _out = plot_generated_grid(_imgs, "Samples from ConvVAE V2 (3 conv blocks)")
    _out
    return


@app.cell
def _(latent_dim_ui, mo, trained_v3):
    if trained_v3 is None:
        _out = mo.md("_Train the models first (Section 5)._")
    else:
        _imgs = sample_and_generate(trained_v3, int(latent_dim_ui.value), 24)
        _out = plot_generated_grid(_imgs, "Samples from ConvVAE V3 (4 conv blocks)")
    _out
    return


@app.cell
def _(latent_dim_ui, mo, trained_attn):
    if trained_attn is None:
        _out = mo.md("_Train the models first (Section 5)._")
    else:
        _imgs = sample_and_generate(trained_attn, int(latent_dim_ui.value), 24)
        _out = plot_generated_grid(_imgs, "Samples from ViT VAE (attention-only)")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Results
    """)
    return


@app.function
def plot_loss_curve_comparison(loss_lists: dict) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=(9, 5))
    _styles = [("b-o", "ConvVAE V1"), ("g-s", "ConvVAE V2"), ("r-^", "ConvVAE V3"), ("m-D", "ViT VAE")]
    for (style, _default_label), (name, losses) in zip(_styles, loss_lists.items()):
        ax.plot(
            range(1, len(losses) + 1),
            losses,
            style,
            lw=2,
            ms=4,
            label=name,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (recon + KL)")
    ax.set_title("Training Loss Comparison — CIFAR-10 VAE variants")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    train_losses_attn: list[float],
    train_losses_v1: list[float],
    train_losses_v2: list[float],
    train_losses_v3: list[float],
):
    if not train_losses_v1:
        _out = mo.md("_Train first (Section 5)._")
    else:
        _loss_map = {
            "ConvVAE V1": train_losses_v1,
            "ConvVAE V2": train_losses_v2,
            "ConvVAE V3": train_losses_v3,
            "ViT VAE": train_losses_attn,
        }
        _out = plot_loss_curve_comparison(_loss_map)
    _out
    return


@app.cell
def _(
    mo,
    param_count_conv_v1,
    param_count_conv_v2,
    param_count_conv_v3,
    param_count_vit,
    train_losses_attn: list[float],
    train_losses_v1: list[float],
    train_losses_v2: list[float],
    train_losses_v3: list[float],
):
    if not train_losses_v1:
        _out = mo.md("_Train first (Section 5) to populate this table._")
    else:
        _rows = [
            {
                "model": "ConvolutionalVariationalAutoEncoderV1",
                "depth_hint": "2 conv blocks",
                "parameters": param_count_conv_v1,
                "final_train_loss": round(train_losses_v1[-1], 4),
            },
            {
                "model": "ConvolutionalVariationalAutoEncoderV2",
                "depth_hint": "3 conv blocks",
                "parameters": param_count_conv_v2,
                "final_train_loss": round(train_losses_v2[-1], 4),
            },
            {
                "model": "ConvolutionalVariationalAutoEncoderV3",
                "depth_hint": "4 conv blocks",
                "parameters": param_count_conv_v3,
                "final_train_loss": round(train_losses_v3[-1], 4),
            },
            {
                "model": "VisionAttentionVariationalAutoEncoderV1",
                "depth_hint": "attention-only (depth 4)",
                "parameters": param_count_vit,
                "final_train_loss": round(train_losses_attn[-1], 4),
            },
        ]
        _out = mo.ui.table(_rows)
    _out
    return


@app.cell
def _(
    mo,
    train_losses_attn: list[float],
    train_losses_v1: list[float],
    train_losses_v2: list[float],
    train_losses_v3: list[float],
):
    if not train_losses_v1:
        _summary = mo.md(
            """
    ### Summary (pending training)

    - **Framework**: MLX
    - **Dataset**: CIFAR-10 (50k train / 10k test, RGB 32×32)
    - **Models**: 3 convolutional depths (2/3/4 conv blocks) + 1 ViT-style attention-only VAE
    - **Method**: identical hyperparameters (lr, batch size, weight decay, epochs, latent dim) across all four
    - **Ranking metric**: final train loss (no validation split by design; see Section 7 for generation-quality inspection)

    Train the models above to populate the ranking here.
    """
        )
    else:
        _finals = {
            "ConvVAE V1": train_losses_v1[-1],
            "ConvVAE V2": train_losses_v2[-1],
            "ConvVAE V3": train_losses_v3[-1],
            "ViT VAE": train_losses_attn[-1],
        }
        _best_name = min(_finals, key=_finals.get)
        _best_loss = _finals[_best_name]
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX
    - **Dataset**: CIFAR-10 (50k train / 10k test, RGB 32×32)
    - **Lowest final train loss**: **{_best_name}** at `{_best_loss:.4f}`

    Deeper convolutional VAEs bottleneck to a smaller flat feature map
    (`8×8·64 = 4096` in V1, `4×4·128 = 2048` in V2, `2×2·256 = 1024` in V3)
    but with more parameters concentrated in the deeper channels. The
    attention-only VAE learns spatial relationships from scratch via
    self-attention over 8×8 = 64 patch tokens plus a `[CLS]` token.

    **Qualitative note**: inspect the generation grids in Section 7 side by
    side. CIFAR-10 is a small (50k) natural-image dataset — VAEs
    typically produce blurry samples on it, and the ViT variant may need
    substantially more epochs than the conv variants to catch up, since it
    lacks the locality/translation-equivariance inductive biases of
    convolutions. Fill in your own observations here after inspecting the
    grids.
    """
        )
    _summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Model(s)

    Select which of the four trained VAEs to persist, choose a filename
    (defaulted per selection), and click **Save Model**. Weights are
    written to the repo-root `models/` directory in MLX's
    `.safetensors` format.
    """)
    return


@app.cell
def _(mo):
    save_model_choice_ui = mo.ui.dropdown(
        options=[
            "ConvVAE V1",
            "ConvVAE V2",
            "ConvVAE V3",
            "ViT VAE",
        ],
        value="ConvVAE V1",
        label="Model to save",
    )
    save_model_choice_ui
    return (save_model_choice_ui,)


@app.function
def default_save_filename(choice: str) -> str:
    return {
        "ConvVAE V1": "cifar10_convvae_v1.safetensors",
        "ConvVAE V2": "cifar10_convvae_v2.safetensors",
        "ConvVAE V3": "cifar10_convvae_v3.safetensors",
        "ViT VAE": "cifar10_vitvae_v1.safetensors",
    }.get(choice, "cifar10_vae.safetensors")


@app.cell
def _(mo, save_model_choice_ui):
    save_filename_ui = mo.ui.text(
        value=default_save_filename(save_model_choice_ui.value),
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_model_btn])
    return save_filename_ui, save_model_btn


@app.function
def resolve_trained_model(choice: str, trained_v1, trained_v2, trained_v3, trained_attn):
    return {
        "ConvVAE V1": trained_v1,
        "ConvVAE V2": trained_v2,
        "ConvVAE V3": trained_v3,
        "ViT VAE": trained_attn,
    }.get(choice)


@app.cell
def _(
    mo,
    save_filename_ui,
    save_model_btn,
    save_model_choice_ui,
    trained_attn,
    trained_v1,
    trained_v2,
    trained_v3,
):
    _selected = resolve_trained_model(
        save_model_choice_ui.value,
        trained_v1, trained_v2, trained_v3, trained_attn,
    )
    if _selected is None:
        _out = mo.md(
            f"_The selected model (**{save_model_choice_ui.value}**) has not "
            f"been trained yet. Run Section 5 first._"
        )
    elif not save_model_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save Model** to write the "
            "trained weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_filename_ui.value
        _selected.save_weights(str(_save_path))
        _out = mo.md(
            f"**Saved!** `{save_model_choice_ui.value}` weights written to "
            f"`{_save_path}`."
        )
    _out
    return


if __name__ == "__main__":
    app.run()
