import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import math
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
    # Vector Quantized Variational Autoencoder with a Vision Transformer Backbone on CIFAR-10 — MLX

    ## Research Goal

    Train a **Vector-Quantized Variational Autoencoder (VQ-VAE)** with a
    **Vision Transformer (ViT)** backbone — *both* the encoder and the
    decoder are transformer stacks — on the CIFAR-10 32x32 RGB dataset
    using Apple's **MLX** framework. This follows

    > van den Oord, A., Vinyals, O. and Kavukcuoglu, K. (2017).
    > _Neural Discrete Representation Learning._ arXiv:1711.00937.
    > <https://arxiv.org/abs/1711.00937>

    ### How this design works

    - A **ViT encoder** patchifies each 32x32x3 image into an 8x8 grid of
      4x4 patches, projects them into an embedding space, runs a stack of
      pre-norm transformer blocks over the 64 tokens, then linearly
      projects each token into a `D`-dimensional pre-code embedding.
    - Every one of the 64 spatial positions is independently **quantized**
      to its nearest of `K` learned codebook vectors — the latent for the
      whole image is a discrete `8 x 8` grid of code indices.
    - The **straight-through estimator** (identity in the forward pass,
      gradients routed past the argmin) lets the encoder train.
    - A **mirror ViT decoder** projects the quantized `D`-vectors back to
      the embedding dimension, adds positional embeddings, runs another
      transformer stack, then linearly reads out 4x4x3 pixel patches and
      folds them back into a 32x32x3 image.
    - **Loss** = `MSE(sigmoid(logits), x) + codebook_loss + beta * commitment_loss`
      — no KL term, so this model cannot suffer from **posterior
      collapse** the way a continuous VAE can. Its failure mode is
      **codebook collapse** (few active codes), which is why we monitor
      **perplexity** and re-seed dead codes every epoch.
    - After the VQ-VAE is trained, we fit a small **autoregressive
      transformer** ("code prior") that models `p(z_1, ..., z_64)` in
      raster order over the 8x8 code grid. Sampling from this prior and
      decoding through the ViT decoder yields **novel synthesized
      images** — this is how the original VQ-VAE paper unlocks
      generation.

    ### Notebook outline

    1. Title & Research Goal
    2. Data Exploration
    3. Dataset Creation
    4. Model Definition (ViT-VQ-VAE)
    5. Training (VQ-VAE)
    6. Hyperparameter Search (Optional)
    7. Validation & Cross-Validation
    8. Saved-Model Diagnostics
    9. Results (loss curves, reconstructions, codebook usage, summary)
    10. Code Prior & Interactive Sampling
    11. Save Trained Models
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


@app.function
def cifar10_class_names() -> list[str]:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


@app.cell
def _(mo, test_ds, train_ds):
    mo.md(f"""
    ### Dataset overview

    CIFAR-10 is loaded via `mlx.data.datasets.load_cifar10` as an
    `mlx.data` Buffer. Each sample is a dict with:

    - `image` — `uint8` array shaped `(32, 32, 3)` (channels-last RGB)
    - `label` — scalar `int64` in `[0, 9]`

    | Split | Size |
    |-------|------|
    | Train (raw, 50k) | {len(train_ds):,} |
    | Test | {len(test_ds):,} |

    Section 3 carves an 85/15 train/val split from the 50k train buffer.
    Labels are shown only for exploration — they are **not** used by the
    VQ-VAE loss. Class names in label order: `{cifar10_class_names()}`.
    """)
    return


@app.function
def plot_sample_grid(dataset, class_names, n_show: int = 40, rows: int = 5, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i in range(n_show):
        sample = dataset[i]
        img = np.asarray(sample["image"])
        label = int(np.asarray(sample["label"]).item())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(class_names[label], fontsize=8)
        axes[r, c].axis("off")
    fig.suptitle("CIFAR-10 training samples", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_sample_grid(train_ds, cifar10_class_names())
    return


@app.function
def plot_class_distribution(dataset, class_names):
    labels = np.array(
        [int(np.asarray(dataset[i]["label"]).item()) for i in range(len(dataset))]
    )
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(class_names)), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 training-set class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c + 40, str(int(c)), ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_class_distribution(train_ds, cifar10_class_names())
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def stream_normalize(x):
    return x.astype("float32") / 255.0


@app.function
def normalize_images(x) -> mx.array:
    return mx.array(x, dtype=mx.float32) / 255.0


@app.function
def split_train_val(train_ds, val_fraction: float = 0.15, seed: int = 0):
    n_total = len(train_ds)
    n_val = int(round(n_total * val_fraction))
    perm = np.random.default_rng(seed).permutation(n_total).tolist()
    val_buf = train_ds.perm(perm[:n_val])
    train_buf = train_ds.perm(perm[n_val:])
    return train_buf, val_buf


@app.function
def make_datasets(
    train_ds,
    test_ds,
    batch_size: int,
    val_fraction: float = 0.15,
    seed: int = 0,
):
    train_buf, val_buf = split_train_val(train_ds, val_fraction=val_fraction, seed=seed)

    train_iter = (
        train_buf
        .to_stream()
        .key_transform("image", stream_normalize)
        .shuffle(8192)
        .batch(batch_size)
    )
    val_iter = (
        val_buf
        .to_stream()
        .key_transform("image", stream_normalize)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", stream_normalize)
        .batch(batch_size)
    )
    return train_iter, val_iter, test_iter, len(train_buf), len(val_buf)


@app.function
def preprocess_image_batch(batch) -> mx.array:
    # Contract: `batch` comes from a make_datasets stream, whose key_transform
    # has already scaled "image" to float32 in [0, 1]. This is only a device
    # wrap, not a normalization — build raw batches with normalize_images().
    return mx.array(batch["image"], dtype=mx.float32)


@app.cell
def _(test_ds, train_ds):
    (
        default_train_iter,
        _default_val_iter,
        _default_test_iter,
        default_n_train,
        default_n_val,
    ) = make_datasets(train_ds, test_ds, batch_size=128, val_fraction=0.15)
    return default_n_train, default_n_val, default_train_iter


@app.cell
def _(default_n_train, default_n_val, default_train_iter, mo):
    default_train_iter.reset()
    _peek_batch = next(default_train_iter)
    _peek = mx.array(_peek_batch["image"])
    mo.md(
        f"""
    ### Split sizes and one-batch inspection

    A seeded permutation carves a genuine held-out val partition from the
    50k train buffer (`Buffer.perm`) — train and val samples never overlap.

    - **Train** (85% of raw 50k): {default_n_train:,}
    - **Val** (15% of raw 50k, held out): {default_n_val:,}
    - **Test** (raw, held-out): 10,000

    A single training batch after
    `train_buf.to_stream().key_transform(norm).shuffle(8192).batch(128)`
    (the shuffle buffer re-randomizes on every epoch `.reset()`):

    - batch image shape: `{tuple(_peek.shape)}`
    - batch dtype: `{_peek.dtype}`
    - value range: `[{float(mx.min(_peek).item()):.3f}, {float(mx.max(_peek).item()):.3f}]`
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition (ViT VQ-VAE)
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
    def __init__(self, dims: int = 256, expansion: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dims, dims * expansion)
        self.fc2 = nn.Linear(dims * expansion, dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


@app.class_definition
class TransformerEncoderBlockV1(nn.Module):
    def __init__(
        self,
        dims: int = 256,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
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
class ViTEncoderV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        code_dim: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.code_dim = code_dim
        self.grid_size = image_size // patch_size
        self.patch_embed = PatchEmbeddingV1(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = mx.zeros((1, self.num_patches, embed_dim))
        self.blocks = [
            TransformerEncoderBlockV1(embed_dim, num_heads, mlp_expansion)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim)
        self.to_code = nn.Linear(embed_dim, code_dim)

    def __call__(self, x: mx.array) -> mx.array:
        b = x.shape[0]
        h = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        z = self.to_code(h)
        return z.reshape(b, self.grid_size, self.grid_size, self.code_dim)


@app.class_definition
class VectorQuantizerV1(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.codebook = mx.random.normal(
            shape=(num_embeddings, embedding_dim)
        ) * (1.0 / embedding_dim ** 0.5)

    def reset_dead_codes(self, z_e_flat: mx.array, hit_counts, min_count: int = 1):
        counts = np.asarray(hit_counts).reshape(-1)
        dead = np.nonzero(counts < min_count)[0]
        pool = np.asarray(z_e_flat)
        if dead.size == 0 or pool.shape[0] == 0:
            return 0
        rng = np.random.default_rng()
        pick = rng.integers(0, pool.shape[0], size=dead.size)
        noise = rng.normal(
            scale=1e-3, size=(dead.size, self.embedding_dim)
        ).astype(pool.dtype)
        new_codebook = np.asarray(self.codebook).copy()
        new_codebook[dead] = pool[pick] + noise
        self.codebook = mx.array(new_codebook)
        mx.eval(self.codebook)
        return int(dead.size)

    def __call__(self, z_e: mx.array):
        _b, _h, _w, _d = z_e.shape
        flat = z_e.reshape(-1, _d)

        z_norm = mx.sum(flat * flat, axis=1, keepdims=True)
        c_norm = mx.sum(self.codebook * self.codebook, axis=1, keepdims=True).T
        dot = flat @ self.codebook.T
        distances = z_norm + c_norm - 2.0 * dot

        encoding_indices = mx.argmin(distances, axis=1)
        z_q_flat = self.codebook[encoding_indices]
        z_q = z_q_flat.reshape(_b, _h, _w, _d)

        codebook_loss = mx.mean((mx.stop_gradient(z_e) - z_q) ** 2)
        commitment_loss = mx.mean((z_e - mx.stop_gradient(z_q)) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q_st = z_e + mx.stop_gradient(z_q - z_e)

        one_hot = mx.zeros((encoding_indices.shape[0], self.num_embeddings))
        one_hot[mx.arange(encoding_indices.shape[0]), encoding_indices] = 1.0
        avg_probs = mx.mean(one_hot, axis=0)
        perplexity = mx.exp(
            -mx.sum(avg_probs * mx.log(avg_probs + 1e-10))
        )

        return (
            z_q_st,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
            encoding_indices.reshape(_b, _h, _w),
        )


@app.class_definition
class ViTDecoderV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        out_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        code_dim: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.code_dim = code_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size * out_channels
        self.from_code = nn.Linear(code_dim, embed_dim)
        self.pos_embed = mx.zeros((1, self.num_patches, embed_dim))
        self.blocks = [
            TransformerEncoderBlockV1(embed_dim, num_heads, mlp_expansion)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.patch_dim)

    def decode_logits(self, z_q: mx.array) -> mx.array:
        b = z_q.shape[0]
        h = z_q.reshape(b, self.num_patches, self.code_dim)
        h = self.from_code(h) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        patches = self.head(h)
        gh = self.grid_size
        gw = self.grid_size
        p = self.patch_size
        c = self.out_channels
        x = patches.reshape(b, gh, gw, p, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, gh * p, gw * p, c)

    def __call__(self, z_q: mx.array) -> mx.array:
        return mx.sigmoid(self.decode_logits(z_q))


@app.class_definition
class ViTVQVAEV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        decoder_depth: int = 6,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.grid_size = image_size // patch_size
        self.encoder = ViTEncoderV1(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_expansion=mlp_expansion,
            code_dim=embedding_dim,
        )
        self.quantizer = VectorQuantizerV1(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = ViTDecoderV1(
            image_size=image_size,
            patch_size=patch_size,
            out_channels=in_channels,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_expansion=mlp_expansion,
            code_dim=embedding_dim,
        )

    def __call__(self, x: mx.array):
        z_e = self.encoder(x)
        z_q_st, vq_loss, codebook_loss, commitment_loss, perplexity, indices = (
            self.quantizer(z_e)
        )
        logits = self.decoder.decode_logits(z_q_st)
        x_hat = mx.sigmoid(logits)
        return (
            x_hat,
            logits,
            vq_loss,
            codebook_loss,
            commitment_loss,
            perplexity,
            indices,
        )

    def encode_to_indices(self, x: mx.array) -> mx.array:
        z_e = self.encoder(x)
        _z_q_st, _vq, _cb, _cm, _ppl, indices = self.quantizer(z_e)
        return indices

    def decode_from_indices(self, indices: mx.array) -> mx.array:
        z_q = self.quantizer.codebook[indices]
        return mx.sigmoid(self.decoder.decode_logits(z_q))


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def reconstruction_loss(logits: mx.array, x: mx.array) -> mx.array:
    # MSE reconstruction on [0, 1] pixels, computed from decoder logits
    # via a sigmoid squashing step (loss is on pixel space, not logits).
    return mx.mean((mx.sigmoid(logits) - x) ** 2)


@app.function
def compute_vqvae_loss(model: nn.Module, x: mx.array) -> mx.array:
    _x_hat, logits, vq_loss, _codebook_loss, _commitment_loss, _perplexity, _idx = model(x)
    return reconstruction_loss(logits, x) + vq_loss


@app.function
def compute_vqvae_metrics(model: nn.Module, x: mx.array):
    _x_hat, logits, vq_loss, codebook_loss, commitment_loss, perplexity, _idx = model(x)
    recon_loss = reconstruction_loss(logits, x)
    total = recon_loss + vq_loss
    return total, recon_loss, codebook_loss, commitment_loss, perplexity


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Architecture — `ViTVQVAEV1`

    Both encoder and decoder are transformer stacks. MLX is channels-last
    (`NHWC`). CIFAR-10 images arrive as `(B, 32, 32, 3)` and are patchified
    into an `8 x 8` grid of `4 x 4 x 3` = 48-D patches.

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Patchify + linear proj | `PatchEmbeddingV1` (p=4) | `(B, 64, embed_dim)` |
    | + learnable pos embed | `mx.zeros(1, 64, embed_dim)` | `(B, 64, embed_dim)` |
    | Encoder transformer x depth | `TransformerEncoderBlockV1` | `(B, 64, embed_dim)` |
    | LayerNorm + linear to code | `LayerNorm` + `Linear(embed_dim, D)` | `(B, 64, D)` |
    | Reshape to grid | (folds tokens back to `8x8`) | `(B, 8, 8, D)` |
    | Vector quantizer | `VectorQuantizerV1(K, D)` | `(B, 8, 8, D)` + `(B, 8, 8)` codes |
    | Linear from code | `Linear(D, embed_dim)` + pos embed | `(B, 64, embed_dim)` |
    | Decoder transformer x depth | `TransformerEncoderBlockV1` | `(B, 64, embed_dim)` |
    | LayerNorm + patch head | `Linear(embed_dim, 4*4*3)` | `(B, 64, 48)` |
    | Fold back to image | reshape + transpose | `(B, 32, 32, 3)` logits |

    **Loss** = `mean( (sigmoid(logits) - x)^2 ) + mean(( sg(z_e) - z_q )^2) + beta * mean(( z_e - sg(z_q) )^2)`

    (`sg` denotes `mx.stop_gradient`; `beta = 0.25` by default. We also
    track **perplexity** = `exp(-sum(p_k log p_k))` where `p_k` is the
    average codebook usage across a batch — max value `K`, meaning all
    codes used equally.)

    **Anti-collapse safeguards.** The codebook is initialized from
    `normal(0, 1) / sqrt(D)`, and at the end of every epoch any code that
    received **zero** assignments is re-seeded from a random live encoder
    output (`VectorQuantizerV1.reset_dead_codes`). Section 5 reports
    `codes N/K` per epoch so collapse is visible during training.
    """)
    return


@app.cell
def _(mo):
    _reference_model = ViTVQVAEV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8,
        mlp_expansion=4,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25,
    )
    mx.eval(_reference_model.parameters())
    reference_param_count = count_parameters(_reference_model)
    mo.md(
        f"""**Reference `ViTVQVAEV1` parameter count** — embed_dim=256,
        depth=6/6, num_heads=8, K=512, D=64, beta=0.25:
        `{reference_param_count:,}` parameters."""
    )
    return (reference_param_count,)


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training (VQ-VAE)
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "5e-4": 5e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="Learning Rate",
    )
    epochs_ui = mo.ui.slider(1, 100, value=20, step=1, label="Epochs")
    bs_ui = mo.ui.dropdown(
        options=[64, 128, 256], value=128, label="Batch Size"
    )
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="0",
        label="Weight Decay",
    )
    beta_ui = mo.ui.dropdown(
        options={"0.1": 0.1, "0.25": 0.25, "0.5": 0.5, "1.0": 1.0},
        value="0.25",
        label="Commitment Cost (beta)",
    )
    num_embeddings_ui = mo.ui.dropdown(
        options=[256, 512, 1024],
        value=512,
        label="Num Embeddings (K)",
    )
    embedding_dim_ui = mo.ui.dropdown(
        options=[32, 64, 128],
        value=64,
        label="Embedding Dim (D)",
    )
    train_btn = mo.ui.run_button(label="Train VQ-VAE")
    mo.vstack(
        [
            mo.md("### Hyperparameters"),
            mo.hstack([lr_ui, epochs_ui, bs_ui]),
            mo.hstack([wd_ui, beta_ui]),
            mo.hstack([num_embeddings_ui, embedding_dim_ui]),
            train_btn,
        ]
    )
    return (
        beta_ui,
        bs_ui,
        embedding_dim_ui,
        epochs_ui,
        lr_ui,
        num_embeddings_ui,
        train_btn,
        wd_ui,
    )


@app.cell
def _(bs_ui, test_ds, train_ds):
    train_iter, val_iter, test_iter, _n_train, _n_val = make_datasets(
        train_ds, test_ds, int(bs_ui.value), val_fraction=0.15
    )
    return test_iter, train_iter, val_iter


@app.function
def run_train_epoch(
    model: nn.Module,
    optimizer,
    train_iter,
    preprocess_fn,
):
    loss_and_grad_fn = nn.value_and_grad(model, compute_vqvae_loss)
    k = model.quantizer.num_embeddings
    d_dim = model.quantizer.embedding_dim
    total_loss = 0.0
    total_recon = 0.0
    total_codebook = 0.0
    total_commit = 0.0
    total_ppl = 0.0
    n_batches = 0
    hit_counts = np.zeros(k, dtype=np.int64)
    last_x = None
    train_iter.reset()
    for batch in train_iter:
        x = preprocess_fn(batch)
        last_x = x
        loss, grads = loss_and_grad_fn(model, x)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        _x_hat, logits, _vq, codebook, commit, ppl, indices = model(x)
        recon = reconstruction_loss(logits, x)
        mx.eval(recon, codebook, commit, ppl, indices)
        hit_counts += np.bincount(
            np.asarray(indices).reshape(-1), minlength=k
        )
        total_loss += loss.item()
        total_recon += recon.item()
        total_codebook += codebook.item()
        total_commit += commit.item()
        total_ppl += ppl.item()
        n_batches += 1
    if last_x is not None:
        _z_e = model.encoder(last_x)
        mx.eval(_z_e)
        _pool = mx.array(np.asarray(_z_e).reshape(-1, d_dim))
        model.quantizer.reset_dead_codes(_pool, hit_counts)
    codes_used = int((hit_counts > 0).sum())
    d = max(n_batches, 1)
    return (
        total_loss / d,
        total_recon / d,
        total_codebook / d,
        total_commit / d,
        total_ppl / d,
        codes_used,
    )


@app.function
def run_evaluate(model: nn.Module, data_iter, preprocess_fn):
    total_loss = 0.0
    total_recon = 0.0
    total_codebook = 0.0
    total_commit = 0.0
    total_ppl = 0.0
    n_batches = 0
    data_iter.reset()
    for batch in data_iter:
        x = preprocess_fn(batch)
        total, recon, codebook, commit, ppl = compute_vqvae_metrics(model, x)
        mx.eval(total, recon, codebook, commit, ppl)
        total_loss += total.item()
        total_recon += recon.item()
        total_codebook += codebook.item()
        total_commit += commit.item()
        total_ppl += ppl.item()
        n_batches += 1
    d = max(n_batches, 1)
    return (
        total_loss / d,
        total_recon / d,
        total_codebook / d,
        total_commit / d,
        total_ppl / d,
    )


@app.cell
def _(
    beta_ui,
    embedding_dim_ui,
    epochs_ui,
    lr_ui,
    mo,
    num_embeddings_ui,
    train_btn,
    train_iter,
    val_iter,
    wd_ui,
):
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    train_perplexities = []
    val_perplexities = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train VQ-VAE** to begin training."))
    else:
        _model = ViTVQVAEV1(
            image_size=32,
            patch_size=4,
            in_channels=3,
            embed_dim=256,
            encoder_depth=6,
            decoder_depth=6,
            num_heads=8,
            mlp_expansion=4,
            num_embeddings=int(num_embeddings_ui.value),
            embedding_dim=int(embedding_dim_ui.value),
            commitment_cost=float(beta_ui.value),
        )
        mx.eval(_model.parameters())
        _optimizer = optim.AdamW(
            learning_rate=float(lr_ui.value),
            weight_decay=float(wd_ui.value),
        )
        _n_epochs = int(epochs_ui.value)
        _K = int(num_embeddings_ui.value)
        for _epoch in range(_n_epochs):
            _tl, _trecon, _tcb, _tcm, _tppl, _tcodes = run_train_epoch(
                _model, _optimizer, train_iter, preprocess_image_batch
            )
            _vl, _vrecon, _vcb, _vcm, _vppl = run_evaluate(
                _model, val_iter, preprocess_image_batch
            )
            train_losses.append(_tl)
            val_losses.append(_vl)
            train_recon_losses.append(_trecon)
            val_recon_losses.append(_vrecon)
            train_perplexities.append(_tppl)
            val_perplexities.append(_vppl)
            mo.output.replace(
                mo.md(
                    f"**Epoch {_epoch + 1}/{_n_epochs}** — "
                    f"train loss: {_tl:.4f} (recon {_trecon:.4f}, "
                    f"codebook {_tcb:.4f}, commit {_tcm:.4f}, ppl {_tppl:.2f}, "
                    f"codes {_tcodes}/{_K}) | "
                    f"val loss: {_vl:.4f} (recon {_vrecon:.4f}, ppl {_vppl:.2f})"
                )
            )
        trained_model = _model
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train loss: "
                f"{train_losses[-1]:.4f} | val loss: {val_losses[-1]:.4f} | "
                f"val perplexity: {val_perplexities[-1]:.2f} / K="
                f"{int(num_embeddings_ui.value)}."
            )
        )
    return (
        train_losses,
        train_perplexities,
        train_recon_losses,
        trained_model,
        val_losses,
        val_perplexities,
        val_recon_losses,
    )


@app.cell
def _(mo):
    mo.md("""
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
def run_hp_config(
    train_iter,
    val_iter,
    n_epochs: int,
    lr: float,
    num_embeddings: int,
    embedding_dim: int,
    commitment_cost: float,
    weight_decay: float = 0.0,
):
    model = ViTVQVAEV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8,
        mlp_expansion=4,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
    )
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        run_train_epoch(model, optimizer, train_iter, preprocess_image_batch)
    val_total, val_recon, _cb, _cm, val_ppl = run_evaluate(
        model, val_iter, preprocess_image_batch
    )
    return val_total, val_recon, val_ppl


@app.cell
def _(embedding_dim_ui, hp_search_cb, mo, train_iter, val_iter):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    _space = {
        "num_embeddings": [256, 512, 1024],
        "beta": [0.1, 0.25, 1.0],
    }
    _results = []
    _n_epochs = 3
    _lr = 1e-3
    _D = int(embedding_dim_ui.value)
    for _K in _space["num_embeddings"]:
        for _beta in _space["beta"]:
            _val_total, _val_recon, _val_ppl = run_hp_config(
                train_iter,
                val_iter,
                n_epochs=_n_epochs,
                lr=_lr,
                num_embeddings=_K,
                embedding_dim=_D,
                commitment_cost=_beta,
            )
            _results.append(
                {
                    "num_embeddings": _K,
                    "embedding_dim": _D,
                    "beta": _beta,
                    "val_loss": round(_val_total, 4),
                    "val_recon": round(_val_recon, 4),
                    "val_perplexity": round(_val_ppl, 2),
                }
            )
            mo.output.replace(
                mo.md(
                    f"K={_K}, beta={_beta}: val_loss={_val_total:.4f}, "
                    f"val_recon={_val_recon:.4f}, ppl={_val_ppl:.2f}"
                )
            )
    _results.sort(key=lambda r: r["val_loss"])
    mo.ui.table(_results)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation
    """)
    return


@app.function
def evaluate_model(model: nn.Module, data_iter, preprocess_fn):
    total, recon, codebook, commit, ppl = run_evaluate(
        model, data_iter, preprocess_fn
    )
    return {
        "total_loss": total,
        "recon_loss": recon,
        "codebook_loss": codebook,
        "commitment_loss": commit,
        "perplexity": ppl,
    }


@app.cell
def _(mo, test_iter, trained_model, val_iter):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to see held-out metrics._")
    else:
        _val_metrics = evaluate_model(trained_model, val_iter, preprocess_image_batch)
        _test_metrics = evaluate_model(trained_model, test_iter, preprocess_image_batch)
        _out = mo.md(
            f"""
    ### Held-out evaluation of `ViTVQVAEV1`

    | Split | Total Loss | Reconstruction (MSE) | Codebook Loss | Commitment Loss | Perplexity |
    |-------|-----------|--------------------|---------------|-----------------|-----------|
    | Val   | {_val_metrics["total_loss"]:.4f} | {_val_metrics["recon_loss"]:.4f} | {_val_metrics["codebook_loss"]:.4f} | {_val_metrics["commitment_loss"]:.4f} | {_val_metrics["perplexity"]:.2f} |
    | Test  | {_test_metrics["total_loss"]:.4f} | {_test_metrics["recon_loss"]:.4f} | {_test_metrics["codebook_loss"]:.4f} | {_test_metrics["commitment_loss"]:.4f} | {_test_metrics["perplexity"]:.2f} |

    Higher **perplexity** (closer to `K`) indicates broader codebook usage;
    values collapsing toward 1 signal codebook collapse.
    """
        )
    _out
    return


@app.cell
def _(mo):
    cv_cb = mo.ui.checkbox(
        label="Enable 5-Fold Cross-Validation", value=False
    )
    cv_cb
    return (cv_cb,)


@app.function
def run_cv_fold(
    train_ds,
    fold_index: int,
    n_folds: int,
    n_epochs: int,
    lr: float,
    num_embeddings: int,
    embedding_dim: int,
    commitment_cost: float,
    batch_size: int = 128,
):
    n_total = len(train_ds)
    fold_size = n_total // n_folds
    v_start = fold_index * fold_size
    v_end = v_start + fold_size
    train_indices = list(range(0, v_start)) + list(range(v_end, n_total))
    val_indices = list(range(v_start, v_end))

    model = ViTVQVAEV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8,
        mlp_expansion=4,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
    )
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, compute_vqvae_loss)
    k = model.quantizer.num_embeddings
    d_dim = model.quantizer.embedding_dim

    for _ in range(n_epochs):
        np.random.shuffle(train_indices)
        hit_counts = np.zeros(k, dtype=np.int64)
        last_x = None
        for i in range(0, len(train_indices), batch_size):
            idx = train_indices[i : i + batch_size]
            imgs = np.stack([train_ds[j]["image"] for j in idx])
            x = normalize_images(imgs)
            last_x = x
            loss, grads = loss_and_grad_fn(model, x)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            _z_e = model.encoder(x)
            *_rest, _indices = model.quantizer(_z_e)
            mx.eval(_indices)
            hit_counts += np.bincount(
                np.asarray(_indices).reshape(-1), minlength=k
            )
        if last_x is not None:
            _pool_z = model.encoder(last_x)
            mx.eval(_pool_z)
            _pool = mx.array(
                np.asarray(_pool_z).reshape(-1, d_dim)
            )
            model.quantizer.reset_dead_codes(_pool, hit_counts)

    val_total = 0.0
    val_recon = 0.0
    val_ppl = 0.0
    n_batches = 0
    for i in range(0, len(val_indices), 256):
        idx = val_indices[i : i + 256]
        imgs = np.stack([train_ds[j]["image"] for j in idx])
        x = normalize_images(imgs)
        total, recon, _cb, _cm, ppl = compute_vqvae_metrics(model, x)
        mx.eval(total, recon, ppl)
        val_total += total.item()
        val_recon += recon.item()
        val_ppl += ppl.item()
        n_batches += 1
    d = max(n_batches, 1)
    return val_total / d, val_recon / d, val_ppl / d


@app.cell
def _(
    beta_ui,
    cv_cb,
    embedding_dim_ui,
    mo,
    num_embeddings_ui,
    train_ds,
    trained_model,
):
    mo.stop(
        not cv_cb.value,
        mo.md("_Enable 5-fold CV above (CV over a ViT is expensive)._"),
    )
    cv_results = {}
    if trained_model is None:
        _out = mo.md("_Train first — 5-fold CV results will appear here._")
    else:
        _k = 5
        _fold_totals = []
        _fold_recons = []
        _fold_ppls = []
        for _fold in range(_k):
            _val_total, _val_recon, _val_ppl = run_cv_fold(
                train_ds,
                fold_index=_fold,
                n_folds=_k,
                n_epochs=2,
                lr=1e-3,
                num_embeddings=int(num_embeddings_ui.value),
                embedding_dim=int(embedding_dim_ui.value),
                commitment_cost=float(beta_ui.value),
                batch_size=128,
            )
            _fold_totals.append(_val_total)
            _fold_recons.append(_val_recon)
            _fold_ppls.append(_val_ppl)
            mo.output.replace(
                mo.md(
                    f"Fold {_fold + 1}/{_k} — val_loss: {_val_total:.4f}, "
                    f"val_recon: {_val_recon:.4f}, ppl: {_val_ppl:.2f}"
                )
            )
        cv_results = {
            "fold_total_losses": _fold_totals,
            "fold_recon_losses": _fold_recons,
            "fold_perplexities": _fold_ppls,
            "mean_total": float(np.mean(_fold_totals)),
            "std_total": float(np.std(_fold_totals)),
            "mean_recon": float(np.mean(_fold_recons)),
            "std_recon": float(np.std(_fold_recons)),
            "mean_ppl": float(np.mean(_fold_ppls)),
            "std_ppl": float(np.std(_fold_ppls)),
        }
        _out = mo.md(
            f"""
    ### 5-Fold Cross-Validation on the training set

    | Metric | Mean | Std |
    |--------|------|-----|
    | Total loss | {cv_results["mean_total"]:.4f} | {cv_results["std_total"]:.4f} |
    | Reconstruction (MSE) | {cv_results["mean_recon"]:.4f} | {cv_results["std_recon"]:.4f} |
    | Perplexity | {cv_results["mean_ppl"]:.2f} | {cv_results["std_ppl"]:.2f} |

    Per-fold total losses: `{[round(v, 4) for v in _fold_totals]}`
    """
        )
    _out
    return (cv_results,)


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Saved-Model Diagnostics

    Load any ViT-VQ-VAE checkpoint from `models/` and inspect what the
    decoder actually produces — numeric statistics plus true-scale and
    contrast-stretched reconstruction rows. This section does **not**
    require training in the current session.
    """)
    return


@app.function
def models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


@app.function
def list_checkpoints() -> list[str]:
    d = models_dir()
    if not d.is_dir():
        return []
    names = [p.name for p in d.glob("*.safetensors")] + [
        p.name for p in d.glob("*.npz")
    ]
    return sorted(names)


@app.function
def infer_vitvqvae_config(state: dict) -> dict:
    if "encoder.to_code.weight" not in state:
        raise ValueError(
            "Checkpoint does not look like a ViTVQVAEV1: "
            "missing 'encoder.to_code.weight' key."
        )
    num_embeddings, embedding_dim = (int(v) for v in state["quantizer.codebook"].shape)
    # nn.Linear weight shape is (out_features, in_features), so in-features is column dim
    embed_dim = int(state["encoder.to_code.weight"].shape[1])
    patch_dim_in = int(state["encoder.patch_embed.proj.weight"].shape[1])
    patch_size = int(round((patch_dim_in / 3) ** 0.5))
    encoder_depth = 1 + max(
        int(k.split(".")[2]) for k in state if k.startswith("encoder.blocks.")
    )
    decoder_depth = 1 + max(
        int(k.split(".")[2]) for k in state if k.startswith("decoder.blocks.")
    )
    return {
        "image_size": 32,
        "patch_size": patch_size,
        "in_channels": 3,
        "embed_dim": embed_dim,
        "encoder_depth": encoder_depth,
        "decoder_depth": decoder_depth,
        # num_heads cannot be recovered from the state dict (heads only reshape
        # activations, they do not change any parameter shape). Assume 8.
        "num_heads": 8,
        "mlp_expansion": 4,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
    }


@app.function
def load_vitvqvae(path):
    state = mx.load(str(path))
    cfg = infer_vitvqvae_config(state)
    model = ViTVQVAEV1(**cfg)
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model, cfg


@app.function
def diagnose_model(model: nn.Module, x: mx.array) -> dict:
    z_e = model.encoder(x)
    z_q_st, _vq, _cb, _cm, perplexity, indices = model.quantizer(z_e)
    logits = model.decoder.decode_logits(z_q_st)
    x_hat = mx.sigmoid(logits)
    mx.eval(z_e, logits, x_hat, perplexity, indices)

    recon = np.asarray(x_hat)
    lg = np.asarray(logits)
    idx = np.asarray(indices).reshape(-1)
    k = int(model.quantizer.num_embeddings)
    codes_used = int(np.unique(idx).size)
    n_nan = int(np.isnan(recon).sum())
    n_inf = int(np.isinf(recon).sum())
    finite = recon[np.isfinite(recon)]
    std = float(finite.std()) if finite.size else float("nan")
    mean = float(finite.mean()) if finite.size else float("nan")

    if n_nan or n_inf:
        verdict = f"NON-FINITE output — {n_nan} NaN / {n_inf} Inf pixels; training diverged"
    elif std < 1e-5:
        verdict = (
            f"COLLAPSED — decoder output is constant (~{mean:.3f}); "
            f"{codes_used}/{k} codebook codes used"
        )
    elif codes_used / k < 0.1:
        verdict = (
            f"SEVERE codebook collapse — only {codes_used}/{k} codes used "
            f"(perplexity {float(perplexity.item()):.1f})"
        )
    else:
        verdict = (
            f"OK — output varies across inputs; {codes_used}/{k} codes used, "
            f"perplexity {float(perplexity.item()):.1f}"
        )

    return {
        "verdict": verdict,
        "recon_min": float(finite.min()) if finite.size else float("nan"),
        "recon_max": float(finite.max()) if finite.size else float("nan"),
        "recon_mean": mean,
        "recon_std": std,
        "frac_black": float((recon == 0.0).mean()),
        "frac_white": float((recon >= 1.0 - 1e-6).mean()),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "presigmoid_min": float(np.nanmin(lg)),
        "presigmoid_max": float(np.nanmax(lg)),
        "z_e_std": float(np.asarray(z_e).std()),
        "codes_used": codes_used,
        "num_embeddings": k,
        "perplexity": float(perplexity.item()),
    }


@app.function
def format_diag_table(diag: dict) -> str:
    rows = [
        ("Recon min / max", f"{diag['recon_min']:.6f} / {diag['recon_max']:.6f}"),
        ("Recon mean / std", f"{diag['recon_mean']:.6f} / {diag['recon_std']:.6f}"),
        ("Pixels exactly 0 (black)", f"{100 * diag['frac_black']:.1f}%"),
        ("Pixels >= 1 (white)", f"{100 * diag['frac_white']:.1f}%"),
        ("NaN / Inf pixels", f"{diag['n_nan']} / {diag['n_inf']}"),
        (
            "Decoder pre-sigmoid min / max",
            f"{diag['presigmoid_min']:.1f} / {diag['presigmoid_max']:.1f}",
        ),
        ("Encoder z_e std", f"{diag['z_e_std']:.4f}"),
        (
            "Codebook codes used",
            f"{diag['codes_used']} / {diag['num_embeddings']}",
        ),
        ("Codebook perplexity", f"{diag['perplexity']:.2f}"),
    ]
    body = "\n".join(f"| {name} | {val} |" for name, val in rows)
    return "| Quantity | Value |\n|----------|-------|\n" + body


@app.function
def plot_model_diagnostics(model: nn.Module, x: mx.array, n_show: int = 8):
    x_hat = model(x)[0]
    mx.eval(x_hat)
    orig = np.nan_to_num(np.asarray(x))
    recon = np.nan_to_num(np.asarray(x_hat))
    n_show = min(n_show, orig.shape[0])

    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    for i in range(n_show):
        axes[0, i].imshow(np.clip(orig[i], 0.0, 1.0))
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(recon[i], 0.0, 1.0))
        axes[1, i].axis("off")

        r = recon[i]
        lo, hi = float(r.min()), float(r.max())
        stretched = (r - lo) / (hi - lo + 1e-8)
        axes[2, i].imshow(np.clip(stretched, 0.0, 1.0))
        axes[2, i].axis("off")
        axes[2, i].set_title(f"[{lo:.3f}, {hi:.3f}]", fontsize=7)

    for row, lbl in enumerate(["Original", "Recon (true 0-1)", "Recon (stretched)"]):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(lbl, fontsize=9)

    fig.suptitle("ViT-VQ-VAE decoder output — diagnostics", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(mo):
    _checkpoints = list_checkpoints() or ["<no checkpoints in models/>"]
    _default = (
        "cifar10_vit_vqvae_v1.safetensors"
        if "cifar10_vit_vqvae_v1.safetensors" in _checkpoints
        else _checkpoints[0]
    )
    ckpt_ui = mo.ui.dropdown(options=_checkpoints, value=_default, label="Checkpoint")
    diagnose_btn = mo.ui.run_button(label="Load & Diagnose")
    mo.vstack(
        [mo.md("### Load a saved ViT-VQ-VAE and inspect its output"), ckpt_ui, diagnose_btn]
    )
    return ckpt_ui, diagnose_btn


@app.cell
def _(ckpt_ui, diagnose_btn, mo, test_ds):
    mo.stop(
        not diagnose_btn.value,
        mo.md("Pick a checkpoint and click **Load & Diagnose**."),
    )

    _ckpt_path = models_dir() / ckpt_ui.value
    try:
        _model, _cfg = load_vitvqvae(_ckpt_path)
    except (ValueError, KeyError) as _e:
        _out = mo.md(
            f"**Error loading checkpoint**: `{ckpt_ui.value}` is not a "
            f"ViTVQVAEV1 checkpoint (`{_e}`)."
        )
        mo.output.replace(_out)
    else:
        _raw = np.stack([np.asarray(test_ds[i]["image"]) for i in range(64)])
        _x = normalize_images(_raw)

        _diag = diagnose_model(_model, _x)
        _is_bad = _diag["verdict"].startswith(("NON-FINITE", "COLLAPSED", "SEVERE"))
        _banner_colour = "#b3261e" if _is_bad else "#1e7d32"
        _banner = mo.Html(
            f'<div style="padding:0.6rem 0.9rem;border-radius:6px;'
            f'background:{_banner_colour};color:white;font-weight:600">'
            f'{_diag["verdict"]}</div>'
        )

        _cfg_line = ", ".join(f"{k}={v}" for k, v in _cfg.items())
        mo.output.replace(
            mo.vstack(
                [
                    _banner,
                    mo.md(
                        f"**Checkpoint**: `{ckpt_ui.value}` &nbsp; "
                        f"**inferred config**: {_cfg_line}\n\n"
                        + format_diag_table(_diag)
                    ),
                    plot_model_diagnostics(_model, _x, n_show=8),
                ]
            )
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Results
    """)
    return


@app.function
def plot_loss_curve(
    train_losses: list,
    val_losses: list,
    train_recon_losses: list,
    val_recon_losses: list,
    train_perplexities: list,
    val_perplexities: list,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train")
    axes[0].plot(epochs, val_losses, "r-s", lw=2, ms=4, label="Val")
    axes[0].set_title("Total loss (recon + vq)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_recon_losses, "b-o", lw=2, ms=4, label="Train")
    axes[1].plot(epochs, val_recon_losses, "r-s", lw=2, ms=4, label="Val")
    axes[1].set_title("Reconstruction loss (MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, train_perplexities, "b-o", lw=2, ms=4, label="Train")
    axes[2].plot(epochs, val_perplexities, "r-s", lw=2, ms=4, label="Val")
    axes[2].set_title("Codebook perplexity")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Perplexity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("ViT-VQ-VAE training curves", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    train_losses,
    train_perplexities,
    train_recon_losses,
    val_losses,
    val_perplexities,
    val_recon_losses,
):
    if not train_losses:
        _out = mo.md("_Train first — loss curves will appear here._")
    else:
        _out = plot_loss_curve(
            train_losses,
            val_losses,
            train_recon_losses,
            val_recon_losses,
            train_perplexities,
            val_perplexities,
        )
    _out
    return


@app.function
def plot_reconstructions(model: nn.Module, x: mx.array, class_names, n_show: int = 8):
    x_hat = model(x)[0]
    mx.eval(x_hat)
    orig = np.asarray(x)
    recon = np.asarray(x_hat)
    n_show = min(n_show, orig.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4.5))
    for i in range(n_show):
        axes[0, i].imshow(np.clip(orig[i], 0.0, 1.0))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(recon[i], 0.0, 1.0))
        axes[1, i].axis("off")
    axes[0, 0].axis("on")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].axis("on")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_ylabel("Reconstruction", fontsize=9)
    fig.suptitle(
        f"ViT-VQ-VAE reconstructions ({len(class_names)}-class CIFAR-10)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(mo, test_iter, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first — reconstructions will appear here._")
    else:
        test_iter.reset()
        _batch = next(test_iter)
        _x = mx.array(_batch["image"], dtype=mx.float32)
        _diag = diagnose_model(trained_model, _x)
        _out = mo.vstack(
            [
                mo.md(f"**{_diag['verdict']}**\n\n" + format_diag_table(_diag)),
                plot_reconstructions(trained_model, _x, cifar10_class_names(), n_show=8),
            ]
        )
    _out
    return


@app.function
def plot_codebook_usage(model: nn.Module, data_iter, preprocess_fn, max_batches: int = 20):
    counts = np.zeros(model.num_embeddings, dtype=np.int64)
    data_iter.reset()
    batches_seen = 0
    for batch in data_iter:
        if batches_seen >= max_batches:
            break
        x = preprocess_fn(batch)
        indices = model.encode_to_indices(x)
        mx.eval(indices)
        idx_np = np.asarray(indices).reshape(-1)
        binc = np.bincount(idx_np, minlength=model.num_embeddings)
        counts += binc
        batches_seen += 1

    total = counts.sum()
    freqs = counts / max(total, 1)
    used = int((counts > 0).sum())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(model.num_embeddings), freqs, color="darkorange", edgecolor="black")
    ax.set_xlabel("Codebook index k")
    ax.set_ylabel("Usage frequency")
    ax.set_title(
        f"Codebook usage over {batches_seen} batches — "
        f"{used}/{model.num_embeddings} codes active"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(mo, trained_model, val_iter):
    if trained_model is None:
        _out = mo.md("_Train first — codebook usage will appear here._")
    else:
        _out = plot_codebook_usage(
            trained_model, val_iter, preprocess_image_batch, max_batches=20
        )
    _out
    return


@app.cell
def _(
    beta_ui,
    cv_results,
    embedding_dim_ui,
    mo,
    num_embeddings_ui,
    reference_param_count,
    train_losses,
    train_perplexities,
    train_recon_losses,
    trained_model,
    val_losses,
    val_perplexities,
    val_recon_losses,
):
    if not train_losses or trained_model is None:
        _summary = mo.md(
            f"""
    ### Summary — pending training

    - **Framework**: MLX (channels-last NHWC)
    - **Model**: `ViTVQVAEV1` — ViT encoder + vector quantizer + ViT decoder
    - **Reference parameter count** (embed_dim=256, depth=6/6, K=512, D=64): `{reference_param_count:,}`

    Train the model in Section 5 to populate final losses, perplexity,
    codebook utilization, and CV statistics here.
    """
        )
    else:
        _cv_line = ""
        if isinstance(cv_results, dict) and cv_results:
            _cv_line = (
                f"- **5-fold CV total loss**: "
                f"{cv_results['mean_total']:.4f} +/- {cv_results['std_total']:.4f}\n"
                f"- **5-fold CV perplexity**: "
                f"{cv_results['mean_ppl']:.2f} +/- {cv_results['std_ppl']:.2f}\n"
            )
        _K = int(num_embeddings_ui.value)
        _D = int(embedding_dim_ui.value)
        _beta = float(beta_ui.value)
        _final_ppl = val_perplexities[-1]
        _ppl_ratio = _final_ppl / _K
        _ppl_note = (
            "healthy codebook utilization"
            if _ppl_ratio > 0.4
            else "partial codebook collapse — consider more epochs, "
            "a smaller K, or codebook re-initialization"
        )
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX (channels-last NHWC)
    - **Dataset**: CIFAR-10 (50k train / 10k test, RGB 32x32x3)
    - **Model**: `ViTVQVAEV1` — ViT encoder (patch 4x4 -> 64 tokens -> depth-6
      transformer -> pre-code linear), vector quantizer, mirror ViT decoder
      (post-code linear + pos embed -> depth-6 transformer -> patch head ->
      fold back to 32x32x3)
    - **Codebook**: K = {_K} entries of D = {_D}-dim vectors
    - **Commitment cost (beta)**: {_beta}
    - **Parameters** (reference config): {reference_param_count:,}

    | Metric | Final Train | Final Val |
    |--------|-------------|-----------|
    | Total loss | {train_losses[-1]:.4f} | {val_losses[-1]:.4f} |
    | Reconstruction (MSE) | {train_recon_losses[-1]:.4f} | {val_recon_losses[-1]:.4f} |
    | Perplexity | {train_perplexities[-1]:.2f} | {val_perplexities[-1]:.2f} |

    {_cv_line}
    **Codebook utilization**: final val perplexity is
    {_final_ppl:.2f} out of a maximum of {_K} — {_ppl_note}. The
    codebook-usage bar chart above provides a per-code histogram to
    visually confirm this. Because the VQ-VAE loss has no KL term
    (uniform categorical prior contributes only a constant `log K`), the
    failure mode this model can suffer from is **codebook collapse**
    (few active codes) rather than the posterior collapse a continuous
    VAE can exhibit — perplexity is what monitors it.
    """
        )
    _summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 10 — Code Prior & Interactive Sampling

    A vanilla VQ-VAE cannot *sample* on its own: quantized latents are
    discrete indices with no learned distribution. Following van den Oord
    et al. (2017, Section 4.1), we fit a small **autoregressive
    transformer prior** over the discrete `8 x 8` code grid (flattened in
    raster order). Once trained, sampling proceeds token-by-token from
    this prior; the resulting `(8, 8)` index grid is fed into the ViT
    decoder to render a novel 32x32 RGB image.
    """)
    return


@app.class_definition
class CausalSelfAttentionV1(nn.Module):
    def __init__(self, dims: int = 256, num_heads: int = 4):
        super().__init__()
        assert dims % num_heads == 0, "dims must be divisible by num_heads"
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dims, dims * 3)
        self.proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.triu(mx.full((n, n), -1e9), k=1)
        scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.proj(out)


@app.class_definition
class CausalTransformerBlockV1(nn.Module):
    def __init__(
        self,
        dims: int = 256,
        num_heads: int = 4,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = CausalSelfAttentionV1(dims, num_heads)
        self.norm2 = nn.LayerNorm(dims)
        self.mlp = MultiLayerPerceptronBlockV1(dims, mlp_expansion)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@app.class_definition
class CodePriorTransformerV1(nn.Module):
    def __init__(
        self,
        vocab_size: int = 512,
        seq_len: int = 64,
        d_model: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.bos = mx.zeros((1, 1, d_model))
        self.pos_embed = mx.zeros((1, seq_len, d_model))
        self.blocks = [
            CausalTransformerBlockV1(d_model, num_heads, mlp_expansion)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        b = tokens.shape[0]
        h = self.token_embed(tokens)
        bos = mx.broadcast_to(self.bos, (b, 1, self.d_model))
        h = mx.concatenate([bos, h[:, :-1, :]], axis=1)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


@app.function
def extract_code_dataset(vqvae, data_iter, preprocess_fn, max_batches=None):
    codes = []
    data_iter.reset()
    n = 0
    for batch in data_iter:
        if max_batches is not None and n >= max_batches:
            break
        x = preprocess_fn(batch)
        idx = vqvae.encode_to_indices(x)
        mx.eval(idx)
        arr = np.asarray(idx).reshape(x.shape[0], -1)
        codes.append(arr)
        n += 1
    if not codes:
        return np.zeros((0, vqvae.grid_size * vqvae.grid_size), dtype=np.int32)
    return np.concatenate(codes, axis=0).astype(np.int32)


@app.function
def compute_prior_loss(prior: nn.Module, tokens: mx.array) -> mx.array:
    logits = prior(tokens)
    return mx.mean(
        nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
        )
    )


@app.function
def run_prior_train_epoch(
    prior: nn.Module,
    optimizer,
    codes: np.ndarray,
    batch_size: int,
    seed: int,
) -> float:
    loss_and_grad_fn = nn.value_and_grad(prior, compute_prior_loss)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(codes.shape[0])
    total = 0.0
    n_batches = 0
    for i in range(0, codes.shape[0], batch_size):
        idx = perm[i : i + batch_size]
        batch = mx.array(codes[idx].astype(np.int32))
        loss, grads = loss_and_grad_fn(prior, batch)
        optimizer.update(prior, grads)
        mx.eval(loss, prior.parameters())
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


@app.function
def run_prior_evaluate(prior: nn.Module, codes: np.ndarray, batch_size: int) -> float:
    total = 0.0
    n_batches = 0
    for i in range(0, codes.shape[0], batch_size):
        batch = mx.array(codes[i : i + batch_size].astype(np.int32))
        loss = compute_prior_loss(prior, batch)
        mx.eval(loss)
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


@app.function
def sample_code_grids(
    prior: nn.Module,
    n_samples: int,
    grid_size: int,
    temperature: float,
    top_k: int,
) -> mx.array:
    seq_len = grid_size * grid_size
    vocab = int(prior.vocab_size)
    seq_np = np.zeros((n_samples, seq_len), dtype=np.int32)
    temp = max(float(temperature), 1e-6)
    for t in range(seq_len):
        logits = prior(mx.array(seq_np))[:, t, :] / temp
        if top_k and 0 < top_k < vocab:
            sorted_logits = mx.sort(logits, axis=-1)
            kth = sorted_logits[:, -top_k]
            logits = mx.where(logits < kth[:, None], mx.array(-1e9), logits)
        next_tok = mx.random.categorical(logits)
        mx.eval(next_tok)
        seq_np[:, t] = np.asarray(next_tok).astype(np.int32)
    return mx.array(seq_np).reshape(n_samples, grid_size, grid_size)


@app.function
def plot_generated_grid(images: np.ndarray, title: str, cols: int = 8):
    n = images.shape[0]
    cols = max(1, min(n, cols))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes_flat = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n:
            ax.imshow(np.clip(images[i], 0.0, 1.0))
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


@app.function
def list_prior_checkpoints() -> list[str]:
    d = models_dir()
    if not d.is_dir():
        return []
    names = []
    for p in d.glob("*.safetensors"):
        if "prior" in p.name.lower() or "code_prior" in p.name.lower():
            names.append(p.name)
    for p in d.glob("*.npz"):
        if "prior" in p.name.lower() or "code_prior" in p.name.lower():
            names.append(p.name)
    return sorted(names)


@app.function
def load_code_prior(path, vocab_size: int, seq_len: int):
    state = mx.load(str(path))
    d_model = int(state["token_embed.weight"].shape[1])
    depths = [
        int(k.split(".")[1]) for k in state if k.startswith("blocks.")
    ]
    depth = 1 + max(depths) if depths else 4
    prior = CodePriorTransformerV1(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        depth=depth,
    )
    prior.load_weights(str(path))
    mx.eval(prior.parameters())
    return prior


@app.cell
def _(mo):
    prior_lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="Prior LR",
    )
    prior_epochs_ui = mo.ui.slider(1, 50, value=15, step=1, label="Prior Epochs")
    prior_bs_ui = mo.ui.dropdown(
        options=[128, 256, 512], value=256, label="Prior Batch Size"
    )
    prior_depth_ui = mo.ui.dropdown(
        options=[2, 4, 6], value=4, label="Prior Depth"
    )
    train_prior_btn = mo.ui.run_button(label="Train Code Prior")
    mo.vstack(
        [
            mo.md("### Code prior hyperparameters"),
            mo.hstack([prior_lr_ui, prior_epochs_ui, prior_bs_ui, prior_depth_ui]),
            train_prior_btn,
        ]
    )
    return (
        prior_bs_ui,
        prior_depth_ui,
        prior_epochs_ui,
        prior_lr_ui,
        train_prior_btn,
    )


@app.cell
def _(mo):
    _options = ["in-session (trained above)"] + list_checkpoints()
    prior_vqvae_source_ui = mo.ui.dropdown(
        options=_options,
        value=_options[0],
        label="VQ-VAE source for prior training",
    )
    prior_vqvae_source_ui
    return (prior_vqvae_source_ui,)


@app.cell
def _(
    mo,
    prior_bs_ui,
    prior_depth_ui,
    prior_epochs_ui,
    prior_lr_ui,
    prior_vqvae_source_ui,
    train_iter,
    train_prior_btn,
    trained_model,
    val_iter,
):
    prior_train_losses = []
    prior_val_losses = []
    trained_prior = None
    prior_vqvae = None

    _source_value = prior_vqvae_source_ui.value
    _vqvae = None
    if _source_value == "in-session (trained above)":
        _vqvae = trained_model
    else:
        _ckpt_path = models_dir() / _source_value
        try:
            _vqvae, _ = load_vitvqvae(_ckpt_path)
        except (ValueError, KeyError) as _e:
            mo.output.replace(
                mo.md(
                    f"**Error loading VQ-VAE checkpoint** `{_source_value}`: "
                    f"`{_e}`. Cannot train prior."
                )
            )
            _vqvae = None

    if _vqvae is None:
        mo.output.replace(
            mo.md(
                "_Train a VQ-VAE in Section 5, or pick a valid checkpoint above._"
            )
        )
    elif not train_prior_btn.value:
        mo.output.replace(
            mo.md(
                "Configure the code-prior hyperparameters above and click "
                "**Train Code Prior**."
            )
        )
    else:
        mo.output.replace(mo.md("Extracting code dataset from the VQ-VAE..."))
        _train_codes = extract_code_dataset(
            _vqvae, train_iter, preprocess_image_batch, max_batches=200
        )
        _val_codes = extract_code_dataset(
            _vqvae, val_iter, preprocess_image_batch, max_batches=40
        )
        _seq_len = int(_vqvae.grid_size ** 2)
        _prior = CodePriorTransformerV1(
            vocab_size=int(_vqvae.num_embeddings),
            seq_len=_seq_len,
            d_model=256,
            depth=int(prior_depth_ui.value),
            num_heads=4,
            mlp_expansion=4,
        )
        mx.eval(_prior.parameters())
        _optimizer = optim.AdamW(learning_rate=float(prior_lr_ui.value))
        _bs = int(prior_bs_ui.value)
        _n_epochs = int(prior_epochs_ui.value)
        for _epoch in range(_n_epochs):
            _tl = run_prior_train_epoch(_prior, _optimizer, _train_codes, _bs, seed=_epoch)
            _vl = run_prior_evaluate(_prior, _val_codes, _bs)
            prior_train_losses.append(_tl)
            prior_val_losses.append(_vl)
            mo.output.replace(
                mo.md(
                    f"**Prior epoch {_epoch + 1}/{_n_epochs}** — "
                    f"train nll: {_tl:.4f} (ppl {math.exp(_tl):.2f}) | "
                    f"val nll: {_vl:.4f} (ppl {math.exp(_vl):.2f})"
                )
            )
        trained_prior = _prior
        prior_vqvae = _vqvae
        mo.output.replace(
            mo.md(
                f"**Prior training complete!** Final train nll: "
                f"{prior_train_losses[-1]:.4f} | val nll: {prior_val_losses[-1]:.4f} "
                f"(val ppl: {math.exp(prior_val_losses[-1]):.2f})."
            )
        )
    return prior_train_losses, prior_val_losses, prior_vqvae, trained_prior


@app.function
def plot_prior_loss_curve(prior_train_losses: list, prior_val_losses: list):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(prior_train_losses) + 1)
    axes[0].plot(epochs, prior_train_losses, "b-o", lw=2, ms=4, label="Train")
    axes[0].plot(epochs, prior_val_losses, "r-s", lw=2, ms=4, label="Val")
    axes[0].set_title("Code prior negative log-likelihood")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL (nats/token)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [math.exp(v) for v in prior_train_losses], "b-o", lw=2, ms=4, label="Train")
    axes[1].plot(epochs, [math.exp(v) for v in prior_val_losses], "r-s", lw=2, ms=4, label="Val")
    axes[1].set_title("Code prior perplexity (exp NLL)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Code prior training curves", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, prior_train_losses, prior_val_losses):
    if not prior_train_losses:
        _out = mo.md("_Train the code prior first — loss curves will appear here._")
    else:
        _out = plot_prior_loss_curve(prior_train_losses, prior_val_losses)
    _out
    return


@app.cell
def _(mo):
    _vqvae_options = ["in-session"] + list_checkpoints()
    _prior_options = ["in-session"] + list_prior_checkpoints()
    sample_vqvae_source_ui = mo.ui.dropdown(
        options=_vqvae_options, value=_vqvae_options[0], label="Sampling VQ-VAE"
    )
    sample_prior_source_ui = mo.ui.dropdown(
        options=_prior_options, value=_prior_options[0], label="Sampling Prior"
    )
    temperature_ui = mo.ui.slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
    top_k_ui = mo.ui.slider(0, 512, value=64, step=8, label="Top-k (0 = off)")
    n_samples_ui = mo.ui.slider(4, 32, value=16, step=4, label="Samples")
    sample_btn = mo.ui.run_button(label="Sample")
    mo.vstack(
        [
            mo.md("### Sampling controls"),
            mo.hstack([sample_vqvae_source_ui, sample_prior_source_ui]),
            mo.hstack([temperature_ui, top_k_ui, n_samples_ui]),
            sample_btn,
        ]
    )
    return (
        n_samples_ui,
        sample_btn,
        sample_prior_source_ui,
        sample_vqvae_source_ui,
        temperature_ui,
        top_k_ui,
    )


@app.cell
def _(
    mo,
    n_samples_ui,
    prior_vqvae,
    sample_btn,
    sample_prior_source_ui,
    sample_vqvae_source_ui,
    temperature_ui,
    top_k_ui,
    trained_model,
    trained_prior,
):
    _vqvae_source = sample_vqvae_source_ui.value
    _prior_source = sample_prior_source_ui.value

    _vqvae = None
    if _vqvae_source == "in-session":
        _vqvae = trained_model if trained_model is not None else prior_vqvae
    else:
        try:
            _vqvae, _ = load_vitvqvae(models_dir() / _vqvae_source)
        except (ValueError, KeyError) as _e:
            _vqvae = None

    _prior = None
    if _vqvae is not None:
        if _prior_source == "in-session":
            _prior = trained_prior
        else:
            try:
                _prior = load_code_prior(
                    models_dir() / _prior_source,
                    vocab_size=int(_vqvae.num_embeddings),
                    seq_len=int(_vqvae.grid_size ** 2),
                )
            except (ValueError, KeyError) as _e:
                _prior = None

    if _vqvae is None:
        _out = mo.md(
            "_No VQ-VAE available. Train in Section 5 or pick a checkpoint._"
        )
    elif _prior is None:
        _out = mo.md(
            "_No code prior available. Train the prior above or pick a checkpoint._"
        )
    elif not sample_btn.value:
        _out = mo.md("Configure and click **Sample** to generate novel images.")
    else:
        _n = int(n_samples_ui.value)
        _T = float(temperature_ui.value)
        _topk = int(top_k_ui.value)
        mo.output.replace(
            mo.md(
                f"Sampling {_n} images (T={_T:.2f}, top_k={_topk})..."
            )
        )
        _grids = sample_code_grids(
            _prior, _n, int(_vqvae.grid_size), _T, _topk
        )
        _imgs = _vqvae.decode_from_indices(_grids)
        mx.eval(_imgs)
        _images_np = np.asarray(_imgs)
        _title = f"Prior samples — T={_T:.2f}, top_k={_topk}, n={_n}"
        _out = plot_generated_grid(_images_np, _title)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 11 — Save Trained Models

    Persist the trained `ViTVQVAEV1` and `CodePriorTransformerV1` weights
    to the project's `models/` directory. The file extension chosen
    determines the on-disk format: `.safetensors` or `.npz` (both
    natively supported by `nn.Module.save_weights`).
    """)
    return


@app.cell
def _(mo):
    vqvae_filename_ui = mo.ui.text(
        value="cifar10_vit_vqvae_v1.safetensors",
        label="VQ-VAE filename (saved into models/)",
        full_width=True,
    )
    save_vqvae_btn = mo.ui.run_button(label="Save VQ-VAE")
    mo.vstack([vqvae_filename_ui, save_vqvae_btn])
    return save_vqvae_btn, vqvae_filename_ui


@app.cell
def _(mo, save_vqvae_btn, trained_model, vqvae_filename_ui):
    if trained_model is None:
        _out = mo.md("_Train the VQ-VAE first (Section 5)._")
    elif not save_vqvae_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save VQ-VAE** to write weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / vqvae_filename_ui.value
        trained_model.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** VQ-VAE weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    prior_filename_ui = mo.ui.text(
        value="cifar10_code_prior_v1.safetensors",
        label="Code prior filename (saved into models/)",
        full_width=True,
    )
    save_prior_btn = mo.ui.run_button(label="Save Code Prior")
    mo.vstack([prior_filename_ui, save_prior_btn])
    return prior_filename_ui, save_prior_btn


@app.cell
def _(mo, prior_filename_ui, save_prior_btn, trained_prior):
    if trained_prior is None:
        _out = mo.md("_Train the code prior first (Section 10)._")
    elif not save_prior_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save Code Prior** to write weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / prior_filename_ui.value
        trained_prior.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Code prior weights written to `{_save_path}`.")
    _out
    return


if __name__ == "__main__":
    app.run()
