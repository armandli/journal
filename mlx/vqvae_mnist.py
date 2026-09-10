import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
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
    mo.md(r"""
    # Vector Quantized Variational Autoencoder (VQ-VAE) on MNIST — MLX

    ## Research Goal

    Train a **Vector Quantized Variational Autoencoder (VQ-VAE)** on the
    MNIST handwritten digit dataset using Apple's **MLX** framework,
    following:

    > van den Oord, A., Vinyals, O. and Kavukcuoglu, K. (2017).
    > _Neural Discrete Representation Learning._ arXiv:1711.00937.
    > <https://arxiv.org/abs/1711.00937>

    The network learns a **discrete** latent representation of 28x28
    grayscale digit images using a convolutional encoder, a
    finite codebook of learned embedding vectors, and a mirror
    convolutional decoder that reconstructs the input from the
    quantized codes.

    ### How VQ-VAE differs from a standard (continuous-latent) VAE

    - The encoder output at each spatial position is **quantized** to
      the nearest of `K` learned codebook vectors — the latent
      representation is discrete, not Gaussian.
    - The argmin nearest-neighbor lookup is non-differentiable; the
      **straight-through estimator** (identity in the forward pass,
      routes gradients past the quantizer) is used to train the
      encoder.
    - There is **no KL divergence** term in the training objective —
      under a uniform categorical prior the KL is the constant
      `log K`, which vanishes from the gradient. This means the model
      cannot suffer from **posterior collapse** the way a continuous
      VAE can.
    - The training loss is a sum of three terms: the **reconstruction
      term** — a Bernoulli negative log-likelihood (binary cross-entropy
      on `[0, 1]` pixels, evaluated from the decoder logits so the
      gradient does not vanish when the decoder saturates) — the
      **codebook loss** that pulls codes toward their assigned encoder
      outputs, and the **commitment loss** (weighted by `beta = 0.25`)
      that pulls encoder outputs toward their assigned codes.

    ### Notebook Outline

    1. Title & Research Goal
    2. Data Exploration
    3. Dataset Creation
    4. Model Definition
    5. Training
    6. Hyperparameter Search (Optional)
    7. Validation & Cross-Validation
    8. Saved-Model Diagnostics
    9. Results
    10. Save Trained Model
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

    MNIST is loaded via `mlx.data.datasets.load_mnist` as an `mlx.data`
    Buffer. Each sample is a dict with:

    - `image` — `uint8` array shaped `(28, 28, 1)` (channels-last)
    - `label` — scalar `uint8` in `[0, 9]`

    | Split | Size |
    |-------|------|
    | Train (raw, 60k) | {len(train_ds):,} |
    | Test | {len(test_ds):,} |

    Section 3 carves an 85/15 train/val split from the 60k train buffer.
    Labels are shown only for exploration — they are **not** used by
    the VQ-VAE loss.
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
        axes[r, c].set_title(f"{label}", fontsize=9)
        axes[r, c].axis("off")
    fig.suptitle("MNIST training samples (labels shown as titles)", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_sample_grid(train_ds)
    return


@app.function
def plot_class_distribution(dataset):
    labels = np.array(
        [int(np.array(dataset[i]["label"]).item()) for i in range(len(dataset))]
    )
    counts = np.bincount(labels, minlength=10)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(10), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(np.arange(10))
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
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.function
def normalize_images(x) -> mx.array:
    return mx.array(x, dtype=mx.float32) / 255.0


@app.function
def make_datasets(
    train_ds,
    test_ds,
    batch_size: int,
    val_fraction: float = 0.15,
    seed: int = 0,
):
    def _normalize(x):
        return x.astype("float32") / 255.0

    n_total = len(train_ds)
    n_val = int(round(n_total * val_fraction))
    perm = np.random.default_rng(seed).permutation(n_total).tolist()
    val_buf = train_ds.perm(perm[:n_val])
    train_buf = train_ds.perm(perm[n_val:])

    train_iter = (
        train_buf
        .to_stream()
        .key_transform("image", _normalize)
        .shuffle(8192)
        .batch(batch_size)
    )
    val_iter = (
        val_buf
        .to_stream()
        .key_transform("image", _normalize)
        .batch(batch_size)
    )
    test_iter = (
        test_ds
        .to_stream()
        .key_transform("image", _normalize)
        .batch(batch_size)
    )
    return train_iter, val_iter, test_iter, len(train_buf), len(val_buf)


@app.cell
def _(test_ds, train_ds):
    (
        default_train_iter,
        default_val_iter,
        default_test_iter,
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
    60k train buffer (`Buffer.perm`) — train and val samples never overlap.

    - **Train** (85% of raw 60k): {default_n_train:,}
    - **Val** (15% of raw 60k, held out): {default_n_val:,}
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
    ## Section 4 — Model Definition
    """)
    return


@app.class_definition
class ConvolutionBlockV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.conv(x))


@app.class_definition
class ResidualBlockV1(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.conv3 = nn.Conv2d(
            channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv1 = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, stride=1, padding=0
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(x)
        h = nn.relu(self.conv3(h))
        h = self.conv1(h)
        return x + h


@app.class_definition
class ConvEncoderV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_residual_blocks: int = 2,
        residual_hidden_channels: int = 32,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.down1 = ConvolutionBlockV1(
            in_channels, base_channels // 2, kernel_size=4, stride=2, padding=1
        )
        self.down2 = ConvolutionBlockV1(
            base_channels // 2, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.pre_res = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, stride=1, padding=1
        )
        self.res_blocks = [
            ResidualBlockV1(base_channels, residual_hidden_channels)
            for _ in range(num_residual_blocks)
        ]
        self.pre_vq = nn.Conv2d(
            base_channels, embedding_dim, kernel_size=1, stride=1, padding=0
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.down1(x)
        h = self.down2(h)
        h = self.pre_res(h)
        for block in self.res_blocks:
            h = block(h)
        h = nn.relu(h)
        return self.pre_vq(h)


@app.class_definition
class VectorQuantizerV1(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 128,
        embedding_dim: int = 32,
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
class ConvDecoderV1(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        num_residual_blocks: int = 2,
        residual_hidden_channels: int = 32,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.post_vq = nn.Conv2d(
            embedding_dim, base_channels, kernel_size=3, stride=1, padding=1
        )
        self.res_blocks = [
            ResidualBlockV1(base_channels, residual_hidden_channels)
            for _ in range(num_residual_blocks)
        ]
        self.up1 = nn.ConvTranspose2d(
            base_channels,
            base_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels // 2,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def decode_logits(self, z_q: mx.array) -> mx.array:
        h = self.post_vq(z_q)
        for block in self.res_blocks:
            h = block(h)
        h = nn.relu(h)
        h = nn.relu(self.up1(h))
        return self.up2(h)

    def __call__(self, z_q: mx.array) -> mx.array:
        return mx.sigmoid(self.decode_logits(z_q))


@app.class_definition
class VQVAEV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_residual_blocks: int = 2,
        residual_hidden_channels: int = 32,
        num_embeddings: int = 128,
        embedding_dim: int = 32,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.encoder = ConvEncoderV1(
            in_channels=in_channels,
            base_channels=base_channels,
            num_residual_blocks=num_residual_blocks,
            residual_hidden_channels=residual_hidden_channels,
            embedding_dim=embedding_dim,
        )
        self.quantizer = VectorQuantizerV1(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = ConvDecoderV1(
            out_channels=in_channels,
            base_channels=base_channels,
            num_residual_blocks=num_residual_blocks,
            residual_hidden_channels=residual_hidden_channels,
            embedding_dim=embedding_dim,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

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


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def reconstruction_loss(logits: mx.array, x: mx.array) -> mx.array:
    # Bernoulli negative log-likelihood on [0, 1] pixels, computed from logits
    # (numerically stable, non-vanishing gradient even when the decoder
    # saturates — this is what lets a collapsed decoder recover).
    return mx.mean(nn.losses.binary_cross_entropy(logits, x, with_logits=True))


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
    ### Model Architecture — `VQVAEV1`

    Convolutional VQ-VAE, MLX channels-last (`NHWC`). For MNIST 28x28x1
    the encoder produces a 7x7 spatial map, quantized to a codebook of
    `K` embedding vectors of dimension `D`.

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Encoder downsample 1 | `ConvolutionBlockV1` (k=4, s=2, p=1) | `(B, 14, 14, base/2)` |
    | Encoder downsample 2 | `ConvolutionBlockV1` (k=4, s=2, p=1) | `(B, 7, 7, base)` |
    | Encoder pre-residual conv | `Conv2d` 3x3 | `(B, 7, 7, base)` |
    | Encoder residual blocks | `ResidualBlockV1` x N | `(B, 7, 7, base)` |
    | Encoder pre-VQ 1x1 conv | `Conv2d` 1x1 | `(B, 7, 7, D)` |
    | Vector quantizer | `VectorQuantizerV1(K, D)` | `(B, 7, 7, D)` + `(B, 7, 7)` codes |
    | Decoder post-VQ 3x3 conv | `Conv2d` 3x3 | `(B, 7, 7, base)` |
    | Decoder residual blocks | `ResidualBlockV1` x N | `(B, 7, 7, base)` |
    | Decoder upsample 1 | `ConvTranspose2d` (k=4, s=2, p=1) + ReLU | `(B, 14, 14, base/2)` |
    | Decoder upsample 2 | `ConvTranspose2d` (k=4, s=2, p=1) -> logits; `sigmoid` for display | `(B, 28, 28, 1)` |

    **Loss** = `BCE_with_logits(decoder_logits, x) + mean(( sg(z_e) - z_q )^2) + beta * mean(( z_e - sg(z_q) )^2)`

    (`sg` denotes `mx.stop_gradient`; `beta = 0.25` by default. The decoder
    exposes `decode_logits`; `x_hat = sigmoid(logits)` is used only for
    display and pixel statistics, never in the loss.)
    We also track **perplexity** = `exp(-sum(p_k log p_k))` where `p_k`
    is the average codebook usage across a batch — a diagnostic for
    codebook collapse (max value `K`, meaning all codes used equally).

    **Anti-collapse safeguards.** The codebook is initialized from
    `normal(0, 1) / sqrt(D)` (unit-scale vectors, so the nearest-neighbor
    lookup is not degenerate at step 0), and at the end of every training
    epoch any code that received **zero** assignments is re-seeded from a
    random live encoder output (`VectorQuantizerV1.reset_dead_codes`). The
    Section 5 progress line reports `codes N/K` per epoch so collapse is
    visible while training, not only afterward.
    """)
    return


@app.cell
def _(mo):
    _reference_model = VQVAEV1(
        in_channels=1,
        base_channels=64,
        num_residual_blocks=2,
        residual_hidden_channels=32,
        num_embeddings=128,
        embedding_dim=32,
        commitment_cost=0.25,
    )
    mx.eval(_reference_model.parameters())
    reference_param_count = count_parameters(_reference_model)
    mo.md(
        f"""**Reference `VQVAEV1` parameter count** — base_channels=64,
        num_residual_blocks=2, K=128, D=32, beta=0.25:
        `{reference_param_count:,}` parameters."""
    )
    return (reference_param_count,)


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "5e-4": 5e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="Learning Rate",
    )
    epochs_ui = mo.ui.slider(1, 50, value=10, step=1, label="Epochs")
    bs_ui = mo.ui.dropdown(
        options=[32, 64, 128, 256], value=128, label="Batch Size"
    )
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="0",
        label="Weight Decay",
    )
    beta_ui = mo.ui.dropdown(
        options={"0.1": 0.1, "0.25": 0.25, "0.5": 0.5, "1.0": 1.0, "2.0": 2.0},
        value="0.25",
        label="Commitment Cost (beta)",
    )
    num_embeddings_ui = mo.ui.dropdown(
        options=[32, 64, 128, 256, 512],
        value=128,
        label="Num Embeddings (K)",
    )
    embedding_dim_ui = mo.ui.dropdown(
        options=[16, 32, 64],
        value=32,
        label="Embedding Dim (D)",
    )
    train_btn = mo.ui.run_button(label="Train")
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
def preprocess_image_batch(batch) -> mx.array:
    # Contract: `batch` comes from a make_datasets stream, whose key_transform
    # has already scaled "image" to float32 in [0, 1]. This is only a wrap, not
    # a normalization — build raw batches with normalize_images() instead.
    return mx.array(batch["image"], dtype=mx.float32)


@app.function
def run_train_epoch(
    model: nn.Module,
    optimizer,
    train_iter,
    preprocess_fn,
):
    loss_and_grad_fn = nn.value_and_grad(model, compute_vqvae_loss)
    k = model.quantizer.num_embeddings
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
        _pool = mx.array(np.asarray(_z_e).reshape(-1, _z_e.shape[-1]))
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
        mo.output.replace(mo.md("Click **Train** to begin training the VQ-VAE."))
    else:
        _model = VQVAEV1(
            in_channels=1,
            base_channels=64,
            num_residual_blocks=2,
            residual_hidden_channels=32,
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
    base_channels: int = 64,
    weight_decay: float = 0.0,
):
    model = VQVAEV1(
        in_channels=1,
        base_channels=base_channels,
        num_residual_blocks=2,
        residual_hidden_channels=32,
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
def _(hp_search_cb, mo, train_iter, val_iter):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    _space = {
        "num_embeddings": [64, 128, 256],
        "beta": [0.1, 0.25, 1.0],
    }
    _results = []
    _n_epochs = 3
    _lr = 1e-3
    _D = 32
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
        _out = mo.md("_Train the model first._")
    else:
        _val_metrics = evaluate_model(trained_model, val_iter, preprocess_image_batch)
        _test_metrics = evaluate_model(trained_model, test_iter, preprocess_image_batch)
        _out = mo.md(
            f"""
    ### Held-out evaluation of `VQVAEV1`

    | Split | Total Loss | Reconstruction (BCE) | Codebook Loss | Commitment Loss | Perplexity |
    |-------|-----------|--------------------|---------------|-----------------|-----------|
    | Val   | {_val_metrics["total_loss"]:.4f} | {_val_metrics["recon_loss"]:.4f} | {_val_metrics["codebook_loss"]:.4f} | {_val_metrics["commitment_loss"]:.4f} | {_val_metrics["perplexity"]:.2f} |
    | Test  | {_test_metrics["total_loss"]:.4f} | {_test_metrics["recon_loss"]:.4f} | {_test_metrics["codebook_loss"]:.4f} | {_test_metrics["commitment_loss"]:.4f} | {_test_metrics["perplexity"]:.2f} |

    Higher **perplexity** (closer to `K`) indicates broader codebook usage;
    values collapsing toward 1 signal codebook collapse.
    """
        )
    _out
    return


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

    model = VQVAEV1(
        in_channels=1,
        base_channels=64,
        num_residual_blocks=2,
        residual_hidden_channels=32,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
    )
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, compute_vqvae_loss)
    k = model.quantizer.num_embeddings

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
                np.asarray(_pool_z).reshape(-1, _pool_z.shape[-1])
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
    embedding_dim_ui,
    mo,
    num_embeddings_ui,
    train_ds,
    trained_model,
):
    if trained_model is None:
        _out = mo.md("_Train first — 5-fold CV results will appear here._")
        cv_results = {}
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
    | Reconstruction (BCE) | {cv_results["mean_recon"]:.4f} | {cv_results["std_recon"]:.4f} |
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

    Load any checkpoint from `models/` and inspect what the decoder
    actually produces — numeric statistics plus true-scale and
    contrast-stretched reconstruction rows. This section does **not**
    require training in the current session; it is the tool for
    confirming (or ruling out) a collapsed model such as
    `mnist_vqvae_v1.safetensors`, whose decoder emits an all-black
    constant.
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
def infer_vqvae_config(state: dict) -> dict:
    num_embeddings, embedding_dim = (int(v) for v in state["quantizer.codebook"].shape)
    base_channels = int(state["encoder.down2.conv.weight"].shape[0])
    residual_hidden_channels = int(
        state["encoder.res_blocks.0.conv3.weight"].shape[0]
    )
    num_residual_blocks = len(
        {
            k.split(".")[2]
            for k in state
            if k.startswith("encoder.res_blocks.")
        }
    )
    out_channels = int(state["decoder.up2.weight"].shape[0])
    return {
        "in_channels": out_channels,
        "base_channels": base_channels,
        "num_residual_blocks": num_residual_blocks,
        "residual_hidden_channels": residual_hidden_channels,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
    }


@app.function
def load_vqvae(path):
    state = mx.load(str(path))
    cfg = infer_vqvae_config(state)
    model = VQVAEV1(**cfg)
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
        verdict = f"❌ NON-FINITE output — {n_nan} NaN / {n_inf} Inf pixels; training diverged"
    elif std < 1e-5:
        verdict = (
            f"❌ COLLAPSED — decoder output is constant (≈ {mean:.3f}); "
            f"{codes_used}/{k} codebook codes used"
        )
    elif codes_used / k < 0.1:
        verdict = (
            f"⚠ SEVERE codebook collapse — only {codes_used}/{k} codes used "
            f"(perplexity {float(perplexity.item()):.1f})"
        )
    else:
        verdict = (
            f"✓ output varies across inputs — {codes_used}/{k} codes used, "
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
    orig = np.nan_to_num(np.asarray(x)).squeeze(-1)
    recon = np.nan_to_num(np.asarray(x_hat)).squeeze(-1)
    n_show = min(n_show, orig.shape[0])

    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    for i in range(n_show):
        axes[0, i].imshow(orig[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")

        axes[1, i].imshow(recon[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")

        r = recon[i]
        lo, hi = float(r.min()), float(r.max())
        stretched = (r - lo) / (hi - lo + 1e-8)
        axes[2, i].imshow(stretched, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, i].axis("off")
        axes[2, i].set_title(f"[{lo:.3f}, {hi:.3f}]", fontsize=7)

    for row, lbl in enumerate(["Original", "Recon (true 0-1)", "Recon (stretched)"]):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(lbl, fontsize=9)

    fig.suptitle("VQ-VAE decoder output — diagnostics", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(mo):
    _checkpoints = list_checkpoints() or ["<no checkpoints in models/>"]
    _default = (
        "mnist_vqvae_v1.safetensors"
        if "mnist_vqvae_v1.safetensors" in _checkpoints
        else _checkpoints[0]
    )
    ckpt_ui = mo.ui.dropdown(options=_checkpoints, value=_default, label="Checkpoint")
    diagnose_btn = mo.ui.run_button(label="Load & Diagnose")
    mo.vstack(
        [mo.md("### Load a saved VQ-VAE and inspect its output"), ckpt_ui, diagnose_btn]
    )
    return ckpt_ui, diagnose_btn


@app.cell
def _(ckpt_ui, diagnose_btn, mo, test_ds):
    mo.stop(
        not diagnose_btn.value,
        mo.md("Pick a checkpoint and click **Load & Diagnose**."),
    )

    _ckpt_path = models_dir() / ckpt_ui.value
    _model, _cfg = load_vqvae(_ckpt_path)

    _raw = np.stack([np.asarray(test_ds[i]["image"]) for i in range(64)])
    _x = normalize_images(_raw)

    _diag = diagnose_model(_model, _x)
    _is_bad = _diag["verdict"].startswith(("❌", "⚠"))
    _banner_colour = "#b3261e" if _is_bad else "#1e7d32"
    _banner = mo.Html(
        f'<div style="padding:0.6rem 0.9rem;border-radius:6px;'
        f'background:{_banner_colour};color:white;font-weight:600">'
        f'{_diag["verdict"]}</div>'
    )

    _cfg_line = ", ".join(f"{k}={v}" for k, v in _cfg.items())
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
    axes[1].set_title("Reconstruction loss (BCE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, train_perplexities, "b-o", lw=2, ms=4, label="Train")
    axes[2].plot(epochs, val_perplexities, "r-s", lw=2, ms=4, label="Val")
    axes[2].set_title("Codebook perplexity")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Perplexity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("VQ-VAE training curves", fontsize=13)
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
                plot_model_diagnostics(trained_model, _x, n_show=8),
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
        *_head, indices = model(x)
        mx.eval(indices)
        idx_np = np.array(indices).reshape(-1)
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
    - **Model**: `VQVAEV1` — convolutional VQ-VAE per van den Oord et al. (2017)
    - **Reference parameter count** (K=128, D=32, base=64): `{reference_param_count:,}`

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
    - **Dataset**: MNIST (60k train / 10k test, grayscale 28x28)
    - **Model**: `VQVAEV1` — conv encoder (28 -> 14 -> 7) + vector
      quantizer + mirror conv decoder (7 -> 14 -> 28), residual blocks
      inside both stacks
    - **Codebook**: K = {_K} entries of D = {_D}-dim vectors
    - **Commitment cost (beta)**: {_beta}
    - **Parameters** (reference config): {reference_param_count:,}

    | Metric | Final Train | Final Val |
    |--------|-------------|-----------|
    | Total loss | {train_losses[-1]:.4f} | {val_losses[-1]:.4f} |
    | Reconstruction (BCE) | {train_recon_losses[-1]:.4f} | {val_recon_losses[-1]:.4f} |
    | Perplexity | {train_perplexities[-1]:.2f} | {val_perplexities[-1]:.2f} |

    {_cv_line}
    **Codebook utilization**: final val perplexity is
    {_final_ppl:.2f} out of a maximum of {_K} — {_ppl_note}. The
    codebook-usage bar chart above provides a per-code histogram to
    visually confirm this. Compared to a standard continuous-latent
    VAE (`mlx/vae.py`), the VQ-VAE learns a discrete bottleneck and
    has no KL term (uniform categorical prior contributes a constant
    `log K`), so posterior collapse does not occur — the failure mode
    is instead **codebook collapse** (few active codes), which is what
    perplexity monitors.
    """
        )
    _summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 10 — Save Trained Model

    Persist the trained `VQVAEV1` weights to the project's `models/`
    directory. The file extension chosen determines the on-disk format:
    `.safetensors` or `.npz` (both natively supported by
    `nn.Module.save_weights`).
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="mnist_vqvae_v1.safetensors",
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
