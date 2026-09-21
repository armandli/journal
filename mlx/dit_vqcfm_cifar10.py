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
    # Purrception vs. Continuous Flow Matching over a ViT-VQ-VAE Latent — CIFAR-10 (MLX)

    ## Research goal

    Train and compare two class-conditional generative models on
    **CIFAR-10**, both operating on a **shared frozen vector-quantized
    (VQ) latent space** produced by a Vision-Transformer VQ-VAE tokenizer:

    - **Model V1 — Purrception** (Ke et al., *"Purrception"*,
      arXiv:2510.01478, ICLR 2026). A DiT backbone conditioned on the
      noisy latent `z_t`, the flow time `t`, and the class label `y`
      predicts a **categorical posterior over codebook indices** at every
      latent grid position. The continuous velocity used for ODE
      integration is derived analytically from that posterior as the
      **barycenter of predicted code embeddings**:
      `mu = sum_k pi_{t,k} * e_k`, `v = (mu - z_t) / (1 - t)`.
      Training loss is cross-entropy against the ground-truth code
      indices plus a small z-loss regularizer.
    - **Model V2 — Continuous Flow Matching (CFM) baseline**. The same
      DiT backbone, but with a direct **velocity-regression head** and a
      plain **MSE loss** against the linear-interpolation target
      `v = z_1 - z_0`. This is the standard flow-matching baseline used
      in the Purrception paper's own ablation (Figure 3) — it is not
      itself a novel method.

    ### Note on scope — paper 2 substitution

    The task that produced this notebook originally asked for a *"model
    version 2"* adapted from a second paper, **arXiv:2511.11418v1**. That
    paper was fetched and read in full: its actual title is *"Low-Bit,
    High-Fidelity: Optimal Transport Quantization for Flow Matching"* —
    a **post-training weight-compression technique** for flow-matching
    models (quantizing already-trained weights to 2-3 bits via an
    optimal-transport binning of parameter distributions). It does not
    describe a VQ+flow-matching generative architecture, does not cite
    Purrception, and is unrelated to this notebook's comparison. Rather
    than fabricate a fictitious "Purrception v2" architecture that does
    not exist in the literature, the second model here is the **plain
    Continuous Flow Matching baseline** described above — a genuine,
    standard method that mirrors the CFM ablation reported in the
    Purrception paper's own Figure 3. This substitution is documented
    here transparently so a future reader understands the notebook does
    not implement arXiv:2511.11418v1.

    ### Reused, verified infrastructure

    - **ViT-VQ-VAE tokenizer** adapted from `mlx/vqvae_cifar10.py` in
      this repo (van den Oord et al. 2017, *Neural Discrete
      Representation Learning*, arXiv:1711.00937). This tokenizer maps a
      `32 x 32 x 3` image to an `8 x 8` grid of `D`-dim latents, each of
      which is quantized against a shared `K`-entry codebook.
    - **DiT backbone** with AdaLN class+time conditioning adapted from
      `mlx/dit_cfm_cifar10.py` in this repo (Peebles & Xie 2023,
      arXiv:2212.09748).
    - **Euler ODE integration** for sampling.

    ### Notebook outline

    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation
    4. Model definition (VQ-VAE tokenizer + shared DiT backbone +
       Purrception head + CFM head)
    5. Training
       - Stage A: train the ViT VQ-VAE tokenizer
       - Stage B: freeze tokenizer, train Purrception (V1)
       - Stage C: freeze tokenizer, train CFM (V2)
    6. Hyperparameter search (optional) — grid over LR x tau_sample for
       Purrception (paper's own distinctive knob; expect U-shaped
       sensitivity)
    7. Validation & cross-validation — common generation-quality proxy
       (small in-notebook classifier + diversity proxy), 5-fold CV
    8. Results — loss curves, convergence speed, sample grid
       comparison, temperature sweep, model comparison table, summary
    9. Save trained models (tokenizer + both flow heads)
    10. Load saved model & generate — end-to-end sampling from persisted
        checkpoints without retraining
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration
    """)
    return


@app.function
def cifar10_class_names() -> list[str]:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


@app.function
def load_cifar10_arrays(root: str = "../data/cifar10"):
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
    x_train_np, y_train_np, x_test_np, y_test_np, class_names = load_cifar10_arrays(
        "../data/cifar10"
    )
    return class_names, x_test_np, x_train_np, y_test_np, y_train_np


@app.cell
def _(mo, x_test_np, x_train_np):
    mo.md(f"""
    ### Dataset overview

    CIFAR-10 is loaded via `mlx.data.datasets.load_cifar10` from
    `../data/cifar10/`. Each image is a `float32` array shaped
    `(32, 32, 3)` in `[0, 1]`; labels are `int32` in `[0, 9]`.

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
    fig.suptitle("CIFAR-10 training samples", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, x_train_np, y_train_np):
    plot_sample_grid(x_train_np, y_train_np, class_names, n_show=40)
    return


@app.function
def plot_class_distribution(labels: np.ndarray, class_names: list):
    counts = np.bincount(labels.astype(np.int64), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(class_names)), counts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 training-set class distribution")
    for i, cnt in enumerate(counts):
        ax.text(i, cnt + 40, str(int(cnt)), ha="center", fontsize=8)
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
    val_fraction: float = 0.15,
    seed: int = 42,
):
    n_total = x_train.shape[0]
    n_val = int(n_total * val_fraction)
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    x_tr = mx.array(x_train[tr_idx])
    y_tr = mx.array(y_train[tr_idx].astype(np.int32))
    x_val = mx.array(x_train[val_idx])
    y_val = mx.array(y_train[val_idx].astype(np.int32))
    x_te = mx.array(x_test)
    y_te = mx.array(y_test.astype(np.int32))
    return x_tr, y_tr, x_val, y_val, x_te, y_te


@app.function
def make_batches(
    x: mx.array,
    y: mx.array,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int | None = None,
) -> list:
    n = x.shape[0]
    if shuffle:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_idx = mx.array(idx[start:end].astype(np.int32))
        batches.append((x[b_idx], y[b_idx]))
    return batches


@app.cell
def _(x_test_np, x_train_np, y_test_np, y_train_np):
    x_tr, y_tr, x_val, y_val, x_te, y_te = make_datasets(
        x_train_np, y_train_np, x_test_np, y_test_np, val_fraction=0.15
    )
    return x_te, x_tr, x_val, y_tr, y_val


@app.cell
def _(mo, x_te, x_tr, x_val, y_tr):
    sample_batches = make_batches(x_tr, y_tr, batch_size=128, shuffle=True, seed=0)
    xb0, yb0 = sample_batches[0]
    mo.md(
        f"""
        ### Split sizes & one-batch inspection

        Splits are seeded (`seed=42`) so train and val never overlap and
        the same partitioning is used across all three training stages
        (tokenizer, Purrception, CFM).

        - Train (85%): `{x_tr.shape[0]:,}`
        - Val (15%, held out): `{x_val.shape[0]:,}`
        - Test (raw held-out): `{x_te.shape[0]:,}`
        - Number of training batches at bs=128: `{len(sample_batches):,}`
        - `x_batch.shape` = `{tuple(xb0.shape)}`, dtype `{xb0.dtype}`
        - `y_batch.shape` = `{tuple(yb0.shape)}`, dtype `{yb0.dtype}`
        - value range: `[{float(mx.min(xb0).item()):.3f}, {float(mx.max(xb0).item()):.3f}]`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition

    Three model families are defined below, in three tiers of
    `@app.class_definition` cells: the frozen ViT VQ-VAE tokenizer
    (encoder + vector quantizer + decoder), a shared DiT backbone,
    and the two flow-matching heads (Purrception vs. plain CFM) that
    compose that shared backbone.
    """)
    return


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
    def __init__(
        self,
        dims: int = 128,
        num_heads: int = 4,
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
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        code_dim: int = 32,
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
        num_embeddings: int = 256,
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

    def nearest_codes(self, z: mx.array) -> mx.array:
        b, h, w, d = z.shape
        flat = z.reshape(-1, d)
        z_norm = mx.sum(flat * flat, axis=1, keepdims=True)
        c_norm = mx.sum(self.codebook * self.codebook, axis=1, keepdims=True).T
        dot = flat @ self.codebook.T
        distances = z_norm + c_norm - 2.0 * dot
        idx = mx.argmin(distances, axis=1)
        return idx.reshape(b, h, w)

    def __call__(self, z_e: mx.array):
        b, h, w, d = z_e.shape
        flat = z_e.reshape(-1, d)

        z_norm = mx.sum(flat * flat, axis=1, keepdims=True)
        c_norm = mx.sum(self.codebook * self.codebook, axis=1, keepdims=True).T
        dot = flat @ self.codebook.T
        distances = z_norm + c_norm - 2.0 * dot

        encoding_indices = mx.argmin(distances, axis=1)
        z_q_flat = self.codebook[encoding_indices]
        z_q = z_q_flat.reshape(b, h, w, d)

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
            encoding_indices.reshape(b, h, w),
        )


@app.class_definition
class ViTDecoderV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        out_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        code_dim: int = 32,
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
        embed_dim: int = 128,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 4,
        num_embeddings: int = 256,
        embedding_dim: int = 32,
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

    def encode_continuous(self, x: mx.array) -> mx.array:
        return self.encoder(x)

    def encode_to_indices(self, x: mx.array) -> mx.array:
        z_e = self.encoder(x)
        return self.quantizer.nearest_codes(z_e)

    def decode_from_indices(self, indices: mx.array) -> mx.array:
        z_q = self.quantizer.codebook[indices]
        return mx.sigmoid(self.decoder.decode_logits(z_q))

    def decode_from_latent(self, z: mx.array) -> mx.array:
        return mx.sigmoid(self.decoder.decode_logits(z))


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, embed_dim: int = 128, time_scale: float = 1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        # The bank spans 1 -> 1e-4, a range that assumes DDPM integer timesteps
        # t in [0, 1000]. Flow matching feeds t in [0, 1], so every angle is <= 1 rad
        # and the embedding is nearly constant in t (std across t = 0.0208 measured).
        # The model then has to learn a ~50x amplifier just to read the clock, which
        # drives the conditioning path into an unstable high-gain regime.
        self.time_scale = time_scale
        half = embed_dim // 2
        self.freqs = mx.exp(
            -math.log(10000.0)
            * mx.arange(0, half, dtype=mx.float32)
            / max(half, 1)
        )
        # MLX registers every non-underscore mx.array attribute as a TRAINABLE
        # parameter (nn.Module.valid_parameter_filter). A sinusoidal frequency
        # bank is fixed geometry: left unfrozen, AdamW grinds its geometric
        # decay into noise and the time embedding loses multi-scale resolution.
        # freeze() is used here rather than a "_freqs" rename so the key stays
        # in parameters() and existing checkpoints still load.
        self.freeze(keys=["freqs"], recurse=False)

    def __call__(self, t: mx.array) -> mx.array:
        angles = (t * self.time_scale)[:, None] * self.freqs[None, :]
        return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)


@app.class_definition
class AdaptiveLayerNormV1(nn.Module):
    def __init__(self, dim: int = 128, cond_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        scale, shift = mx.split(self.proj(nn.silu(cond))[:, None, :], 2, axis=-1)
        return self.norm(x) * (1.0 + scale) + shift


@app.class_definition
class QKNormAttentionV1(nn.Module):
    """Multi-head attention with QK normalization.

    Replaces `nn.MultiHeadAttention`, which offers no QK normalization. AdaLN feeds
    each block `LayerNorm(x) * (1 + scale)`, so attention logits grow as `scale^2`.
    Nothing bounds `scale`, and sharpening attention lowers the loss, so the optimizer
    drives it up until softmax saturates into a hard argmax (entropy 0) through which
    no gradient flows. `final_norm` then renormalizes the blow-up, so the loss stays
    finite and the failure is silent. Normalizing q and k per head decouples attention
    sharpness from the AdaLN scale (the ViT-22B / SD3 / Flux remedy).

    Parameter count is essentially unchanged: MLX's `MultiHeadAttention` also defaults
    to `bias=False`, so only the two RMSNorm gains (2 * head_dim per block) are added.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, d = x.shape
        heads, hd = self.num_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(b, n, heads, hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, n, heads, hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, heads, hd).transpose(0, 2, 1, 3)
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5), axis=-1)
        return self.out_proj((attn @ v).transpose(0, 2, 1, 3).reshape(b, n, d))


@app.class_definition
class DiTLatentBlockV1(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.attn_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.attn = QKNormAttentionV1(dim, num_heads)
        self.mlp_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        h = self.attn_norm(x, cond)
        x = x + self.attn(h)
        return x + self.mlp(self.mlp_norm(x, cond))


@app.class_definition
class DiffusionTransformerBackboneV1(nn.Module):
    def __init__(
        self,
        grid_size: int = 8,
        latent_dim: int = 32,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        num_tokens = grid_size * grid_size
        self.num_tokens = num_tokens
        self.proj_in = nn.Linear(latent_dim, embed_dim)
        self.pos_embed = mx.zeros((1, num_tokens, embed_dim))
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.blocks = [
            DiTLatentBlockV1(embed_dim, num_heads, mlp_dim, embed_dim)
            for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(embed_dim)

    def __call__(self, z: mx.array, t: mx.array, y: mx.array) -> mx.array:
        b, gh, gw, d = z.shape
        tokens = self.proj_in(z.reshape(b, gh * gw, d)) + self.pos_embed
        cond = self.time_embed(t) + self.class_embed(y)
        for block in self.blocks:
            tokens = block(tokens, cond)
        return self.final_norm(tokens)


@app.class_definition
class PurrceptionFlowHeadV1(nn.Module):
    """Purrception (arXiv:2510.01478): categorical-posterior head over
    VQ codebook indices. Continuous velocity is derived at sample time
    as the barycenter of predicted code embeddings (not a parameter of
    the model itself)."""

    def __init__(
        self,
        grid_size: int = 8,
        latent_dim: int = 32,
        num_codes: int = 256,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.backbone = DiffusionTransformerBackboneV1(
            grid_size=grid_size,
            latent_dim=latent_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
        )
        self.head = nn.Linear(embed_dim, num_codes)

    def __call__(self, z: mx.array, t: mx.array, y: mx.array) -> mx.array:
        h = self.backbone(z, t, y)
        return self.head(h)


@app.class_definition
class ContinuousFlowMatchingHeadV1(nn.Module):
    """Standard Continuous Flow Matching (CFM) baseline: same DiT
    backbone as Purrception, but the head directly regresses the
    continuous velocity `v = z_1 - z_0` under MSE loss."""

    def __init__(
        self,
        grid_size: int = 8,
        latent_dim: int = 32,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.backbone = DiffusionTransformerBackboneV1(
            grid_size=grid_size,
            latent_dim=latent_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
        )
        self.head = nn.Linear(embed_dim, latent_dim)

    def __call__(self, z: mx.array, t: mx.array, y: mx.array) -> mx.array:
        b = z.shape[0]
        h = self.backbone(z, t, y)
        v = self.head(h)
        return v.reshape(b, self.grid_size, self.grid_size, self.latent_dim)


@app.class_definition
class TinyCifarClassifierV1(nn.Module):
    """Small in-notebook CIFAR-10 CNN classifier used as an
    Inception-substitute generation-quality proxy (conditional
    generation accuracy). Not the paper's evaluation protocol."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.pool(nn.relu(self.conv1(x)))
        h = self.pool(nn.relu(self.conv2(h)))
        h = self.pool(nn.relu(self.conv3(h)))
        b = h.shape[0]
        h = h.reshape(b, -1)
        h = nn.relu(self.fc1(h))
        return self.fc2(h)


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.cell
def _(mo):
    mo.md(r"""
    ### Architecture summary

    MLX is channels-last (NHWC). Both flow heads operate on the
    tokenizer's `(B, 8, 8, D)` continuous latent grid.

    | Component | Module | Output shape |
    |-----------|--------|--------------|
    | Tokenizer encoder | `ViTEncoderV1` | `(B, 8, 8, D)` continuous latent |
    | Tokenizer quantizer | `VectorQuantizerV1(K, D)` | `(B, 8, 8, D)` + `(B, 8, 8)` codes |
    | Tokenizer decoder | `ViTDecoderV1` | `(B, 32, 32, 3)` image |
    | Shared DiT backbone | `DiffusionTransformerBackboneV1` | `(B, 64, embed_dim)` tokens |
    | Purrception head V1 | `Linear(embed_dim, K)` | `(B, 64, K)` categorical logits |
    | CFM head V2 | `Linear(embed_dim, D)` + reshape | `(B, 8, 8, D)` velocity |

    Defaults chosen for on-device training on Apple Silicon:
    `embed_dim=128, num_heads=4, num_layers=4, K=256, D=32`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What is Purrception's contribution, and what is reused

    **Paper's contribution (`PurrceptionFlowHeadV1` + its loss and
    sampling logic in this notebook):**

    1. Supervising the flow model with **categorical cross-entropy
       over VQ code indices** rather than the standard MSE regression
       on velocity.
    2. Deriving the continuous velocity **analytically from the
       predicted categorical posterior** via the barycenter of
       predicted code embeddings — no separate velocity head:
       `mu_t = sum_k pi_{t,k} e_k` and `v_t = (mu_t - z_t) / (1 - t)`.
    3. **Temperature-controlled sampling** exploiting that categorical
       posterior (paper reports a U-shaped sensitivity to
       `tau_sample`, with an optimum around 0.8-0.9 — Section 6
       sweeps this).
    4. A small **z-loss regularizer** `1e-5 * (logsumexp(logits))^2`
       to prevent logit drift.

    **Reused infrastructure (not Purrception's contribution — used
    because it is standard and battle-tested):**

    - The **ViT VQ-VAE tokenizer** itself (van den Oord et al. 2017,
      arXiv:1711.00937), adapted from this repo's
      `mlx/vqvae_cifar10.py`. Purrception treats the tokenizer as a
      fixed feature extractor.
    - The **DiT backbone** (Peebles & Xie 2023, arXiv:2212.09748),
      adapted from this repo's `mlx/dit_cfm_cifar10.py`.
    - **Euler ODE integration** for sampling.

    **What Model V2 (CFM) shares vs. differs:** V2 uses the *same*
    `DiffusionTransformerBackboneV1` and the *same* frozen
    tokenizer. The differences are exactly the head class
    (`ContinuousFlowMatchingHeadV1`) and the training loss
    (MSE on velocity vs. cross-entropy on code indices). This
    isolates the effect of Purrception's categorical-supervision
    idea from any backbone/tokenizer choice.
    """)
    return


@app.cell
def _():
    reference_tokenizer = ViTVQVAEV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=4,
        mlp_expansion=4,
        num_embeddings=256,
        embedding_dim=32,
        commitment_cost=0.25,
    )
    mx.eval(reference_tokenizer.parameters())
    reference_purrception = PurrceptionFlowHeadV1(
        grid_size=8,
        latent_dim=32,
        num_codes=256,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
    )
    mx.eval(reference_purrception.parameters())
    reference_cfm = ContinuousFlowMatchingHeadV1(
        grid_size=8,
        latent_dim=32,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
    )
    mx.eval(reference_cfm.parameters())
    tokenizer_param_count = count_parameters(reference_tokenizer)
    purrception_param_count = count_parameters(reference_purrception)
    cfm_param_count = count_parameters(reference_cfm)
    return cfm_param_count, purrception_param_count, tokenizer_param_count


@app.cell
def _(cfm_param_count, mo, purrception_param_count, tokenizer_param_count):
    mo.md(f"""
    ### Reference parameter counts

    | Model | Parameters |
    |-------|-----------|
    | `ViTVQVAEV1` (tokenizer, embed_dim=128, depth=4/4, K=256, D=32) | `{tokenizer_param_count:,}` |
    | `PurrceptionFlowHeadV1` (embed_dim=128, num_layers=4, K=256) | `{purrception_param_count:,}` |
    | `ContinuousFlowMatchingHeadV1` (embed_dim=128, num_layers=4, D=32) | `{cfm_param_count:,}` |

    Both flow heads share the same `DiffusionTransformerBackboneV1`
    architecture — the count difference above comes purely from
    Purrception's `Linear(embed_dim, K=256)` head vs. CFM's
    `Linear(embed_dim, D=32)` head.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training

    Three gated stages, each with its own `mo.ui.run_button`:

    - **Stage A** trains the VQ-VAE tokenizer to convergence.
    - **Stage B** freezes the tokenizer and trains **Purrception** on
      top of the tokenizer's continuous latents.
    - **Stage C** freezes the same tokenizer and trains the **CFM**
      baseline for a matched epoch budget, for a fair comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Stage A — Train the ViT VQ-VAE tokenizer

    #### Tokenizer capacity — which knob fixes blocky reconstructions

    Blockiness here is **structural, not a capacity shortfall**. The decoder ends in

    ```python
    self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
    ```

    so every `patch_size x patch_size` block of pixels is emitted by a single linear
    projection of a single token, with no cross-patch mixing afterwards. Neighbouring
    patches are therefore free to disagree at their shared border, and that
    discontinuity is the blockiness you see. Shrinking the patch shrinks the seam.

    Measured over 3 epochs on 8 000 images at `lr=3e-4` (comparative only — the
    absolute numbers are low because the budget is short):

    | Change | Params | Tokens | Recon PSNR | Wall clock |
    |---|---|---|---|---|
    | baseline `patch 4, dim 128, K 256` | 1 627 984 | 64 | 17.01 dB | 9 s |
    | `K 256 -> 1024` | 1 652 560 | 64 | 11.99 dB | 11 s |
    | `dim 128 -> 256` | 6 393 424 | 64 | 17.21 dB | 23 s |
    | **`patch 4 -> 2`** | 1 667 884 | **256** | **19.85 dB** | 58 s |

    **Patch size is the lever.** Halving it bought +2.84 dB for 2.5% more parameters,
    while quadrupling the width bought +0.20 dB for 4x the parameters. Extra width and
    depth refine *within* a patch; they cannot remove a seam between patches.

    Two honest caveats on that table. The larger codebook looks worse only because it
    is **undertrained** at 3 epochs — `K=1024` has 4x as many code vectors to place, and
    its perplexity was still climbing (362 of 1024); give it a real budget before
    judging it. And `patch 2` is not free: it produces **256 tokens instead of 64**, so
    the Stage B and C flow heads pay **16x** the attention cost, which is why this run
    took 6.4x as long.

    A reasonable ladder: try `patch 2` first at the current width; if reconstructions
    are sharp enough but colours band, raise `K`; only then raise width and depth.
    """)
    return


@app.cell
def _(mo):
    tok_lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "5e-4": 5e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Tokenizer LR",
    )
    tok_epochs_ui = mo.ui.slider(1, 50, value=6, step=1, label="Tokenizer Epochs")
    tok_bs_ui = mo.ui.dropdown(options=[64, 128, 256], value=128, label="Batch Size")
    tok_wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="0.0",
        label="Tokenizer Weight Decay",
    )
    tok_patch_ui = mo.ui.dropdown(
        options={"2": 2, "4": 4, "8": 8}, value="4", label="Patch Size"
    )
    tok_dim_ui = mo.ui.dropdown(
        options={"128": 128, "192": 192, "256": 256, "384": 384},
        value="128",
        label="Width (embed_dim)",
    )
    tok_depth_ui = mo.ui.slider(2, 10, value=4, step=1, label="Enc/Dec Depth")
    tok_mlp_ui = mo.ui.dropdown(
        options={"2": 2, "4": 4, "6": 6}, value="4", label="MLP Expansion"
    )
    tok_codes_ui = mo.ui.dropdown(
        options={"256": 256, "512": 512, "1024": 1024, "2048": 2048},
        value="256",
        label="Codebook Size K",
    )
    tok_codedim_ui = mo.ui.dropdown(
        options={"16": 16, "32": 32, "64": 64}, value="32", label="Code Dim"
    )
    tok_train_btn = mo.ui.run_button(label="Train VQ-VAE Tokenizer")
    mo.vstack(
        [
            mo.md("Optimizer settings:"),
            mo.hstack([tok_lr_ui, tok_epochs_ui, tok_bs_ui, tok_wd_ui]),
            mo.md("Tokenizer capacity:"),
            mo.hstack([tok_patch_ui, tok_dim_ui, tok_depth_ui]),
            mo.hstack([tok_mlp_ui, tok_codes_ui, tok_codedim_ui]),
            tok_train_btn,
        ]
    )
    return (
        tok_bs_ui,
        tok_codedim_ui,
        tok_codes_ui,
        tok_depth_ui,
        tok_dim_ui,
        tok_epochs_ui,
        tok_lr_ui,
        tok_mlp_ui,
        tok_patch_ui,
        tok_train_btn,
        tok_wd_ui,
    )


@app.cell
def _(
    mo,
    tok_codedim_ui,
    tok_codes_ui,
    tok_depth_ui,
    tok_dim_ui,
    tok_mlp_ui,
    tok_patch_ui,
):
    _ps = int(tok_patch_ui.value)
    _preview = ViTVQVAEV1(
        image_size=32,
        patch_size=_ps,
        in_channels=3,
        embed_dim=int(tok_dim_ui.value),
        encoder_depth=int(tok_depth_ui.value),
        decoder_depth=int(tok_depth_ui.value),
        num_heads=4,
        mlp_expansion=int(tok_mlp_ui.value),
        num_embeddings=int(tok_codes_ui.value),
        embedding_dim=int(tok_codedim_ui.value),
        commitment_cost=0.25,
    )
    mx.eval(_preview.parameters())
    _tokens = (32 // _ps) ** 2
    _bits = _tokens * math.log2(int(tok_codes_ui.value))
    mo.md(f"""
    **Selected tokenizer** — `{_ps}x{_ps}` patches, width `{int(tok_dim_ui.value)}`,
    depth `{int(tok_depth_ui.value)}`, `K={int(tok_codes_ui.value)}`,
    code dim `{int(tok_codedim_ui.value)}`

    | | |
    |---|---|
    | Tokenizer parameters | `{count_parameters(_preview):,}` |
    | Tokens per image | `{_tokens}` (grid `{32 // _ps}x{32 // _ps}`) |
    | Latent code budget | `{_bits:,.0f}` bits/image ({3072 * 8 / _bits:.1f}x compression vs 8-bit RGB) |
    | Flow-head attention cost | `{(_tokens / 64) ** 2:.0f}x` the 64-token default |

    The flow heads in Stages B and C read `grid_size`, `latent_dim` and `num_codes`
    straight off the trained tokenizer, so they follow these settings automatically.
    Every knob here is recoverable from the saved weights by `infer_tokenizer_config`,
    so checkpoints stay loadable at any capacity.
    """)
    return


@app.function
def reconstruction_loss(logits: mx.array, x: mx.array) -> mx.array:
    return mx.mean((mx.sigmoid(logits) - x) ** 2)


@app.function
def compute_vqvae_loss(model: nn.Module, x: mx.array) -> mx.array:
    _x_hat, logits, vq_loss, _cb, _cm, _ppl, _idx = model(x)
    return reconstruction_loss(logits, x) + vq_loss


@app.function
def run_train_epoch_vqvae(
    model: nn.Module,
    optimizer,
    train_batches: list,
):
    loss_and_grad_fn = nn.value_and_grad(model, compute_vqvae_loss)
    k = model.quantizer.num_embeddings
    d_dim = model.quantizer.embedding_dim
    total_loss = 0.0
    total_recon = 0.0
    total_ppl = 0.0
    n = 0
    hit_counts = np.zeros(k, dtype=np.int64)
    last_x = None
    for xb, _yb in train_batches:
        last_x = xb
        loss, grads = loss_and_grad_fn(model, xb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        _x_hat, logits, _vq, _cb, _cm, ppl, indices = model(xb)
        recon = reconstruction_loss(logits, xb)
        mx.eval(recon, ppl, indices)
        hit_counts += np.bincount(
            np.asarray(indices).reshape(-1), minlength=k
        )
        total_loss += loss.item()
        total_recon += recon.item()
        total_ppl += ppl.item()
        n += 1
    if last_x is not None:
        z_e = model.encoder(last_x)
        mx.eval(z_e)
        pool = mx.array(np.asarray(z_e).reshape(-1, d_dim))
        model.quantizer.reset_dead_codes(pool, hit_counts)
    codes_used = int((hit_counts > 0).sum())
    d = max(n, 1)
    return total_loss / d, total_recon / d, total_ppl / d, codes_used


@app.function
def run_evaluate_vqvae(model: nn.Module, batches: list):
    total_loss = 0.0
    total_recon = 0.0
    total_ppl = 0.0
    n = 0
    for xb, _yb in batches:
        _x_hat, logits, vq_loss, _cb, _cm, ppl, _idx = model(xb)
        recon = reconstruction_loss(logits, xb)
        total = recon + vq_loss
        mx.eval(total, recon, ppl)
        total_loss += total.item()
        total_recon += recon.item()
        total_ppl += ppl.item()
        n += 1
    d = max(n, 1)
    return total_loss / d, total_recon / d, total_ppl / d


@app.cell
def _(
    mo,
    tok_bs_ui,
    tok_codedim_ui,
    tok_codes_ui,
    tok_depth_ui,
    tok_dim_ui,
    tok_epochs_ui,
    tok_lr_ui,
    tok_mlp_ui,
    tok_patch_ui,
    tok_train_btn,
    tok_wd_ui,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    tok_train_losses = []
    tok_val_losses = []
    tok_train_recons = []
    tok_val_recons = []
    tok_train_ppls = []
    tok_val_ppls = []
    trained_tokenizer = None

    if not tok_train_btn.value:
        mo.output.replace(
            mo.md("Click **Train VQ-VAE Tokenizer** to begin Stage A.")
        )
    else:
        tokenizer_model = ViTVQVAEV1(
            image_size=32,
            patch_size=int(tok_patch_ui.value),
            in_channels=3,
            embed_dim=int(tok_dim_ui.value),
            encoder_depth=int(tok_depth_ui.value),
            decoder_depth=int(tok_depth_ui.value),
            num_heads=4,
            mlp_expansion=int(tok_mlp_ui.value),
            num_embeddings=int(tok_codes_ui.value),
            embedding_dim=int(tok_codedim_ui.value),
            commitment_cost=0.25,
        )
        mx.eval(tokenizer_model.parameters())
        tok_optimizer = optim.AdamW(
            learning_rate=float(tok_lr_ui.value),
            weight_decay=float(tok_wd_ui.value),
        )
        tok_n_epochs = int(tok_epochs_ui.value)
        tok_val_batches = make_batches(
            x_val, y_val, batch_size=int(tok_bs_ui.value), shuffle=False
        )
        for tok_epoch in range(tok_n_epochs):
            tok_train_batches = make_batches(
                x_tr, y_tr, batch_size=int(tok_bs_ui.value), shuffle=True, seed=tok_epoch
            )
            tl, tr, tp, tc = run_train_epoch_vqvae(
                tokenizer_model, tok_optimizer, tok_train_batches
            )
            vl, vr, vp = run_evaluate_vqvae(tokenizer_model, tok_val_batches)
            tok_train_losses.append(tl)
            tok_val_losses.append(vl)
            tok_train_recons.append(tr)
            tok_val_recons.append(vr)
            tok_train_ppls.append(tp)
            tok_val_ppls.append(vp)
            mo.output.replace(
                mo.md(
                    f"**Tokenizer epoch {tok_epoch + 1}/{tok_n_epochs}** — "
                    f"train loss {tl:.4f} (recon {tr:.4f}, ppl {tp:.1f}, "
                    f"codes {tc}/256) | val loss {vl:.4f} (recon {vr:.4f}, ppl {vp:.1f})"
                )
            )
        trained_tokenizer = tokenizer_model
        mo.output.replace(
            mo.md(
                f"**Tokenizer training complete.** Final val recon "
                f"{tok_val_recons[-1]:.4f}, val perplexity "
                f"{tok_val_ppls[-1]:.1f} / 256."
            )
        )
    return (trained_tokenizer,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Stage B — Train Purrception (V1)

    The tokenizer trained in Stage A is treated as frozen. For each
    batch:

    ```
    z1 = tokenizer.encoder(x)        # continuous latent, (B, 8, 8, D)
    c  = tokenizer.quantizer(z1).indices  # ground-truth codes, (B, 8, 8)
    z0 ~ N(0, I)  same shape as z1
    t  ~ U(0, 1)
    zt = t*z1 + (1-t)*z0
    logits = PurrceptionFlowHeadV1(zt, t, y)   # (B, 64, K)
    loss = CE(c.flatten, softmax(logits / tau_train).flatten)
         + 1e-5 * mean( logsumexp(logits)^2 )
    ```
    """)
    return


@app.cell
def _(mo):
    purr_lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="Purrception LR",
    )
    purr_epochs_ui = mo.ui.slider(1, 100, value=10, step=1, label="Purrception Epochs")
    purr_bs_ui = mo.ui.dropdown(options=[64, 128, 256], value=128, label="Batch Size")
    purr_wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    purr_tau_train_ui = mo.ui.dropdown(
        options={"0.5": 0.5, "1.0": 1.0, "1.5": 1.5},
        value="1.0",
        label="tau_train",
    )
    purr_train_btn = mo.ui.run_button(label="Train Purrception (V1)")
    mo.vstack(
        [
            mo.md("Hyperparameters for Purrception:"),
            mo.hstack([purr_lr_ui, purr_epochs_ui, purr_bs_ui]),
            mo.hstack([purr_wd_ui, purr_tau_train_ui]),
            purr_train_btn,
        ]
    )
    return (
        purr_bs_ui,
        purr_epochs_ui,
        purr_lr_ui,
        purr_tau_train_ui,
        purr_train_btn,
        purr_wd_ui,
    )


@app.function
def compute_purrception_loss(
    head: nn.Module,
    zt: mx.array,
    t: mx.array,
    y: mx.array,
    codes: mx.array,
    tau_train: float = 1.0,
    z_loss_weight: float = 1e-5,
) -> mx.array:
    logits = head(zt, t, y)
    k = logits.shape[-1]
    scaled = logits / tau_train
    ce = nn.losses.cross_entropy(
        scaled.reshape(-1, k), codes.reshape(-1)
    ).mean()
    lse = mx.logsumexp(logits, axis=-1)
    z_loss = z_loss_weight * mx.mean(lse * lse)
    return ce + z_loss


@app.function
def tokenize_batch(tokenizer: nn.Module, xb: mx.array):
    z1 = tokenizer.encoder(xb)
    codes = tokenizer.quantizer.nearest_codes(z1)
    mx.eval(z1, codes)
    return mx.stop_gradient(z1), mx.stop_gradient(codes)


@app.function
def sample_flow_pair(z1: mx.array):
    z0 = mx.random.normal(shape=z1.shape)
    t = mx.random.uniform(shape=(z1.shape[0],))
    t_view = t.reshape(-1, 1, 1, 1)
    zt = t_view * z1 + (1.0 - t_view) * z0
    return z0, t, zt


@app.function
def run_train_epoch_purrception(
    head: nn.Module,
    tokenizer: nn.Module,
    optimizer,
    train_batches: list,
    tau_train: float = 1.0,
) -> float:
    loss_and_grad_fn = nn.value_and_grad(head, compute_purrception_loss)
    total = 0.0
    n = 0
    for xb, yb in train_batches:
        z1, codes = tokenize_batch(tokenizer, xb)
        _z0, t, zt = sample_flow_pair(z1)
        loss, grads = loss_and_grad_fn(head, zt, t, yb, codes, tau_train)
        optimizer.update(head, grads)
        mx.eval(loss, head.parameters())
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.function
def run_evaluate_purrception(
    head: nn.Module,
    tokenizer: nn.Module,
    batches: list,
    tau_train: float = 1.0,
) -> float:
    total = 0.0
    n = 0
    for xb, yb in batches:
        z1, codes = tokenize_batch(tokenizer, xb)
        _z0, t, zt = sample_flow_pair(z1)
        loss = compute_purrception_loss(head, zt, t, yb, codes, tau_train)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.cell
def _(
    mo,
    purr_bs_ui,
    purr_epochs_ui,
    purr_lr_ui,
    purr_tau_train_ui,
    purr_train_btn,
    purr_wd_ui,
    trained_tokenizer,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    purr_train_losses = []
    purr_val_losses = []
    trained_purrception = None

    if trained_tokenizer is None:
        mo.output.replace(
            mo.md("_Train the VQ-VAE tokenizer in Stage A before training Purrception._")
        )
    elif not purr_train_btn.value:
        mo.output.replace(
            mo.md("Click **Train Purrception (V1)** to begin Stage B.")
        )
    else:
        purr_head = PurrceptionFlowHeadV1(
            grid_size=int(trained_tokenizer.grid_size),
            latent_dim=int(trained_tokenizer.embedding_dim),
            num_codes=int(trained_tokenizer.num_embeddings),
            num_classes=10,
            embed_dim=128,
            num_heads=4,
            mlp_dim=512,
            num_layers=4,
        )
        mx.eval(purr_head.parameters())
        purr_optimizer = optim.AdamW(
            learning_rate=float(purr_lr_ui.value),
            weight_decay=float(purr_wd_ui.value),
        )
        purr_n_epochs = int(purr_epochs_ui.value)
        purr_tau = float(purr_tau_train_ui.value)
        purr_val_batches = make_batches(
            x_val, y_val, batch_size=int(purr_bs_ui.value), shuffle=False
        )
        for purr_epoch in range(purr_n_epochs):
            purr_train_batches = make_batches(
                x_tr, y_tr, batch_size=int(purr_bs_ui.value), shuffle=True, seed=purr_epoch
            )
            ptl = run_train_epoch_purrception(
                purr_head, trained_tokenizer, purr_optimizer, purr_train_batches, purr_tau
            )
            pvl = run_evaluate_purrception(
                purr_head, trained_tokenizer, purr_val_batches, purr_tau
            )
            purr_train_losses.append(ptl)
            purr_val_losses.append(pvl)
            mo.output.replace(
                mo.md(
                    f"**Purrception epoch {purr_epoch + 1}/{purr_n_epochs}** — "
                    f"train CE+zloss {ptl:.4f} | val CE+zloss {pvl:.4f}"
                )
            )
        trained_purrception = purr_head
        mo.output.replace(
            mo.md(
                f"**Purrception training complete.** Final train "
                f"{purr_train_losses[-1]:.4f} | val {purr_val_losses[-1]:.4f}."
            )
        )
    return purr_train_losses, purr_val_losses, trained_purrception


@app.cell
def _(mo):
    mo.md(r"""
    ### Stage C — Train Continuous Flow Matching baseline (V2)

    Same frozen tokenizer, same DiT backbone architecture as Stage
    B. The head regresses velocity directly and the loss is plain
    MSE against `v = z_1 - z_0`.
    """)
    return


@app.cell
def _(mo):
    cfm_lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="CFM LR",
    )
    cfm_epochs_ui = mo.ui.slider(1, 100, value=10, step=1, label="CFM Epochs")
    cfm_bs_ui = mo.ui.dropdown(options=[64, 128, 256], value=128, label="Batch Size")
    cfm_wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    cfm_train_btn = mo.ui.run_button(label="Train CFM (V2)")
    mo.vstack(
        [
            mo.md("Hyperparameters for the CFM baseline:"),
            mo.hstack([cfm_lr_ui, cfm_epochs_ui, cfm_bs_ui, cfm_wd_ui]),
            cfm_train_btn,
        ]
    )
    return cfm_bs_ui, cfm_epochs_ui, cfm_lr_ui, cfm_train_btn, cfm_wd_ui


@app.function
def compute_cfm_loss(
    head: nn.Module,
    zt: mx.array,
    t: mx.array,
    y: mx.array,
    v_target: mx.array,
) -> mx.array:
    v_pred = head(zt, t, y)
    return mx.mean((v_pred - v_target) ** 2)


@app.function
def run_train_epoch_cfm(
    head: nn.Module,
    tokenizer: nn.Module,
    optimizer,
    train_batches: list,
) -> float:
    loss_and_grad_fn = nn.value_and_grad(head, compute_cfm_loss)
    total = 0.0
    n = 0
    for xb, yb in train_batches:
        z1, _codes = tokenize_batch(tokenizer, xb)
        z0, t, zt = sample_flow_pair(z1)
        v_target = z1 - z0
        loss, grads = loss_and_grad_fn(head, zt, t, yb, v_target)
        optimizer.update(head, grads)
        mx.eval(loss, head.parameters())
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.function
def run_evaluate_cfm(head: nn.Module, tokenizer: nn.Module, batches: list) -> float:
    total = 0.0
    n = 0
    for xb, yb in batches:
        z1, _codes = tokenize_batch(tokenizer, xb)
        z0, t, zt = sample_flow_pair(z1)
        v_target = z1 - z0
        loss = compute_cfm_loss(head, zt, t, yb, v_target)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.cell
def _(
    cfm_bs_ui,
    cfm_epochs_ui,
    cfm_lr_ui,
    cfm_train_btn,
    cfm_wd_ui,
    mo,
    trained_tokenizer,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    cfm_train_losses = []
    cfm_val_losses = []
    trained_cfm = None

    if trained_tokenizer is None:
        mo.output.replace(
            mo.md("_Train the VQ-VAE tokenizer in Stage A before training CFM._")
        )
    elif not cfm_train_btn.value:
        mo.output.replace(
            mo.md("Click **Train CFM (V2)** to begin Stage C.")
        )
    else:
        cfm_head = ContinuousFlowMatchingHeadV1(
            grid_size=int(trained_tokenizer.grid_size),
            latent_dim=int(trained_tokenizer.embedding_dim),
            num_classes=10,
            embed_dim=128,
            num_heads=4,
            mlp_dim=512,
            num_layers=4,
        )
        mx.eval(cfm_head.parameters())
        cfm_optimizer = optim.AdamW(
            learning_rate=float(cfm_lr_ui.value),
            weight_decay=float(cfm_wd_ui.value),
        )
        cfm_n_epochs = int(cfm_epochs_ui.value)
        cfm_val_batches = make_batches(
            x_val, y_val, batch_size=int(cfm_bs_ui.value), shuffle=False
        )
        for cfm_epoch in range(cfm_n_epochs):
            cfm_train_batches = make_batches(
                x_tr, y_tr, batch_size=int(cfm_bs_ui.value), shuffle=True, seed=cfm_epoch
            )
            ctl = run_train_epoch_cfm(
                cfm_head, trained_tokenizer, cfm_optimizer, cfm_train_batches
            )
            cvl = run_evaluate_cfm(cfm_head, trained_tokenizer, cfm_val_batches)
            cfm_train_losses.append(ctl)
            cfm_val_losses.append(cvl)
            mo.output.replace(
                mo.md(
                    f"**CFM epoch {cfm_epoch + 1}/{cfm_n_epochs}** — "
                    f"train MSE {ctl:.4f} | val MSE {cvl:.4f}"
                )
            )
        trained_cfm = cfm_head
        mo.output.replace(
            mo.md(
                f"**CFM training complete.** Final train "
                f"{cfm_train_losses[-1]:.4f} | val {cfm_val_losses[-1]:.4f}."
            )
        )
    return cfm_train_losses, cfm_val_losses, trained_cfm


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (Optional)
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(
        label="Enable Hyperparameter Search (Purrception LR x tau_sample sweep)",
        value=False,
    )
    hp_search_cb
    return (hp_search_cb,)


@app.function
def purrception_sample_latents(
    head: nn.Module,
    codebook: mx.array,
    y: mx.array,
    num_steps: int,
    tau_sample: float,
    grid_size: int,
    latent_dim: int,
    t_max: float = 0.999,
) -> mx.array:
    b = y.shape[0]
    z = mx.random.normal(shape=(b, grid_size, grid_size, latent_dim))
    for s in range(num_steps):
        t_val = s / num_steps
        t = mx.full((b,), t_val, dtype=mx.float32)
        logits = head(z, t, y)
        pi = mx.softmax(logits / max(tau_sample, 1e-4), axis=-1)
        mu_flat = pi @ codebook
        mu = mu_flat.reshape(b, grid_size, grid_size, latent_dim)
        denom = max(1.0 - t_val, 1.0 - t_max)
        v = (mu - z) / denom
        z = z + (1.0 / num_steps) * v
        mx.eval(z)
    return z


@app.function
def cfm_sample_latents(
    head: nn.Module,
    y: mx.array,
    num_steps: int,
    grid_size: int,
    latent_dim: int,
) -> mx.array:
    b = y.shape[0]
    dt = 1.0 / num_steps
    z = mx.random.normal(shape=(b, grid_size, grid_size, latent_dim))
    for s in range(num_steps):
        t = mx.full((b,), s * dt, dtype=mx.float32)
        v = head(z, t, y)
        z = z + dt * v
        mx.eval(z)
    return z


@app.function
def latents_to_images(
    tokenizer: nn.Module, z: mx.array, use_hard_quantize: bool = True
) -> mx.array:
    if use_hard_quantize:
        codes = tokenizer.quantizer.nearest_codes(z)
        return tokenizer.decode_from_indices(codes)
    return tokenizer.decode_from_latent(z)


@app.function
def sample_purrception_images(
    tokenizer: nn.Module,
    head: nn.Module,
    y: mx.array,
    num_steps: int,
    tau_sample: float,
) -> mx.array:
    z = purrception_sample_latents(
        head,
        tokenizer.quantizer.codebook,
        y,
        num_steps=num_steps,
        tau_sample=tau_sample,
        grid_size=int(tokenizer.grid_size),
        latent_dim=int(tokenizer.embedding_dim),
    )
    return latents_to_images(tokenizer, z, use_hard_quantize=True)


@app.function
def sample_cfm_images(
    tokenizer: nn.Module,
    head: nn.Module,
    y: mx.array,
    num_steps: int,
) -> mx.array:
    z = cfm_sample_latents(
        head,
        y,
        num_steps=num_steps,
        grid_size=int(tokenizer.grid_size),
        latent_dim=int(tokenizer.embedding_dim),
    )
    return latents_to_images(tokenizer, z, use_hard_quantize=True)


@app.function
def image_diversity_score(images: mx.array) -> float:
    arr = np.asarray(images)
    return float(arr.reshape(arr.shape[0], -1).std(axis=0).mean())


@app.function
def run_purrception_hp_config(
    tokenizer: nn.Module,
    x_tr_arr: mx.array,
    y_tr_arr: mx.array,
    x_val_arr: mx.array,
    y_val_arr: mx.array,
    lr: float,
    tau_sample: float,
    n_epochs: int,
    batch_size: int,
    num_steps: int = 20,
    n_gen: int = 40,
) -> dict:
    head = PurrceptionFlowHeadV1(
        grid_size=int(tokenizer.grid_size),
        latent_dim=int(tokenizer.embedding_dim),
        num_codes=int(tokenizer.num_embeddings),
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
    )
    mx.eval(head.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    val_batches = make_batches(x_val_arr, y_val_arr, batch_size=batch_size, shuffle=False)
    for e in range(n_epochs):
        tr = make_batches(x_tr_arr, y_tr_arr, batch_size=batch_size, shuffle=True, seed=e)
        run_train_epoch_purrception(head, tokenizer, optimizer, tr, tau_train=1.0)
    val_loss = run_evaluate_purrception(head, tokenizer, val_batches, tau_train=1.0)
    y_gen = mx.array(
        np.tile(np.arange(10, dtype=np.int32), int(math.ceil(n_gen / 10)))[:n_gen]
    )
    imgs = sample_purrception_images(tokenizer, head, y_gen, num_steps, tau_sample)
    mx.eval(imgs)
    div = image_diversity_score(imgs)
    return {"val_loss": val_loss, "diversity": div}


@app.cell
def _(hp_search_cb, mo, trained_tokenizer, x_tr, x_val, y_tr, y_val):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    if trained_tokenizer is None:
        mo.output.replace(
            mo.md("_Train the VQ-VAE tokenizer (Stage A) before running the HP search._")
        )
    else:
        hp_lrs = [1e-4, 3e-4, 1e-3]
        hp_taus = [0.6, 0.8, 1.0, 1.2]
        hp_epochs = 3
        hp_batch = 128
        hp_sub_n = min(6000, x_tr.shape[0])
        hp_sub_x = x_tr[:hp_sub_n]
        hp_sub_y = y_tr[:hp_sub_n]
        hp_results = []
        for hp_lr in hp_lrs:
            for hp_tau in hp_taus:
                hp_metrics = run_purrception_hp_config(
                    trained_tokenizer,
                    hp_sub_x,
                    hp_sub_y,
                    x_val,
                    y_val,
                    lr=hp_lr,
                    tau_sample=hp_tau,
                    n_epochs=hp_epochs,
                    batch_size=hp_batch,
                )
                hp_results.append(
                    {
                        "lr": hp_lr,
                        "tau_sample": hp_tau,
                        "val_loss": round(hp_metrics["val_loss"], 4),
                        "sample_diversity": round(hp_metrics["diversity"], 4),
                    }
                )
                mo.output.replace(
                    mo.md(
                        f"lr={hp_lr}, tau={hp_tau}: val={hp_metrics['val_loss']:.4f}, "
                        f"diversity={hp_metrics['diversity']:.4f}"
                    )
                )
        hp_results.sort(key=lambda r: r["val_loss"])
        mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation

    Because V1 (cross-entropy over codes) and V2 (MSE on velocity)
    optimize incomparable losses, evaluation uses a **common
    generation-quality proxy**:

    - A small `TinyCifarClassifierV1` is trained here for a few
      epochs on real CIFAR-10 (~70-80% test accuracy). It serves as
      the graders. **This is a substitute for FID** — an
      Inception-based feature extractor is not readily available
      offline in this environment. It does not reproduce the
      paper's actual FID numbers, only their relative behaviour on
      this notebook's scale.
    - **Conditional generation accuracy**: fraction of samples the
      classifier assigns to the requested class label.
    - **Sample diversity**: per-pixel std across generated samples,
      averaged over all pixel channels — a floor detector for mode
      collapse.

    Then a **5-fold CV** re-trains both heads on training-data folds
    with a reduced schedule, reporting the proxy accuracy per fold.
    """)
    return


@app.function
def compute_classifier_loss(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()


@app.function
def run_train_epoch_classifier(
    model: nn.Module, optimizer, train_batches: list
) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_classifier_loss)
    total = 0.0
    n = 0
    for xb, yb in train_batches:
        loss, grads = loss_and_grad_fn(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.function
def evaluate_classifier_accuracy(model: nn.Module, batches: list) -> float:
    correct = 0
    total = 0
    for xb, yb in batches:
        logits = model(xb)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        correct += int(mx.sum(preds == yb).item())
        total += int(yb.shape[0])
    return correct / max(total, 1)


@app.function
def train_proxy_classifier(
    x_tr_arr: mx.array,
    y_tr_arr: mx.array,
    x_val_arr: mx.array,
    y_val_arr: mx.array,
    n_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    clf = TinyCifarClassifierV1(num_classes=10)
    mx.eval(clf.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    val_batches = make_batches(x_val_arr, y_val_arr, batch_size=batch_size, shuffle=False)
    best_acc = 0.0
    for e in range(n_epochs):
        tr = make_batches(x_tr_arr, y_tr_arr, batch_size=batch_size, shuffle=True, seed=e)
        run_train_epoch_classifier(clf, optimizer, tr)
        acc = evaluate_classifier_accuracy(clf, val_batches)
        best_acc = max(best_acc, acc)
    return clf, best_acc


@app.cell
def _(mo):
    proxy_btn = mo.ui.run_button(label="Train Generation-Quality Proxy Classifier")
    proxy_epochs_ui = mo.ui.slider(1, 15, value=5, step=1, label="Proxy classifier epochs")
    mo.vstack([mo.hstack([proxy_epochs_ui]), proxy_btn])
    return proxy_btn, proxy_epochs_ui


@app.cell
def _(mo, proxy_btn, proxy_epochs_ui, x_tr, x_val, y_tr, y_val):
    proxy_classifier = None
    proxy_val_accuracy = 0.0

    if not proxy_btn.value:
        mo.output.replace(
            mo.md("Click **Train Generation-Quality Proxy Classifier** to fit the grader.")
        )
    else:
        proxy_classifier, proxy_val_accuracy = train_proxy_classifier(
            x_tr, y_tr, x_val, y_val, n_epochs=int(proxy_epochs_ui.value), batch_size=128
        )
        mo.output.replace(
            mo.md(
                f"**Proxy classifier ready** — best val acc "
                f"{proxy_val_accuracy:.4f} (higher is a stricter grader)."
            )
        )
    return (proxy_classifier,)


@app.function
def generation_quality_proxy(
    classifier: nn.Module,
    images: mx.array,
    y: mx.array,
) -> dict:
    logits = classifier(images)
    preds = mx.argmax(logits, axis=-1)
    mx.eval(preds)
    match = int(mx.sum(preds == y).item())
    total = int(y.shape[0])
    diversity = image_diversity_score(images)
    return {
        "cond_gen_accuracy": match / max(total, 1),
        "diversity": diversity,
        "n_samples": total,
    }


@app.function
def evaluate_model(
    kind: str,
    tokenizer: nn.Module,
    head: nn.Module,
    classifier: nn.Module,
    num_steps: int,
    num_per_class: int = 4,
    tau_sample: float = 0.9,
) -> dict:
    y = mx.array(
        np.repeat(np.arange(10, dtype=np.int32), num_per_class)
    )
    if kind == "purrception":
        imgs = sample_purrception_images(tokenizer, head, y, num_steps, tau_sample)
    else:
        imgs = sample_cfm_images(tokenizer, head, y, num_steps)
    mx.eval(imgs)
    return generation_quality_proxy(classifier, imgs, y)


@app.cell
def _(
    mo,
    proxy_classifier,
    trained_cfm,
    trained_purrception,
    trained_tokenizer,
):
    if (
        trained_tokenizer is None
        or trained_purrception is None
        or trained_cfm is None
        or proxy_classifier is None
    ):
        out_val = mo.md(
            "_Complete Stages A, B, C and train the proxy classifier before evaluating._"
        )
    else:
        v1_metrics = evaluate_model(
            "purrception",
            trained_tokenizer,
            trained_purrception,
            proxy_classifier,
            num_steps=25,
            num_per_class=8,
            tau_sample=0.9,
        )
        v2_metrics = evaluate_model(
            "cfm",
            trained_tokenizer,
            trained_cfm,
            proxy_classifier,
            num_steps=25,
            num_per_class=8,
        )
        out_val = mo.md(
            f"""
            ### Held-out generation-quality proxy metrics

            | Model | Conditional-generation accuracy | Sample diversity (px std) | Samples |
            |-------|--------------------------------|---------------------------|---------|
            | Purrception V1 (tau=0.9, 25 steps) | {v1_metrics['cond_gen_accuracy']:.4f} | {v1_metrics['diversity']:.4f} | {v1_metrics['n_samples']} |
            | CFM V2 (25 steps) | {v2_metrics['cond_gen_accuracy']:.4f} | {v2_metrics['diversity']:.4f} | {v2_metrics['n_samples']} |

            Chance-level conditional accuracy is `0.1`. Higher is
            better; diversity that collapses toward `0` indicates mode
            collapse.
            """
        )
    out_val
    return


@app.function
def run_cv_fold_both(
    tokenizer: nn.Module,
    classifier: nn.Module,
    x_fold: mx.array,
    y_fold: mx.array,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_epochs: int,
    batch_size: int,
    num_steps: int,
    tau_sample: float,
) -> dict:
    tr_i = mx.array(train_idx.astype(np.int32))
    va_i = mx.array(val_idx.astype(np.int32))
    x_tr_f = x_fold[tr_i]
    y_tr_f = y_fold[tr_i]
    x_va_f = x_fold[va_i]
    y_va_f = y_fold[va_i]

    grid = int(tokenizer.grid_size)
    d = int(tokenizer.embedding_dim)
    k = int(tokenizer.num_embeddings)

    v1 = PurrceptionFlowHeadV1(
        grid_size=grid,
        latent_dim=d,
        num_codes=k,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
    )
    mx.eval(v1.parameters())
    v1_opt = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)

    v2 = ContinuousFlowMatchingHeadV1(
        grid_size=grid,
        latent_dim=d,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        num_layers=4,
    )
    mx.eval(v2.parameters())
    v2_opt = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)

    for e in range(n_epochs):
        tr_batches = make_batches(x_tr_f, y_tr_f, batch_size=batch_size, shuffle=True, seed=e)
        run_train_epoch_purrception(v1, tokenizer, v1_opt, tr_batches, tau_train=1.0)
        tr_batches = make_batches(x_tr_f, y_tr_f, batch_size=batch_size, shuffle=True, seed=e + 1000)
        run_train_epoch_cfm(v2, tokenizer, v2_opt, tr_batches)

    y_gen = mx.array(np.repeat(np.arange(10, dtype=np.int32), 4))
    imgs_v1 = sample_purrception_images(tokenizer, v1, y_gen, num_steps, tau_sample)
    mx.eval(imgs_v1)
    m_v1 = generation_quality_proxy(classifier, imgs_v1, y_gen)
    imgs_v2 = sample_cfm_images(tokenizer, v2, y_gen, num_steps)
    mx.eval(imgs_v2)
    m_v2 = generation_quality_proxy(classifier, imgs_v2, y_gen)

    va_batches = make_batches(x_va_f, y_va_f, batch_size=batch_size, shuffle=False)
    val_ce = run_evaluate_purrception(v1, tokenizer, va_batches, tau_train=1.0)
    val_mse = run_evaluate_cfm(v2, tokenizer, va_batches)

    return {
        "v1_acc": m_v1["cond_gen_accuracy"],
        "v1_div": m_v1["diversity"],
        "v1_val_ce": val_ce,
        "v2_acc": m_v2["cond_gen_accuracy"],
        "v2_div": m_v2["diversity"],
        "v2_val_mse": val_mse,
    }


@app.cell
def _(mo):
    cv_cb = mo.ui.checkbox(label="Enable 5-Fold Cross-Validation (expensive)", value=False)
    cv_cb
    return (cv_cb,)


@app.cell
def _(cv_cb, mo, proxy_classifier, trained_tokenizer, x_tr, y_tr):
    mo.stop(
        not cv_cb.value,
        mo.md("_Enable 5-fold CV above to run this section (retrains both heads per fold)._"),
    )
    cv_results = {}
    if trained_tokenizer is None or proxy_classifier is None:
        cv_out = mo.md("_Train the tokenizer (Stage A) and proxy classifier first._")
    else:
        k_folds = 5
        cv_n = min(4000, int(x_tr.shape[0]))
        cv_x = x_tr[:cv_n]
        cv_y = y_tr[:cv_n]
        cv_rng = np.random.default_rng(seed=7)
        perm = cv_rng.permutation(cv_n)
        folds = np.array_split(perm, k_folds)
        fold_records = []
        for f in range(k_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != f])
            rec = run_cv_fold_both(
                trained_tokenizer,
                proxy_classifier,
                cv_x,
                cv_y,
                train_idx,
                val_idx,
                n_epochs=2,
                batch_size=128,
                num_steps=15,
                tau_sample=0.9,
            )
            rec["fold"] = f + 1
            fold_records.append(rec)
            mo.output.replace(
                mo.md(
                    f"Fold {f + 1}/{k_folds} — V1 acc {rec['v1_acc']:.4f}, div {rec['v1_div']:.4f} | "
                    f"V2 acc {rec['v2_acc']:.4f}, div {rec['v2_div']:.4f}"
                )
            )
        v1_accs = [r["v1_acc"] for r in fold_records]
        v2_accs = [r["v2_acc"] for r in fold_records]
        v1_divs = [r["v1_div"] for r in fold_records]
        v2_divs = [r["v2_div"] for r in fold_records]
        cv_results = {
            "fold_records": fold_records,
            "v1_acc_mean": float(np.mean(v1_accs)),
            "v1_acc_std": float(np.std(v1_accs)),
            "v2_acc_mean": float(np.mean(v2_accs)),
            "v2_acc_std": float(np.std(v2_accs)),
            "v1_div_mean": float(np.mean(v1_divs)),
            "v2_div_mean": float(np.mean(v2_divs)),
        }
        cv_out = mo.md(
            f"""
            ### {k_folds}-Fold CV on training-data folds (2 epochs/fold)

            | Model | Conditional acc mean +/- std | Diversity mean |
            |-------|------------------------------|----------------|
            | Purrception V1 | {cv_results['v1_acc_mean']:.4f} +/- {cv_results['v1_acc_std']:.4f} | {cv_results['v1_div_mean']:.4f} |
            | CFM V2 | {cv_results['v2_acc_mean']:.4f} +/- {cv_results['v2_acc_std']:.4f} | {cv_results['v2_div_mean']:.4f} |
            """
        )
    cv_out
    return (cv_results,)


@app.function
def diagnose_flow_head_health(head: nn.Module, n_probe: int = 8) -> dict:
    bb = head.backbone
    t = mx.array(np.linspace(0.05, 0.95, n_probe).astype(np.float32))
    y = mx.array(np.zeros(n_probe, dtype=np.int32))
    z = mx.random.normal(shape=(n_probe, bb.grid_size, bb.grid_size, bb.latent_dim))

    def _rms(v):
        return float(mx.sqrt(mx.mean(v.astype(mx.float32) ** 2)))

    sinus = bb.time_embed.layers[0](t)
    cond = bb.time_embed(t) + bb.class_embed(y)
    tokens = bb.proj_in(z.reshape(n_probe, bb.num_tokens, bb.latent_dim)) + bb.pos_embed
    scale, _ = mx.split(bb.blocks[0].attn_norm.proj(nn.silu(cond))[:, None, :], 2, axis=-1)

    entropies = []
    h = tokens
    for block in bb.blocks:
        hn = block.attn_norm(h, cond)
        attn = block.attn
        b, n, _d = hn.shape
        heads, hd = attn.num_heads, attn.head_dim
        q = attn.q_norm(attn.q_proj(hn).reshape(b, n, heads, hd)).transpose(0, 2, 1, 3)
        k = attn.k_norm(attn.k_proj(hn).reshape(b, n, heads, hd)).transpose(0, 2, 1, 3)
        probs = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5), axis=-1)
        entropies.append(float(mx.mean(-mx.sum(probs * mx.log(probs + 1e-9), axis=-1))))
        h = block(h, cond)

    return {
        "time_signal_std": float(mx.mean(mx.std(sinus, axis=0))),
        "cond_rms": _rms(cond),
        "adaln_scale_rms": _rms(scale),
        "final_residual_rms": _rms(h),
        "min_attn_entropy": min(entropies),
        "uniform_attn_entropy": math.log(bb.num_tokens),
    }


@app.cell
def _(mo, trained_cfm, trained_purrception):
    _rows = []
    for _tag, _m in (("Purrception (V1)", trained_purrception), ("CFM (V2)", trained_cfm)):
        if _m is None:
            continue
        _d = diagnose_flow_head_health(_m)
        _v = (
            "DIVERGED"
            if _d["adaln_scale_rms"] > 50 or _d["final_residual_rms"] > 1e3
            else ("SATURATED" if _d["min_attn_entropy"] < 0.2 else "PASS")
        )
        _rows.append(
            f"| {_tag} | `{_d['adaln_scale_rms']:.2f}` | `{_d['final_residual_rms']:.2f}` | "
            f"`{_d['min_attn_entropy']:.3f}` | `{_d['time_signal_std']:.4f}` | **{_v}** |"
        )
    if not _rows:
        _out = mo.md("_Train a flow head (Stage B or C) to run the training-health probe._")
    else:
        _out = mo.md(
            "### Training-Health Probe\n\n"
            "| Model | AdaLN scale RMS | Final residual RMS | Min attn entropy | Time-signal std | Verdict |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(_rows) + "\n\n"
            "Healthy: scale < 50, residual < 1e3, entropy > 0.2 (uniform = `ln 64 = 4.159`), "
            "time-signal std > 0.1.\n\n"
            "A cross-entropy stuck near `ln(256) = 5.5452` with **zero attention entropy** is a "
            "*diverged* model, not an untrained one: AdaLN feeds each block "
            "`LayerNorm(x) * (1 + scale)`, so attention logits grow as `scale^2` until softmax "
            "becomes a hard argmax through which no gradient flows. `final_norm` renormalizes "
            "the blow-up, so the loss stays finite and the failure is silent."
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
def plot_loss_curves(
    purr_train: list,
    purr_val: list,
    cfm_train: list,
    cfm_val: list,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if purr_train:
        ep = range(1, len(purr_train) + 1)
        axes[0].plot(ep, purr_train, "b-o", lw=2, ms=4, label="Purrception train")
        axes[0].plot(ep, purr_val, "r-s", lw=2, ms=4, label="Purrception val")
    axes[0].set_title("Purrception V1 (categorical CE + z-loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (nats)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if cfm_train:
        ep = range(1, len(cfm_train) + 1)
        axes[1].plot(ep, cfm_train, "g-o", lw=2, ms=4, label="CFM train")
        axes[1].plot(ep, cfm_val, "m-s", lw=2, ms=4, label="CFM val")
    axes[1].set_title("CFM V2 (velocity MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training loss curves", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(
    cfm_train_losses,
    cfm_val_losses,
    mo,
    purr_train_losses,
    purr_val_losses,
):
    if not purr_train_losses and not cfm_train_losses:
        loss_out = mo.md("_Train at least one flow head to plot loss curves._")
    else:
        loss_out = plot_loss_curves(
            purr_train_losses, purr_val_losses, cfm_train_losses, cfm_val_losses
        )
    loss_out
    return


@app.function
def normalize_curve_to_first(values: list) -> list:
    if not values:
        return []
    v0 = values[0] if values[0] != 0 else 1.0
    return [v / v0 for v in values]


@app.function
def plot_convergence_speed(purr_val: list, cfm_val: list):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    purr_norm = normalize_curve_to_first(purr_val)
    cfm_norm = normalize_curve_to_first(cfm_val)
    if purr_norm:
        ax.plot(range(1, len(purr_norm) + 1), purr_norm, "b-o", lw=2, ms=4, label="Purrception V1")
    if cfm_norm:
        ax.plot(range(1, len(cfm_norm) + 1), cfm_norm, "g-s", lw=2, ms=4, label="CFM V2")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="50% of initial val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss (normalized to epoch 1)")
    ax.set_title("Convergence speed — normalized validation loss vs. epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(cfm_val_losses, mo, purr_val_losses):
    if not purr_val_losses and not cfm_val_losses:
        conv_out = mo.md("_Train at least one flow head to plot convergence speed._")
    else:
        conv_out = plot_convergence_speed(purr_val_losses, cfm_val_losses)
    conv_out
    return


@app.function
def plot_sample_grid_comparison(
    tokenizer: nn.Module,
    purr_head: nn.Module | None,
    cfm_head: nn.Module | None,
    class_names: list,
    num_per_class: int = 2,
    num_steps: int = 25,
    tau_sample: float = 0.9,
    seed: int = 42,
):
    n_classes = len(class_names)
    total = n_classes * num_per_class
    y = mx.array(np.repeat(np.arange(n_classes, dtype=np.int32), num_per_class))
    mx.random.seed(seed)
    imgs_v1 = None
    if purr_head is not None:
        imgs_v1 = sample_purrception_images(tokenizer, purr_head, y, num_steps, tau_sample)
        mx.eval(imgs_v1)
    mx.random.seed(seed)
    imgs_v2 = None
    if cfm_head is not None:
        imgs_v2 = sample_cfm_images(tokenizer, cfm_head, y, num_steps)
        mx.eval(imgs_v2)

    cols = num_per_class
    rows_per_model = n_classes
    n_panels = 0
    if imgs_v1 is not None:
        n_panels += 1
    if imgs_v2 is not None:
        n_panels += 1
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No models trained yet", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(
        rows_per_model, cols * n_panels, figsize=(cols * n_panels * 1.6, rows_per_model * 1.7)
    )
    if rows_per_model == 1:
        axes = np.expand_dims(axes, 0)
    if cols * n_panels == 1:
        axes = np.expand_dims(axes, 1)
    panel_offset = 0
    if imgs_v1 is not None:
        imgs_v1_np = np.asarray(imgs_v1)
        for i in range(total):
            r, c = divmod(i, cols)
            ax = axes[r, panel_offset + c]
            ax.imshow(np.clip(imgs_v1_np[i], 0.0, 1.0))
            if c == 0 and panel_offset == 0:
                ax.set_ylabel(class_names[r], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and c == cols // 2:
                ax.set_title("Purrception V1", fontsize=10)
        panel_offset += cols
    if imgs_v2 is not None:
        imgs_v2_np = np.asarray(imgs_v2)
        for i in range(total):
            r, c = divmod(i, cols)
            ax = axes[r, panel_offset + c]
            ax.imshow(np.clip(imgs_v2_np[i], 0.0, 1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and c == cols // 2:
                ax.set_title("CFM V2", fontsize=10)
    fig.suptitle(
        f"Class-conditional samples (num_steps={num_steps}, tau={tau_sample})",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, mo, trained_cfm, trained_purrception, trained_tokenizer):
    if trained_tokenizer is None:
        sample_out = mo.md("_Train the VQ-VAE tokenizer to compare samples._")
    elif trained_purrception is None and trained_cfm is None:
        sample_out = mo.md("_Train at least one flow head to compare samples._")
    else:
        sample_out = plot_sample_grid_comparison(
            trained_tokenizer,
            trained_purrception,
            trained_cfm,
            class_names,
            num_per_class=2,
            num_steps=25,
            tau_sample=0.9,
        )
    sample_out
    return


@app.function
def plot_temperature_sweep(
    tokenizer: nn.Module,
    head: nn.Module,
    classifier: nn.Module,
    taus: list,
    num_steps: int = 20,
    num_per_class: int = 4,
):
    y = mx.array(np.repeat(np.arange(10, dtype=np.int32), num_per_class))
    accs = []
    divs = []
    for tau in taus:
        imgs = sample_purrception_images(tokenizer, head, y, num_steps, tau)
        mx.eval(imgs)
        m = generation_quality_proxy(classifier, imgs, y)
        accs.append(m["cond_gen_accuracy"])
        divs.append(m["diversity"])
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(taus, accs, "b-o", lw=2, ms=6, label="Conditional gen accuracy")
    ax1.set_xlabel("tau_sample")
    ax1.set_ylabel("Conditional-generation accuracy", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(taus, divs, "r-s", lw=2, ms=6, label="Sample diversity")
    ax2.set_ylabel("Sample diversity (px std)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title(
        "Purrception V1: temperature sweep "
        "(paper reports U-shaped sensitivity around 0.8-0.9)"
    )
    fig.tight_layout()
    return fig


@app.cell
def _(mo, proxy_classifier, trained_purrception, trained_tokenizer):
    if (
        trained_tokenizer is None
        or trained_purrception is None
        or proxy_classifier is None
    ):
        temp_out = mo.md(
            "_Train the tokenizer, Purrception V1, and the proxy classifier to view "
            "the temperature sweep._"
        )
    else:
        temp_out = plot_temperature_sweep(
            trained_tokenizer,
            trained_purrception,
            proxy_classifier,
            taus=[0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5],
            num_steps=20,
            num_per_class=4,
        )
    temp_out
    return


@app.function
def first_epoch_below(values: list, threshold: float) -> int | None:
    for i, v in enumerate(values):
        if v <= threshold:
            return i + 1
    return None


@app.cell
def _(
    cfm_param_count,
    cfm_train_losses,
    cfm_val_losses,
    cv_results,
    mo,
    purr_train_losses,
    purr_val_losses,
    purrception_param_count,
    tokenizer_param_count,
    trained_tokenizer,
):
    if trained_tokenizer is None:
        summary_out = mo.md("_Train Stage A to see the results summary._")
    else:
        purr_final = purr_val_losses[-1] if purr_val_losses else float("nan")
        cfm_final = cfm_val_losses[-1] if cfm_val_losses else float("nan")
        purr_conv = None
        cfm_conv = None
        if purr_val_losses:
            purr_conv = first_epoch_below(purr_val_losses, purr_val_losses[0] * 0.5)
        if cfm_val_losses:
            cfm_conv = first_epoch_below(cfm_val_losses, cfm_val_losses[0] * 0.5)
        purr_conv_str = "n/a" if purr_conv is None else str(purr_conv)
        cfm_conv_str = "n/a" if cfm_conv is None else str(cfm_conv)
        if isinstance(cv_results, dict) and cv_results:
            cv_line = (
                f"- **CV (5-fold) conditional-gen accuracy** — "
                f"V1: {cv_results['v1_acc_mean']:.4f} +/- {cv_results['v1_acc_std']:.4f} | "
                f"V2: {cv_results['v2_acc_mean']:.4f} +/- {cv_results['v2_acc_std']:.4f}\n"
            )
        else:
            cv_line = ""

        summary_out = mo.md(
            f"""
            ### Model comparison

            | Metric | Purrception V1 | CFM V2 |
            |--------|----------------|--------|
            | Loss family | Categorical CE + z-loss (nats) | MSE on velocity |
            | Final val loss | {purr_final:.4f} | {cfm_final:.4f} |
            | Trained epochs | {len(purr_train_losses)} | {len(cfm_train_losses)} |
            | Epochs to reach 50% of initial val loss | {purr_conv_str} | {cfm_conv_str} |
            | Head parameters | {purrception_param_count:,} | {cfm_param_count:,} |
            | Backbone parameters (shared) | identical `DiffusionTransformerBackboneV1` | identical `DiffusionTransformerBackboneV1` |
            | Tokenizer parameters (frozen, shared) | {tokenizer_param_count:,} | {tokenizer_param_count:,} |

            {cv_line}

            ### Summary

            - **Framework**: MLX (channels-last NHWC).
            - **Shared frozen tokenizer** (`ViTVQVAEV1`) maps images to
              an 8x8 grid of continuous D=32 latents, plus a
              256-entry codebook of code indices.
            - **Model V1 — Purrception**: DiT backbone outputs categorical
              logits over codebook indices; the continuous velocity is
              derived analytically as the barycenter of predicted code
              embeddings (see Section 4). Trained with cross-entropy on
              true codes + z-loss regularizer.
            - **Model V2 — CFM baseline**: same DiT backbone, direct
              velocity regression, plain MSE.
            - **Comparison metric**: a small in-notebook CIFAR-10
              classifier grades class-conditional generated samples as
              an FID substitute (see Section 7). This is *not* the
              paper's protocol; it is the closest offline stand-in
              available here.
            - **What to look for** — Purrception's own paper reports (a)
              faster convergence than CFM at matched compute and
              (b) U-shaped sensitivity to `tau_sample` around 0.8-0.9.
              The convergence-speed and temperature-sweep plots above
              are the empirical checks of those claims *at this
              notebook's small scale* — do not read them as a
              reproduction of the paper's ImageNet-256 FID numbers,
              which require several orders of magnitude more compute.
            - **Reused vs. contributed** — the tokenizer and DiT
              backbone are standard infrastructure; Purrception's
              actual contribution isolated by this comparison is the
              categorical-cross-entropy + barycenter-velocity idea plus
              the temperature knob (Section 4 markdown cell).
            """
        )
    summary_out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Models

    Persist the tokenizer and both flow heads to the project-root
    `models/` directory. The three files can each be reloaded
    independently by Section 10.
    """)
    return


@app.cell
def _(mo):
    save_tokenizer_filename_ui = mo.ui.text(
        value="cifar10_vqtokenizer_v1.safetensors",
        label="Tokenizer filename",
        full_width=True,
    )
    save_tokenizer_btn = mo.ui.run_button(label="Save Tokenizer")
    mo.vstack([save_tokenizer_filename_ui, save_tokenizer_btn])
    return save_tokenizer_btn, save_tokenizer_filename_ui


@app.cell
def _(mo, save_tokenizer_btn, save_tokenizer_filename_ui, trained_tokenizer):
    if trained_tokenizer is None:
        save_tok_out = mo.md("_Train the VQ-VAE tokenizer (Stage A) first._")
    elif not save_tokenizer_btn.value:
        save_tok_out = mo.md(
            "Enter a filename and click **Save Tokenizer** to write "
            "weights to `models/`."
        )
    else:
        models_dir_a = Path(__file__).resolve().parent.parent / "models"
        models_dir_a.mkdir(parents=True, exist_ok=True)
        save_path_a = models_dir_a / save_tokenizer_filename_ui.value
        trained_tokenizer.save_weights(str(save_path_a))
        save_tok_out = mo.md(
            f"**Saved!** Tokenizer weights written to `{save_path_a}`."
        )
    save_tok_out
    return


@app.cell
def _(mo):
    save_purr_filename_ui = mo.ui.text(
        value="cifar10_purrception_v1.safetensors",
        label="Purrception filename",
        full_width=True,
    )
    save_purr_btn = mo.ui.run_button(label="Save Purrception")
    mo.vstack([save_purr_filename_ui, save_purr_btn])
    return save_purr_btn, save_purr_filename_ui


@app.cell
def _(mo, save_purr_btn, save_purr_filename_ui, trained_purrception):
    if trained_purrception is None:
        save_purr_out = mo.md("_Train Purrception (Stage B) first._")
    elif not save_purr_btn.value:
        save_purr_out = mo.md(
            "Enter a filename and click **Save Purrception** to write "
            "weights to `models/`."
        )
    else:
        models_dir_b = Path(__file__).resolve().parent.parent / "models"
        models_dir_b.mkdir(parents=True, exist_ok=True)
        save_path_b = models_dir_b / save_purr_filename_ui.value
        trained_purrception.save_weights(str(save_path_b))
        save_purr_out = mo.md(
            f"**Saved!** Purrception weights written to `{save_path_b}`."
        )
    save_purr_out
    return


@app.cell
def _(mo):
    save_cfm_filename_ui = mo.ui.text(
        value="cifar10_cfm_v1.safetensors",
        label="CFM filename",
        full_width=True,
    )
    save_cfm_btn = mo.ui.run_button(label="Save CFM")
    mo.vstack([save_cfm_filename_ui, save_cfm_btn])
    return save_cfm_btn, save_cfm_filename_ui


@app.cell
def _(mo, save_cfm_btn, save_cfm_filename_ui, trained_cfm):
    if trained_cfm is None:
        save_cfm_out = mo.md("_Train the CFM baseline (Stage C) first._")
    elif not save_cfm_btn.value:
        save_cfm_out = mo.md(
            "Enter a filename and click **Save CFM** to write "
            "weights to `models/`."
        )
    else:
        models_dir_c = Path(__file__).resolve().parent.parent / "models"
        models_dir_c.mkdir(parents=True, exist_ok=True)
        save_path_c = models_dir_c / save_cfm_filename_ui.value
        trained_cfm.save_weights(str(save_path_c))
        save_cfm_out = mo.md(
            f"**Saved!** CFM weights written to `{save_path_c}`."
        )
    save_cfm_out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 10 — Load Saved Model & Generate

    Reload the three checkpoints and use them to sample fresh
    class-conditional images end-to-end, without depending on the
    in-memory training-session models. Guards check that files
    exist and match the expected architecture keys before invoking
    the sampling routines.
    """)
    return


@app.function
def models_dir_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


@app.function
def list_saved_models() -> list[str]:
    d = models_dir_path()
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.safetensors")) + sorted(
        p.name for p in d.glob("*.npz")
    )


@app.function
def infer_tokenizer_config(state: dict) -> dict:
    if "encoder.to_code.weight" not in state:
        raise ValueError("Not a ViTVQVAEV1 checkpoint: missing 'encoder.to_code.weight'.")
    num_embeddings, embedding_dim = (int(v) for v in state["quantizer.codebook"].shape)
    embed_dim = int(state["encoder.to_code.weight"].shape[1])
    patch_dim_in = int(state["encoder.patch_embed.proj.weight"].shape[1])
    patch_size = int(round((patch_dim_in / 3) ** 0.5))
    encoder_depth = 1 + max(
        int(k.split(".")[2]) for k in state if k.startswith("encoder.blocks.")
    )
    decoder_depth = 1 + max(
        int(k.split(".")[2]) for k in state if k.startswith("decoder.blocks.")
    )
    fc1_shape = state["encoder.blocks.0.mlp.fc1.weight"].shape
    mlp_expansion = int(fc1_shape[0] // fc1_shape[1])
    return {
        "image_size": 32,
        "patch_size": patch_size,
        "in_channels": 3,
        "embed_dim": embed_dim,
        "encoder_depth": encoder_depth,
        "decoder_depth": decoder_depth,
        "num_heads": 4,
        "mlp_expansion": mlp_expansion,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
    }


@app.function
def infer_head_config(state: dict, num_codes: int, latent_dim: int) -> dict:
    embed_dim = int(state["backbone.proj_in.weight"].shape[0])
    num_layers = 1 + max(
        int(k.split(".")[2]) for k in state if k.startswith("backbone.blocks.")
    )
    mlp_dim = int(state["backbone.blocks.0.mlp.layers.0.weight"].shape[0])
    return {
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "num_codes": num_codes,
        "latent_dim": latent_dim,
    }


@app.function
def load_tokenizer_ckpt(path):
    state = mx.load(str(path))
    cfg = infer_tokenizer_config(state)
    model = ViTVQVAEV1(**cfg)
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model, cfg


@app.function
def load_purrception_ckpt(path, tokenizer: nn.Module):
    state = mx.load(str(path))
    cfg = infer_head_config(
        state,
        num_codes=int(tokenizer.num_embeddings),
        latent_dim=int(tokenizer.embedding_dim),
    )
    head = PurrceptionFlowHeadV1(
        grid_size=int(tokenizer.grid_size),
        latent_dim=cfg["latent_dim"],
        num_codes=cfg["num_codes"],
        num_classes=10,
        embed_dim=cfg["embed_dim"],
        num_heads=4,
        mlp_dim=cfg["mlp_dim"],
        num_layers=cfg["num_layers"],
    )
    head.load_weights(str(path))
    mx.eval(head.parameters())
    return head, cfg


@app.function
def load_cfm_ckpt(path, tokenizer: nn.Module):
    state = mx.load(str(path))
    cfg = infer_head_config(
        state,
        num_codes=int(tokenizer.num_embeddings),
        latent_dim=int(tokenizer.embedding_dim),
    )
    head = ContinuousFlowMatchingHeadV1(
        grid_size=int(tokenizer.grid_size),
        latent_dim=cfg["latent_dim"],
        num_classes=10,
        embed_dim=cfg["embed_dim"],
        num_heads=4,
        mlp_dim=cfg["mlp_dim"],
        num_layers=cfg["num_layers"],
    )
    head.load_weights(str(path))
    mx.eval(head.parameters())
    return head, cfg


@app.cell
def _(mo):
    saved_choices = list_saved_models() or ["<no checkpoints under models/>"]

    def pick_default(name, choices):
        return name if name in choices else choices[0]

    load_tok_ui = mo.ui.dropdown(
        options=saved_choices,
        value=pick_default("cifar10_vqtokenizer_v1.safetensors", saved_choices),
        label="Tokenizer checkpoint",
    )
    load_purr_ui = mo.ui.dropdown(
        options=saved_choices,
        value=pick_default("cifar10_purrception_v1.safetensors", saved_choices),
        label="Purrception checkpoint",
    )
    load_cfm_ui = mo.ui.dropdown(
        options=saved_choices,
        value=pick_default("cifar10_cfm_v1.safetensors", saved_choices),
        label="CFM checkpoint",
    )
    load_steps_ui = mo.ui.slider(5, 200, value=25, step=1, label="Sampling steps")
    load_tau_ui = mo.ui.slider(0.3, 1.6, value=0.9, step=0.05, label="Purrception tau_sample")
    load_generate_btn = mo.ui.run_button(label="Load & Generate")
    mo.vstack(
        [
            mo.md("### Load checkpoints and generate a fresh grid"),
            mo.hstack([load_tok_ui, load_purr_ui, load_cfm_ui]),
            mo.hstack([load_steps_ui, load_tau_ui]),
            load_generate_btn,
        ]
    )
    return (
        load_cfm_ui,
        load_generate_btn,
        load_purr_ui,
        load_steps_ui,
        load_tau_ui,
        load_tok_ui,
    )


@app.cell
def _(
    class_names,
    load_cfm_ui,
    load_generate_btn,
    load_purr_ui,
    load_steps_ui,
    load_tau_ui,
    load_tok_ui,
    mo,
):
    mo.stop(
        not load_generate_btn.value,
        mo.md("Pick checkpoints and click **Load & Generate** to sample from saved weights."),
    )
    d = models_dir_path()
    tok_path = d / load_tok_ui.value
    purr_path = d / load_purr_ui.value
    cfm_path = d / load_cfm_ui.value
    missing = [p for p in [tok_path, purr_path, cfm_path] if not p.is_file()]
    if missing:
        gen_out = mo.md(
            f"**Missing checkpoint files**: `{[str(p) for p in missing]}` — "
            f"complete Section 9 first."
        )
    else:
        try:
            tokenizer_loaded, tok_cfg = load_tokenizer_ckpt(tok_path)
        except (ValueError, KeyError) as tok_err:
            gen_out = mo.md(
                f"**Error loading tokenizer** from `{tok_path.name}`: `{tok_err}`."
            )
            mo.output.replace(gen_out)
        else:
            try:
                purr_loaded, purr_cfg = load_purrception_ckpt(purr_path, tokenizer_loaded)
            except (ValueError, KeyError) as purr_err:
                purr_loaded = None
                purr_cfg = None
                mo.output.append(
                    mo.md(
                        f"**Warning** — could not load Purrception `{purr_path.name}`: "
                        f"`{purr_err}`."
                    )
                )
            try:
                cfm_loaded, cfm_cfg = load_cfm_ckpt(cfm_path, tokenizer_loaded)
            except (ValueError, KeyError) as cfm_err:
                cfm_loaded = None
                cfm_cfg = None
                mo.output.append(
                    mo.md(
                        f"**Warning** — could not load CFM `{cfm_path.name}`: `{cfm_err}`."
                    )
                )
            fig = plot_sample_grid_comparison(
                tokenizer_loaded,
                purr_loaded,
                cfm_loaded,
                class_names,
                num_per_class=2,
                num_steps=int(load_steps_ui.value),
                tau_sample=float(load_tau_ui.value),
                seed=123,
            )
            cfg_summary_bits = [
                f"tokenizer={tok_cfg}",
            ]
            if purr_cfg is not None:
                cfg_summary_bits.append(f"purrception={purr_cfg}")
            if cfm_cfg is not None:
                cfg_summary_bits.append(f"cfm={cfm_cfg}")
            gen_out = mo.vstack(
                [
                    mo.md(
                        "**Loaded from disk (no in-memory training-session models used).**"
                    ),
                    mo.md("Inferred configs: " + " ; ".join(cfg_summary_bits)),
                    fig,
                ]
            )
    gen_out
    return


if __name__ == "__main__":
    app.run()
