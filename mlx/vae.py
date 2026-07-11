import marimo

__generated_with = "0.23.13"
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
    mo.md("""
    # Variational Autoencoder on MNIST — MLX

    ## Research Goal

    Train a **Variational Autoencoder (VAE)** on the MNIST handwritten digit
    dataset using Apple's **MLX** framework. The model learns a compact
    low-dimensional latent representation of 28×28 grayscale digit images
    and reconstructs them by sampling from the learned Gaussian latent
    distribution.

    This notebook mirrors the reference MLX VAE example and reports the
    final **train / validation / test ELBO loss** together with visual
    reconstruction quality.

    ### Method
    - Fully connected encoder / decoder built from `MultiLayerPerceptronBlockV1`
    - 32-dimensional Gaussian latent space with the reparameterization trick
    - ELBO loss = squared-error reconstruction loss + KL divergence to `N(0, I)`
    - Optimizer: **AdamW**
    - Metric: mean ELBO per sample on validation / test splits
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

    - `image` — `uint8` array shaped `(28, 28, 1)`
    - `label` — scalar `uint8` in `[0, 9]`

    | Split | Size |
    |-------|------|
    | Train (raw) | {len(train_ds):,} |
    | Test | {len(test_ds):,} |
    """)
    return


@app.function
def plot_mnist_samples(train_ds, n_show=40, rows=5, cols=8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    for i in range(n_show):
        sample = train_ds[i]
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
    plot_mnist_samples(train_ds)
    return


@app.function
def plot_class_distribution(train_ds):
    labels = np.array(
        [int(np.array(train_ds[i]["label"]).item()) for i in range(len(train_ds))]
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
def make_datasets(train_ds, test_ds, batch_size, val_fraction=0.15):
    def normalize(x):
        return x.astype("float32") / 255.0

    n_total = len(train_ds)
    n_val = int(round(n_total * val_fraction))
    shuffled = train_ds.shuffle()
    train_iter = (
        shuffled
        .to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    val_iter = (
        shuffled
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
    return train_iter, val_iter, test_iter


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition
    """)
    return


@app.class_definition
class MultiLayerPerceptronBlockV1(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.fc2(nn.relu(self.fc1(x))))


@app.class_definition
class VariationalEncoderV1(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.mlp = MultiLayerPerceptronBlockV1(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def __call__(self, x: mx.array):
        h = self.mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)


@app.class_definition
class VariationalDecoderV1(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512, output_dim: int = 784):
        super().__init__()
        self.mlp = MultiLayerPerceptronBlockV1(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, z: mx.array) -> mx.array:
        return mx.sigmoid(self.fc_out(self.mlp(z)))


@app.class_definition
class VariationalAutoEncoderV1(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 512, latent_dim: int = 32):
        super().__init__()
        self.encoder = VariationalEncoderV1(input_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoderV1(latent_dim, hidden_dim, input_dim)
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.conv(x))


@app.class_definition
class ConvolutionalEncoderV1(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 32):
        super().__init__()
        self.block1 = ConvolutionBlockV1(in_channels, base_channels, stride=2)
        self.block2 = ConvolutionBlockV1(base_channels, base_channels * 2, stride=2)
        self.fc_mu = nn.Linear(7 * 7 * base_channels * 2, latent_dim)
        self.fc_logvar = nn.Linear(7 * 7 * base_channels * 2, latent_dim)

    def __call__(self, x: mx.array):
        h = self.block1(x)
        h = self.block2(h)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_logvar(h)


@app.class_definition
class ConvolutionalDecoderV1(nn.Module):
    def __init__(self, latent_dim: int = 32, base_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * base_channels * 2)
        self.deconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.spatial_c = base_channels * 2

    def __call__(self, z: mx.array) -> mx.array:
        h = nn.relu(self.fc(z))
        h = h.reshape(h.shape[0], 7, 7, self.spatial_c)
        h = nn.relu(self.deconv1(h))
        return mx.sigmoid(self.deconv2(h))


@app.class_definition
class ConvolutionalVariationalAutoEncoderV1(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 32):
        super().__init__()
        self.encoder = ConvolutionalEncoderV1(in_channels, base_channels, latent_dim)
        self.decoder = ConvolutionalDecoderV1(latent_dim, base_channels, in_channels)
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
def train_and_validate(
    model: nn.Module,
    loss_fn,
    train_iter,
    val_iter,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    preprocess_fn,
    on_epoch_end=None,
):
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        tl = run_train_epoch(model, loss_fn, optimizer, train_iter, preprocess_fn)
        vl = run_evaluate(model, loss_fn, val_iter, preprocess_fn)
        train_losses.append(tl)
        val_losses.append(vl)
        if on_epoch_end is not None:
            on_epoch_end(epoch, n_epochs, tl, vl)
    return train_losses, val_losses


@app.cell
def _(mo):
    mo.md("""
    ### Model Architecture — VariationalAutoEncoderV1

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Encoder MLP block | `MultiLayerPerceptronBlockV1` | `(B, hidden_dim)` |
    | `mu` head | `Linear(hidden_dim, latent_dim)` | `(B, latent_dim)` |
    | `logvar` head | `Linear(hidden_dim, latent_dim)` | `(B, latent_dim)` |
    | Reparameterization | `mu + eps * exp(0.5 * logvar)` | `(B, latent_dim)` |
    | Decoder MLP block | `MultiLayerPerceptronBlockV1` | `(B, hidden_dim)` |
    | Output projection | `Linear(hidden_dim, 784)` + `sigmoid` | `(B, 784)` |

    **Loss**: `mean_batch( sum_pixel (x - x_hat)^2 ) + KL(q(z|x) || N(0, I))`
    """)
    return


@app.cell
def _(mo):
    _reference_model = VariationalAutoEncoderV1(
        input_dim=784, hidden_dim=512, latent_dim=32
    )
    mx.eval(_reference_model.parameters())
    reference_param_count = count_parameters(_reference_model)
    mo.md(
        f"**Reference model parameter count** "
        f"(input=784, hidden=512, latent=32): `{reference_param_count:,}`"
    )
    return (reference_param_count,)


@app.cell
def _(mo):
    mo.md("""
    ### Model Architecture — ConvolutionalVariationalAutoEncoderV1

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Conv block 1 | `ConvolutionBlockV1` (k=3, stride=2, pad=1) | `(B, 14, 14, 32)` |
    | Conv block 2 | `ConvolutionBlockV1` (k=3, stride=2, pad=1) | `(B, 7, 7, 64)` |
    | Flatten | — | `(B, 3136)` |
    | `mu` head | `Linear(3136, latent_dim)` | `(B, latent_dim)` |
    | `logvar` head | `Linear(3136, latent_dim)` | `(B, latent_dim)` |
    | Reparameterize | `mu + eps * exp(0.5 * logvar)` | `(B, latent_dim)` |
    | Project + reshape | `Linear → ReLU → (B, 7, 7, 64)` | `(B, 7, 7, 64)` |
    | Deconv block 1 | `ConvTranspose2d` (k=4, stride=2, pad=1) + ReLU | `(B, 14, 14, 32)` |
    | Deconv block 2 | `ConvTranspose2d` (k=4, stride=2, pad=1) + sigmoid | `(B, 28, 28, 1)` |
    """)
    return


@app.cell
def _(mo):
    _conv_ref = ConvolutionalVariationalAutoEncoderV1(
        in_channels=1, base_channels=32, latent_dim=32
    )
    mx.eval(_conv_ref.parameters())
    conv_reference_param_count = count_parameters(_conv_ref)
    mo.md(
        f"**Conv VAE reference parameter count** "
        f"(base_channels=32, latent=32): `{conv_reference_param_count:,}`"
    )
    return (conv_reference_param_count,)


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="1e-3",
        label="Learning Rate",
    )
    epochs_ui = mo.ui.slider(1, 50, value=20, step=1, label="Epochs")
    bs_ui = mo.ui.dropdown(
        options=[32, 64, 128, 256], value=128, label="Batch Size"
    )
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    latent_dim_ui = mo.ui.dropdown(
        options=[8, 16, 32, 64], value=32, label="Latent Dim"
    )
    hidden_dim_ui = mo.ui.dropdown(
        options=[256, 512, 1024], value=512, label="Hidden Dim"
    )
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Hyperparameters"),
            mo.hstack([lr_ui, epochs_ui, bs_ui]),
            mo.hstack([wd_ui, latent_dim_ui, hidden_dim_ui]),
            train_btn,
        ]
    )
    return (
        bs_ui,
        epochs_ui,
        hidden_dim_ui,
        latent_dim_ui,
        lr_ui,
        train_btn,
        wd_ui,
    )


@app.cell
def _(bs_ui, test_ds, train_ds):
    train_iter, val_iter, test_iter = make_datasets(train_ds, test_ds, int(bs_ui.value))
    return test_iter, train_iter, val_iter


@app.cell
def _(
    epochs_ui,
    hidden_dim_ui,
    latent_dim_ui,
    lr_ui,
    mo,
    train_btn,
    train_iter,
    val_iter,
    wd_ui,
):
    train_losses = []
    val_losses = []
    conv_train_losses = []
    conv_val_losses = []
    trained_model = None
    conv_trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training both MLP and Conv VAE."))
    else:
        _mlp = VariationalAutoEncoderV1(
            input_dim=784,
            hidden_dim=int(hidden_dim_ui.value),
            latent_dim=int(latent_dim_ui.value),
        )
        _conv = ConvolutionalVariationalAutoEncoderV1(
            in_channels=1,
            base_channels=32,
            latent_dim=int(latent_dim_ui.value),
        )
        mx.eval(_mlp.parameters())
        mx.eval(_conv.parameters())

        def _mlp_preprocess(batch):
            return mx.array(batch["image"]).reshape(-1, 784)

        def _conv_preprocess(batch):
            return mx.array(batch["image"])

        def _mlp_progress(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"MLP — Epoch {epoch + 1}/{n_epochs}  train: {tl:.4f} | val: {vl:.4f}")
            )

        def _conv_progress(epoch, n_epochs, ctl, cvl):
            mo.output.replace(
                mo.md(f"Conv — Epoch {epoch + 1}/{n_epochs}  train: {ctl:.4f} | val: {cvl:.4f}")
            )

        train_losses, val_losses = train_and_validate(
            _mlp, compute_vae_loss, train_iter, val_iter,
            epochs_ui.value, lr_ui.value, wd_ui.value,
            _mlp_preprocess, _mlp_progress,
        )
        conv_train_losses, conv_val_losses = train_and_validate(
            _conv, compute_vae_loss, train_iter, val_iter,
            epochs_ui.value, lr_ui.value, wd_ui.value,
            _conv_preprocess, _conv_progress,
        )
        trained_model = _mlp
        conv_trained_model = _conv
        mo.output.replace(
            mo.md(
                f"**Training complete!**  \n"
                f"MLP  — final train: {train_losses[-1]:.4f} | val: {val_losses[-1]:.4f}  \n"
                f"Conv — final train: {conv_train_losses[-1]:.4f} | val: {conv_val_losses[-1]:.4f}"
            )
        )
    return (
        conv_train_losses,
        conv_trained_model,
        conv_val_losses,
        train_losses,
        trained_model,
        val_losses,
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


@app.cell
def _(hp_search_cb, mo, train_iter, val_iter):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above._"),
    )
    _space = {"lr": [1e-4, 1e-3, 3e-3], "latent_dim": [16, 32, 64]}
    _results = []
    _preprocess = lambda batch: mx.array(batch["image"]).reshape(-1, 784)
    for _lr in _space["lr"]:
        for _ld in _space["latent_dim"]:
            _m = VariationalAutoEncoderV1(
                input_dim=784, hidden_dim=512, latent_dim=_ld
            )
            mx.eval(_m.parameters())
            _, _val_losses = train_and_validate(
                _m, compute_vae_loss, train_iter, val_iter,
                5, _lr, 0.0, _preprocess,
            )
            _vl = _val_losses[-1]
            _results.append(
                {"lr": _lr, "latent_dim": _ld, "val_loss": round(_vl, 4)}
            )
            mo.output.replace(
                mo.md(f"lr={_lr}, latent_dim={_ld} → val={_vl:.4f}")
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


@app.cell
def _(conv_trained_model, mo, test_iter, trained_model, val_iter):
    if trained_model is None:
        _out = mo.md("_Train the models first._")
    else:
        _mlp_preprocess = lambda batch: mx.array(batch["image"]).reshape(-1, 784)
        _conv_preprocess = lambda batch: mx.array(batch["image"])
        _val_elbo = run_evaluate(trained_model, compute_vae_loss, val_iter, _mlp_preprocess)
        _test_elbo = run_evaluate(trained_model, compute_vae_loss, test_iter, _mlp_preprocess)
        _conv_val_elbo = run_evaluate(conv_trained_model, compute_vae_loss, val_iter, _conv_preprocess)
        _conv_test_elbo = run_evaluate(conv_trained_model, compute_vae_loss, test_iter, _conv_preprocess)
        _out = mo.md(
            f"""
    **Evaluation Results**

    | Model | Val ELBO | Test ELBO |
    |-------|----------|-----------|
    | MLP VAE | {_val_elbo:.4f} | {_test_elbo:.4f} |
    | Conv VAE | {_conv_val_elbo:.4f} | {_conv_test_elbo:.4f} |
    """
        )
    _out
    return


@app.cell
def _(mo, train_ds, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first._")
        cv_results = {}
    else:
        _k = 5
        _n_full = len(train_ds)
        _fold_size = _n_full // _k
        _fold_losses = []

        for _fold in range(_k):
            _vs = _fold * _fold_size
            _ve = _vs + _fold_size

            _fold_model = VariationalAutoEncoderV1(
                input_dim=784, hidden_dim=512, latent_dim=32
            )
            mx.eval(_fold_model.parameters())
            _fold_opt = optim.AdamW(learning_rate=1e-3)
            _vg = nn.value_and_grad(_fold_model, compute_vae_loss)

            for _ep in range(3):
                _train_indices = list(range(0, _vs)) + list(range(_ve, _n_full))
                np.random.shuffle(_train_indices)
                for _i in range(0, len(_train_indices), 128):
                    _idx = _train_indices[_i : _i + 128]
                    _imgs = np.stack([train_ds[_j]["image"] for _j in _idx])
                    _x = (
                        mx.array(_imgs, dtype=mx.float32).reshape(-1, 784)
                        / 255.0
                    )
                    _l, _g = _vg(_fold_model, _x)
                    _fold_opt.update(_fold_model, _g)
                    mx.eval(_l, _fold_model.parameters())

            _val_total = 0.0
            _val_n = 0
            _val_indices = list(range(_vs, _ve))
            for _i in range(0, len(_val_indices), 256):
                _idx = _val_indices[_i : _i + 256]
                _imgs = np.stack([train_ds[_j]["image"] for _j in _idx])
                _x = (
                    mx.array(_imgs, dtype=mx.float32).reshape(-1, 784) / 255.0
                )
                _l = compute_vae_loss(_fold_model, _x)
                mx.eval(_l)
                _val_total += _l.item()
                _val_n += 1
            _fold_losses.append(_val_total / max(_val_n, 1))
            mo.output.replace(
                mo.md(
                    f"Fold {_fold + 1}/{_k} — "
                    f"val loss: {_fold_losses[-1]:.4f}"
                )
            )

        _mean = float(np.mean(_fold_losses))
        _std = float(np.std(_fold_losses))
        cv_results = {
            "fold_losses": _fold_losses,
            "mean": _mean,
            "std": _std,
        }
        _out = mo.md(
            f"**{_k}-Fold CV ELBO: {_mean:.4f} ± {_std:.4f}**"
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Results
    """)
    return


@app.cell
def _(conv_train_losses, conv_val_losses, mo, train_losses, val_losses):
    if not train_losses:
        _out = mo.md("_Train first._")
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
        _axes[0].plot(range(1, len(train_losses) + 1), train_losses, "b-o", lw=2, ms=4, label="Train")
        _axes[0].plot(range(1, len(val_losses) + 1), val_losses, "r-s", lw=2, ms=4, label="Val")
        _axes[0].set_title("MLP VAE")
        _axes[0].set_xlabel("Epoch")
        _axes[0].set_ylabel("ELBO Loss")
        _axes[0].legend()
        _axes[0].grid(True, alpha=0.3)
        _axes[1].plot(range(1, len(conv_train_losses) + 1), conv_train_losses, "b-o", lw=2, ms=4, label="Train")
        _axes[1].plot(range(1, len(conv_val_losses) + 1), conv_val_losses, "r-s", lw=2, ms=4, label="Val")
        _axes[1].set_title("Conv VAE")
        _axes[1].set_xlabel("Epoch")
        _axes[1].set_ylabel("ELBO Loss")
        _axes[1].legend()
        _axes[1].grid(True, alpha=0.3)
        _fig.suptitle("MLP VAE vs Conv VAE — Training Comparison", fontsize=13)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(conv_trained_model, mo, test_iter, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first._")
    else:
        _n_show = 8
        test_iter.reset()
        _batch = next(test_iter)

        _orig_flat = mx.array(_batch["image"]).reshape(-1, 784)[:_n_show]
        _mlp_recon, _, _ = trained_model(_orig_flat)
        mx.eval(_mlp_recon)

        _orig_4d = mx.array(_batch["image"])[:_n_show]
        _conv_recon, _, _ = conv_trained_model(_orig_4d)
        mx.eval(_conv_recon)

        _orig_np = np.array(_orig_flat).reshape(-1, 28, 28)
        _mlp_np = np.array(_mlp_recon).reshape(-1, 28, 28)
        _conv_np = np.array(_conv_recon).reshape(-1, 28, 28)

        _fig, _axes = plt.subplots(3, _n_show, figsize=(16, 6))
        for _i in range(_n_show):
            _axes[0, _i].imshow(_orig_np[_i], cmap="gray")
            _axes[0, _i].axis("off")
            _axes[1, _i].imshow(_mlp_np[_i], cmap="gray")
            _axes[1, _i].axis("off")
            _axes[2, _i].imshow(_conv_np[_i], cmap="gray")
            _axes[2, _i].axis("off")
        _axes[0, 0].set_title("Original", fontsize=9, loc="left")
        _axes[1, 0].set_title("MLP VAE", fontsize=9, loc="left")
        _axes[2, 0].set_title("Conv VAE", fontsize=9, loc="left")
        _fig.suptitle("Reconstructions: MLP VAE vs Conv VAE (test set)", fontsize=12)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(
    conv_reference_param_count,
    conv_train_losses,
    conv_val_losses,
    mo,
    reference_param_count,
    train_losses,
    val_losses,
):
    if not train_losses:
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX
    - **Dataset**: MNIST (60k train / 10k test, grayscale 28×28)
    - **MLP VAE** (`VariationalAutoEncoderV1`): {reference_param_count:,} parameters — treats pixels independently
    - **Conv VAE** (`ConvolutionalVariationalAutoEncoderV1`): {conv_reference_param_count:,} parameters — exploits spatial structure via stride-2 convolutions

    Train the models above to populate the final loss comparison here.
    """
        )
    else:
        _mlp_wins = train_losses[-1] < conv_train_losses[-1]
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX
    - **Dataset**: MNIST (60k train / 10k test, grayscale 28×28)

    | Model | Parameters | Final Train ELBO | Final Val ELBO |
    |-------|-----------|-----------------|----------------|
    | MLP VAE | {reference_param_count:,} | {train_losses[-1]:.4f} | {val_losses[-1]:.4f} |
    | Conv VAE | {conv_reference_param_count:,} | {conv_train_losses[-1]:.4f} | {conv_val_losses[-1]:.4f} |

    **Lower final train ELBO**: {"MLP VAE" if _mlp_wins else "Conv VAE"}

    The MLP VAE flattens images to 784-dim vectors; the Conv VAE encodes spatial
    structure with two stride-2 conv blocks (28→14→7) and decodes with two
    transposed-conv blocks (7→14→28). Compare the reconstruction grids above
    to see qualitative differences. Enable the hyperparameter search checkbox
    for a sweep over lr × latent_dim; the cross-validation cell reports 5-fold
    ELBO with mean ± std.
    """
        )
    _summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Models

    Persist the trained `VariationalAutoEncoderV1` (MLP) and
    `ConvolutionalVariationalAutoEncoderV1` (Conv) weights to the
    project's `models/` directory. The file extension chosen determines
    the on-disk format: `.safetensors` or `.npz` (both natively
    supported by MLX).
    """)
    return


@app.cell
def _(mo):
    mlp_save_filename_ui = mo.ui.text(
        value="vae_mlp_mnist_v1.safetensors",
        label="MLP VAE filename (saved into models/)",
        full_width=True,
    )
    mlp_save_btn = mo.ui.run_button(label="Save MLP VAE")
    conv_save_filename_ui = mo.ui.text(
        value="vae_conv_mnist_v1.safetensors",
        label="Conv VAE filename (saved into models/)",
        full_width=True,
    )
    conv_save_btn = mo.ui.run_button(label="Save Conv VAE")
    mo.vstack(
        [
            mo.hstack([mlp_save_filename_ui, mlp_save_btn]),
            mo.hstack([conv_save_filename_ui, conv_save_btn]),
        ]
    )
    return (
        conv_save_btn,
        conv_save_filename_ui,
        mlp_save_btn,
        mlp_save_filename_ui,
    )


@app.cell
def _(mlp_save_btn, mlp_save_filename_ui, mo, trained_model):
    if trained_model is None:
        _mlp_out = mo.md("_Train the MLP VAE first (Section 5) before saving._")
    elif not mlp_save_btn.value:
        _mlp_out = mo.md(
            "Enter a filename and click **Save MLP VAE** to write the "
            "trained weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / mlp_save_filename_ui.value
        trained_model.save_weights(str(_save_path))
        _mlp_out = mo.md(f"**Saved!** MLP VAE weights written to `{_save_path}`.")
    _mlp_out
    return


@app.cell
def _(conv_save_btn, conv_save_filename_ui, conv_trained_model, mo):
    if conv_trained_model is None:
        _conv_out = mo.md("_Train the Conv VAE first (Section 5) before saving._")
    elif not conv_save_btn.value:
        _conv_out = mo.md(
            "Enter a filename and click **Save Conv VAE** to write the "
            "trained weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / conv_save_filename_ui.value
        conv_trained_model.save_weights(str(_save_path))
        _conv_out = mo.md(f"**Saved!** Conv VAE weights written to `{_save_path}`.")
    _conv_out
    return


if __name__ == "__main__":
    app.run()
