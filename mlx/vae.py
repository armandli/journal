import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
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
    mo.md(
        f"""
        ### Dataset overview

        MNIST is loaded as an `mlx.data` Buffer. Each sample is a dict with:

        - `image` — `uint8` array shaped `(28, 28, 1)`
        - `label` — scalar `uint8` in `[0, 9]`

        | Split | Size |
        |-------|------|
        | Train (raw) | {len(train_ds):,} |
        | Test | {len(test_ds):,} |
        """
    )
    return


@app.cell
def _(train_ds):
    _n_show = 40
    _rows, _cols = 5, 8
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(12, 7))
    for _i in range(_n_show):
        _sample = train_ds[_i]
        _img = np.array(_sample["image"]).squeeze()
        _label = int(np.array(_sample["label"]).item())
        _r, _c = divmod(_i, _cols)
        _axes[_r, _c].imshow(_img, cmap="gray")
        _axes[_r, _c].set_title(f"{_label}", fontsize=9)
        _axes[_r, _c].axis("off")
    _fig.suptitle("MNIST training samples (labels shown as titles)", fontsize=13)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(train_ds):
    _labels = np.array(
        [int(np.array(train_ds[_i]["label"]).item()) for _i in range(len(train_ds))]
    )
    _counts = np.bincount(_labels, minlength=10)
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.bar(np.arange(10), _counts, color="steelblue", edgecolor="black")
    _ax.set_xticks(np.arange(10))
    _ax.set_xlabel("Digit")
    _ax.set_ylabel("Count")
    _ax.set_title("MNIST training-set class distribution")
    for _i, _c in enumerate(_counts):
        _ax.text(_i, _c + 50, str(int(_c)), ha="center", fontsize=8)
    _ax.grid(True, alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation
    """)
    return


@app.cell
def _(train_ds):
    _rng = np.random.default_rng(seed=42)
    _n_total = len(train_ds)
    _perm = _rng.permutation(_n_total)
    _n_val = int(round(_n_total * 0.15))
    _val_indices = np.sort(_perm[:_n_val])
    _train_indices = np.sort(_perm[_n_val:])
    train_buf = train_ds[_train_indices]
    val_buf = train_ds[_val_indices]
    return train_buf, val_buf


@app.cell
def _(test_ds):
    test_buf = test_ds
    return (test_buf,)


@app.cell
def _(mo, test_buf, train_buf, val_buf):
    mo.md(
        f"""
        ### Split sizes

        | Split | Buffer | Size |
        |-------|--------|------|
        | Train (85%) | `train_buf` | {len(train_buf):,} |
        | Validation (15%) | `val_buf` | {len(val_buf):,} |
        | Test | `test_buf` | {len(test_buf):,} |

        Batching is done with the mlx.data streaming API:

        - Training:   `train_buf.shuffle().to_stream().batch(batch_size)`
        - Val / test: `buf.to_stream().batch(batch_size)`

        Images are normalized to `[0, 1]` and reshaped to `(-1, 784)`
        inside every batch loop.
        """
    )
    return


@app.cell
def _(train_buf):
    _peek = next(iter(train_buf.to_stream().batch(4)))
    _peek_x = mx.array(_peek["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
    batch_shape = tuple(_peek_x.shape)
    batch_dtype = str(_peek_x.dtype)
    batch_min = float(_peek_x.min().item())
    batch_max = float(_peek_x.max().item())
    return batch_dtype, batch_max, batch_min, batch_shape


@app.cell
def _(batch_dtype, batch_max, batch_min, batch_shape, mo):
    mo.md(
        f"""
        ### Batch sanity check

        | Field | Value |
        |-------|-------|
        | `x.shape` after reshape | `{batch_shape}` |
        | `x.dtype` | `{batch_dtype}` |
        | min value | `{batch_min:.4f}` |
        | max value | `{batch_max:.4f}` |
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition
    """)
    return


@app.cell
def _():
    class MultiLayerPerceptronBlockV1(nn.Module):
        """Two-layer fully connected block with ReLU activations."""

        def __init__(self, input_dim: int = 784, hidden_dim: int = 512):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        def __call__(self, x: mx.array) -> mx.array:
            return nn.relu(self.fc2(nn.relu(self.fc1(x))))

    class VariationalEncoderV1(nn.Module):
        """MLP encoder producing (mu, logvar) for the diagonal Gaussian posterior."""

        def __init__(
            self,
            input_dim: int = 784,
            hidden_dim: int = 512,
            latent_dim: int = 32,
        ):
            super().__init__()
            self.mlp = MultiLayerPerceptronBlockV1(input_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        def __call__(self, x: mx.array):
            h = self.mlp(x)
            return self.fc_mu(h), self.fc_logvar(h)

    class VariationalDecoderV1(nn.Module):
        """MLP decoder mapping z back to pixels in [0, 1] via sigmoid."""

        def __init__(
            self,
            latent_dim: int = 32,
            hidden_dim: int = 512,
            output_dim: int = 784,
        ):
            super().__init__()
            self.mlp = MultiLayerPerceptronBlockV1(latent_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def __call__(self, z: mx.array) -> mx.array:
            return mx.sigmoid(self.fc_out(self.mlp(z)))

    class VariationalAutoEncoderV1(nn.Module):
        """Top-level VAE composing an encoder, decoder, and reparameterization."""

        def __init__(
            self,
            input_dim: int = 784,
            hidden_dim: int = 512,
            latent_dim: int = 32,
        ):
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

    return (VariationalAutoEncoderV1,)


@app.cell
def _():
    def count_parameters(model: nn.Module) -> int:
        return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))

    def compute_vae_loss(model: nn.Module, x: mx.array) -> mx.array:
        recon, mu, logvar = model(x)
        recon_loss = mx.mean(mx.sum((recon - x) ** 2, axis=-1))
        kl_loss = -0.5 * mx.mean(
            mx.sum(1 + logvar - mu ** 2 - mx.exp(logvar), axis=-1)
        )
        return recon_loss + kl_loss

    def train_epoch(
        model: nn.Module, optimizer, train_buffer, batch_size: int
    ) -> float:
        loss_and_grad_fn = nn.value_and_grad(model, compute_vae_loss)
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_buffer.shuffle().to_stream().batch(batch_size):
            x = mx.array(batch["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
            loss, grads = loss_and_grad_fn(model, x)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            epoch_loss += loss.item()
            n_batches += 1
        return epoch_loss / max(n_batches, 1)

    def evaluate_elbo(model: nn.Module, buf, batch_size: int = 256) -> float:
        total = 0.0
        n = 0
        for batch in buf.to_stream().batch(batch_size):
            x = mx.array(batch["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
            loss = compute_vae_loss(model, x)
            mx.eval(loss)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    return compute_vae_loss, count_parameters, evaluate_elbo, train_epoch


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
def _(VariationalAutoEncoderV1, count_parameters, mo):
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
def _(
    VariationalAutoEncoderV1,
    bs_ui,
    epochs_ui,
    evaluate_elbo,
    hidden_dim_ui,
    latent_dim_ui,
    lr_ui,
    mo,
    train_btn,
    train_buf,
    train_epoch,
    val_buf,
    wd_ui,
):
    train_losses = []
    val_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin."))
    else:
        _model = VariationalAutoEncoderV1(
            input_dim=784,
            hidden_dim=int(hidden_dim_ui.value),
            latent_dim=int(latent_dim_ui.value),
        )
        mx.eval(_model.parameters())
        _optimizer = optim.AdamW(
            learning_rate=lr_ui.value, weight_decay=wd_ui.value
        )
        _n = epochs_ui.value
        _bs = int(bs_ui.value)
        for _epoch in range(_n):
            _tl = train_epoch(_model, _optimizer, train_buf, _bs)
            _vl = evaluate_elbo(_model, val_buf, _bs)
            train_losses.append(_tl)
            val_losses.append(_vl)
            mo.output.replace(
                mo.md(
                    f"**Epoch {_epoch + 1}/{_n}** — "
                    f"train: {_tl:.4f} | val: {_vl:.4f}"
                )
            )
        trained_model = _model
        mo.output.replace(
            mo.md(
                f"**Done!** Final train loss: {train_losses[-1]:.4f} | "
                f"val loss: {val_losses[-1]:.4f}"
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
    hp_search_cb = mo.ui.checkbox(
        label="Enable Hyperparameter Search", value=False
    )
    hp_search_cb
    return (hp_search_cb,)


@app.cell
def _(
    VariationalAutoEncoderV1,
    evaluate_elbo,
    hp_search_cb,
    mo,
    train_buf,
    train_epoch,
    val_buf,
):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above._"),
    )
    _space = {"lr": [1e-4, 1e-3, 3e-3], "latent_dim": [16, 32, 64]}
    _results = []
    for _lr in _space["lr"]:
        for _ld in _space["latent_dim"]:
            _m = VariationalAutoEncoderV1(
                input_dim=784, hidden_dim=512, latent_dim=_ld
            )
            mx.eval(_m.parameters())
            _opt = optim.AdamW(learning_rate=_lr)
            for _ in range(5):
                train_epoch(_m, _opt, train_buf, 128)
            _vl = evaluate_elbo(_m, val_buf)
            _results.append(
                {"lr": _lr, "latent_dim": _ld, "val_loss": round(_vl, 4)}
            )
            mo.output.replace(
                mo.md(f"lr={_lr}, latent_dim={_ld} → val={_vl:.4f}")
            )
    _results.sort(key=lambda r: r["val_loss"])
    hp_results = _results
    mo.ui.table(_results)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation
    """)
    return


@app.cell
def _(evaluate_elbo, mo, test_buf, trained_model, val_buf):
    if trained_model is None:
        _out = mo.md("_Train the model first._")
    else:
        _test_elbo = evaluate_elbo(trained_model, test_buf)
        _val_elbo = evaluate_elbo(trained_model, val_buf)
        _out = mo.md(
            f"""
    **Evaluation Results**

    | Split | ELBO Loss |
    |-------|-----------|
    | Validation | {_val_elbo:.4f} |
    | Test | {_test_elbo:.4f} |
    """
        )
    _out
    return


@app.cell
def _(VariationalAutoEncoderV1, compute_vae_loss, mo, train_ds, trained_model):
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
def _(mo, train_losses, val_losses):
    if not train_losses:
        _out = mo.md("_Train first._")
    else:
        _fig, _ax = plt.subplots(figsize=(9, 4))
        _ax.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            "b-o",
            lw=2,
            ms=4,
            label="Train ELBO",
        )
        _ax.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            "r-s",
            lw=2,
            ms=4,
            label="Val ELBO",
        )
        _ax.set_xlabel("Epoch")
        _ax.set_ylabel("ELBO Loss")
        _ax.set_title("VAE Training Curve")
        _ax.legend()
        _ax.grid(True, alpha=0.3)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo, test_buf, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first._")
    else:
        _n_show = 8
        _batch = next(iter(test_buf.to_stream().batch(_n_show)))
        _orig = mx.array(_batch["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
        _recon, _, _ = trained_model(_orig)
        mx.eval(_recon)
        _orig_np = np.array(_orig).reshape(-1, 28, 28)
        _recon_np = np.array(_recon).reshape(-1, 28, 28)
        _fig, _axes = plt.subplots(2, _n_show, figsize=(16, 4))
        for _i in range(_n_show):
            _axes[0, _i].imshow(_orig_np[_i], cmap="gray")
            _axes[0, _i].axis("off")
            _axes[1, _i].imshow(_recon_np[_i], cmap="gray")
            _axes[1, _i].axis("off")
        _axes[0, 0].set_title("Original", fontsize=9, loc="left")
        _axes[1, 0].set_title("Reconstructed", fontsize=9, loc="left")
        _fig.suptitle("VAE Reconstructions (test set)", fontsize=12)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo, reference_param_count, train_losses, val_losses):
    if not train_losses:
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX
    - **Dataset**: MNIST (60k train / 10k test, grayscale 28×28)
    - **Model**: `VariationalAutoEncoderV1` — MLP encoder / decoder with a
      32-dim Gaussian latent space
    - **Reference parameter count**: {reference_param_count:,}
    - **Loss**: pixel-sum squared error + KL(q(z|x) || N(0, I))

    Train the model above to populate the final loss numbers here.
    """
        )
    else:
        _summary = mo.md(
            f"""
    ### Summary

    - **Framework**: MLX
    - **Dataset**: MNIST (60k train / 10k test, grayscale 28×28)
    - **Model**: `VariationalAutoEncoderV1` — MLP encoder / decoder with a
      configurable Gaussian latent space
    - **Reference parameter count**: {reference_param_count:,}
    - **Final train ELBO loss**: `{train_losses[-1]:.4f}`
    - **Final validation ELBO loss**: `{val_losses[-1]:.4f}`

    The training curve above shows train / validation ELBO per epoch. The
    reconstruction grid demonstrates qualitative fidelity on held-out test
    digits. Enable the hyperparameter search checkbox for a small sweep over
    learning rate and latent dimension; the cross-validation cell reports
    5-fold ELBO with mean ± std.
    """
        )
    _summary
    return


if __name__ == "__main__":
    app.run()
