# Training, Validation & Hyperparameter Search Patterns

## MLX Training Loop

```python
# Training cell template (MLX)
@app.cell
def _(ModelClass, train_buf, lr_ui, epochs_ui, bs_ui, wd_ui, train_btn):
    train_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training."))
    else:
        _model = ModelClass(...)          # instantiate with hyperparams
        mx.eval(_model.parameters())
        _optimizer = optim.AdamW(
            learning_rate=lr_ui.value,
            weight_decay=wd_ui.value,
        )

        def _loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y).mean()

        _vg_fn = nn.value_and_grad(_model, _loss_fn)
        _n_epochs = epochs_ui.value
        _bs = int(bs_ui.value)

        for _epoch in range(_n_epochs):
            _epoch_loss = 0.0
            _n_batches = 0
            _stream = train_buf.shuffle().to_stream().batch(_bs)
            for _batch in _stream:
                _x = mx.array(_batch["image"], dtype=mx.float32) / 255.0
                _y = mx.array(_batch["label"])
                _loss, _grads = _vg_fn(_model, _x, _y)
                _optimizer.update(_model, _grads)
                mx.eval(_loss, _model.parameters())
                _epoch_loss += _loss.item()
                _n_batches += 1
            _avg = _epoch_loss / max(_n_batches, 1)
            train_losses.append(_avg)
            mo.output.replace(mo.md(f"**Epoch {_epoch+1}/{_n_epochs}** — loss: {_avg:.4f}"))

        trained_model = _model
        mo.output.replace(mo.md(f"**Training complete!** Final loss: {train_losses[-1]:.4f}"))

    return train_losses, trained_model
```

## PyTorch Training Loop

```python
@app.cell
def _(ModelClass, train_loader, lr_ui, epochs_ui, wd_ui, train_btn, device):
    train_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training."))
    else:
        _model = ModelClass(...).to(device)
        _optimizer = torch.optim.AdamW(
            _model.parameters(), lr=lr_ui.value, weight_decay=wd_ui.value
        )
        _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            _optimizer, T_max=epochs_ui.value
        )
        _loss_fn = torch.nn.CrossEntropyLoss()
        _scaler = torch.cuda.amp.GradScaler()
        _n_epochs = epochs_ui.value

        _model.train()
        for _epoch in range(_n_epochs):
            _epoch_loss = 0.0
            for _x, _y in train_loader:
                _x, _y = _x.to(device), _y.to(device)
                _optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type):
                    _logits = _model(_x)
                    _loss = _loss_fn(_logits, _y)
                _scaler.scale(_loss).backward()
                _scaler.unscale_(_optimizer)
                torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
                _scaler.step(_optimizer)
                _scaler.update()
                _epoch_loss += _loss.item()
            _avg = _epoch_loss / len(train_loader)
            train_losses.append(_avg)
            _scheduler.step()
            mo.output.replace(mo.md(f"**Epoch {_epoch+1}/{_n_epochs}** — loss: {_avg:.4f}"))

        trained_model = _model
        mo.output.replace(mo.md(f"**Training complete!** Final loss: {train_losses[-1]:.4f}"))

    return train_losses, trained_model
```

## Hyperparameter Search Pattern

```python
# Gate cell
@app.cell
def _():
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)

# Search cell (MLX example)
@app.cell
def _(hp_search_cb, ModelClass, train_buf, val_buf):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._")
    )

    _search_space = {
        "lr": [1e-4, 1e-3, 3e-3],
        "latent_dim": [16, 32, 64],
    }
    _n_search_epochs = 5
    _bs = 128
    _hp_results = []

    for _lr in _search_space["lr"]:
        for _ld in _search_space["latent_dim"]:
            _model = ModelClass(latent_dim=_ld)
            mx.eval(_model.parameters())
            _opt = optim.Adam(learning_rate=_lr)
            _vg_fn = nn.value_and_grad(_model, _loss_fn_for_search)

            for _ep in range(_n_search_epochs):
                _stream = train_buf.shuffle().to_stream().batch(_bs)
                for _batch in _stream:
                    _x = mx.array(_batch["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
                    _loss, _grads = _vg_fn(_model, _x)
                    _opt.update(_model, _grads)
                    mx.eval(_loss, _model.parameters())

            # evaluate on val set
            _val_loss = 0.0
            _val_n = 0
            for _batch in val_buf.to_stream().batch(_bs):
                _x = mx.array(_batch["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
                _l, _, _ = _model(_x)
                mx.eval(_l)
                _val_loss += float(mx.mean(_l).item())
                _val_n += 1

            _hp_results.append({
                "lr": _lr,
                "latent_dim": _ld,
                "val_loss": _val_loss / max(_val_n, 1),
            })
            mo.output.replace(mo.md(f"lr={_lr}, latent_dim={_ld} — val_loss={_val_loss/_val_n:.4f}"))

    _hp_results.sort(key=lambda r: r["val_loss"])
    hp_results = _hp_results
    mo.ui.table(_hp_results)
    return (hp_results,)
```

## MLX Evaluation Pattern

```python
@app.cell
def _(trained_model, test_buf):
    if trained_model is None:
        _out = mo.md("_Train the model first._")
    else:
        _correct = 0
        _total = 0
        _stream = test_buf.to_stream().batch(256)
        for _batch in _stream:
            _x = mx.array(_batch["image"], dtype=mx.float32).reshape(-1, 784) / 255.0
            _y = mx.array(_batch["label"])
            _logits = trained_model(_x)
            mx.eval(_logits)
            _preds = mx.argmax(_logits, axis=-1)
            _correct += int(mx.sum(_preds == _y).item())
            _total += _y.shape[0]
        _acc = _correct / _total
        _out = mo.md(f"**Test Accuracy: {_acc:.4f}** ({_correct}/{_total} correct)")
    _out
    return
```

## PyTorch Evaluation Pattern

```python
@app.cell
def _(trained_model, test_loader, device):
    if trained_model is None:
        _out = mo.md("_Train the model first._")
    else:
        trained_model.eval()
        _correct = 0
        _total = 0
        with torch.no_grad():
            for _x, _y in test_loader:
                _x, _y = _x.to(device), _y.to(device)
                _preds = trained_model(_x).argmax(dim=1)
                _correct += (_preds == _y).sum().item()
                _total += _y.size(0)
        _acc = _correct / _total
        _out = mo.md(f"**Test Accuracy: {_acc:.4f}** ({_correct}/{_total} correct)")
    _out
    return
```

## k-Fold Cross-Validation Pattern (PyTorch)

```python
@app.cell
def _(ModelClass, full_dataset, device, lr_ui, epochs_ui):
    if trained_model is None:
        _out = mo.md("_Train first, then cross-validation results will appear._")
    else:
        from sklearn.model_selection import KFold
        import numpy as np

        _k = 5
        _kf = KFold(n_splits=_k, shuffle=True, random_state=42)
        _indices = np.arange(len(full_dataset))
        _fold_metrics = []

        for _fold, (_train_idx, _val_idx) in enumerate(_kf.split(_indices)):
            _train_sub = torch.utils.data.Subset(full_dataset, _train_idx)
            _val_sub = torch.utils.data.Subset(full_dataset, _val_idx)
            _tl = DataLoader(_train_sub, batch_size=64, shuffle=True)
            _vl = DataLoader(_val_sub, batch_size=256)

            _model = ModelClass(...).to(device)
            _opt = torch.optim.Adam(_model.parameters(), lr=lr_ui.value)
            _loss_fn = torch.nn.CrossEntropyLoss()

            _model.train()
            for _ep in range(epochs_ui.value):
                for _x, _y in _tl:
                    _x, _y = _x.to(device), _y.to(device)
                    _opt.zero_grad()
                    _loss_fn(_model(_x), _y).backward()
                    _opt.step()

            _model.eval()
            _correct = _total = 0
            with torch.no_grad():
                for _x, _y in _vl:
                    _x, _y = _x.to(device), _y.to(device)
                    _correct += (_model(_x).argmax(1) == _y).sum().item()
                    _total += _y.size(0)
            _fold_metrics.append(_correct / _total)
            mo.output.replace(mo.md(f"Fold {_fold+1}/{_k} — acc: {_fold_metrics[-1]:.4f}"))

        _mean = float(np.mean(_fold_metrics))
        _std = float(np.std(_fold_metrics))
        cv_results = {"fold_accuracies": _fold_metrics, "mean": _mean, "std": _std}
        _out = mo.md(f"**{_k}-Fold CV Accuracy: {_mean:.4f} ± {_std:.4f}**")
    _out
    return (cv_results,)
```

## Results Plot Patterns

```python
# Loss curve
_fig, _ax = plt.subplots(figsize=(8, 4))
_ax.plot(range(1, len(train_losses) + 1), train_losses, "b-o", linewidth=2, markersize=4, label="Train")
_ax.set_xlabel("Epoch"); _ax.set_ylabel("Loss"); _ax.set_title("Training Loss")
_ax.legend(); _ax.grid(True, alpha=0.3)
_fig.tight_layout()
_fig

# Confusion matrix (classification)
from sklearn.metrics import confusion_matrix
import seaborn as sns
_cm = confusion_matrix(all_labels, all_preds)
_fig, _ax = plt.subplots(figsize=(8, 7))
sns.heatmap(_cm, annot=True, fmt="d", cmap="Blues", ax=_ax)
_ax.set_xlabel("Predicted"); _ax.set_ylabel("True")
_ax.set_title("Confusion Matrix")
_fig.tight_layout()
_fig
```
