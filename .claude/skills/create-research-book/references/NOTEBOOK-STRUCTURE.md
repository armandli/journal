# Research Notebook Structure Reference

## Marimo File Format (0.23.13+)

```python
import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    # All third-party library imports go here
    # Do NOT import marimo here

@app.cell
def _():
    import marimo as mo
    return

@app.cell
def _():
    # ... cell content ...
    return (exported_var,)
```

### Cell Decorators

| Decorator | Purpose |
|-----------|---------|
| `@app.cell` | Reactive cell — the standard cell for logic, UI, and display |
| `@app.function` | Standalone reusable function — **one function per cell** |
| `@app.class_definition` | Reusable class definition — **one class per cell** |

`@app.function` and `@app.class_definition` are globally visible across all cells without being in any return tuple. Use them for every helper function and every `nn.Module` subclass.

### Rules
- `with app.setup:` — third-party imports only (mlx, torch, numpy, matplotlib, etc.)
- First `@app.cell` — `import marimo as mo` with plain `return`
- **Avoid `_`-prefixed variables** — extract logic into `@app.function` instead of accumulating `_work_vars` inside a cell. The only acceptable `_` usage: `_out` in conditional-display cells, and one-off lambdas (e.g. `_preprocess`) that are genuinely cell-specific.
- Cross-cell variables: returned explicitly in `return (var1, var2)`
- No variable redeclaration across cells
- Last expression in a cell is displayed (if not inside an `if` block)
- Never use `global`

### Conditional Display Pattern

Use `_out` as the single cell-local variable in conditional-display cells. Extract the actual computation (plotting, evaluation) into a `@app.function` above so the cell stays minimal.

```python
@app.function
def plot_results(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data)
    fig.tight_layout()
    return fig


@app.cell
def _(data, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first._")
    else:
        _out = plot_results(data)
    _out
    return
```

### Training Cell Pattern

```python
@app.cell
def _(train_btn, model_class, train_data, lr_ui, epochs_ui, bs_ui, ...):
    train_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin."))
    else:
        # build model, optimizer, run loop
        ...
        mo.output.replace(mo.md(f"Training complete — final loss: {train_losses[-1]:.4f}"))

    return train_losses, trained_model
```

### Hyperparameter Search Gate Pattern

```python
@app.cell
def _():
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)

@app.cell
def _(hp_search_cb, ...):
    mo.stop(not hp_search_cb.value, mo.md("_Enable hyperparameter search above to run this section._"))
    # ... hp search code ...
    return (hp_results,)
```

## Section Order

| # | Section | Key outputs |
|---|---------|-------------|
| 1 | Title & Research Goal | Markdown only |
| 2 | Data Exploration | `train_buf`/`train_ds`, sample visualizations |
| 3 | Dataset Creation | `train_loader`, `val_loader`, `test_loader` |
| 4 | Model Definition | `ModelClass`, `model_summary` |
| 5 | Training | `train_losses`, `trained_model` |
| 6 | Hyperparameter Search | `hp_results` (optional) |
| 7 | Validation & Cross-Validation | `val_metrics`, `cv_results` |
| 8 | Results | Plots, tables, summary markdown |

## Matplotlib Display Rules

Always extract matplotlib code into a `@app.function` that returns the `fig` object. The calling cell is then a single-line expression. Never call `plt.show()`.

```python
@app.function
def plot_sample_grid(dataset, n_show: int = 40, rows: int = 5, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    for i in range(n_show):
        sample = dataset[i]
        img = np.array(sample["image"]).squeeze()
        label = int(np.array(sample["label"]).item())
        r, c = divmod(i, cols)
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].set_title(str(label), fontsize=9)
        axes[r, c].axis("off")
    fig.suptitle("Sample images", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(train_ds):
    plot_sample_grid(train_ds)
    return
```

For conditional plots (depends on a trained model), use `_out`:

```python
@app.cell
def _(trained_model, test_batch):
    if trained_model is None:
        _out = mo.md("_Train first._")
    else:
        _out = plot_reconstructions(trained_model, test_batch)
    _out
    return
```

## UI Element Rules

- Define UI elements in one cell, access `.value` in a later cell
- Never access `.value` in the same cell where the element is defined
- Use `mo.hstack([...])` / `mo.vstack([...])` for layout
- Standard controls for training:

```python
lr_ui = mo.ui.dropdown(
    options={"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "3e-3": 3e-3},
    value="1e-3", label="Learning Rate"
)
epochs_ui = mo.ui.slider(1, 100, value=20, step=1, label="Epochs")
bs_ui = mo.ui.dropdown(options=[32, 64, 128, 256], value=64, label="Batch Size")
wd_ui = mo.ui.dropdown(
    options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2},
    value="1e-4", label="Weight Decay"
)
mo.vstack([
    mo.md("### Hyperparameters"),
    mo.hstack([lr_ui, epochs_ui]),
    mo.hstack([bs_ui, wd_ui]),
])
```
