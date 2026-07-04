# Conversion Patterns Reference

Detailed before/after examples for each conversion category applied during `jupyter-to-marimo`.

---

## C01 — Magic Commands

### Display-only magics: delete entirely

```python
# BEFORE (converted cell)
@app.cell
def _():
    %matplotlib inline
    return

# AFTER
@app.cell
def _():
    return
```

Magics to silently delete (no replacement needed):
- `%matplotlib inline`
- `%matplotlib notebook`
- `%matplotlib agg`
- `%autoreload 2` (and other autoreload variants)
- `%load_ext autoreload`

### Timing magics: delete the magic line, keep the body

```python
# BEFORE
@app.cell
def _(model, X_test):
    %%time
    predictions = model.predict(X_test)
    return (predictions,)

# AFTER
@app.cell
def _(model, X_test):
    predictions = model.predict(X_test)
    return (predictions,)
```

### `%pip install` → PEP 723 dependency

```python
# BEFORE (cell with pip magic)
@app.cell
def _():
    %pip install numpy pandas torch
    return

# AFTER: delete the cell body (or the whole cell if it only contained the magic)
# AND add to PEP 723 block at top of file:
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "torch",
# ]
# ///
```

### Unknown cell-level magics → `# REVIEW:` comment

```python
# BEFORE
@app.cell
def _():
    %%bash
    echo "hello"
    return

# AFTER
@app.cell
def _():
    # REVIEW: magic command: %%bash
    # echo "hello"
    return
```

```python
# BEFORE
@app.cell
def _():
    %load data.py
    return

# AFTER
@app.cell
def _():
    # REVIEW: magic command: %load data.py
    return
```

---

## C02 — IPython Display Calls

### Single object → final expression

```python
# BEFORE
@app.cell
def _(df):
    display(df)
    return

# AFTER
@app.cell
def _(df):
    df
    return
```

### `HTML` → `mo.Html`

```python
# BEFORE
@app.cell
def _():
    from IPython.display import HTML
    display(HTML("<h1>Title</h1>"))
    return

# AFTER
@app.cell
def _(mo):
    mo.Html("<h1>Title</h1>")
    return
```

### `Markdown` → `mo.md`

```python
# BEFORE
@app.cell
def _():
    from IPython.display import Markdown
    display(Markdown("## Section"))
    return

# AFTER
@app.cell
def _(mo):
    mo.md("## Section")
    return
```

### `Image` → `mo.image`

```python
# BEFORE
@app.cell
def _():
    from IPython.display import Image
    display(Image("plot.png"))
    return

# AFTER
@app.cell
def _(mo):
    mo.image("plot.png")
    return
```

### Multiple displays → `mo.vstack`

```python
# BEFORE
@app.cell
def _(df, fig, mo):
    display(df)
    display(fig)
    return

# AFTER
@app.cell
def _(df, fig, mo):
    mo.vstack([df, fig])
    return
```

### Remove IPython imports after replacing all uses

After replacing all `display`, `HTML`, `Markdown`, and `Image` calls, remove the IPython import lines:

```python
# Remove these lines:
from IPython.display import display
from IPython.display import HTML, Markdown, Image
import IPython
```

If a cell only contained IPython imports and no other logic, make it return nothing:

```python
@app.cell
def _():
    return
```

---

## C03 — Setup Cell Import Consolidation

### Standard setup cell structure

```python
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import typer
    import numpy as np
    import pandas as pd
    import torch
    from pathlib import Path
```

### What to move into the setup cell

Move any `import` or `from ... import` statement that:
- Has no meaningful side effects tied to that cell's position
- Is used in multiple cells
- Is a standard library or third-party package import

### What to keep in its original cell

Keep an import in its original cell only when:
- It conditionally imports based on runtime logic (e.g., `if torch.cuda.is_available(): import cupy`)
- It has a required ordering side effect (e.g., `import matplotlib; matplotlib.use('Agg')` before other matplotlib imports)

### Detecting a missing setup cell

A converted notebook has no setup cell when the pattern `with app.setup:` does not appear anywhere in the file.

**Create a setup cell** immediately after `app = marimo.App(...)`:

```python
# BEFORE (no setup cell)
app = marimo.App(width="medium")

@app.cell
def _():
    import numpy as np
    import pandas as pd
    ...

# AFTER
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import typer
    import numpy as np
    import pandas as pd
```

Then remove the import statements from the individual cells and update each cell's argument list to remove the now-redundant locally-imported names.

### Updating cell arguments after consolidation

After moving imports to setup, remove the imported names from `def _(...)` argument lists only if they were brought in by that cell's own import. Names that flow from other cells via return values must stay in the argument list.

```python
# BEFORE (np imported locally)
@app.cell
def _():
    import numpy as np
    arr = np.zeros((3, 3))
    return (arr,)

# AFTER (np is now in setup cell, so no arg needed)
@app.cell
def _():
    arr = np.zeros((3, 3))
    return (arr,)
```

---

## C04 — Anti-Pattern: try/except Control Flow

### Remove when except silences all errors

```python
# BEFORE
@app.cell
def _(data_loader):
    try:
        batch = next(data_loader)
    except Exception:
        batch = None
    return (batch,)

# AFTER
@app.cell
def _(data_loader):
    batch = next(data_loader)
    return (batch,)
```

```python
# BEFORE
@app.cell
def _(model, X):
    try:
        predictions = model.predict(X)
    except Exception as e:
        print(f"Error: {e}")
        predictions = []
    return (predictions,)

# AFTER
@app.cell
def _(model, X):
    predictions = model.predict(X)
    return (predictions,)
```

### Keep try/except when catching a specific exception with real recovery

```python
# KEEP — specific exception, meaningful recovery
@app.cell
def _(model_path):
    try:
        model = torch.load(model_path)
    except FileNotFoundError:
        model = MyModel()  # create default if checkpoint missing
    return (model,)
```

The rule: if `except` catches `Exception` (or bare `except`) and the recovery is `return None/[]` or printing, remove it. If it catches a named specific exception and does something meaningful, keep it.

---

## C05 — Anti-Pattern: Indented Final Expression

### Ternary fix when the condition is a simple boolean

```python
# BEFORE
@app.cell
def _(mo, show_plot, fig):
    if show_plot:
        fig  # won't render — indented

# AFTER
@app.cell
def _(mo, show_plot, fig):
    fig if show_plot else None
```

### Ternary fix for conditional output

```python
# BEFORE
@app.cell
def _(mo, results):
    if results:
        mo.md(f"Accuracy: {results['acc']:.2%}")

# AFTER
@app.cell
def _(mo, results):
    mo.md(f"Accuracy: {results['acc']:.2%}") if results else None
```

### Remove guard when the cell has a hard dependency anyway

If the cell already receives `results` as an argument, marimo will not run it until `results` is available — the guard is redundant:

```python
# BEFORE
@app.cell
def _(results):
    if results is not None:
        acc = results["acc"]
        loss = results["loss"]
        acc, loss

# AFTER
@app.cell
def _(results):
    acc = results["acc"]
    loss = results["loss"]
    acc, loss
```

---

## C06 — Anti-Pattern: Overly Guarded Cells

### Remove `if dependency:` wrapping the entire cell body

```python
# BEFORE
@app.cell
def _(model, X_test, y_test):
    if model:
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        score

# AFTER
@app.cell
def _(model, X_test, y_test):
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    score
```

```python
# BEFORE
@app.cell
def _(df):
    if df is not None:
        summary = df.describe()
        summary

# AFTER
@app.cell
def _(df):
    summary = df.describe()
    summary
```

**Rationale:** Marimo only executes a cell when all its arguments are available and non-erroring. Guarding on `if model:` or `if df is not None:` is redundant — if `model` is `None` or `df` is `None`, the upstream cell that produced it was the appropriate place to handle that case.

---

## C07 — PEP 723 Block

### Full structure

Place at the very top of the file, before `import marimo`:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "typer",
#     "numpy",
#     "pandas",
# ]
# ///
```

### Import → PyPI name mapping

| Import statement | PyPI package name |
|-----------------|-------------------|
| `import numpy` | `numpy` |
| `import pandas` | `pandas` |
| `import polars` | `polars` |
| `import torch` | `torch` |
| `import torchvision` | `torchvision` |
| `from sklearn` | `scikit-learn` |
| `import cv2` | `opencv-python` |
| `from PIL` | `pillow` |
| `import matplotlib` | `matplotlib` |
| `import altair` | `altair` |
| `import scipy` | `scipy` |
| `import seaborn` | `seaborn` |
| `import plotly` | `plotly` |
| `import requests` | `requests` |
| `import typer` | `typer` |
| `import marimo` | `marimo` |
| `import tqdm` | `tqdm` |
| `import yaml` | `pyyaml` |
| `import bs4` | `beautifulsoup4` |
| `import sklearn` | `scikit-learn` |
| `import xgboost` | `xgboost` |
| `import lightgbm` | `lightgbm` |
| `import transformers` | `transformers` |
| `import datasets` | `datasets` |
| `import accelerate` | `accelerate` |
| `import diffusers` | `diffusers` |

For import names not in this table, use the import name as the PyPI package name and add a `# REVIEW:` comment.

### Version pinning

Only add version constraints when the notebook explicitly requires a minimum version (e.g., a feature introduced in a specific release). Otherwise leave bare package names.

---

## C08 — `marimo check` Error Recovery

### SyntaxError near `%`

A magic command was not replaced. Find any remaining `%` or `%%` lines and apply the rules from C01.

```
SyntaxError: invalid syntax
  File "notebook.py", line 42
    %matplotlib inline
    ^
```

Fix: delete `%matplotlib inline` from that cell.

### NameError: name not defined

The variable was used in IPython's flat namespace but is now not in scope.

```
NameError: name 'display' is not defined
```

Fix: find remaining `display(...)` calls and apply C02 transformations.

```
NameError: name 'np' is not defined
```

Fix: check that `numpy as np` is in the setup cell and that the cell's `def _(...)` does not need `np` as an explicit argument (it won't, since setup cell names are available everywhere).

### Cell name collision

```
ValueError: duplicate cell name '_'
```

Fix: rename duplicate `@app.cell` function names. Append `_2`, `_3`, etc. to later occurrences:

```python
# BEFORE (two cells named _)
@app.cell
def _():
    x = 1
    return (x,)

@app.cell
def _():
    y = 2
    return (y,)

# AFTER
@app.cell
def _():
    x = 1
    return (x,)

@app.cell
def _2():
    y = 2
    return (y,)
```

### TypeError on display

```
TypeError: display() takes 0 positional arguments
```

Fix: find remaining `display(...)` calls that were not converted and apply C02 transformations.

### NameError for IPython globals

```
NameError: name 'In' is not defined
NameError: name 'Out' is not defined
NameError: name 'get_ipython' is not defined
```

Fix: delete any cells or lines that reference `In`, `Out`, `_i`, `_ii`, `_oh`, `_ih`, or `get_ipython(...)`. These are IPython history globals with no marimo equivalent.
