---
name: jupyter-to-marimo
description: Converts a Jupyter notebook (.ipynb) to a marimo notebook (.py) by running marimo convert, then reviewing and fixing the output for magic commands, IPython display calls, anti-patterns, setup cell completeness, and missing PEP 723 metadata. Use when the user asks to "convert a Jupyter notebook to marimo", "migrate an ipynb to marimo", "turn this notebook into a marimo file", or "convert ipynb to py marimo". Do NOT use for creating new marimo notebooks from scratch (use marimo-notebook), for explaining marimo concepts, or for converting formats other than .ipynb to marimo .py.
argument-hint: "[notebook.ipynb]"
---

# Jupyter to Marimo Conversion

## Step 1: Validate Input

- Confirm `$ARGUMENTS` ends in `.ipynb`; if not, stop and ask the user for the correct filename.
- Confirm the file exists; if not, stop and report the missing file.
- Derive the output path by replacing `.ipynb` with `.py` (same directory).

## Step 2: Run marimo convert

```bash
uvx marimo convert <input.ipynb> -o <output.py>
```

If the command fails, stop and report the error. Do not proceed with a partial or missing output file.

## Step 3: Read and Audit the Converted File

Read the full output file before making any edits. Scan for:

- Magic commands: lines starting with `%` or `%%`
- IPython display calls: `display(...)`, `HTML(...)`, `Markdown(...)`, `Image(...)`
- Missing or incomplete `with app.setup:` block
- Imports scattered across non-setup cells
- `try/except` used for control flow (bare `except Exception` returning None)
- Final expressions that are indented inside `if` blocks
- Overly guarded cells: `if dependency:` wrapping the entire cell body
- Missing PEP 723 `# /// script` block at the top of the file
- Duplicate `@app.cell` function names

## Step 4: Fix Issues

Apply fixes in this order. Read the full file before each category of edits.

### 4a: Magic Commands

| Magic | Action |
|-------|--------|
| `%matplotlib inline` | Delete (marimo renders matplotlib automatically) |
| `%matplotlib notebook` | Delete |
| `%load_ext` | Delete or `# REVIEW:` comment |
| `%%time`, `%timeit` | Delete the magic line, keep the cell body |
| `%pip install pkg` | Move `pkg` to PEP 723 dependencies; delete the magic line |
| `%conda install pkg` | Move `pkg` to PEP 723 dependencies; delete the magic line |
| `%%bash`, `%%sh` | Replace with `subprocess.run(...)` or `# REVIEW:` comment |
| Other `%` / `%%` | Replace line with `# REVIEW: magic command: <original line>` |

### 4b: IPython Display Calls

- `display(obj)` as the only statement → make `obj` the final (unindented) expression
- `display(HTML(s))` → `mo.Html(s)`
- `display(Markdown(s))` → `mo.md(s)`
- `display(Image(path))` → `mo.image(path)`
- Multiple `display(...)` calls in one cell → `mo.vstack([item1, item2, ...])`
- Remove `from IPython.display import ...` and `import IPython` lines after replacing all uses

### 4c: Import Consolidation

Move all top-level `import` and `from ... import` statements into the `with app.setup:` block.

**Keep imports in their own cell only if:** the import has a meaningful side effect that must run at that point (e.g., setting a backend before other code runs).

If no `with app.setup:` block exists, create one immediately after `app = marimo.App(...)`:

```python
with app.setup:
    import marimo as mo
    import typer
    # ... other imports
```

Always include `import marimo as mo` in the setup cell. Add `import typer` if the notebook has CLI-style argument cells.

### 4d: Anti-Pattern Fixes

**try/except control flow** — remove when the except clause just silences errors:
```python
# BEFORE
try:
    result = compute()
except Exception:
    result = None

# AFTER
result = compute()
```
Keep try/except only when catching a specific named exception with real recovery logic.

**Indented final expression** — fix when the only output is inside an `if` block:
```python
# BEFORE
if condition:
    mo.md("text")  # won't render

# AFTER
mo.md("text") if condition else None
```

**Overly guarded cells** — remove `if dependency:` wrapping the entire cell body:
```python
# BEFORE
if model:
    predictions = model.predict(X)
    predictions

# AFTER
predictions = model.predict(X)
predictions
```

**Duplicate cell function names** — append `_2`, `_3`, etc. to later duplicates.

## Step 5: Add PEP 723 Block

If the file does not already have a `# /// script` block, add one at the very top (before any other content):

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "typer",
#     "<package>",
# ]
# ///
```

Populate `dependencies` by scanning the setup cell imports. Use this import→PyPI name mapping:

| Import name | PyPI package |
|-------------|--------------|
| `numpy` | `numpy` |
| `pandas` | `pandas` |
| `polars` | `polars` |
| `torch` | `torch` |
| `sklearn` | `scikit-learn` |
| `cv2` | `opencv-python` |
| `PIL` | `pillow` |
| `matplotlib` | `matplotlib` |
| `altair` | `altair` |
| `scipy` | `scipy` |
| `seaborn` | `seaborn` |
| `plotly` | `plotly` |
| `requests` | `requests` |
| `typer` | `typer` |

Always include `marimo` and `typer` in dependencies.

See [references/CONVERSION-PATTERNS.md](references/CONVERSION-PATTERNS.md) for full pattern examples.

## Step 6: Validate

```bash
uvx marimo check <output.py>
```

If errors are reported, fix and re-run up to **3 cycles**. Common errors and fixes:

| Error | Fix |
|-------|-----|
| `SyntaxError` near `%` | A magic command was missed; remove it |
| `NameError: <name>` | Variable defined in IPython global scope; move to setup cell or add as cell arg |
| Cell name collision | Rename duplicate `@app.cell` function |
| `TypeError` on display | Replace remaining `display(...)` call |

If validation still fails after 3 cycles, leave remaining issues as `# REVIEW:` comments and report them in Step 7.

## Step 7: Report

Provide a structured summary:

```
## Conversion Summary: <input.ipynb> → <output.py>

**Removed:** <n> magic commands, <n> IPython imports
**Converted:** <n> display() calls, <n> magic commands → PEP 723
**Fixed:** <n> anti-patterns (try/except, indented expr, guards)
**Added:** PEP 723 block with dependencies: [<list>]
**Validation:** PASSED / FAILED (see # REVIEW comments)

**Review Items:** (if any)
- Line <n>: <description of unrecognized pattern>
```

## Safety Rules

- Read the full file before making edits in each category.
- Never delete entire cells — only modify or comment out problematic lines within cells.
- Preserve the original cell ordering exactly.
- Never alter the `__generated_with` version string.
- Leave `# REVIEW: <original>` comments for any pattern you cannot confidently convert.
- Apply one edit pass per category (4a → 4b → 4c → 4d) without re-reading between minor edits.
