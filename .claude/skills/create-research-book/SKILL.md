---
name: create-research-book
description: Generate a complete ML research marimo notebook (.py file) from a research goal. Produces sections for data exploration, dataset creation, versioned reusable neural network modules (MLX or PyTorch), hyperparameter-controlled training, optional hyperparameter search, cross-validation, and results reporting. Use when user asks to "create a research notebook", "generate a research book for [topic]", "set up an ML experiment for [goal]", or "make a marimo research template". Arguments: [notebook_name.py] [research goal sentence]. Do NOT use for general marimo editing (use marimo-notebook), for non-ML topics, or when no research goal is given.
argument-hint: "[notebook_name.py] [research goal description]"
---

# create-research-book

Generate a structured ML research marimo notebook from two arguments:
- `$ARGUMENTS[0]` — output filename (e.g. `mlx/vae_experiment.py`)
- `$ARGUMENTS[1]` — research goal sentence (e.g. "train a VAE on MNIST to learn a disentangled latent space")

## Step 1 — Parse Arguments

Extract:
- `notebook_path` = `$ARGUMENTS[0]` (the `.py` file path, relative to repo root)
- `research_goal` = `$ARGUMENTS[1]` (the research question / objective)

Infer from the research goal:
- **Framework**: default to MLX if the goal mentions Apple Silicon, MLX, or on-device; otherwise use PyTorch. If unclear, use MLX.
- **Data domain**: image, text, tabular, audio, or time-series — infer from the goal
- **Model type**: CNN, Transformer, MLP, RNN, VAE, GAN, diffusion, etc.

## Step 2 — Create the Notebook File

Write a marimo notebook to `notebook_path`. The notebook must use the modern marimo format:

```python
import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    # all third-party imports here (NOT marimo itself)

@app.cell
def _():
    import marimo as mo
    return

# ... all other cells as @app.cell functions
```

Follow all rules from CLAUDE.md and the [notebook structure reference](references/NOTEBOOK-STRUCTURE.md).

## Step 3 — Write All Notebook Sections in Order

Write every section below. Do not skip any. Each section is one or more marimo cells.

### Section 1 — Title & Research Goal

A markdown cell with:
- Notebook title (derived from research goal)
- Research goal statement
- Brief outline of the sections

### Section 2 — Data Exploration

- Load or download the dataset. Save raw data to `../data/<dataset_name>/`
- Use mlx-data (`mlx.data.datasets`) if a built-in loader exists, otherwise use torchvision, HuggingFace datasets, or direct download
- Show 20–60 sample data points (image grid for images, head() for tabular, waveform plot for audio)
- For **tabular data**: show `.describe()` statistics, null counts, column types, correlation heatmap
- For **image data**: show sample grid with labels, class distribution bar chart, examples per class grid
- For **text data**: show sample sentences, token length distribution, vocabulary statistics
- Display dataset size (train/val/test split sizes)

### Section 3 — Dataset Creation

- Define train/val/test splits (default: 70/15/15 or use standard splits if they exist)
- Apply any preprocessing: normalization, tokenization, augmentation
- Build data pipeline using `mlx.data` streams (for MLX) or `torch.utils.data.DataLoader` (for PyTorch)
- Show one preprocessed batch to confirm shapes and dtypes
- For MLX: use `.shuffle().to_stream().batch(batch_size)` pattern
- For PyTorch: use `DataLoader` with `num_workers`, `pin_memory`

### Section 4 — Model Definition

Follow the **Module Design Rules** in [references/MODULE-PATTERNS.md](references/MODULE-PATTERNS.md) strictly:
- Class names use PascalCase with the version number embedded at the end: `MultiLayerPerceptronV1`, `ResidualBlockV1`, `TransformerBlockV1`. Never a separate `VERSION` attribute.
- To upgrade a module, define a new class with the incremented suffix: `MultiLayerPerceptronV2`.
- Modules must be fully self-contained — no references to global variables or outer scope.
- Modules must compose other custom modules (not just raw framework primitives).
- Each module's `__init__` must have explicit typed parameters with defaults.
- All standalone functions use snake_case and receive every dependency as an explicit parameter — no global variable access: `def train_model(model, data, lr, epochs):` not `def train_model():`.
- Provide a markdown cell documenting the architecture (table listing each component class, its version, and output shape).

Write at minimum:
1. One or more **building-block modules** (e.g. `ResidualBlockV1`, `AttentionHeadV1`, `MultiLayerPerceptronV1`)
2. One **top-level model class** that composes the building blocks (e.g. `ResearchCnnV1`)
3. Standalone helper functions: `count_parameters(model)`, `compute_loss(model, batch)`, `train_model(model, optimizer, data, epochs)`, `evaluate_model(model, data)`
4. A cell that instantiates the model and shows parameter count

### Section 5 — Training

- UI controls (sliders/dropdowns) for: learning rate, batch size, epochs, weight decay
- A `mo.ui.run_button(label="Train")` to trigger training
- Training loop with per-epoch loss logged via `mo.output.replace()`
- For MLX: use `nn.value_and_grad`, `mx.eval` at each step; see [references/TRAINING-PATTERNS.md](references/TRAINING-PATTERNS.md)
- For PyTorch: use `autocast`, `GradScaler`, `clip_grad_norm_`
- Return `(train_losses, trained_model)` from the training cell; initialize to `([], None)` when button not clicked

### Section 6 — Hyperparameter Search (Optional)

- A `mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)` gate
- Use `mo.stop` to skip this section when unchecked
- Grid-search or random-search over 2–3 key hyperparameters (lr, hidden_dim, latent_dim)
- Run short training runs (3–5 epochs) per configuration
- Show results as a table (`mo.ui.table`) sorted by validation loss

### Section 7 — Validation & Cross-Validation

- Evaluate the trained model on the held-out test set
- Compute and display relevant metrics (accuracy, F1, MSE, ELBO, FID — match to task)
- Implement k-fold cross-validation (k=5 default) on the full dataset
- Show per-fold metrics and mean ± std
- For MLX: evaluate with `model.eval()` (if applicable) and `mx.eval()`
- For PyTorch: evaluate with `model.eval()` and `torch.no_grad()`

### Section 8 — Results

- Training loss curve (matplotlib line plot)
- Validation metrics summary table
- For image models: show reconstructions or generated samples
- For classification: confusion matrix heatmap
- Final summary markdown cell stating what was learned / achieved

## Step 4 — Verify

After writing the file, run:

```bash
/home/armandli/journal/env/bin/marimo check --fix "$ARGUMENTS[0]"
```

If errors remain after `--fix`, read the error output and fix them manually, then re-run.

## Step 5 — Report

Tell the user:
- The file path created
- The framework used (MLX or PyTorch)
- The model architecture summary
- Any assumptions made about the dataset

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "create-research-book"
```
