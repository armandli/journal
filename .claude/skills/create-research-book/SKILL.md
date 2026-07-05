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

## Step 2 — Delegate to the notebook-researcher Agent

Spawn the `notebook-researcher` subagent with the following self-contained prompt. Replace the placeholders before passing:

```
Write a complete ML research marimo notebook to the file `<notebook_path>` (path is relative to /home/armandli/journal, the repo root).

Research goal: <research_goal>

Framework: <MLX or PyTorch — inferred from goal>
Data domain: <image / text / tabular / audio / time-series>
Model type: <CNN / MLP / VAE / Transformer / etc.>

Working directory: /home/armandli/journal
Python environment: /home/armandli/journal/env/bin/python
marimo binary: /home/armandli/journal/env/bin/marimo

---

Read these reference files before writing (they are in /home/armandli/journal/.claude/skills/create-research-book/references/):
- NOTEBOOK-STRUCTURE.md   — marimo file format rules, cell patterns, UI element rules
- MODULE-PATTERNS.md      — neural network module naming and design rules
- TRAINING-PATTERNS.md    — training loop, HP search, evaluation, CV patterns

Also read /home/armandli/journal/.claude/CLAUDE.md for project-wide marimo conventions.

---

## Code Organization Rules (apply to every section)

These rules are mandatory. Violations are bugs, not style choices.

1. **One class per cell** — every `nn.Module` subclass gets its own `@app.class_definition` cell. Never put two classes in one cell.
2. **One function per cell** — every standalone helper gets its own `@app.function` cell. Never bundle multiple functions.
3. **No `_`-prefixed variables in cells** — extract logic into `@app.function` instead of writing local `_` work vars inside a cell. The only acceptable `_` vars are: `_out` in conditional-display cells, and short-lived lambdas (e.g. `_preprocess`) that are specific to one cell and genuinely not reusable.
4. **All dependencies explicit** — every function receives every value it needs as a parameter. No captured globals, no module-level state.
5. **Maximize reusability** — design every function and class to work for the general case, not just the immediate use. Parameterize what varies; use sensible defaults.

---

Write ALL 8 sections below. Do not skip any. Do not leave stubs or TODO comments.

### Section 1 — Title & Research Goal
A markdown cell with the notebook title, research goal statement, and a brief outline of all sections.

### Section 2 — Data Exploration
- Load or download the dataset to `../data/<dataset_name>/`
- For MLX + image data: use `mlx.data.datasets.load_mnist(root="../data/mnist", train=True)` if MNIST
- Extract each visualization as its own `@app.function` (e.g. `plot_sample_grid(dataset)`, `plot_class_distribution(dataset)`). These functions return a `fig` object — never call `plt.show()`. The cell that calls them is a single line returning the figure.
- Print train / val / test split sizes in a markdown cell

### Section 3 — Dataset Creation
- Define train/val/test splits (MNIST standard: use the built-in 60k train + 10k test; split train 85/15 for train/val)
- Normalize images to [0, 1] float32
- MLX pipeline: `.shuffle().to_stream().batch(batch_size)`
- Put the split/iterator creation logic in a single `@app.function` (e.g. `make_datasets(train_ds, test_ds, batch_size)`)
- Show one batch shape and dtype

### Section 4 — Model Definition
Follow MODULE-PATTERNS.md strictly:
- Class names embed version at end: `MultiLayerPerceptronBlockV1`, `ConvolutionBlockV1`, `VariationalAutoEncoderV1`
- Each class in its own `@app.class_definition` cell — never group classes
- No VERSION attribute; no globals; typed parameters with defaults; composition required
- Each standalone helper (`count_parameters`, `compute_loss`, etc.) in its own `@app.function` cell
- Write building blocks + at least one top-level model
- Add a markdown architecture table cell
- Instantiate model + show parameter count via `count_parameters(model)`

### Section 5 — Training
- `mo.ui.dropdown` for lr, batch_size, weight_decay; `mo.ui.slider` for epochs
- `mo.ui.run_button(label="Train")` gate
- Extract `run_train_epoch(model, loss_fn, optimizer, train_iter, preprocess_fn)` and `run_evaluate(model, loss_fn, data_iter, preprocess_fn)` as `@app.function` cells
- The training cell calls those helpers; use `mo.output.replace()` for live progress
- Return `(train_losses, trained_model)` — initialize to `([], None)` when button not clicked

### Section 6 — Hyperparameter Search (Optional)
- `mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)` gate
- `mo.stop(not hp_search_cb.value, ...)` to skip
- Grid over 2–3 hyperparameters (lr × latent_dim or similar); 3–5 epochs per config
- Results in `mo.ui.table` sorted by val_loss

### Section 7 — Validation & Cross-Validation
- Extract `evaluate_model(model, loss_fn, data_iter, preprocess_fn)` as `@app.function`
- Evaluate on the test set; report key metrics in a markdown table
- k=5 fold CV on train data; per-fold metrics + mean ± std

### Section 8 — Results
- Extract each plot as its own `@app.function` (e.g. `plot_loss_curve(train_losses, val_losses)`, `plot_reconstructions(model, test_batch)`)
- For VAE/generative models: show a grid of original vs. reconstructed images
- Model comparison table if multiple variants were defined
- Final summary markdown

---

After writing the file, run:
  /home/armandli/journal/env/bin/marimo check --fix "<notebook_path>"

Fix any remaining errors manually and re-run until clean. Report the file path, framework, architecture summary, and any assumptions made.
```

Wait for the agent to complete and relay its report to the user.

## Step 3 — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "create-research-book"
```
