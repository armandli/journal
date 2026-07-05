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

Write ALL 8 sections below. Do not skip any. Do not leave stubs or TODO comments.

### Section 1 — Title & Research Goal
A markdown cell with the notebook title, research goal statement, and a brief outline of all sections.

### Section 2 — Data Exploration
- Load or download the dataset to `../data/<dataset_name>/`
- For MLX + image data: use `mlx.data.datasets.load_mnist(root="../data/mnist", train=True)` if MNIST
- Show a grid of 20–60 sample images (matplotlib, return `_fig`, never call plt.show())
- Show a class distribution bar chart
- Print train / val / test split sizes

### Section 3 — Dataset Creation
- Define train/val/test splits (MNIST standard: use the built-in 60k train + 10k test; split train 85/15 for train/val)
- Normalize images to [0, 1] float32
- MLX pipeline: `.shuffle().to_stream().batch(batch_size)` 
- Show one batch shape and dtype

### Section 4 — Model Definition
Follow MODULE-PATTERNS.md strictly:
- Class names embed version at end: `MultiLayerPerceptronBlockV1`, `ConvolutionBlockV1`, `VariationalAutoEncoderV1`
- No VERSION attribute; no globals; typed parameters with defaults; composition required
- All standalone functions snake_case; all deps explicit params
- Write building blocks + at least one top-level model
- Add a markdown architecture table cell
- Instantiate model + show parameter count via `count_parameters(model)`

### Section 5 — Training
- `mo.ui.dropdown` for lr, batch_size, weight_decay; `mo.ui.slider` for epochs
- `mo.ui.run_button(label="Train")` gate
- MLX: `nn.value_and_grad` + `mx.eval` each step; live progress via `mo.output.replace()`
- Return `(train_losses, trained_model)` — initialize to `([], None)` when button not clicked

### Section 6 — Hyperparameter Search (Optional)
- `mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)` gate
- `mo.stop(not hp_search_cb.value, ...)` to skip
- Grid over 2–3 hyperparameters (lr × latent_dim or similar); 3–5 epochs per config
- Results in `mo.ui.table` sorted by val_loss

### Section 7 — Validation & Cross-Validation
- Evaluate ELBO (reconstruction + KL) on the test set
- k=5 fold CV on train data; per-fold metrics + mean ± std

### Section 8 — Results
- Training loss curve (matplotlib, return `_fig`)
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
