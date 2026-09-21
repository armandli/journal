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
    mo.md("""
    # Flow Matching with a Diffusion Transformer on CIFAR-10 (MLX)

    ## Research Goal

    Train a **class-conditional flow-matching generative model** on
    **CIFAR-10** using a **Diffusion Transformer (DiT)** backbone
    implemented in **MLX**. The DiT depth (`num_layers`) is a first-class
    UI parameter. At inference time we integrate the learned velocity field
    with three ODE solvers — **Euler**, **Midpoint / Heun**, and
    **classical Runge-Kutta 4 (RK4)** — with a UI-controlled step count.

    ### Method
    - Linear interpolation path: `x_t = (1 - t) * x0 + t * x1` with
      `x0 ~ N(0, I)` and `x1` a real CIFAR-10 image
    - Target velocity is constant along the path: `v_t = x1 - x0`
    - Training loss: `MSE(model(x_t, t, y), v_t)`
    - Inference: solve `dx/dt = v_theta(x, t, y)` from `t=0` to `t=1`

    ### Notebook Outline
    1. Title & research goal (this cell) — followed by the **Debugging Log**
    2. Data exploration
    3. Dataset creation
    4. Model definition (DiT + flow-matching loss + ODE solvers)
    5. Training loop
    6. Optional hyperparameter search
    7. Validation & cross-validation
    8. Results — loss curves, generated samples, solver comparison
    9. Save trained model
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 1b — Debugging Log

    Both models in this notebook once trained to a validation loss of ~1.5 and stayed
    there. No exception, no NaN, no warning — just a flat curve. Five defects were
    responsible. Each is recorded here with its symptom, root cause, why it was
    invisible, and the measurement that exposed it.

    | # | Defect | Symptom | Fix |
    |---|---|---|---|
    | 1 | Fixed constants registered as trainable parameters | V2 frozen at 1.4142 for 30 epochs | `_`-prefix the constants |
    | 2 | Rotation tables rebuilt inside the gradient graph | wasted compute per step | cache in `__init__` |
    | 3 | Timestep embedding with no dynamic range | forces a ~50x gain in the conditioning path | `time_scale=1000.0` |
    | 4 | Unbounded attention logits (AdaLN runaway) | silent divergence, both models | QK normalization |
    | 5 | No instrument to detect 1–4 | every failure looked identical | two verification rites |

    ---

    ### The number that unmasked all of it: `pi/2`

    Every failure parked itself near **1.5**. That is not a coincidence and not "the
    model is still learning". For `x_t = (1-t) x0 + t x1` with unit-variance data, the
    best possible predictor that sees only a **single pixel** of `x_t` and the time `t`
    — no spatial structure whatsoever — achieves

    $$\mathbb{E}_t\!\left[\frac{1}{(1-t)^2 + t^2}\right] = \int_0^1 \frac{dt}{2t^2 - 2t + 1} = \frac{\pi}{2} \approx 1.5708$$

    So a flow-matching loss sitting at ~1.5 means the network is extracting **zero**
    spatial information. Deriving that floor turned a vague "it isn't learning" into a
    precise claim: *the spatial pathway is dead*. Compute the floor for your own loss;
    it converts a plateau into a diagnosis.

    ---

    ### Defect 1 — MLX silently trains any array you name without a `_`

    `mlx.nn.Module.valid_parameter_filter` is literally:

    ```python
    return isinstance(value, (dict, list, mx.array)) and not key.startswith("_")
    ```

    MLX has **no `register_buffer`**. Registration is decided by the attribute *name*.
    So `self.freqs = ...` handed the RoPE frequency bank and the sinusoidal timestep
    bank straight to AdamW. Read back out of the trained checkpoints:

    | | canonical | after training |
    |---|---|---|
    | RoPE `theta_0..7` (block 0) | `[1, .316, .1, .0316, .01, .00316, .001, .000316]` | `[1.19, .55, .048, .015, .077, .26, -.026, -.106]` |
    | timestep `theta_0..5` | `[1.0, .931, .866, .806, .750, .698]` | `[1.258, 1.280, 1.232, 1.033, 1.128, .985]` |

    The RoPE bank lost its sign as well as its scale. The timestep bank lost its
    *monotonicity* — the decay is no longer a ladder, so the embedding's multi-scale
    time resolution is gone (L2 drift from canonical: 3.07).

    At `lr=1e-3` AdamW moves each entry ~1e-3 per step. The geometric decay is ground
    into noise within one epoch, the rotation angles shift on *every* step, and
    attention can never settle on a spatial pattern. **Fix:** `_freqs`, `_row_cos`,
    `_row_sin`, `_col_cos`, `_col_sin`. Deliberately learnable tables such as
    `pos_embed` correctly stay bare.

    **Audit any MLX model before training:**

    ```python
    print([k for k, _ in mlx.utils.tree_flatten(model.trainable_parameters())])
    ```

    *Where checkpoints already exist,* prefer `self.freeze(keys=["freqs"], recurse=False)`
    instead: it keeps the key in `parameters()` so old weights still load, while
    excluding it from gradients and weight decay.

    ---

    ### Defect 2 — Recomputing fixed geometry every forward pass

    Grid positions and their cos/sin tables were rebuilt on every call, inside the
    gradient graph, in all six blocks. They depend only on `grid_size`, so they are
    now built once in `__init__`. The old code also did `repeat(cos, 2)` and then
    immediately strided it back with `[::2]` — a round trip that cancelled itself.
    Not a correctness bug, but waste is its own kind of defect.

    ---

    ### Defect 3 — A clock the model could not read

    The bank `theta_i = 10000^(-i/d)` spans `1 -> 1e-4`. That range assumes **DDPM
    integer timesteps** `t in [0, 1000]`. Flow matching feeds `t in [0, 1]`, so every
    angle is at most 1 radian and the embedding is nearly constant in `t`:

    ```
    std of the sinusoidal embedding across t  =  0.0199      (before)
                                              =  0.4690      (after time_scale=1000)
    ```

    With a signal that weak, the network must learn a ~50x amplifier just to tell one
    timestep from another — and that high-gain conditioning path is what made training
    fragile. **Fix:** `time_scale=1000.0`, restoring the range the formula was designed
    for. A formula copied from one regime does not carry its assumptions with it.

    ---

    ### Defect 4 — Unbounded attention logits, in *both* models

    AdaLN feeds each block `LayerNorm(x) * (1 + scale)`, so attention logits grow as
    `scale^2`. Nothing bounded `scale`, and sharpening attention lowers the loss — so
    the optimizer drove it up until softmax became a hard argmax. Measured in the two
    stalled checkpoints against a healthy run:

    | Signal | healthy | stalled V2 | stalled V1 |
    |---|---|---|---|
    | `cond` RMS | 6.40 | 45.43 | 22.64 |
    | AdaLN `scale` RMS | 0.89 | **423.3** | **148.8** |
    | final residual RMS | 3.87 | **325 820** | **24 650** |
    | attention entropy | not measured | **0.0000** | **0.0000** |
    | max attention logit | — | — | **5.8e7** |

    (The healthy column is the original V1 checkpoint, which predates the entropy
    probe. Healthy runs measured afterwards hold entropy between 1.8 and 3.6, against
    a uniform-attention maximum of `ln 64 = 4.159`.)

    Entropy of exactly zero means every query attends to precisely one key. No gradient
    flows through a saturated softmax: the model is **dead**, not slow. It still
    reported a finite ~1.5 loss because `final_norm` renormalizes the blow-up — which
    is exactly why the failure was invisible.

    **Fix:** QK normalization — `nn.RMSNorm(head_dim)` applied to q and k per head, the
    ViT-22B / SD3 / Flux remedy. It decouples attention sharpness from the AdaLN scale.
    `nn.MultiHeadAttention` offers no such option, so V1 uses `QKNormAttentionV1`
    (identical to `RoPE2DAttentionV2` minus the rotation).

    **The hardest lesson here.** This was first diagnosed as a V2-only problem, on the
    theory that V2 lacks `pos_embed` and so has no other lever to sharpen attention.
    That theory was wrong: V1 diverged the same way as soon as its trajectory was
    perturbed. V1's original success was **luck, not immunity**. A single run that
    happens to work is not evidence that a design is stable.

    Also note what did **not** work — each was tested in isolation and still diverged:

    | Attempted fix alone | Result |
    |---|---|
    | `time_scale=1000` only | `scale` 1.67 -> 17.7, diverged |
    | zero-init the AdaLN projection only | `scale` 2.45 -> 11.1, diverged |
    | QK norm only | survived, but residual stayed ~530 and convergence was slow |
    | QK norm **+** `time_scale` | `val 0.42`, residual self-stabilized to ~3 |

    ---

    ### Defect 5 — There was no instrument

    Every one of the above produced the same flat ~1.5 curve. A loss value alone cannot
    distinguish "untrained", "unlucky" and "catastrophically diverged". Two rites now
    live in this notebook:

    - **`verify_rope2d_v2`** (Section 4b) — asserts no fixed constant reaches
      `trainable_parameters()`, and confirms numerically that the attention score
      depends only on the *relative* patch offset (holds to ~1e-6, float32's limit).
    - **`diagnose_training_health`** (Sections 8 and 8b) — reports AdaLN `scale` RMS,
      final residual RMS, minimum attention entropy, and time-signal strength. Run it
      on **every** trained model before trusting a loss number.

    ---

    ### Known remaining gap — reproducibility

    This notebook does **not** reproduce run to run. `make_batches` calls unseeded
    `np.random.permutation`, and `compute_flow_loss` draws `t` and `x0` from the
    unseeded global MLX RNG. Two byte-identical runs under `mx.random.seed(7)` gave
    epoch-2 losses of **1.6414 vs 1.4615**.

    Consequence: **any A/B gap smaller than ~0.2 at short horizons is noise.** V1 and V2
    both land near 0.42 at 20 000 images / 12 epochs; that is not evidence they are
    equivalent. To fix this properly, thread an explicit `mx.random.key` and a
    `np.random.Generator` through `make_batches` and `compute_flow_loss` rather than
    relying on global state.

    ---

    ### Five transferable rules

    1. **Compute your loss's floor.** A plateau at an unexplained value is a clue, not
       a mystery. `pi/2` turned "not learning" into "the spatial pathway is dead".
    2. **Print `trainable_parameters()` before training.** In MLX the parameter set is
       decided by attribute naming, and getting it wrong fails silently.
    3. **Measure activations, not just loss.** `final_norm` will hide a 10^5 blow-up
       behind a perfectly finite number.
    4. **Attention entropy is the vital sign of a transformer.** Zero means a hard
       argmax and dead gradients, long before the loss admits anything is wrong.
    5. **One run proves nothing.** An intermittent failure is still a failure; a lucky
       success is still a bug.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration
    """)
    return


@app.function
def cifar10_class_names() -> list:
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


@app.function
def load_cifar10_arrays(
    root: str = "../data/cifar10",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
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
    x_train_np, y_train_np, x_test_np, y_test_np, class_names = load_cifar10_arrays("../data/cifar10")
    return class_names, x_test_np, x_train_np, y_test_np, y_train_np


@app.cell
def _(mo, x_test_np, x_train_np):
    mo.md(f"""
    ### Dataset overview

    CIFAR-10 is loaded via `mlx.data.datasets.load_cifar10` from
    `../data/cifar10/` (each split's Buffer is collapsed into a single
    full batch, then materialised to NumPy). Each image is a `float32`
    array shaped `(32, 32, 3)` in `[0, 1]`; labels are `int32` in `[0, 9]`.

    | Split | Size | Shape |
    |-------|------|-------|
    | Train (raw) | {x_train_np.shape[0]:,} | {tuple(x_train_np.shape[1:])} |
    | Test | {x_test_np.shape[0]:,} | {tuple(x_test_np.shape[1:])} |
    """)
    return


@app.function
def plot_sample_grid(images: np.ndarray, labels: np.ndarray, class_names: list, n_show: int = 40, cols: int = 8):
    rows = int(math.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        if i < n_show:
            ax.imshow(np.clip(images[i], 0.0, 1.0))
            ax.set_title(class_names[int(labels[i])], fontsize=8)
        ax.axis("off")
    fig.suptitle("CIFAR-10 sample images", fontsize=13)
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, x_train_np, y_train_np):
    plot_sample_grid(x_train_np, y_train_np, class_names, n_show=40)
    return


@app.function
def plot_class_distribution(labels: np.ndarray, class_names: list):
    counts = np.bincount(labels.astype(np.int64), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(class_names)), counts, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 class distribution (train)")
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
    val_fraction: float = 0.1,
):
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    n_total = x_train_norm.shape[0]
    n_val = int(n_total * val_fraction)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    x_tr = mx.array(x_train_norm[tr_idx])
    y_tr = mx.array(y_train[tr_idx].astype(np.int32))
    x_val = mx.array(x_train_norm[val_idx])
    y_val = mx.array(y_train[val_idx].astype(np.int32))
    x_te = mx.array(x_test_norm)
    y_te = mx.array(y_test.astype(np.int32))
    return x_tr, y_tr, x_val, y_val, x_te, y_te


@app.cell
def _(x_test_np, x_train_np, y_test_np, y_train_np):
    x_tr, y_tr, x_val, y_val, x_te, y_te = make_datasets(
        x_train_np, y_train_np, x_test_np, y_test_np, val_fraction=0.1
    )
    return x_te, x_tr, x_val, y_te, y_tr, y_val


@app.function
def make_batches(x: mx.array, y: mx.array, batch_size: int = 128, shuffle: bool = True) -> list:
    n = x.shape[0]
    if shuffle:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_idx = mx.array(idx[start:end].astype(np.int32))
        batches.append((x[b_idx], y[b_idx]))
    return batches


@app.cell
def _(mo, x_tr, y_tr):
    _sample_batches = make_batches(x_tr, y_tr, batch_size=128, shuffle=True)
    _xb, _yb = _sample_batches[0]
    mo.md(
        f"""
        ### One-batch check

        - Number of training batches (bs=128): **{len(_sample_batches):,}**
        - `x_batch.shape` = `{tuple(_xb.shape)}`, dtype = `{_xb.dtype}`
        - `y_batch.shape` = `{tuple(_yb.shape)}`, dtype = `{_yb.dtype}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition
    """)
    return


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, embed_dim: int = 256, time_scale: float = 1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        # The frequency bank spans 1 -> 1e-4, which assumes DDPM-style integer
        # timesteps t in [0, 1000]. Flow matching feeds t in [0, 1], so without
        # rescaling every angle is <= 1 rad, the embedding is nearly constant in t
        # (std across t ~ 0.02), and the model must learn a ~50x amplifier just to
        # read the clock. That high-gain conditioning path is what destabilises
        # training. Scaling t back to [0, 1000] restores the intended range.
        self.time_scale = time_scale
        half = embed_dim // 2
        # underscore prefix keeps this out of Module.parameters(): the sinusoidal
        # frequency bank is a fixed constant, never an optimizer target
        self._freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / max(half, 1))

    def __call__(self, t: mx.array) -> mx.array:
        a = (t * self.time_scale)[:, None] * self._freqs[None, :]
        return mx.concatenate([mx.sin(a), mx.cos(a)], axis=-1)


@app.class_definition
class AdaptiveLayerNormV1(nn.Module):
    def __init__(self, dim: int = 256, cond_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        scale, shift = mx.split(self.proj(nn.silu(cond))[:, None, :], 2, axis=-1)
        return self.norm(x) * (1.0 + scale) + shift


@app.class_definition
class PatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x).reshape(x.shape[0], -1, self.proj.weight.shape[0])


@app.class_definition
class QKNormAttentionV1(nn.Module):
    """Multi-head attention with QK normalization and no positional injection.

    Identical to `RoPE2DAttentionV2` except that it applies no rotation: V1 receives
    position from the learned `pos_embed` table added to the patch embeddings. Keeping
    the two attentions otherwise identical is what makes the V1-vs-V2 comparison a
    clean ablation of the positional encoding alone.

    This replaces `nn.MultiHeadAttention`, which offers no QK normalization. Without
    it, AdaLN feeds the block `LayerNorm(x) * (1 + scale)`, attention logits grow as
    `scale^2`, and training diverges to a saturated hard-argmax attention (entropy 0)
    from which no gradient escapes. Parameter count is unchanged: MLX's
    `MultiHeadAttention` also defaults to `bias=False`, so only the two RMSNorm gains
    (2 * head_dim per block) are added.
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
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(B, N, H, hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, N, H, hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, hd).transpose(0, 2, 1, 3)
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5), axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


@app.class_definition
class DiTBlockV1(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, cond_dim: int = 256):
        super().__init__()
        self.attn_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.attn = QKNormAttentionV1(dim, num_heads)
        self.mlp_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        h = self.attn_norm(x, cond)
        x = x + self.attn(h)
        return x + self.mlp(self.mlp_norm(x, cond))


@app.class_definition
class UnpatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 256, out_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.grid = image_size // patch_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        b = x.shape[0]
        p, c, g = self.patch_size, self.out_channels, self.grid
        h = self.proj(x).reshape(b, g, g, p, p, c)
        return h.transpose(0, 1, 3, 2, 4, 5).reshape(b, g * p, g * p, c)


@app.class_definition
class DiffusionTransformerV1(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2
        self.patchify = PatchifyV1(patch_size, embed_dim, in_channels)
        self.pos_embed = mx.zeros((1, num_patches, embed_dim))
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.blocks = [DiTBlockV1(embed_dim, num_heads, mlp_dim, embed_dim) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatchify = UnpatchifyV1(patch_size, embed_dim, in_channels, image_size)

    def __call__(self, x: mx.array, t: mx.array, y: mx.array) -> mx.array:
        h = self.patchify(x) + self.pos_embed
        cond = self.time_embed(t) + self.class_embed(y)
        for block in self.blocks:
            h = block(h, cond)
        return self.unpatchify(self.final_norm(h))


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def compute_flow_loss(model: nn.Module, x1: mx.array, y: mx.array, rng_key_unused=None) -> mx.array:
    t = mx.random.uniform(shape=(x1.shape[0],))
    x0 = mx.random.normal(shape=x1.shape)
    t_view = t.reshape(-1, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = model(x_t, t, y)
    return mx.mean((v_pred - v_target) ** 2)


@app.function
def euler_solve(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = model(x, t, y)
        x = x + dt * v
        mx.eval(x)
    return x


@app.function
def euler_solve_trajectory(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> list:
    dt = 1.0 / num_steps
    x = x0
    trajectory = [x]
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = model(x, t, y)
        x = x + dt * v
        mx.eval(x)
        trajectory.append(x)
    return trajectory


@app.function
def midpoint_solve(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        t_mid = mx.full((x.shape[0],), i * dt + 0.5 * dt, dtype=mx.float32)
        k1 = model(x, t, y)
        x_mid = x + 0.5 * dt * k1
        k2 = model(x_mid, t_mid, y)
        x = x + dt * k2
        mx.eval(x)
    return x


@app.function
def rk4_solve(model: nn.Module, x0: mx.array, y: mx.array, num_steps: int = 50) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t0 = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        t_mid = mx.full((x.shape[0],), i * dt + 0.5 * dt, dtype=mx.float32)
        t_end = mx.full((x.shape[0],), (i + 1) * dt, dtype=mx.float32)
        k1 = model(x, t0, y)
        k2 = model(x + 0.5 * dt * k1, t_mid, y)
        k3 = model(x + 0.5 * dt * k2, t_mid, y)
        k4 = model(x + dt * k3, t_end, y)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        mx.eval(x)
    return x


@app.cell
def _(mo):
    mo.md("""
    ### Model Architecture — `DiffusionTransformerV1`

    | Component | Module | Output Shape |
    |-----------|--------|--------------|
    | Patch embedding | `PatchifyV1` (Conv2d, stride=patch) | `(B, N, D)` where `N = (H/p)^2` |
    | Positional embedding | learnable table `(1, N, D)` | `(B, N, D)` |
    | Time embedding | `SinusoidalTimestepEmbeddingV1` + MLP | `(B, D)` |
    | Class embedding | `nn.Embedding(C+1, D)` (null token for CFG) | `(B, D)` |
    | Backbone | `DiTBlockV1 x num_layers` (AdaLN + `QKNormAttentionV1`, AdaLN + MLP) | `(B, N, D)` |
    | Head | `LayerNorm` + `UnpatchifyV1` | `(B, H, W, C)` |

    Defaults: `image_size=32, patch_size=4, embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6`.
    """)
    return


@app.cell
def _():
    default_model = DiffusionTransformerV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=6,
        dropout=0.0,
    )
    mx.eval(default_model.parameters())
    default_param_count = count_parameters(default_model)
    return (default_param_count,)


@app.cell
def _(default_param_count, mo):
    mo.md(f"""
    **Default model parameter count**: `{default_param_count:,}`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Section 4b — Positional Encoding V2: 2D Rotary Position Embedding

    #### 1-D RoPE — Foundations

    RoPE (Su et al., 2021) encodes token position $m$ by **rotating** consecutive
    head-dimension pairs $(x_{2i},\; x_{2i+1})$ of the Q and K vectors by angle
    $m\theta_i$, where the base frequencies decay geometrically:

    $$\theta_i = 10000^{-2i/d}, \quad i \in \left[0,\; \tfrac{d}{2}\right)$$

    The 2×2 rotation applied to pair $i$ at position $m$:

    $$\begin{pmatrix}\tilde{x}_{2i}\\\tilde{x}_{2i+1}\end{pmatrix}
    = \begin{pmatrix}\cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}
    \begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}$$

    **Relative-position property.** Since $\mathbf{R}(m)^{\top}\mathbf{R}(n) = \mathbf{R}(m-n)$,
    the rotated inner product $\langle\mathbf{R}(m)\mathbf{q},\;\mathbf{R}(n)\mathbf{k}\rangle$
    depends only on the relative offset $m - n$, giving translation-equivariance for free.

    ---

    #### 2-D Extension — VisionLLaMA (arxiv:2403.13298)

    For image patches at 2-D grid position $(r, c)$ we must encode **two** coordinates
    while preserving the relative-position property in both directions.

    **Key idea**: split the head dimension $d$ into two halves and apply independent 1-D
    RoPE to each, using a different position argument:

    | Head-dim slice | Position used | Encodes |
    |---|---|---|
    | $[0,\; d/2)$ — *row half* | row index $r$ | vertical relative offset $r - r'$ |
    | $[d/2,\; d)$ — *col half* | column index $c$ | horizontal relative offset $c - c'$ |

    The attention score between patches $(r, c)$ and $(r', c')$ factorises:

    $$\langle\mathbf{q},\mathbf{k}\rangle
    = \underbrace{\langle\mathbf{q}_r^{(r)},\;\mathbf{k}_r^{(r')}\rangle}_{\text{depends on }r-r'}
    + \underbrace{\langle\mathbf{q}_c^{(c)},\;\mathbf{k}_c^{(c')}\rangle}_{\text{depends on }c-c'}$$

    Both terms individually satisfy the 1-D relative-position property, so the full 2-D
    attention score depends only on the relative displacement $(r - r',\; c - c')$.

    ---

    #### Frequency Formula for 2-D RoPE

    Each half has size $d/2$.  Substituting $d/2$ as the effective dimension into the
    base-frequency formula:

    $$\theta_i = 10000^{-2i/(d/2)} = 10000^{-4i/d}, \quad i \in \left[0,\; \tfrac{d}{4}\right)$$

    Both halves share the same frequency bank $\{\theta_i\}$ — only the position argument
    differs ($r$ vs $c$).  With `embed_dim=256, num_heads=8` → `head_dim=32` → `quarter=8`
    unique frequencies per head.

    ---

    #### Implementation: `_apply_rotary`

    The rotation tables are **built once in `__init__`** — positions and
    frequencies are fixed geometry, so there is nothing to recompute per step.

    ```python
    quarter   = head_dim // 4
    freqs[i]  = 10000 ** (-4*i / head_dim)          # (quarter,)

    # for position vector p ∈ {row_pos, col_pos}, shape (N,):
    angles    = p[:, None] * freqs[None, :]         # (N, quarter)
    cos, sin  = cos(angles), sin(angles)            # (N, quarter) — cached

    # _apply_rotary(x, cos, sin),  x: (B, N, H, half):
    pairs  = x.reshape(B, N, H, quarter, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]           # even / odd of each pair
    rot_0  = x0 * cos - x1 * sin
    rot_1  = x0 * sin + x1 * cos
    return stack([rot_0, rot_1], -1).reshape(B, N, H, half)
    ```

    `DiffusionTransformerV2` drops the learnable `pos_embed` table
    (≈ `num_patches × embed_dim` = `64 × 256 = 16 384` scalars) and injects
    position via fixed RoPE rotations directly into every Q and K projection.

    ---

    #### ⚠ MLX pitfall — fixed geometry must not become a parameter

    `mlx.nn.Module.valid_parameter_filter` registers **every** `mx.array`
    attribute as a learnable parameter unless its key starts with `_`:

    ```python
    return isinstance(value, (dict, list, mx.array)) and not key.startswith("_")
    ```

    A RoPE frequency bank stored as `self.freqs` is therefore handed to the
    optimizer. At `lr=1e-3` AdamW moves each θ_i by ~1e-3 per step, so after a
    single epoch the geometric decay `[1, 0.32, …, 3.2e-4]` is ground into
    noise (observed: `[1.19, 0.55, …, -0.11]`). The rotation angles then shift
    on *every* step, the attention lattice can never settle on a spatial
    pattern, and the loss stalls at the **no-spatial-information floor**

    $$\mathbb{E}_t\!\left[\tfrac{1}{(1-t)^2 + t^2}\right] = \int_0^1 \frac{dt}{2t^2 - 2t + 1} = \frac{\pi}{2} \approx 1.571$$

    which is the MSE of the best predictor that sees only `(x_t, t)` per pixel.
    All RoPE constants here are `_`-prefixed for exactly this reason.

    ---

    #### Second failure mode — silent divergence to the same floor

    Fixing the frequency bank is necessary but **not sufficient**. V2 still stalls at
    the same `pi/2` floor, intermittently, for an unrelated reason: it *diverges*, and
    `final_norm` hides it. Two compounding defects:

    **1. The timestep embedding has almost no dynamic range.** The bank spans
    `1 -> 1e-4`, which assumes DDPM integer timesteps `t in [0, 1000]`. Flow matching
    feeds `t in [0, 1]`, so every angle is <= 1 rad and the embedding is nearly constant:
    `std across t = 0.0199`. The model must learn a ~50x amplifier just to read the
    clock, which forces the conditioning path into a high-gain regime
    (measured `cond` RMS 45.4 in the stalled run, vs 6.4 in healthy V1).

    **2. Attention logits are unbounded.** AdaLN feeds each block
    `LayerNorm(x) * (1 + scale)`, so logits grow as `scale^2`. V2 has no `pos_embed`,
    so its only lever for sharpening attention is that shared `scale` — and it drives
    it up until softmax saturates. Measured in the stalled checkpoint:

    | Signal | healthy V1 | stalled V2 |
    |---|---|---|
    | `cond` RMS | 6.40 | 45.43 |
    | AdaLN `scale` RMS | 0.89 | **423.3** |
    | final residual RMS | 3.87 | **325 820** |
    | attention entropy | — | **0.0000** (uniform = 4.159) |

    Entropy 0 means every query attends to exactly one key: a hard argmax, through
    which no gradient flows. The model is dead, but the loss reads a finite 1.50
    because `final_norm` renormalizes the blow-up.

    **The fixes**, both applied above and both required:
    `time_scale=1000.0` in `SinusoidalTimestepEmbeddingV1` restores the intended
    angle range, and **QK normalization** (`nn.RMSNorm` on q and k per head — the
    ViT-22B / SD3 / Flux remedy) decouples attention sharpness from the AdaLN scale.
    Measured over 14 epochs on 8 192 images: monotone descent to `val 0.4502` with the
    network self-stabilizing (residual `213 -> 3.7`, scale `3.2 -> 1.5`, entropy never
    below 1.77). Neither fix alone was sufficient — both were tested in isolation and
    both still diverged.

    **This is not a V2 problem.** The AdaLN runaway is architecture-independent; V1
    merely got lucky on its first run. Given the corrected timestep range but still
    using `nn.MultiHeadAttention` (no QK norm), V1 diverged the same way — AdaLN
    `scale` 148.8, residual 24 650, attention entropy 0.0000 in all six blocks, logits
    reaching 5.8e7, final loss 1.4326. So V1 uses `QKNormAttentionV1`, which is
    `RoPE2DAttentionV2` minus the rotation. With it, V1 reaches `val 0.4212` in 12
    epochs on 20 000 images and `scale` settles at 0.89 — the same value the original
    healthy V1 held.

    Keeping QK norm in **both** models also preserves the ablation: V1 and V2 now
    differ by exactly the `pos_embed` table and the RoPE rotation, nothing else.
    """)
    return


@app.class_definition
class RoPE2DAttentionV2(nn.Module):
    def __init__(self, dim: int, num_heads: int, grid_size: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        quarter = self.head_dim // 4
        # θ_i = 10000^(-4i/head_dim), shared by row-half and col-half.
        # Every attribute below is underscore-prefixed so MLX keeps it OUT of
        # Module.parameters(): RoPE frequencies and grid positions are fixed
        # geometry. Registering them as parameters lets AdamW scramble the
        # frequency bank every step, which destroys positional structure.
        freqs = 1.0 / (10000.0 ** (mx.arange(0, quarter, dtype=mx.float32) * 4.0 / self.head_dim))
        # row-major patch order: patch (r, c) → index r*grid_size + c
        row_pos = mx.array([r for r in range(grid_size) for _ in range(grid_size)], dtype=mx.float32)
        col_pos = mx.array([c for _ in range(grid_size) for c in range(grid_size)], dtype=mx.float32)
        # cached rotation tables, (N, quarter): one angle per (position, frequency)
        self._row_cos, self._row_sin = self._rotation_table(row_pos, freqs)
        self._col_cos, self._col_sin = self._rotation_table(col_pos, freqs)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # QK normalization (ViT-22B / SD3 / Flux). AdaLN feeds this block
        # LayerNorm(x) * (1 + scale), so attention logits grow as scale^2. With no
        # bound, the model drives `scale` up to sharpen attention until softmax
        # saturates (entropy -> 0) and every gradient through it dies. Normalizing
        # q and k per head decouples attention sharpness from the AdaLN scale.
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    @staticmethod
    def _rotation_table(positions: mx.array, freqs: mx.array):
        # positions: (N,) row or col grid indices; freqs: (quarter,)
        # angles → cos/sin: (N, quarter), one entry per rotated dimension-pair
        angles = positions[:, None] * freqs[None, :]
        return mx.cos(angles), mx.sin(angles)

    def _apply_rotary(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x:   (B, N, H, half)   where half = head_dim // 2
        # cos/sin: (N, quarter)  where quarter = half // 2
        B, N, H, half = x.shape
        pairs = x.reshape(B, N, H, half // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]          # (B, N, H, half//2) each
        c = cos[None, :, None, :]                       # (1, N, 1, half//2)
        s = sin[None, :, None, :]                       # (1, N, 1, half//2)
        rot0 = x0 * c - x1 * s
        rot1 = x0 * s + x1 * c
        return mx.stack([rot0, rot1], axis=-1).reshape(B, N, H, half)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        half = hd // 2

        q = self.q_norm(self.q_proj(x).reshape(B, N, H, hd))
        k = self.k_norm(self.k_proj(x).reshape(B, N, H, hd))
        v = self.v_proj(x).reshape(B, N, H, hd)

        # split each head into row-half [0, half) and col-half [half, hd)
        q_r, q_c = q[..., :half], q[..., half:]
        k_r, k_c = k[..., :half], k[..., half:]

        # apply 1-D RoPE independently on each half with the matching cached table
        q_r = self._apply_rotary(q_r, self._row_cos, self._row_sin)
        k_r = self._apply_rotary(k_r, self._row_cos, self._row_sin)
        q_c = self._apply_rotary(q_c, self._col_cos, self._col_sin)
        k_c = self._apply_rotary(k_c, self._col_cos, self._col_sin)

        q = mx.concatenate([q_r, q_c], axis=-1).transpose(0, 2, 1, 3)  # (B, H, N, hd)
        k = mx.concatenate([k_r, k_c], axis=-1).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


@app.class_definition
class DiTBlockV2(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, cond_dim: int = 256, grid_size: int = 8):
        super().__init__()
        self.attn_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.attn = RoPE2DAttentionV2(dim, num_heads, grid_size)
        self.mlp_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        h = self.attn_norm(x, cond)
        x = x + self.attn(h)
        return x + self.mlp(self.mlp_norm(x, cond))


@app.class_definition
class DiffusionTransformerV2(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.patchify = PatchifyV1(patch_size, embed_dim, in_channels)
        # no pos_embed table — position injected via 2D RoPE into every Q/K projection
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.blocks = [DiTBlockV2(embed_dim, num_heads, mlp_dim, embed_dim, self.grid_size) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatchify = UnpatchifyV1(patch_size, embed_dim, in_channels, image_size)

    def __call__(self, x: mx.array, t: mx.array, y: mx.array) -> mx.array:
        h = self.patchify(x)
        cond = self.time_embed(t) + self.class_embed(y)
        for block in self.blocks:
            h = block(h, cond)
        return self.unpatchify(self.final_norm(h))


@app.cell
def _(mo):
    _v1 = DiffusionTransformerV1(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6,
    )
    _v2 = DiffusionTransformerV2(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6,
    )
    mx.eval(_v1.parameters())
    mx.eval(_v2.parameters())
    _n1 = count_parameters(_v1)
    _n2 = count_parameters(_v2)
    mo.md(f"""
    ### V1 vs V2 Parameter Counts

    | Model | Parameters | Notes |
    |-------|-----------|-------|
    | `DiffusionTransformerV1` | `{_n1:,}` | learnable `pos_embed` table + `QKNormAttentionV1` |
    | `DiffusionTransformerV2` | `{_n2:,}` | 2-D RoPE, no `pos_embed` + `RoPE2DAttentionV2` |
    | Difference | `{_n1 - _n2:+,}` | exactly `num_patches × embed_dim` = `{64 * 256:,}` |

    The two attentions are identical apart from the RoPE rotation, and both carry QK
    normalization, so this difference is *entirely* the `pos_embed` table — the
    comparison is a clean ablation of the positional encoding alone.
    """)
    return


@app.function
def verify_rope2d_v2(model: nn.Module) -> dict:
    fixed_names = ("freqs", "_freqs", "_row_cos", "_row_sin", "_col_cos", "_col_sin")
    trainable = [k for k, _ in mlx.utils.tree_flatten(model.trainable_parameters())]
    leaked = [k for k in trainable if k.rsplit(".", 1)[-1] in fixed_names]

    attn = model.blocks[0].attn
    g = model.grid_size
    n = g * g
    hd = attn.head_dim
    half = hd // 2
    # identical content in every patch: any variation in score is purely positional
    x = mx.repeat(mx.random.normal(shape=(1, 1, model.embed_dim)), n, axis=1)
    q = attn.q_norm(attn.q_proj(x).reshape(1, n, attn.num_heads, hd))
    k = attn.k_norm(attn.k_proj(x).reshape(1, n, attn.num_heads, hd))
    q = mx.concatenate(
        [
            attn._apply_rotary(q[..., :half], attn._row_cos, attn._row_sin),
            attn._apply_rotary(q[..., half:], attn._col_cos, attn._col_sin),
        ],
        axis=-1,
    ).transpose(0, 2, 1, 3)
    k = mx.concatenate(
        [
            attn._apply_rotary(k[..., :half], attn._row_cos, attn._row_sin),
            attn._apply_rotary(k[..., half:], attn._col_cos, attn._col_sin),
        ],
        axis=-1,
    ).transpose(0, 2, 1, 3)
    scores = np.array((q @ k.transpose(0, 1, 3, 2))[0, 0])

    offset_groups = {}
    for i in range(n):
        for j in range(n):
            offset_groups.setdefault((i // g - j // g, i % g - j % g), []).append(scores[i, j])
    spread = max(float(np.max(v) - np.min(v)) for v in offset_groups.values())
    return {
        "leaked": leaked,
        "offset_spread": spread,
        "score_range": float(scores.max() - scores.min()),
    }


@app.cell
def _(mo):
    _v2_check = DiffusionTransformerV2(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=256, num_heads=8, mlp_dim=512, num_layers=6,
    )
    mx.eval(_v2_check.parameters())
    _rep = verify_rope2d_v2(_v2_check)
    _p1 = "PASS" if not _rep["leaked"] else "FAIL"
    _p2 = "PASS" if _rep["offset_spread"] < 1e-3 else "FAIL"
    _p3 = "PASS" if _rep["score_range"] > 0.1 else "FAIL"
    mo.md(f"""
    ### Rite of Verification — 2-D RoPE

    | Check | Result | Verdict |
    |---|---|---|
    | Fixed constants leaked into `trainable_parameters()` | `{_rep["leaked"] or "none"}` | **{_p1}** |
    | Max score spread within one relative-offset class | `{_rep["offset_spread"]:.2e}` | **{_p2}** |
    | Positional score dynamic range | `{_rep["score_range"]:.3f}` | **{_p3}** |

    Placing identical content in every patch makes any variation in the attention
    score purely positional. Check 2 confirms that
    $\\langle \\mathbf{{R}}(m)\\mathbf{{q}},\\; \\mathbf{{R}}(n)\\mathbf{{k}}\\rangle$
    depends only on the relative displacement $m - n$, to float32 precision.
    Check 1 guards the failure mode that previously stalled V2: a frequency bank
    silently registered as a trainable parameter.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch Size",
    )
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    epochs_ui = mo.ui.slider(1, 100, value=30, step=1, label="Epochs")
    num_layers_ui = mo.ui.slider(1, 12, value=6, step=1, label="DiT num_layers")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Hyperparameters"),
            mo.hstack([lr_ui, bs_ui, wd_ui]),
            mo.hstack([epochs_ui, num_layers_ui]),
            train_btn,
        ]
    )
    return bs_ui, epochs_ui, lr_ui, num_layers_ui, train_btn, wd_ui


@app.function
def run_train_epoch(model: nn.Module, loss_and_grad_fn, optimizer, train_batches: list) -> float:
    epoch_loss = 0.0
    n = 0
    for xb, yb in train_batches:
        loss, grads = loss_and_grad_fn(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += loss.item()
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def run_evaluate(model: nn.Module, val_batches: list) -> float:
    total = 0.0
    n = 0
    for xb, yb in val_batches:
        loss = compute_flow_loss(model, xb, yb)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@app.cell
def _(
    bs_ui,
    epochs_ui,
    lr_ui,
    mo,
    num_layers_ui,
    train_btn,
    wd_ui,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses = []
    val_losses = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to begin training."))
    else:
        _model = DiffusionTransformerV1(
            image_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=256,
            num_heads=8,
            mlp_dim=512,
            num_layers=num_layers_ui.value,
            dropout=0.0,
        )
        mx.eval(_model.parameters())
        _optimizer = optim.AdamW(learning_rate=lr_ui.value, weight_decay=wd_ui.value)
        _loss_and_grad_fn = nn.value_and_grad(_model, compute_flow_loss)
        _n_epochs = epochs_ui.value
        _val_batches = make_batches(x_val, y_val, batch_size=bs_ui.value, shuffle=False)
        for epoch in range(_n_epochs):
            _train_batches = make_batches(x_tr, y_tr, batch_size=bs_ui.value, shuffle=True)
            tl = run_train_epoch(_model, _loss_and_grad_fn, _optimizer, _train_batches)
            vl = run_evaluate(_model, _val_batches)
            train_losses.append(tl)
            val_losses.append(vl)
            mo.output.replace(
                mo.md(f"**Epoch {epoch + 1}/{_n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model = _model
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train: {train_losses[-1]:.4f} | val: {val_losses[-1]:.4f}"
            )
        )
    return train_losses, trained_model, val_losses


@app.cell
def _(mo):
    mo.md("""
    ## Section 5b — Training V2 (2D RoPE)
    """)
    return


@app.cell
def _(mo):
    lr_ui_v2 = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Learning Rate",
    )
    bs_ui_v2 = mo.ui.dropdown(
        options={"64": 64, "128": 128, "256": 256},
        value="128",
        label="Batch Size",
    )
    wd_ui_v2 = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight Decay",
    )
    epochs_ui_v2 = mo.ui.slider(1, 100, value=30, step=1, label="Epochs")
    num_layers_ui_v2 = mo.ui.slider(1, 12, value=6, step=1, label="DiT num_layers")
    train_btn_v2 = mo.ui.run_button(label="Train V2")
    mo.vstack(
        [
            mo.md("### Hyperparameters (V2 — 2D RoPE)"),
            mo.hstack([lr_ui_v2, bs_ui_v2, wd_ui_v2]),
            mo.hstack([epochs_ui_v2, num_layers_ui_v2]),
            train_btn_v2,
        ]
    )
    return (
        bs_ui_v2,
        epochs_ui_v2,
        lr_ui_v2,
        num_layers_ui_v2,
        train_btn_v2,
        wd_ui_v2,
    )


@app.cell
def _(
    bs_ui_v2,
    epochs_ui_v2,
    lr_ui_v2,
    mo,
    num_layers_ui_v2,
    train_btn_v2,
    wd_ui_v2,
    x_tr,
    x_val,
    y_tr,
    y_val,
):
    train_losses_v2 = []
    val_losses_v2 = []
    trained_model_v2 = None

    if not train_btn_v2.value:
        mo.output.replace(mo.md("Click **Train V2** to begin training the 2D RoPE model."))
    else:
        _model_v2 = DiffusionTransformerV2(
            image_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=256,
            num_heads=8,
            mlp_dim=512,
            num_layers=num_layers_ui_v2.value,
            dropout=0.0,
        )
        mx.eval(_model_v2.parameters())
        _optimizer_v2 = optim.AdamW(learning_rate=lr_ui_v2.value, weight_decay=wd_ui_v2.value)
        _loss_and_grad_fn_v2 = nn.value_and_grad(_model_v2, compute_flow_loss)
        _n_epochs_v2 = epochs_ui_v2.value
        _val_batches_v2 = make_batches(x_val, y_val, batch_size=bs_ui_v2.value, shuffle=False)
        for _epoch_v2 in range(_n_epochs_v2):
            _train_batches_v2 = make_batches(x_tr, y_tr, batch_size=bs_ui_v2.value, shuffle=True)
            _tl_v2 = run_train_epoch(_model_v2, _loss_and_grad_fn_v2, _optimizer_v2, _train_batches_v2)
            _vl_v2 = run_evaluate(_model_v2, _val_batches_v2)
            train_losses_v2.append(_tl_v2)
            val_losses_v2.append(_vl_v2)
            mo.output.replace(
                mo.md(f"**Epoch {_epoch_v2 + 1}/{_n_epochs_v2}** — train: {_tl_v2:.4f} | val: {_vl_v2:.4f}")
            )
        trained_model_v2 = _model_v2
        mo.output.replace(
            mo.md(
                f"**V2 training complete!** Final train: {train_losses_v2[-1]:.4f} | val: {val_losses_v2[-1]:.4f}"
            )
        )
    return train_losses_v2, trained_model_v2, val_losses_v2


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (Optional)
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_config(
    x_tr_arr: mx.array,
    y_tr_arr: mx.array,
    x_val_arr: mx.array,
    y_val_arr: mx.array,
    lr: float,
    num_layers: int,
    n_epochs: int,
    batch_size: int,
) -> float:
    model = DiffusionTransformerV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=num_layers,
        dropout=0.0,
    )
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    val_batches = make_batches(x_val_arr, y_val_arr, batch_size=batch_size, shuffle=False)
    for _ in range(n_epochs):
        train_batches = make_batches(x_tr_arr, y_tr_arr, batch_size=batch_size, shuffle=True)
        run_train_epoch(model, loss_and_grad_fn, optimizer, train_batches)
    return run_evaluate(model, val_batches)


@app.cell
def _(hp_search_cb, mo, x_tr, x_val, y_tr, y_val):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    _search_space = {"lr": [1e-4, 3e-4], "num_layers": [4, 6, 8]}
    _hp_epochs = 5
    _hp_batch_size = 128
    hp_results = []
    _sub_n = min(6000, x_tr.shape[0])
    _sub_x = x_tr[:_sub_n]
    _sub_y = y_tr[:_sub_n]
    for _lr in _search_space["lr"]:
        for _nl in _search_space["num_layers"]:
            _vl = run_hp_config(_sub_x, _sub_y, x_val, y_val, _lr, _nl, _hp_epochs, _hp_batch_size)
            hp_results.append({"lr": _lr, "num_layers": _nl, "val_loss": round(_vl, 4)})
            mo.output.replace(mo.md(f"lr={_lr}, num_layers={_nl} -> val={_vl:.4f}"))
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation & Cross-Validation
    """)
    return


@app.function
def evaluate_model(model: nn.Module, batches: list) -> float:
    return run_evaluate(model, batches)


@app.cell
def _(bs_ui, mo, trained_model, x_te, y_te):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before evaluating._")
    else:
        _test_batches = make_batches(x_te, y_te, batch_size=bs_ui.value, shuffle=False)
        test_loss = evaluate_model(trained_model, _test_batches)
        _out = mo.md(f"**Test set flow-matching loss**: `{test_loss:.4f}`")
    _out
    return


@app.function
def run_cv_fold(
    x_fold: mx.array,
    y_fold: mx.array,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    lr: float,
    num_layers: int,
    n_epochs: int,
    batch_size: int,
) -> float:
    tr_i = mx.array(train_idx.astype(np.int32))
    va_i = mx.array(val_idx.astype(np.int32))
    x_tr_fold = x_fold[tr_i]
    y_tr_fold = y_fold[tr_i]
    x_va_fold = x_fold[va_i]
    y_va_fold = y_fold[va_i]
    model = DiffusionTransformerV1(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_dim=512,
        num_layers=num_layers,
        dropout=0.0,
    )
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss)
    val_batches = make_batches(x_va_fold, y_va_fold, batch_size=batch_size, shuffle=False)
    for _ in range(n_epochs):
        train_batches = make_batches(x_tr_fold, y_tr_fold, batch_size=batch_size, shuffle=True)
        run_train_epoch(model, loss_and_grad_fn, optimizer, train_batches)
    return run_evaluate(model, val_batches)


@app.cell
def _(mo, trained_model, x_tr, y_tr):
    if trained_model is None:
        _out = mo.md("_Train first, then k-fold cross-validation results will appear here._")
    else:
        _cv_n = min(6000, x_tr.shape[0])
        _cv_x = x_tr[:_cv_n]
        _cv_y = y_tr[:_cv_n]
        _k = 3
        _rng = np.random.default_rng(seed=0)
        _perm = _rng.permutation(_cv_n)
        _folds = np.array_split(_perm, _k)
        cv_fold_losses = []
        for _f in range(_k):
            _val_idx = _folds[_f]
            _train_idx = np.concatenate([_folds[j] for j in range(_k) if j != _f])
            _vl = run_cv_fold(_cv_x, _cv_y, _train_idx, _val_idx, 3e-4, 6, 3, 128)
            cv_fold_losses.append(_vl)
            mo.output.replace(mo.md(f"Fold {_f + 1}/{_k} — val loss: {_vl:.4f}"))
        _mean = float(np.mean(cv_fold_losses))
        _std = float(np.std(cv_fold_losses))
        cv_results = {"fold_losses": cv_fold_losses, "mean": _mean, "std": _std}
        _out = mo.md(
            f"**{len(cv_fold_losses)}-Fold CV flow loss**: `{_mean:.4f} ± {_std:.4f}` "
            f"(folds: {[round(v, 4) for v in cv_fold_losses]})"
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
def plot_loss_curve(train_losses: list, val_losses: list | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", lw=2, ms=4, label="Train")
    if val_losses:
        ax.plot(epochs, val_losses, "r-s", lw=2, ms=4, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss (MSE)")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses, trained_model, val_losses):
    if trained_model is None:
        _out = mo.md("_Train the model first to see the loss curve._")
    else:
        _out = plot_loss_curve(train_losses, val_losses)
    _out
    return


@app.cell
def _(mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train V1 (Section 5) first to run the training-health probe._")
    else:
        _h = diagnose_training_health(trained_model)
        _h1 = "PASS" if _h["adaln_scale_rms"] < 50 else "DIVERGED"
        _h2 = "PASS" if _h["final_residual_rms"] < 1e3 else "DIVERGED"
        _h3 = "PASS" if _h["min_attn_entropy"] > 0.2 else "SATURATED"
        _out = mo.md(f"""
        ### Training-Health Probe — V1

        | Signal | Value | Healthy range | Verdict |
        |---|---|---|---|
        | AdaLN `scale` RMS | `{_h["adaln_scale_rms"]:.3f}` | < 50 | **{_h1}** |
        | Final residual RMS | `{_h["final_residual_rms"]:.3f}` | < 1e3 | **{_h2}** |
        | Min attention entropy | `{_h["min_attn_entropy"]:.3f}` | > 0.2 (uniform = `{_h["uniform_attn_entropy"]:.3f}`) | **{_h3}** |
        | `cond` RMS | `{_h["cond_rms"]:.3f}` | — | — |

        Run this on **both** models. The AdaLN runaway is architecture-independent:
        V1 diverged here too until it was given QK normalization.
        """)
    _out
    return


@app.function
def denormalize_cifar(x: mx.array) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    img = np.array(x) * std + mean
    return np.clip(img, 0.0, 1.0)


@app.function
def plot_generated_grid(
    model: nn.Module,
    solver_fn,
    num_steps: int,
    class_names: list,
    num_per_class: int = 1,
):
    n_classes = len(class_names)
    total = n_classes * num_per_class
    labels = np.repeat(np.arange(n_classes, dtype=np.int32), num_per_class)
    y = mx.array(labels)
    x0 = mx.random.normal(shape=(total, 32, 32, 3))
    x1 = solver_fn(model, x0, y, num_steps)
    mx.eval(x1)
    imgs = denormalize_cifar(x1)
    cols = num_per_class
    rows = n_classes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    for i in range(total):
        r, c = divmod(i, cols)
        ax = axes[r, c] if cols > 1 else axes[r]
        ax.imshow(imgs[i])
        if c == 0:
            ax.set_ylabel(class_names[r], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Generated samples ({solver_fn.__name__}, {num_steps} steps)", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def plot_solver_comparison(
    model: nn.Module,
    class_idx: int,
    class_names: list,
    steps_list: list,
):
    solvers = [("Euler", euler_solve), ("Midpoint", midpoint_solve), ("RK4", rk4_solve)]
    rows = len(solvers)
    cols = len(steps_list)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.9))
    y = mx.array(np.array([class_idx], dtype=np.int32))
    seed_noise = mx.random.normal(shape=(1, 32, 32, 3))
    for r, (name, fn) in enumerate(solvers):
        for c, ns in enumerate(steps_list):
            x1 = fn(model, seed_noise, y, ns)
            mx.eval(x1)
            img = denormalize_cifar(x1)[0]
            ax = axes[r, c] if rows > 1 and cols > 1 else (axes[c] if rows == 1 else axes[r])
            ax.imshow(img)
            ax.set_title(f"{name} — {ns} steps", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Solver comparison — class `{class_names[class_idx]}`", fontsize=12)
    fig.tight_layout()
    return fig


@app.function
def plot_euler_trajectory(
    model: nn.Module,
    class_idx: int,
    class_names: list,
    num_steps: int,
    num_frames: int = 10,
):
    y = mx.array(np.array([class_idx], dtype=np.int32))
    x0 = mx.random.normal(shape=(1, 32, 32, 3))
    trajectory = euler_solve_trajectory(model, x0, y, num_steps)
    frame_idx = sorted(set(np.linspace(0, num_steps, num_frames).round().astype(int).tolist()))
    n = len(frame_idx)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.7, 2.0))
    for col, step in enumerate(frame_idx):
        img = denormalize_cifar(trajectory[step])[0]
        ax = axes[col] if n > 1 else axes
        ax.imshow(img)
        ax.set_title(f"t={step / num_steps:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Euler solver trajectory — class `{class_names[class_idx]}` ({num_steps} steps)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


@app.cell
def _(class_names, mo):
    solver_ui = mo.ui.dropdown(
        options={"Euler": "euler", "Midpoint": "midpoint", "RK4": "rk4"},
        value="Euler",
        label="ODE Solver",
    )
    steps_ui = mo.ui.slider(5, 200, value=50, step=1, label="Solver Steps")
    class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value=class_names[0],
        label="Class to generate",
    )
    sample_btn = mo.ui.run_button(label="Sample")
    compare_btn = mo.ui.run_button(label="Compare Solvers")
    mo.vstack(
        [
            mo.md("### Sampling controls"),
            mo.hstack([solver_ui, steps_ui, class_ui]),
            mo.hstack([sample_btn, compare_btn]),
        ]
    )
    return class_ui, compare_btn, sample_btn, solver_ui, steps_ui


@app.function
def resolve_solver(name: str):
    return {"euler": euler_solve, "midpoint": midpoint_solve, "rk4": rk4_solve}[name]


@app.cell
def _(class_names, mo, sample_btn, solver_ui, steps_ui, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to generate samples._")
    elif not sample_btn.value:
        _out = mo.md("Click **Sample** to generate a grid of images for every class.")
    else:
        _out = plot_generated_grid(
            trained_model, resolve_solver(solver_ui.value), steps_ui.value, class_names, num_per_class=4
        )
    _out
    return


@app.cell
def _(class_names, class_ui, compare_btn, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first to compare solvers._")
    elif not compare_btn.value:
        _out = mo.md("Click **Compare Solvers** to run Euler / Midpoint / RK4 at several step counts.")
    else:
        _out = plot_solver_comparison(
            trained_model, int(class_ui.value), class_names, [10, 25, 50, 100]
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### Euler Solver Trajectory — Watching the Model Denoise

    The Euler solver integrates the learned velocity field one small step
    at a time, starting from pure Gaussian noise (`t=0`) and arriving at a
    generated image (`t=1`). The frames below sample intermediate states
    `x_t` along a single trajectory, making the model's step-by-step
    progress toward a coherent image directly visible.
    """)
    return


@app.cell
def _(class_names, mo):
    traj_class_ui = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value=class_names[0],
        label="Class to generate",
    )
    traj_steps_ui = mo.ui.slider(10, 200, value=50, step=1, label="Euler Steps")
    traj_frames_ui = mo.ui.slider(4, 20, value=10, step=1, label="Frames to display")
    traj_btn = mo.ui.run_button(label="Generate Trajectory")
    mo.vstack(
        [
            mo.hstack([traj_class_ui, traj_steps_ui, traj_frames_ui]),
            traj_btn,
        ]
    )
    return traj_btn, traj_class_ui, traj_frames_ui, traj_steps_ui


@app.cell
def _(
    class_names,
    mo,
    trained_model,
    traj_btn,
    traj_class_ui,
    traj_frames_ui,
    traj_steps_ui,
):
    if trained_model is None:
        _out = mo.md("_Train the model first to visualize the sampling trajectory._")
    elif not traj_btn.value:
        _out = mo.md("Click **Generate Trajectory** to watch the Euler solver denoise step by step.")
    else:
        _out = plot_euler_trajectory(
            trained_model,
            int(traj_class_ui.value),
            class_names,
            traj_steps_ui.value,
            num_frames=traj_frames_ui.value,
        )
    _out
    return


@app.cell
def _(mo, num_layers_ui, train_losses, trained_model, val_losses):
    if trained_model is None:
        _out = mo.md("_Train the model to see a results summary._")
    else:
        _out = mo.md(
            f"""
            ### Summary

            - Backbone: **DiffusionTransformerV1** with `num_layers = {num_layers_ui.value}`,
              `embed_dim=256`, `num_heads=8`, `mlp_dim=512`, patch size `4`.
            - Trained for `{len(train_losses)}` epochs; final train loss
              `{train_losses[-1]:.4f}`, final val loss `{val_losses[-1]:.4f}`.
            - Flow-matching objective: linear-path MSE between predicted and
              target velocity `v = x1 - x0`.
            - Inference supports **Euler**, **Midpoint / Heun**, and **RK4**
              solvers over a UI-controlled number of steps. RK4 typically
              produces the best samples at fewer steps at the cost of 4x more
              network evaluations per step.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8b — V2 (2D RoPE) Results
    """)
    return


@app.function
def plot_loss_curves_v1_v2(
    train_losses_v1: list,
    val_losses_v1: list,
    train_losses_v2: list,
    val_losses_v2: list,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    e1 = range(1, len(train_losses_v1) + 1)
    e2 = range(1, len(train_losses_v2) + 1)
    ax.plot(e1, train_losses_v1, "b-o", lw=2, ms=4, label="V1 Train (learned pos)")
    ax.plot(e1, val_losses_v1,   "b--s", lw=2, ms=4, label="V1 Val")
    ax.plot(e2, train_losses_v2, "r-o", lw=2, ms=4, label="V2 Train (2D RoPE)")
    ax.plot(e2, val_losses_v2,   "r--s", lw=2, ms=4, label="V2 Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss (MSE)")
    ax.set_title("V1 (learned positional embedding) vs V2 (2D RoPE) training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    train_losses,
    train_losses_v2,
    trained_model,
    trained_model_v2,
    val_losses,
    val_losses_v2,
):
    if trained_model is None and trained_model_v2 is None:
        _out = mo.md("_Train both V1 (Section 5) and V2 (Section 5b) to compare loss curves._")
    elif trained_model is None:
        _out = plot_loss_curve(train_losses_v2, val_losses_v2)
    elif trained_model_v2 is None:
        _out = plot_loss_curve(train_losses, val_losses)
    else:
        _out = plot_loss_curves_v1_v2(train_losses, val_losses, train_losses_v2, val_losses_v2)
    _out
    return


@app.function
def diagnose_training_health(model: nn.Module, n_probe: int = 8) -> dict:
    t = mx.array(np.linspace(0.05, 0.95, n_probe).astype(np.float32))
    y = mx.array(np.zeros(n_probe, dtype=np.int32))
    x = mx.random.normal(shape=(n_probe, 32, 32, 3))

    def _rms(v):
        return float(mx.sqrt(mx.mean(v.astype(mx.float32) ** 2)))

    sinus = model.time_embed.layers[0](t)
    cond = model.time_embed(t) + model.class_embed(y)
    h = model.patchify(x)
    if hasattr(model, "pos_embed"):
        h = h + model.pos_embed
    scale, _ = mx.split(model.blocks[0].attn_norm.proj(nn.silu(cond))[:, None, :], 2, axis=-1)

    entropies = []
    for block in model.blocks:
        attn = block.attn
        hn = block.attn_norm(h, cond)
        b, n, _d = hn.shape
        heads, hd = attn.num_heads, attn.head_dim
        half = hd // 2
        q = attn.q_norm(attn.q_proj(hn).reshape(b, n, heads, hd))
        k = attn.k_norm(attn.k_proj(hn).reshape(b, n, heads, hd))
        if hasattr(attn, "_row_cos"):
            q = mx.concatenate(
                [
                    attn._apply_rotary(q[..., :half], attn._row_cos, attn._row_sin),
                    attn._apply_rotary(q[..., half:], attn._col_cos, attn._col_sin),
                ],
                axis=-1,
            )
            k = mx.concatenate(
                [
                    attn._apply_rotary(k[..., :half], attn._row_cos, attn._row_sin),
                    attn._apply_rotary(k[..., half:], attn._col_cos, attn._col_sin),
                ],
                axis=-1,
            )
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        probs = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5), axis=-1)
        entropies.append(float(mx.mean(-mx.sum(probs * mx.log(probs + 1e-9), axis=-1))))
        h = block(h, cond)

    n_tokens = h.shape[1] if h.ndim == 3 else (32 // 4) ** 2
    return {
        "time_signal_std": float(mx.mean(mx.std(sinus, axis=0))),
        "cond_rms": _rms(cond),
        "adaln_scale_rms": _rms(scale),
        "final_residual_rms": _rms(h),
        "min_attn_entropy": min(entropies) if entropies else float("nan"),
        "uniform_attn_entropy": math.log(64),
    }


@app.cell
def _(mo, trained_model_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train V2 (Section 5b) first to run the training-health probe._")
    else:
        _d = diagnose_training_health(trained_model_v2)
        _c1 = "PASS" if _d["adaln_scale_rms"] < 50 else "DIVERGED"
        _c2 = "PASS" if _d["final_residual_rms"] < 1e3 else "DIVERGED"
        _c3 = "PASS" if _d["min_attn_entropy"] > 0.2 else "SATURATED"
        _c4 = "PASS" if _d["time_signal_std"] > 0.1 else "WEAK"
        _out = mo.md(f"""
        ### Training-Health Probe — V2

        | Signal | Value | Healthy range | Verdict |
        |---|---|---|---|
        | AdaLN `scale` RMS | `{_d["adaln_scale_rms"]:.3f}` | < 50 | **{_c1}** |
        | Final residual RMS | `{_d["final_residual_rms"]:.3f}` | < 1e3 | **{_c2}** |
        | Min attention entropy | `{_d["min_attn_entropy"]:.3f}` | > 0.2 (uniform = `{_d["uniform_attn_entropy"]:.3f}`) | **{_c3}** |
        | Time-signal std across `t` | `{_d["time_signal_std"]:.4f}` | > 0.1 | **{_c4}** |
        | `cond` RMS | `{_d["cond_rms"]:.3f}` | — | — |

        A flow-matching loss stuck near `pi/2 = 1.571` with **zero attention entropy**
        is a *diverged* model, not an untrained one: `final_norm` renormalizes the
        blow-up, so the loss stays finite and the failure is silent. This probe makes
        that state visible.
        """)
    _out
    return


@app.cell
def _(class_names, mo):
    solver_ui_v2 = mo.ui.dropdown(
        options={"Euler": "euler", "Midpoint": "midpoint", "RK4": "rk4"},
        value="Euler",
        label="ODE Solver",
    )
    steps_ui_v2 = mo.ui.slider(5, 200, value=50, step=1, label="Solver Steps")
    class_ui_v2 = mo.ui.dropdown(
        options={name: i for i, name in enumerate(class_names)},
        value=class_names[0],
        label="Class to generate",
    )
    sample_btn_v2 = mo.ui.run_button(label="Sample V2")
    compare_btn_v2 = mo.ui.run_button(label="Compare Solvers V2")
    mo.vstack(
        [
            mo.md("### V2 Sampling Controls"),
            mo.hstack([solver_ui_v2, steps_ui_v2, class_ui_v2]),
            mo.hstack([sample_btn_v2, compare_btn_v2]),
        ]
    )
    return (
        class_ui_v2,
        compare_btn_v2,
        sample_btn_v2,
        solver_ui_v2,
        steps_ui_v2,
    )


@app.cell
def _(
    class_names,
    mo,
    sample_btn_v2,
    solver_ui_v2,
    steps_ui_v2,
    trained_model_v2,
):
    if trained_model_v2 is None:
        _out = mo.md("_Train V2 (Section 5b) first to generate samples._")
    elif not sample_btn_v2.value:
        _out = mo.md("Click **Sample V2** to generate a grid of images for every class.")
    else:
        _out = plot_generated_grid(
            trained_model_v2,
            resolve_solver(solver_ui_v2.value),
            steps_ui_v2.value,
            class_names,
            num_per_class=4,
        )
    _out
    return


@app.cell
def _(class_names, class_ui_v2, compare_btn_v2, mo, trained_model_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train V2 (Section 5b) first to compare solvers._")
    elif not compare_btn_v2.value:
        _out = mo.md("Click **Compare Solvers V2** to run Euler / Midpoint / RK4 at several step counts.")
    else:
        _out = plot_solver_comparison(
            trained_model_v2, int(class_ui_v2.value), class_names, [10, 25, 50, 100]
        )
    _out
    return


@app.cell
def _(mo, num_layers_ui_v2, train_losses_v2, trained_model_v2, val_losses_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train the V2 model to see a results summary._")
    else:
        _out = mo.md(
            f"""
            ### V2 Summary

            - Backbone: **DiffusionTransformerV2** (2D RoPE) with `num_layers = {num_layers_ui_v2.value}`,
              `embed_dim=256`, `num_heads=8`, `mlp_dim=512`, patch size `4`.
            - Trained for `{len(train_losses_v2)}` epochs; final train loss
              `{train_losses_v2[-1]:.4f}`, final val loss `{val_losses_v2[-1]:.4f}`.
            - Positional encoding: **2-D RoPE** — row and column grid positions are
              injected into every Q/K projection via rotation; no learnable position
              table is present in the model.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Model
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="cifar10_dit_flow_v1.safetensors",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_model_btn])
    return save_filename_ui, save_model_btn


@app.cell
def _(mo, save_filename_ui, save_model_btn, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before saving._")
    elif not save_model_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save Model** to write the trained "
            "weights to `models/`."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_filename_ui.value
        trained_model.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Model weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9b — Save V2 Trained Model
    """)
    return


@app.cell
def _(mo):
    save_filename_ui_v2 = mo.ui.text(
        value="cifar10_dit_flow_rope2d_v2.safetensors",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn_v2 = mo.ui.run_button(label="Save V2 Model")
    mo.vstack([save_filename_ui_v2, save_model_btn_v2])
    return save_filename_ui_v2, save_model_btn_v2


@app.cell
def _(mo, save_filename_ui_v2, save_model_btn_v2, trained_model_v2):
    if trained_model_v2 is None:
        _out = mo.md("_Train the V2 model first (Section 5b) before saving._")
    elif not save_model_btn_v2.value:
        _out = mo.md(
            "Enter a filename and click **Save V2 Model** to write the trained "
            "weights to `models/`."
        )
    else:
        _models_dir_v2 = Path(__file__).resolve().parent.parent / "models"
        _models_dir_v2.mkdir(parents=True, exist_ok=True)
        _save_path_v2 = _models_dir_v2 / save_filename_ui_v2.value
        trained_model_v2.save_weights(str(_save_path_v2))
        _out = mo.md(f"**Saved!** V2 model weights written to `{_save_path_v2}`.")
    _out
    return


if __name__ == "__main__":
    app.run()
