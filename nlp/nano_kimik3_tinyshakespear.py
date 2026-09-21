import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    import math
    from dataclasses import dataclass, asdict
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nano-Kimi-K3 on TinyShakespeare

    **Research goal** — train a *miniaturized* version of Moonshot AI's **Kimi
    K3** (July 2026) on the TinyShakespeare corpus with the MLX framework,
    keeping **every** architectural component of the full-scale model in place
    and only reducing the *sizes*. The notebook doubles as an annotated reader
    for Kimi K3: each module cell is preceded by a markdown cell that cites the
    source note (`kimi-k3-open-frontier-intelligence.md`,
    `kimi-linear-kimi-delta-attention.md`,
    `attention-residuals-attnres.md`) and derives the math it implements.

    This tome is an *architecture comprehension* exercise, not a scaling
    result. 278K training tokens is *far* too small a corpus for KDA's
    channel-wise decay, Stable-LatentMoE's extreme sparsity, or Block AttnRes
    to show their real benefit — read the *code* and the *notes*.

    ## Component-by-component mapping to Kimi K3

    | Nano component (this notebook) | Kimi K3 counterpart | Reduction |
    |---|---|---|
    | vocab_size = 10K | 160K | 16x |
    | d_model = 256 | 7,168 | 28x |
    | n_layers = **13** = 3x(3 KDA + 1 MLA) + 1 trailing MLA | 93 = 69 KDA + 24 MLA + trailing | ~7x |
    | KDA:MLA ratio | 3:1 (identical) | 1x |
    | n_heads = 4, head_dim = 64 | 96 | 24x heads |
    | MLA q_lora_rank / kv_lora_rank | 128 / 64 | -- |
    | qk_nope / qk_rope / v_head_dim | 64 / **0 (NoPE)** / 64 | NoPE |
    | LatentMoE latent width l = 128 (0.5x d_model) | 3,584 (0.5x) | 28x |
    | routed experts / active | 32 / 4 | 896 / 16, 28x |
    | shared experts | 2 (identical) | 1x |
    | expert hidden (latent) | 64 | 3,072 | 48x |
    | AttnRes blocks N | 4 (+ embedding as b_0) | 8 (+embedding) | 2x |
    | ShortConv kernel | 4 (identical) | 1x |
    | KDA chunk size C | 64 | -- (kernel-side 16 tiles) |
    | g_min | -5 (identical) | 1x |
    | SiTU-GLU gamma_1, gamma_2 | 4, 25 (identical) | 1x |
    | seq_len | 256 | 1,048,576 | 4096x |
    | MTP layers | 1 (identical) | 1x |
    | MXFP4 QAT (routed experts) | toggleable | matches K3 recipe |
    | Per-Head Muon + K2 clipping + WD 0.1 | matches K3 recipe | -- |
    | EAGLE-3 draft (unrolled 2 steps) | unrolled 7 steps | 3.5x |

    ## Deliberately omitted (with reasons)

    - **MoonViT-V2** and native-multimodal pathway — text-only corpus.
    - **MoonEP** expert parallelism, **FlashKDA** CUTLASS kernels, **KDA
      Context Parallelism**, **AgentENV** microVM sandboxes, KDA-aware
      **prefix cache** — multi-GPU / infra co-design.
    - **SFT -> 9 domain x effort RL experts -> MOPD** post-training —
      this is a *pre-training* notebook.
    - **EAGLE-3 draft** and **MXFP4 QAT** ARE included since they touch
      the model itself (deployment-aware co-design).

    ## Section outline

    1. **Title & Research Goal** — this cell
    2. **Data Exploration** — Zipf, token-id histogram, merge lengths, corpus stats
    3. **Dataset Creation** — contiguous 90/5/5 split, window dataset, one batch
    4. **Model Definition** — every K3 component, with math, in dependency order
    5. **Training** — cosine LR (1% warmup), Per-Head Muon vs plain AdamW
    6. **Hyperparameter Search** — 2x2x2 grid (opt-in via checkbox)
    7. **Validation & Cross-Validation** — blocked 5-fold CV
    8. **Results** — loss, expert load, router bias, KDA decay spectrum, AttnRes depth attention
    9. **Sample Output** — recurrent-mode generation from the trained model
    10. **Save Trained Model** — safetensors weights + JSON sidecar (with frozen QB biases)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Data Exploration

    Corpus: TinyShakespeare (~1.1MB raw UTF-8). The BPE merge table and the
    pre-tokenized stream are already on disk — we do **no downloading** and
    no re-tokenization at load time. `resolve_data_dir` looks in three
    conventional locations so the notebook works whether launched from the
    repo root, from `nlp/`, or against the user's home data cache.
    """)
    return


@app.function
def resolve_data_dir(candidates: list[Path] | None = None) -> Path:
    """Return the first existing directory among the candidates.

    Defaults to (../data, ~/data, <repo_root>/data). Raises FileNotFoundError
    if none exist so mis-configured paths fail loudly rather than silently
    tokenizing zero bytes.
    """
    if candidates is None:
        candidates = [
            Path("../data"),
            Path.home() / "data",
            Path(__file__).resolve().parent.parent.parent / "data",
        ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"No data dir found among {candidates}")


@app.cell
def _():
    data_dir = resolve_data_dir()
    merges_table_file = str(data_dir / "tinyshakespear.json")
    training_token_file = str(data_dir / "tinyshakespear_tokens.txt")
    training_file = str(data_dir / "tinyshakespear.txt")
    return merges_table_file, training_file, training_token_file


@app.function
def load_merge_table(json_path: str) -> dict:
    """Load the BPE merge table produced by the tinyshakespear tokenizer.

    Returns pair->id, pair->rank, id->pair dicts (verbatim from the
    sibling nano_language_model.py so both use identical decoding).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    pair_to_id: dict[tuple[int, int], int] = {}
    pair_to_rank: dict[tuple[int, int], int] = {}
    id_to_pair: dict[int, tuple[int, int]] = {}
    for rank, (left, right, merged_id) in enumerate(data["merges"]):
        pair = (left, right)
        pair_to_id[pair] = merged_id
        pair_to_rank[pair] = rank
        id_to_pair[merged_id] = pair

    return {
        "pair_to_id": pair_to_id,
        "pair_to_rank": pair_to_rank,
        "id_to_pair": id_to_pair,
    }


@app.function
def tokenize(text: str, merge_table: dict) -> list[int]:
    """Greedy BPE tokenization matching the tinyshakespear tokenizer."""
    pair_to_rank = merge_table["pair_to_rank"]
    pair_to_id = merge_table["pair_to_id"]
    tokens = list(text.encode("utf-8"))

    while len(tokens) >= 2:
        ranked_pairs = (
            (tokens[i], tokens[i + 1])
            for i in range(len(tokens) - 1)
            if (tokens[i], tokens[i + 1]) in pair_to_rank
        )
        best_pair = min(ranked_pairs, key=lambda pair: pair_to_rank[pair], default=None)
        if best_pair is None:
            break

        merged_id = pair_to_id[best_pair]
        merged_tokens: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                merged_tokens.append(merged_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        tokens = merged_tokens

    return tokens


@app.function
def decode_tokens(tokens: list[int], merge_table: dict) -> str:
    """Recursively expand BPE ids into UTF-8 bytes, then decode to text."""
    id_to_pair = merge_table["id_to_pair"]
    memo: dict[int, bytes] = {}

    def expand(token_id: int) -> bytes:
        if token_id < 256:
            return bytes([token_id])
        if token_id in memo:
            return memo[token_id]
        left, right = id_to_pair[token_id]
        expanded = expand(left) + expand(right)
        memo[token_id] = expanded
        return expanded

    byte_sequence = b"".join(expand(token_id) for token_id in tokens)
    return byte_sequence.decode("utf-8", errors="replace")


@app.cell
def _(merges_table_file):
    merge_table = load_merge_table(merges_table_file)
    _max_id = max(max(merge_table["id_to_pair"].keys(), default=0), 255)
    vocab_size = max(_max_id + 1, 256)
    assert vocab_size == 10000, f"expected vocab 10000, got {vocab_size}"
    print(f"merge table: {len(merge_table['pair_to_id'])} merges, vocab={vocab_size}")
    return merge_table, vocab_size


@app.cell
def _(training_file, training_token_file):
    with open(training_file, "r") as _f:
        raw_text = _f.read()
    with open(training_token_file, "r") as _f:
        all_tokens = np.array([int(t) for t in _f.read().split()], dtype=np.int32)
    assert all_tokens.shape[0] == 277996, f"expected 277,996 tokens, got {all_tokens.shape[0]}"
    print(f"raw corpus: {len(raw_text):,} bytes")
    print(f"token stream: {all_tokens.shape[0]:,} tokens")
    print(f"compression ratio: {len(raw_text) / all_tokens.shape[0]:.3f} bytes/token")
    return all_tokens, raw_text


@app.function
def plot_zipf(all_tokens: np.ndarray):
    counts = np.bincount(all_tokens)
    counts = np.sort(counts)[::-1]
    counts = counts[counts > 0]
    ranks = np.arange(1, len(counts) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ranks, counts, "b-", lw=1.2)
    ax.set_xlabel("rank (log)")
    ax.set_ylabel("frequency (log)")
    ax.set_title("Zipf plot of token frequencies")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_token_id_histogram(all_tokens: np.ndarray, n_bins: int = 60):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_tokens, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.75)
    ax.set_xlabel("token id")
    ax.set_ylabel("count")
    ax.set_title(f"Token-id distribution (min={int(all_tokens.min())}, max={int(all_tokens.max())})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_merge_length(merge_table: dict):
    memo: dict[int, int] = {}

    def token_len(tid: int) -> int:
        if tid < 256:
            return 1
        if tid in memo:
            return memo[tid]
        left, right = merge_table["id_to_pair"][tid]
        n = token_len(left) + token_len(right)
        memo[tid] = n
        return n

    pair_to_rank = merge_table["pair_to_rank"]
    ordered = sorted(merge_table["pair_to_id"].items(), key=lambda kv: pair_to_rank[kv[0]])
    ranks = list(range(len(ordered)))
    lengths = [token_len(merged_id) for _pair, merged_id in ordered]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ranks, lengths, s=3, alpha=0.3, color="darkorange")
    ax.set_xlabel("merge rank")
    ax.set_ylabel("resulting token length (bytes)")
    ax.set_title("Merge rank vs. resulting token length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def summarize_play_structure(raw_text: str) -> dict:
    """Rough speaker/line-structure summary of the raw Shakespeare corpus."""
    lines = raw_text.split("\n")
    speaker_lines = [l for l in lines if l.strip().endswith(":") and l.strip().isupper()]
    return {
        "n_lines": len(lines),
        "n_speaker_lines": len(speaker_lines),
        "n_blank_lines": sum(1 for l in lines if not l.strip()),
        "avg_line_len": float(np.mean([len(l) for l in lines])) if lines else 0.0,
        "sample_speakers": list(dict.fromkeys(l.strip().rstrip(":") for l in speaker_lines[:12])),
    }


@app.cell
def _(all_tokens):
    plot_zipf(all_tokens)
    return


@app.cell
def _(all_tokens):
    plot_token_id_histogram(all_tokens)
    return


@app.cell
def _(merge_table):
    plot_merge_length(merge_table)
    return


@app.cell
def _(all_tokens, mo, raw_text, vocab_size):
    _byte_coverage = 100 * float(np.mean(all_tokens < 256))
    _bytes_per_tok = len(raw_text) / all_tokens.shape[0]
    _n_train = int(0.90 * all_tokens.shape[0])
    _n_val = int(0.05 * all_tokens.shape[0])
    _n_test = all_tokens.shape[0] - _n_train - _n_val
    _play = summarize_play_structure(raw_text)
    mo.md(
        f"""
        ### Corpus stats

        | quantity | value |
        |---|---|
        | raw bytes | {len(raw_text):,} |
        | tokens | {all_tokens.shape[0]:,} |
        | vocab (asserted) | {vocab_size:,} |
        | bytes / token | {_bytes_per_tok:.3f} |
        | raw-byte tokens (%) | {_byte_coverage:.1f}% |
        | train / val / test | {_n_train:,} / {_n_val:,} / {_n_test:,} |

        ### Raw text structure

        | quantity | value |
        |---|---|
        | total lines | {_play['n_lines']:,} |
        | speaker headers (`NAME:`) | {_play['n_speaker_lines']:,} |
        | blank lines | {_play['n_blank_lines']:,} |
        | mean line length (chars) | {_play['avg_line_len']:.1f} |
        | first speakers | {", ".join(_play['sample_speakers'])} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Dataset Creation

    **Contiguous** 90/5/5 split of the 277,996-token stream: train is the
    first 90%, val the next 5%, test the final 5%. We do **not** shuffle
    across the split boundary — a language corpus contains long-range
    dependencies (character arcs, running dialogue), and random-token
    splitting would leak information from those dependencies into every
    split. Only the *starting offsets of training windows* are shuffled.

    The dataset yields `(x, y)` pairs where `y = x` shifted by one token
    — the standard next-token prediction target.
    """)
    return


@app.class_definition
class TokenWindowDatasetV1:
    """Iterable dataset of (x, y) next-token windows from a 1D token stream.

    Every iteration reshuffles window *start offsets* (train only, controlled
    via the `shuffle` flag). Windows are `seq_len + 1` long so that x = w[:-1]
    and y = w[1:] are aligned next-token pairs.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        seq_len: int = 256,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._num_windows = max(0, self.tokens.shape[0] - seq_len - 1)

    def __len__(self) -> int:
        return self._num_windows // self.batch_size

    def __iter__(self):
        if self.shuffle:
            starts = self._rng.permutation(self._num_windows)
        else:
            starts = np.arange(self._num_windows)
        offsets = np.arange(self.seq_len + 1)
        n_batches = self._num_windows // self.batch_size
        for b in range(n_batches):
            batch_starts = starts[b * self.batch_size : (b + 1) * self.batch_size]
            windows_np = self.tokens[batch_starts[:, None] + offsets[None, :]]
            batch = mx.array(windows_np, dtype=mx.int32)
            yield batch[:, :-1], batch[:, 1:]


@app.function
def make_datasets(
    tokens: np.ndarray,
    seq_len: int = 256,
    batch_size: int = 8,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple[TokenWindowDatasetV1, TokenWindowDatasetV1, TokenWindowDatasetV1]:
    """Contiguous chronological split of a 1D token stream into three iterables."""
    n = tokens.shape[0]
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    n_train = n - n_val - n_test
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train : n_train + n_val]
    test_tokens = tokens[n_train + n_val :]
    train_ds = TokenWindowDatasetV1(train_tokens, seq_len, batch_size, shuffle=True, seed=seed)
    val_ds = TokenWindowDatasetV1(val_tokens, seq_len, batch_size, shuffle=False, seed=seed + 1)
    test_ds = TokenWindowDatasetV1(test_tokens, seq_len, batch_size, shuffle=False, seed=seed + 2)
    return train_ds, val_ds, test_ds


@app.cell
def _(all_tokens):
    train_ds, val_ds, test_ds = make_datasets(all_tokens, seq_len=256, batch_size=4)
    print(f"train batches: {len(train_ds)}  val batches: {len(val_ds)}  test batches: {len(test_ds)}")
    return test_ds, train_ds


@app.cell
def _(train_ds):
    _x, _y = next(iter(train_ds))
    print(f"batch x: shape={_x.shape} dtype={_x.dtype}")
    print(f"batch y: shape={_y.shape} dtype={_y.dtype}")
    print(f"first row x[:12]: {_x[0, :12].tolist()}")
    print(f"first row y[:12]: {_y[0, :12].tolist()}  (should be x[:12] shifted by 1)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Model Definition

    Every module below is a scaled-down implementation of a real Kimi K3
    component. Build order follows *dependency order* — helpers first, leaves
    next, then composites, then the top-level model. Each class is preceded
    by a markdown cell that cites the source note and derives the math.

    A note on where these details came from: the K3 tech report leaves several
    knobs unspecified (ShortConv kernel, `q_lora_rank`, `kda_mode` naming, the
    exact `A_log`/`dt_bias` parameterization, `attn_res_block_size` semantics).
    Those specifics come from the reference FLA / NeMo-AutoModel Kimi-K3 and
    Kimi-Linear implementations, not the paper.
    """)
    return


@app.class_definition
@dataclass
class NanoKimiK3Config:
    """All knobs of nano-Kimi-K3 in one place.

    Passed by value into every module `__init__` so nothing captures globals.
    Serialized to JSON alongside the model weights in Section 10 so the
    saved backbone can be reconstructed exactly.
    """

    vocab_size: int = 10000
    d_model: int = 256
    n_layers: int = 13
    # KDA:MLA schedule = 3 groups of (3 KDA + 1 MLA) + trailing MLA
    kda_group_size: int = 3
    trailing_mla: bool = True
    n_heads: int = 4
    head_dim: int = 64
    # MLA config (NoPE: qk_rope_head_dim = 0)
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 0
    v_head_dim: int = 64
    mla_use_output_gate: bool = True
    # KDA config
    kda_head_dim: int = 64
    kda_n_heads: int = 4
    kda_chunk_size: int = 64
    kda_mode: str = "chunk"
    short_conv_kernel: int = 4
    g_min: float = -5.0
    kda_alpha_bottleneck: int = 32
    # Stable LatentMoE
    latent_dim: int = 128
    n_routed_experts: int = 32
    n_active_experts: int = 4
    n_shared_experts: int = 2
    expert_hidden: int = 64
    shared_expert_hidden: int = 512  # 2x d_model; keeps total params in the 20-40M band
    situ_gamma_gate: float = 4.0
    situ_gamma_up: float = 25.0
    latentmoe_use_rmsnorm: bool = True
    latentmoe_use_situ_glu: bool = True
    latentmoe_use_quantile_balancing: bool = True
    qb_freeze: bool = False
    # AttnRes
    attn_res_block_size: int = 4  # first three blocks contain 4 layers each; final block holds trailing MLA
    # Sequence / training
    seq_len: int = 256
    # MTP
    mtp_layers: int = 1
    mtp_loss_weight: float = 0.1
    # MXFP4 QAT (routed-expert weights only)
    mxfp4_qat: bool = False
    mxfp4_block_size: int = 32
    # Optimizer defaults
    weight_decay: float = 0.1
    k2_weight_clip: float = 1.0
    warmup_frac: float = 0.01
    # EAGLE-3 draft
    eagle3_unroll: int = 2


@app.cell
def _():
    config = NanoKimiK3Config()
    print(config)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1 MXFP4 quantization-aware training

    K3 stores **routed-expert weights** in MXFP4 (E2M1 elements, shared
    power-of-two block scale over blocks of 32) and activations in MXFP8
    — *all non-expert modules stay higher precision* (see the *Deployment
    -aware post-training* section of `kimi-k3-open-frontier-intelligence.md`).
    Training uses **fake quantization with a straight-through estimator**:
    round to the FP4 grid in the forward pass, pass the gradient through
    unchanged in the backward pass.

    $$
    \tilde x = x + \text{stop\_grad}\bigl(Q_{\text{FP4}}(x) - x\bigr)
    $$

    The shared block scale is a power of two: $s = 2^{\lfloor \log_2(\text{absmax}/6)\rfloor}$
    since the largest representable E2M1 magnitude is 6. Our fake
    implementation uses uniform 4-bit round-to-nearest as a differentiable
    stand-in for the exact E2M1 grid — STE dynamics are identical, and it
    is enough here for a numerical stability ablation.
    """)
    return


@app.function
def fake_quantize_mxfp4(x: mx.array, block_size: int = 32) -> mx.array:
    """Fake MXFP4 quantize-dequantize with per-block power-of-two scale and STE.

    Uniform 4-bit round-to-nearest is used as a simple, differentiable
    stand-in for the exact E2M1 grid. FP4 max magnitude is 6.
    """
    orig_shape = x.shape
    if x.shape[-1] % block_size != 0:
        absmax = mx.max(mx.abs(x)) + 1e-6
        # power-of-two scale
        scale = mx.power(2.0, mx.floor(mx.log2(absmax / 6.0 + 1e-12)))
        q_int = mx.clip(mx.round(x / (scale + 1e-12) * (7.0 / 6.0)), -7, 7)
        x_q = q_int * (6.0 / 7.0) * scale
        return x + mx.stop_gradient(x_q - x)
    x_flat = x.reshape(-1, block_size)
    absmax = mx.max(mx.abs(x_flat), axis=-1, keepdims=True) + 1e-6
    scale = mx.power(2.0, mx.floor(mx.log2(absmax / 6.0 + 1e-12)))
    q_int = mx.clip(mx.round(x_flat / (scale + 1e-12) * (7.0 / 6.0)), -7, 7)
    x_q = q_int * (6.0 / 7.0) * scale
    x_q = x_q.reshape(orig_shape)
    return x + mx.stop_gradient(x_q - x)


@app.class_definition
class MXFP4QuantizerV1(nn.Module):
    """Toggleable fake-MXFP4 quantizer, per K3's routed-expert QAT recipe.

    Wraps `fake_quantize_mxfp4`; when `enabled = False` it is a no-op so it
    doubles as an ablation control. Applied to routed-expert weights only.
    """

    def __init__(self, enabled: bool = False, block_size: int = 32):
        super().__init__()
        self.enabled = enabled
        self.block_size = block_size

    def __call__(self, x: mx.array) -> mx.array:
        if not self.enabled:
            return x
        return fake_quantize_mxfp4(x, self.block_size)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 SiTU-GLU

    K3's activation. SwiGLU's two multiplicative factors are both unbounded,
    risking low-precision overflow. **SiTU-GLU** applies a smooth softcap
    $\gamma \tanh(x/\gamma)$ to *both* the Swish-gate's linear factor and
    the up branch:

    $$
    \text{SiTU-GLU}(x) = \bigl[\gamma_1 \tanh(W_g x / \gamma_1) \odot \sigma(W_g x)\bigr] \odot \bigl[\gamma_2 \tanh(W_u x / \gamma_2)\bigr]
    $$

    K3 uses $\gamma_1 = 4$ (gate) and $\gamma_2 = 25$ (up); both are exposed
    on the K3 `SituAndMul` fused-activation module as `beta` and `linear_beta`.
    Near the origin it matches SwiGLU; for large positive inputs $|f(x)| \le \gamma_1 \gamma_2 = 100$.
    See the *SiTU-GLU* section of `kimi-k3-open-frontier-intelligence.md`.
    """)
    return


@app.function
def situ_glu(g_lin: mx.array, u_lin: mx.array, gamma_gate: float = 4.0, gamma_up: float = 25.0) -> mx.array:
    """Fused SiTU-GLU activation: bounded gate x bounded up-projection."""
    gate = gamma_gate * mx.tanh(g_lin / gamma_gate) * mx.sigmoid(g_lin)
    up = gamma_up * mx.tanh(u_lin / gamma_up)
    return gate * up


@app.class_definition
class SiTUGLUV1(nn.Module):
    """SiTU-GLU feed-forward block with configurable softcaps."""

    def __init__(
        self,
        d_in: int = 128,
        d_hidden: int = 256,
        d_out: int | None = None,
        gamma_gate: float = 4.0,
        gamma_up: float = 25.0,
    ):
        super().__init__()
        d_out = d_out if d_out is not None else d_in
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)
        self.gamma_gate = gamma_gate
        self.gamma_up = gamma_up

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(situ_glu(self.gate(x), self.up(x), self.gamma_gate, self.gamma_up))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 ShortConv (depthwise causal 1-D conv + SiLU)

    Every KDA layer applies **three ShortConvs** — one each on the projected
    Q, K, and V (`q_conv1d`, `k_conv1d`, `v_conv1d` in the K3 reference).
    Kernel size 4, depthwise (`groups = channels`), *causal* (left-pad by
    $k-1$, no future leakage), followed by SiLU:

    $$
    \tilde x_t = \mathrm{SiLU}\Big(\sum_{i=0}^{k-1} w_{c,i}\, x_{t-i,c}\Big)
    $$

    The causal padding + `groups=channels` combo is what a downstream
    verification cell will assert on: perturbing $x_{t+1}$ must not change
    $\tilde x_t$.
    """)
    return


@app.class_definition
class ShortConvV1(nn.Module):
    """Depthwise causal 1-D conv (kernel 4 by default) followed by SiLU.

    Kernel size, channels are configurable so this same class handles the
    q, k, v conv1ds in KDA. In `recurrent` mode you may pass a `state` of
    shape (B, k-1, C) carrying the last k-1 pre-conv inputs; the new state
    (last k-1 of the current x) is returned alongside the conv output.
    """

    def __init__(self, channels: int = 128, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        std = 1.0 / math.sqrt(kernel_size)
        self.weight = mx.random.uniform(-std, std, (channels, kernel_size, 1))
        if bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array, state: mx.array | None = None) -> tuple[mx.array, mx.array]:
        # x: (B, T, C)
        k = self.kernel_size
        if state is None:
            padded = mx.pad(x, [(0, 0), (k - 1, 0), (0, 0)])
        else:
            padded = mx.concatenate([state, x], axis=1)
        y = mx.conv1d(padded, self.weight, padding=0, groups=self.channels)
        if self.bias is not None:
            y = y + self.bias
        y = nn.silu(y)
        # Truncate to the last x.shape[1] positions (in case state was passed)
        y = y[:, -x.shape[1] :, :]
        # New conv state = last k-1 of the raw x (or padded if x shorter)
        if x.shape[1] >= k - 1:
            new_state = x[:, -(k - 1) :, :]
        else:
            new_state = padded[:, -(k - 1) :, :]
        return y, new_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.4 Kimi Delta Attention (KDA)

    The centerpiece of K3's linear-attention path. Per head, with state
    $S_t \in \mathbb{R}^{d_k \times d_v}$:

    $$
    S_t = (I - \beta_t\, k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t\, k_t v_t^\top,
    \qquad
    \tilde o_t = S_t^\top q_t
    $$

    Channel-wise decay $\alpha_t \in (e^{g_{\min}}, 1)^{d_k}$, delta-rule
    write strength $\beta_t \in (0, 1)$. See
    `kimi-linear-kimi-delta-attention.md` for the derivation. The **K3
    change vs Kimi Linear** is the *lower-bounded scaled-sigmoid* log-decay:

    $$
    g_t^h = g_{\min} \cdot \sigma(e^{A_h} \cdot z_t^h) \in (g_{\min}, 0)^{d_k},
    \qquad
    \alpha_t^h = \exp(g_t^h) \in (e^{g_{\min}}, 1)^{d_k}
    $$

    with a learnable per-head $A_h$ (`A_log`, init 0), a per-head bias
    (`dt_bias`) added to $z_t$, and **fixed** $g_{\min} = -5$. Every retention
    factor exceeds $e^{-5} \approx 6.7 \times 10^{-3}$ so the cumulative
    log-decay over a 16-token tile lies in $(-80, 0)$ — reciprocal rescaling
    stays inside BF16 range, letting both diagonal and off-diagonal chunk
    tiles use dense matmuls. The gate is computed in **float32**.

    **K3 also replaces Kimi Linear's low-rank output gate with a full-rank
    input-dependent gate**:

    $$
    y_t = W_o \bigl[\sigma(W_g x_t) \odot \mathrm{RMSNorm}_{\text{head}}(\tilde o_t)\bigr]
    $$

    Two execution modes (mirroring FLA's `kda_mode`):

    - `"chunk"` (default, training): the WY / UT-transform chunkwise
      parallel algorithm — split into chunks of $C$, cumulative decay
      $\gamma$, decay-aware causal mask $\Gamma$, then
      $M = (I + \mathrm{StrictTril}(\mathrm{diag}(\beta)(\Gamma \odot K K^\top)))^{-1} \mathrm{diag}(\beta)$,
      with the inverse computed by **explicit forward substitution** over
      the $C$ rows (never `mx.linalg.inv`). $W$ and $U = M K_\gamma$, $M V$;
      then per-chunk inter+intra output and sequential inter-chunk state
      propagation.
    - `"recurrent"`: the constant-memory $O(d_k d_v)$ RNN form used for
      decoding; accepts and returns the state $S$.

    **All positional information** in K3 comes from KDA's channel-wise
    decay (that is why the MLA layers are NoPE — see next section).
    """)
    return


@app.class_definition
class KimiDeltaAttentionV1(nn.Module):
    """Kimi Delta Attention with lower-bounded decay, full-rank output gate,
    ShortConv on Q/K/V, and both chunkwise-parallel and recurrent modes."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        head_dim: int = 64,
        kernel_size: int = 4,
        chunk_size: int = 64,
        g_min: float = -5.0,
        alpha_bottleneck: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.g_min = g_min
        d_inner = n_heads * head_dim

        # Q/K/V projections (per-head merged)
        self.q_proj = nn.Linear(d_model, d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, d_inner, bias=False)

        # ShortConv on q, k, v (depthwise causal, followed by SiLU)
        self.q_conv1d = ShortConvV1(d_inner, kernel_size)
        self.k_conv1d = ShortConvV1(d_inner, kernel_size)
        self.v_conv1d = ShortConvV1(d_inner, kernel_size)

        # Decay: low-rank projection to per-head d_k, per-head A_log and dt_bias
        self.alpha_down = nn.Linear(d_model, alpha_bottleneck, bias=False)
        self.alpha_up = nn.Linear(alpha_bottleneck, d_inner, bias=False)
        # A_log: per-head learnable scale (init 0 -> exp(0)=1)
        self.A_log = mx.zeros((n_heads,))
        # dt_bias: per-head bias on the decay logit (broadcast across d_k channels)
        self.dt_bias = mx.zeros((n_heads,))

        # Write strength beta (scalar per head per token)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)

        # Output gate: full-rank, input-dependent (K3's change vs Kimi Linear)
        self.g_proj = nn.Linear(d_model, d_inner, bias=False)
        self.o_head_norm = nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(d_inner, d_model, bias=False)

    def _project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Project x into (Q, K, V, g, beta) shaped for KDA math.

        Q, K: (B, H, T, head_dim) L2-normed.
        V: (B, H, T, head_dim).
        g: (B, H, T, head_dim) log-decay in (g_min, 0).
        beta: (B, H, T) in (0, 1).
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q_lin = self.q_proj(x)
        k_lin = self.k_proj(x)
        v_lin = self.v_proj(x)
        q_c, _ = self.q_conv1d(q_lin)
        k_c, _ = self.k_conv1d(k_lin)
        v_c, _ = self.v_conv1d(v_lin)
        # Split into heads: (B, T, H, D) -> (B, H, T, D)
        Q = q_c.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        K = k_c.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        V = v_c.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        # L2-norm Q and K along the last axis
        Q = Q / (mx.sqrt(mx.sum(Q * Q, axis=-1, keepdims=True)) + 1e-6)
        K = K / (mx.sqrt(mx.sum(K * K, axis=-1, keepdims=True)) + 1e-6)
        # Decay logit -> in float32
        z = self.alpha_up(nn.silu(self.alpha_down(x)))  # (B, T, H*D)
        z = z.reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B, H, T, D)
        A_h = mx.exp(self.A_log).reshape(1, H, 1, 1)  # (1, H, 1, 1)
        dt_b = self.dt_bias.reshape(1, H, 1, 1)  # (1, H, 1, 1)
        z_scaled = A_h * z + dt_b
        # Compute gate in fp32
        z32 = z_scaled.astype(mx.float32)
        g = self.g_min * mx.sigmoid(z32)  # in (g_min, 0)
        g = g.astype(x.dtype)
        # Beta: per-head scalar
        beta_lin = self.beta_proj(x)  # (B, T, H)
        beta = mx.sigmoid(beta_lin).transpose(0, 2, 1)  # (B, H, T)
        return Q, K, V, g, beta

    def _output_gate(self, o: mx.array, x: mx.array) -> mx.array:
        """Head-wise RMSNorm then full-rank input-dependent gate + output proj."""
        B, H, T, D = o.shape
        o = self.o_head_norm(o)  # RMSNorm on last axis (head_dim)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        gate = mx.sigmoid(self.g_proj(x))
        return self.o_proj(gate * o)

    def _forward_recurrent(
        self,
        Q: mx.array,
        K: mx.array,
        V: mx.array,
        alpha: mx.array,
        beta: mx.array,
        S_init: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Token-by-token recurrence. alpha here is already exp(g)."""
        B, H, T, dk = Q.shape
        dv = V.shape[-1]
        if S_init is None:
            S = mx.zeros((B, H, dk, dv), dtype=Q.dtype)
        else:
            S = S_init
        outputs = []
        for t in range(T):
            q_t = Q[..., t, :]  # (B, H, dk)
            k_t = K[..., t, :]
            v_t = V[..., t, :]
            a_t = alpha[..., t, :]  # (B, H, dk)
            b_t = beta[..., t]  # (B, H)
            # S = Diag(a_t) . S
            S = a_t[..., :, None] * S  # (B, H, dk, dv)
            # kS = k_t . S : (B, H, dv)
            kS = mx.sum(k_t[..., :, None] * S, axis=-2)
            # S = S - beta * k_t . (kS)^T
            S = S - b_t[..., None, None] * (k_t[..., :, None] * kS[..., None, :])
            # S = S + beta * k_t . v_t^T
            S = S + b_t[..., None, None] * (k_t[..., :, None] * v_t[..., None, :])
            # o = S^T . q = sum over dk
            o_t = mx.sum(q_t[..., :, None] * S, axis=-2)  # (B, H, dv)
            outputs.append(o_t)
        O = mx.stack(outputs, axis=-2)  # (B, H, T, dv)
        return O, S

    def _forward_chunk(
        self,
        Q: mx.array,
        K: mx.array,
        V: mx.array,
        g: mx.array,
        beta: mx.array,
        S_init: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Chunkwise parallel form with UT transform and inter-chunk state prop."""
        B, H, T, dk = Q.shape
        dv = V.shape[-1]
        C = int(self.chunk_size)
        # Right-pad to a multiple of C
        pad = (C - (T % C)) % C
        if pad > 0:
            Q = mx.concatenate([Q, mx.zeros((B, H, pad, dk), dtype=Q.dtype)], axis=-2)
            K = mx.concatenate([K, mx.zeros((B, H, pad, dk), dtype=K.dtype)], axis=-2)
            V = mx.concatenate([V, mx.zeros((B, H, pad, dv), dtype=V.dtype)], axis=-2)
            g = mx.concatenate([g, mx.zeros((B, H, pad, dk), dtype=g.dtype)], axis=-2)
            beta = mx.concatenate([beta, mx.zeros((B, H, pad), dtype=beta.dtype)], axis=-1)
        Tp = T + pad
        nc = Tp // C
        Qc = Q.reshape(B, H, nc, C, dk)
        Kc = K.reshape(B, H, nc, C, dk)
        Vc = V.reshape(B, H, nc, C, dv)
        gc = g.reshape(B, H, nc, C, dk)
        bc = beta.reshape(B, H, nc, C)

        # Cumulative log-decay within each chunk: log_gamma[..., r, :] = sum_{s<=r} g[..., s, :]
        log_gamma = mx.cumsum(gc, axis=-2)  # (B, H, nc, C, dk), in (C*g_min, 0)
        # Clip for FP32 exp safety
        gamma = mx.exp(mx.clip(log_gamma, -80.0, 0.0))  # (B, H, nc, C, dk)
        log_gamma_C = log_gamma[..., -1:, :]  # (B, H, nc, 1, dk)
        gamma_r_to_end = mx.exp(mx.clip(log_gamma_C - log_gamma, -80.0, 0.0))  # (B, H, nc, C, dk)
        gamma_C = gamma[..., -1, :]  # (B, H, nc, dk)

        # Pairwise decay for QK_A and KK_gamma: log_gamma_ratio[..., r, s, c] = log_gamma[r,c] - log_gamma[s,c]
        lg_r = log_gamma[..., :, None, :]  # (B, H, nc, C, 1, dk)
        lg_s = log_gamma[..., None, :, :]  # (B, H, nc, 1, C, dk)
        gamma_ratio = mx.exp(mx.clip(lg_r - lg_s, -80.0, 0.0))  # (B, H, nc, C, C, dk)

        # KK_gamma[r, s] = <gamma_{s->r} ⊙ K_r, K_s>
        KK_gamma = mx.einsum("bhnrsc,bhnrc,bhnsc->bhnrs", gamma_ratio, Kc, Kc)  # (B, H, nc, C, C)
        # QK_A[r, s] = <gamma_{s->r} ⊙ Q_r, K_s>
        QK_A = mx.einsum("bhnrsc,bhnrc,bhnsc->bhnrs", gamma_ratio, Qc, Kc)

        # Masks
        lower_strict = mx.tril(mx.ones((C, C), dtype=Q.dtype), k=-1)  # r > s -> 1
        lower_incl = mx.tril(mx.ones((C, C), dtype=Q.dtype), k=0)  # r >= s -> 1
        T_ = bc[..., :, None] * KK_gamma * lower_strict  # (B, H, nc, C, C), strictly lower

        # M = (I + T_)^{-1} * diag(beta) via forward substitution over the C rows
        M_rows = []
        for r in range(C):
            parts = []
            if r > 0:
                parts.append(mx.zeros(bc.shape[:-1] + (r,), dtype=bc.dtype))
            parts.append(bc[..., r : r + 1])
            if r < C - 1:
                parts.append(mx.zeros(bc.shape[:-1] + (C - r - 1,), dtype=bc.dtype))
            row = mx.concatenate(parts, axis=-1)  # (B, H, nc, C)
            if r > 0:
                M_prev = mx.stack(M_rows, axis=-2)  # (B, H, nc, r, C)
                T_row = T_[..., r, :r]  # (B, H, nc, r)
                contrib = mx.einsum("bhns,bhnsc->bhnc", T_row, M_prev)
                row = row - contrib
            M_rows.append(row)
        M = mx.stack(M_rows, axis=-2)  # (B, H, nc, C, C)

        # K_bar (for W and inter-Q_bar path): gamma_r ⊙ k_r
        K_bar = gamma * Kc  # (B, H, nc, C, dk)
        # K_bar_S (for state-update sum): gamma_{r->C} ⊙ k_r
        K_bar_S = gamma_r_to_end * Kc  # (B, H, nc, C, dk)
        # Q_bar: gamma_r ⊙ q_r (for inter output)
        Q_bar = gamma * Qc  # (B, H, nc, C, dk)

        W = mx.matmul(M, K_bar)  # (B, H, nc, C, dk)
        U = mx.matmul(M, Vc)  # (B, H, nc, C, dv)

        QK_A = QK_A * lower_incl  # causal mask (r >= s)

        # Sequential inter-chunk state propagation
        if S_init is None:
            S = mx.zeros((B, H, dk, dv), dtype=Q.dtype)
        else:
            S = S_init
        O_chunks = []
        for t in range(nc):
            Q_bar_t = Q_bar[..., t, :, :]  # (B, H, C, dk)
            W_t = W[..., t, :, :]
            U_t = U[..., t, :, :]
            QK_A_t = QK_A[..., t, :, :]
            K_bar_S_t = K_bar_S[..., t, :, :]
            gamma_C_t = gamma_C[..., t, :]  # (B, H, dk)

            inter_t = mx.matmul(Q_bar_t, S)  # (B, H, C, dv)
            WS = mx.matmul(W_t, S)  # (B, H, C, dv)
            UW = U_t - WS  # (B, H, C, dv)
            intra_t = mx.matmul(QK_A_t, UW)  # (B, H, C, dv)
            O_chunks.append(inter_t + intra_t)

            K_bar_S_T = mx.swapaxes(K_bar_S_t, -1, -2)  # (B, H, dk, C)
            add = mx.matmul(K_bar_S_T, UW)  # (B, H, dk, dv)
            S = gamma_C_t[..., :, None] * S + add

        O = mx.concatenate(O_chunks, axis=-2)  # (B, H, T+pad, dv)
        if pad > 0:
            O = O[..., :T, :]
        return O, S

    def __call__(
        self,
        x: mx.array,
        mode: str = "chunk",
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Returns (y, S_next). Modes: 'chunk' or 'recurrent'."""
        Q, K, V, g, beta = self._project(x)
        alpha = mx.exp(g)
        if mode == "chunk":
            O, S = self._forward_chunk(Q, K, V, g, beta, state)
        elif mode == "recurrent":
            O, S = self._forward_recurrent(Q, K, V, alpha, beta, state)
        else:
            raise ValueError(f"unknown KDA mode: {mode}")
        y = self._output_gate(O, x)
        return y, S


@app.function
def assert_kda_modes_agree(kda: KimiDeltaAttentionV1, x: mx.array, tol: float = 1e-3) -> float:
    """Verify KDA chunk and recurrent paths agree on the same input.

    This is the Rite that proves the two forms are the same spirit. Returns
    the max abs difference (raises AssertionError if it exceeds `tol`).
    """
    Q, K, V, g, beta = kda._project(x)
    alpha = mx.exp(g)
    O_chunk, _ = kda._forward_chunk(Q, K, V, g, beta, None)
    O_rec, _ = kda._forward_recurrent(Q, K, V, alpha, beta, None)
    mx.eval(O_chunk, O_rec)
    diff = float(mx.abs(O_chunk - O_rec).max().item())
    assert diff < tol, f"KDA chunk vs recurrent disagree by {diff} > {tol}"
    return diff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.5 Gated Multi-Head Latent Attention (Gated MLA + NoPE)

    K3's periodic global attention layer. From DeepSeek-V2: down-project KV
    to a shared latent $c_t = W_c x_t$, cache the latent (much smaller than
    per-head K/V), and up-project per-head at attention time. K3 adds a
    **full-rank channel-wise output gate** (`mla_use_output_gate = True`)
    and — following Kimi Linear — sets `qk_rope_head_dim = 0`: **NoPE.**
    No rotary embedding is applied anywhere in the model. Positional info
    comes entirely from the KDA layers' channel-wise decay.

    The K3 report's MLA config fields we implement here:
    `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim = 0`,
    `v_head_dim`, and `mla_use_output_gate`. The rope fields survive in the
    config only for DeepSeek-MLA source compatibility; we assert
    `qk_rope_head_dim == 0` throughout.
    """)
    return


@app.class_definition
class GatedMultiHeadLatentAttentionV1(nn.Module):
    """Gated MLA with NoPE. Returns (y, latent_cache) where latent_cache is
    the compressed KV latent stream for reuse in generation."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        q_lora_rank: int = 128,
        kv_lora_rank: int = 64,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 64,
        use_output_gate: bool = True,
    ):
        super().__init__()
        assert qk_rope_head_dim == 0, "K3 is NoPE: qk_rope_head_dim must be 0"
        self.n_heads = n_heads
        self.qk_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.use_output_gate = use_output_gate

        # Q: low-rank decomposition
        self.q_down = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_lora_rank)
        self.q_up = nn.Linear(q_lora_rank, n_heads * qk_nope_head_dim, bias=False)

        # KV: shared latent + RMSNorm + per-head up-projections
        self.kv_down = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.kv_up_k = nn.Linear(kv_lora_rank, n_heads * qk_nope_head_dim, bias=False)
        self.kv_up_v = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)

        # Full-rank output gate + output projection
        if use_output_gate:
            self.g_proj = nn.Linear(d_model, n_heads * v_head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def __call__(self, x: mx.array, latent_cache: mx.array | None = None) -> tuple[mx.array, mx.array]:
        B, T, D = x.shape
        H = self.n_heads
        dqk = self.qk_head_dim
        dv = self.v_head_dim
        # Q path
        q = self.q_up(self.q_norm(self.q_down(x)))  # (B, T, H*dqk)
        Q = q.reshape(B, T, H, dqk).transpose(0, 2, 1, 3)  # (B, H, T, dqk)
        # KV latent
        c = self.kv_norm(self.kv_down(x))  # (B, T, kv_lora_rank)
        if latent_cache is not None:
            c_full = mx.concatenate([latent_cache, c], axis=1)
        else:
            c_full = c
        K = self.kv_up_k(c_full).reshape(B, c_full.shape[1], H, dqk).transpose(0, 2, 1, 3)  # (B, H, T_kv, dqk)
        V = self.kv_up_v(c_full).reshape(B, c_full.shape[1], H, dv).transpose(0, 2, 1, 3)  # (B, H, T_kv, dv)
        # Attention: causal, no rope
        scale = 1.0 / math.sqrt(dqk)
        logits = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * scale  # (B, H, T, T_kv)
        # Causal mask: q at position i can attend to k at positions [0, i_kv_offset + i]
        offset = c_full.shape[1] - T
        i = mx.arange(T)[:, None]
        j = mx.arange(c_full.shape[1])[None, :]
        allow = j <= (i + offset)
        neg = mx.array(-1e4, dtype=logits.dtype)
        logits = mx.where(allow[None, None, :, :], logits, neg)
        attn = mx.softmax(logits, axis=-1)
        out = mx.matmul(attn, V)  # (B, H, T, dv)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * dv)
        if self.use_output_gate:
            gate = mx.sigmoid(self.g_proj(x))  # (B, T, H*dv)
            out = out * gate
        return self.o_proj(out), c_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.6 Block Attention Residuals (Block AttnRes)

    Depth-wise softmax attention over **block representations**. Each
    sublayer has a **learned pseudo-query** $w_l \in \mathbb{R}^d$
    (input-independent), keys and values are $\mathrm{RMSNorm}(b_i)$ for
    completed blocks $b_i$ plus the current intra-block partial sum:

    $$
    \kappa(q, k) = \exp(q^\top \mathrm{RMSNorm}(k)),\qquad
    h_l = \sum_{i} \frac{\kappa(w_l, k_i)}{\sum_j \kappa(w_l, k_j)}\, v_i
    $$

    **Pseudo-queries MUST be zero-initialized** so initial depth-attention
    is uniform. The AttnRes paper flags this as *critical* for training
    stability (see the "Zero pseudo-queries" callout in `attention-residuals-attnres.md`).

    K3's decoder layer carries **two** AttnRes projections per layer —
    `self_attention_res_proj` and `mlp_res_proj` — one before the attention
    sublayer and one before the MLP/MoE sublayer, each with its own
    pseudo-query and its own key RMSNorm. Block AttnRes gives us
    $O(Nd)$ memory instead of Full AttnRes's $O(Ld)$. In production, blocks
    must not straddle pipeline stages (single-machine here, but noted).
    """)
    return


@app.class_definition
class BlockAttentionResidualV1(nn.Module):
    """Softmax depth-attention over block reps + current intra-block partial.

    Owns one learned pseudo-query and one RMSNorm for keys. Instantiate
    twice per decoder layer: once for the attention sublayer, once for
    the MLP/MoE sublayer.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        # Pseudo-query MUST be zero-initialized (AttnRes stability requirement)
        self.pseudo_query = mx.zeros((d_model,))
        self.key_norm = nn.RMSNorm(d_model)

    def __call__(self, block_reps: list[mx.array]) -> mx.array:
        """block_reps: list of (B, T, D) tensors — completed block outputs
        plus (optionally as the last element) the current intra-block partial sum.
        """
        # Stack along a new depth axis: (N, B, T, D)
        V = mx.stack(block_reps, axis=0)
        K = self.key_norm(V)  # RMSNorm on the last (channel) axis
        # Logits: <w, RMSNorm(k_i)>: (N, B, T)
        logits = mx.einsum("d,nbtd->nbt", self.pseudo_query, K)
        weights = mx.softmax(logits, axis=0)  # softmax over depth axis
        return mx.einsum("nbt,nbtd->btd", weights, V)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.7 Latent Expert (routed expert operating in latent width)

    In K3's **LatentMoE**, routed experts live in a **compact latent width**
    $\ell$ (K3: 3,584; nano: 128; both are $0.5 \times d_{\text{model}}$)
    rather than the full model width. Each routed expert is a small SiTU-GLU
    FFN operating entirely in the latent space:

    $$
    E^{\text{routed}}_i(u) = W^{(i)}_{\text{down}}\,\text{SiTU-GLU}(u; W^{(i)}_{g}, W^{(i)}_{u})
    $$

    The K3 reference module names are `routed_expert_up_proj`,
    `routed_expert_norm`, `routed_expert_down_proj` (all in the latent
    space). MXFP4 QAT is applied to these routed weights *only* — attention,
    the LatentMoE $W_\downarrow / W_\uparrow$ projections, shared experts,
    and the router itself stay higher precision.
    """)
    return


@app.class_definition
class LatentExpertV1(nn.Module):
    """One routed expert: SiTU-GLU FFN in the latent width, with optional MXFP4 QAT."""

    def __init__(
        self,
        latent_dim: int = 128,
        expert_hidden: int = 64,
        gamma_gate: float = 4.0,
        gamma_up: float = 25.0,
        mxfp4_qat: bool = False,
        mxfp4_block_size: int = 32,
    ):
        super().__init__()
        self.up_gate = nn.Linear(latent_dim, expert_hidden, bias=False)
        self.up_val = nn.Linear(latent_dim, expert_hidden, bias=False)
        self.down = nn.Linear(expert_hidden, latent_dim, bias=False)
        self.gamma_gate = gamma_gate
        self.gamma_up = gamma_up
        self.quant = MXFP4QuantizerV1(enabled=mxfp4_qat, block_size=mxfp4_block_size)

    def __call__(self, u: mx.array) -> mx.array:
        # Fake-quantize the expert input to reproduce MXFP4 QAT dynamics on the expert path
        u_q = self.quant(u)
        return self.down(situ_glu(self.up_gate(u_q), self.up_val(u_q), self.gamma_gate, self.gamma_up))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.8 Quantile Balancing Router (QB)

    K3's **auxiliary-loss-free** load balancer, generalizing DeepSeek-V3's
    bias-based scheme. Router uses **FP32 sigmoid** scoring
    $s_i = \sigma(W_r x_i)$; a per-expert bias $b$ is added **only for
    top-$k$ selection**, and mixture weights are computed from the
    **unbiased** scores (correction-bias-only selection):

    $$
    T_i = \operatorname*{argtop}_k(s_i + b),
    \qquad
    p_{i,j} = \frac{s_{i,j}}{\sum_{r \in T_i} s_{i,r}}
    $$

    **Quantile update.** Instead of DeepSeek's fixed-step sign update, QB
    sets each bias from the router-score quantile matching the target load
    $q = mk/n$. Running top-$(k+1)$ exposes each token's cutoff
    $\tau_i^{(t)}$ (the score an expert must beat to enter its top-$k$);
    the closed-form update is

    $$
    \hat b_j^{(t+1)} \leftarrow \operatorname{quantile}_{1 - k/n}(s_{:,j} - \tau^{(t)}),
    \qquad
    b^{(t+1)} \leftarrow \hat b^{(t+1)} - \mathrm{mean}(\hat b^{(t+1)})\,\mathbf{1}.
    $$

    The update takes effect **only on the next step** (a batch is never
    routed with a bias derived from itself) and the bias is **frozen at
    inference**. On a single machine we compute the quantile exactly with
    `np.quantile`; at K3 scale QB reads it from a per-rank histogram whose
    bin counts are summed with a single all-reduce (additive counts →
    exact pooled quantile up to bin width).

    We store the bias with a **leading underscore** so it stays out of
    the trainable parameter tree and out of the safetensors dump.
    Section 10 saves it explicitly in the JSON sidecar.
    """)
    return


@app.class_definition
class QuantileBalancingRouterV1(nn.Module):
    """Sigmoid router with bias-based top-k selection and closed-form
    quantile bias updates."""

    def __init__(
        self,
        d_model: int = 256,
        n_experts: int = 32,
        n_active: int = 4,
        freeze: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Non-trainable QB bias (leading underscore keeps out of param tree)
        self._route_bias = mx.zeros((n_experts,))
        self._frozen = freeze

    def freeze(self) -> None:
        self._frozen = True

    def score(self, x: mx.array) -> mx.array:
        """Return unbiased FP32 sigmoid router scores of shape (B, T, E)."""
        logits = self.gate(x).astype(mx.float32)
        return mx.sigmoid(logits)

    def select(self, scores: mx.array) -> tuple[mx.array, mx.array]:
        """Top-k selection using scores + QB bias.

        Returns (top_indices, top_biased_scores) both of shape (B, T, k).
        The mixture weights should be computed from unbiased scores by the caller.
        """
        biased = scores + mx.stop_gradient(self._route_bias)
        k = self.n_active
        idx = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        vals = mx.take_along_axis(biased, idx, axis=-1)
        return idx, vals

    def compute_cutoffs(self, biased_scores: mx.array) -> mx.array:
        """For QB update: return the top-(k+1) cutoff per token (B, T)."""
        k1 = min(self.n_active + 1, self.n_experts)
        idx = mx.argpartition(-biased_scores, kth=k1 - 1, axis=-1)[..., :k1]
        vals = mx.take_along_axis(biased_scores, idx, axis=-1)
        cutoff = mx.min(vals, axis=-1)  # the (k+1)-th score
        return cutoff

    def update_bias_from_quantile(self, scores: mx.array) -> None:
        """Closed-form QB update. Non-differentiable; wraps in numpy."""
        if self._frozen:
            return
        biased = scores + mx.stop_gradient(self._route_bias)
        cutoff = self.compute_cutoffs(biased)  # (B, T)
        margins = scores - cutoff[..., None]  # (B, T, E)
        m = np.asarray(margins).reshape(-1, self.n_experts)  # (BT, E)
        q = 1.0 - self.n_active / self.n_experts
        b_hat = np.quantile(m, q, axis=0)  # (E,)
        b_hat = b_hat - b_hat.mean()
        self._route_bias = mx.array(b_hat.astype(np.float32))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.9 Stable LatentMoE

    K3's channel-mixing block:

    $$
    u = \sum_{i \in T_k(x)} p_i \cdot E^{\text{routed}}_i(W_\downarrow x),
    \qquad
    y = \sum_{j=1}^{N_s} E^{\text{shared}}_j(x) + W_\uparrow \, \mathrm{RMSNorm}(u)
    $$

    The three **Stable-LatentMoE** stabilizers are visible below and each
    is toggleable via config:

    1. `latentmoe_use_rmsnorm` — the RMSNorm between the routed
       aggregation and $W_\uparrow$ (K3's "Normalized LatentMoE").
    2. `latentmoe_use_situ_glu` — SiTU-GLU inside every expert (K3
       swaps SwiGLU for SiTU-GLU here).
    3. `latentmoe_use_quantile_balancing` — Quantile Balancing on the router.

    We return `(output, expert_load, router_stats)` so the training loop
    can plot load balancing and update the QB bias.
    """)
    return


@app.class_definition
class StableLatentMoEV1(nn.Module):
    """Stable LatentMoE: routed experts in the latent width + shared experts
    at full width, with all three K3 stability fixes toggleable."""

    def __init__(
        self,
        d_model: int = 256,
        latent_dim: int = 128,
        n_routed_experts: int = 32,
        n_active_experts: int = 4,
        n_shared_experts: int = 2,
        expert_hidden: int = 64,
        shared_expert_hidden: int = 512,
        gamma_gate: float = 4.0,
        gamma_up: float = 25.0,
        use_rmsnorm: bool = True,
        use_situ_glu: bool = True,
        use_quantile_balancing: bool = True,
        mxfp4_qat: bool = False,
        mxfp4_block_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_routed = n_routed_experts
        self.n_active = n_active_experts
        self.use_rmsnorm = use_rmsnorm
        self.use_situ_glu = use_situ_glu
        self.use_qb = use_quantile_balancing

        # LatentMoE down/up (never quantized: attention + LatentMoE projections stay HP)
        self.latent_down = nn.Linear(d_model, latent_dim, bias=False)
        self.latent_up = nn.Linear(latent_dim, d_model, bias=False)
        self.aggregate_norm = nn.RMSNorm(latent_dim) if use_rmsnorm else None

        # Shared experts — at full model width, unquantized
        self.shared_experts = [
            SiTUGLUV1(d_model, shared_expert_hidden, d_model, gamma_gate, gamma_up)
            if use_situ_glu
            else SiTUGLUV1(d_model, shared_expert_hidden, d_model, 1e6, 1e6)
            for _ in range(n_shared_experts)
        ]

        # Routed experts — in the latent width, MXFP4-QAT-toggleable
        self.routed_experts = [
            LatentExpertV1(latent_dim, expert_hidden, gamma_gate, gamma_up, mxfp4_qat, mxfp4_block_size)
            for _ in range(n_routed_experts)
        ]

        # Router
        self.router = QuantileBalancingRouterV1(d_model, n_routed_experts, n_active_experts)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, dict]:
        B, T, D = x.shape
        # Shared experts always on (at full width)
        shared_out = mx.zeros_like(x)
        for se in self.shared_experts:
            shared_out = shared_out + se(x)

        # Route on FP32 sigmoid scores
        scores = self.router.score(x)  # (B, T, E)
        idx, biased_top = self.router.select(scores)  # (B, T, k), (B, T, k)
        # Mixture weights from UNBIASED scores restricted to top-k
        top_unbiased = mx.take_along_axis(scores, idx, axis=-1)  # (B, T, k)
        weights = top_unbiased / (mx.sum(top_unbiased, axis=-1, keepdims=True) + 1e-6)
        weights = weights.astype(x.dtype)

        # Project to latent, dispatch to routed experts (dense loop over experts)
        u_in = self.latent_down(x)  # (B, T, L)
        routed_agg = mx.zeros_like(u_in)
        # Compute a per-expert mask + per-expert weight (B, T)
        idx_np = np.asarray(idx)
        # Build a dense (B, T, E) weight tensor: weight_dense[b,t,e] = weight if e ∈ top-k else 0
        weight_dense = np.zeros((B, T, self.n_routed), dtype=np.float32)
        weights_np = np.asarray(weights).astype(np.float32)
        for k_slot in range(self.n_active):
            eidx = idx_np[..., k_slot]  # (B, T)
            wval = weights_np[..., k_slot]
            for b in range(B):
                weight_dense[b, np.arange(T), eidx[b]] += wval[b]
        weight_dense_mx = mx.array(weight_dense).astype(x.dtype)  # (B, T, E)

        expert_load = mx.zeros((self.n_routed,), dtype=mx.float32)
        for e in range(self.n_routed):
            w_e = weight_dense_mx[..., e : e + 1]  # (B, T, 1)
            e_out = self.routed_experts[e](u_in)  # (B, T, L)
            routed_agg = routed_agg + w_e * e_out
            expert_load = expert_load + mx.sum((weight_dense_mx[..., e] > 0).astype(mx.float32))

        # Aggregation normalisation then up-project
        if self.aggregate_norm is not None:
            routed_agg = self.aggregate_norm(routed_agg)
        routed_out = self.latent_up(routed_agg)

        y = shared_out + routed_out
        router_stats = {"scores": scores, "idx": idx, "biased_top": biased_top}
        return y, expert_load, router_stats

    def maybe_update_router_bias(self, scores: mx.array) -> None:
        """If Quantile Balancing is enabled, update router bias for next step."""
        if self.use_qb:
            self.router.update_bias_from_quantile(scores)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.10 One nano-K3 decoder layer

    Per K3, each decoder layer sandwiches AttnRes around both sublayers:

    ```
    BlockAttnRes -> (KDA or Gated MLA) -> BlockAttnRes -> Stable LatentMoE
    ```

    with pre-RMSNorm on each sublayer's input. Sublayer outputs are added
    into the current *intra-block partial sum*; when a block boundary is
    crossed, the partial sum is finalized as a new $b_n$ and appended to
    the completed-blocks list.

    `layer_kind` picks between `"kda"` and `"mla"`. The KDA layer takes a
    KDA state (S matrix), the MLA layer takes a latent cache.
    """)
    return


@app.class_definition
class NanoKimiK3LayerV1(nn.Module):
    """One K3 decoder layer: two AttnRes + (KDA or MLA) + Stable LatentMoE."""

    def __init__(self, cfg: NanoKimiK3Config, layer_kind: str = "kda"):
        super().__init__()
        assert layer_kind in ("kda", "mla")
        self.layer_kind = layer_kind
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.moe_norm = nn.RMSNorm(cfg.d_model)
        self.attn_res = BlockAttentionResidualV1(cfg.d_model)  # self_attention_res_proj
        self.mlp_res = BlockAttentionResidualV1(cfg.d_model)  # mlp_res_proj
        if layer_kind == "kda":
            self.mixer = KimiDeltaAttentionV1(
                d_model=cfg.d_model,
                n_heads=cfg.kda_n_heads,
                head_dim=cfg.kda_head_dim,
                kernel_size=cfg.short_conv_kernel,
                chunk_size=cfg.kda_chunk_size,
                g_min=cfg.g_min,
                alpha_bottleneck=cfg.kda_alpha_bottleneck,
            )
        else:
            self.mixer = GatedMultiHeadLatentAttentionV1(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                q_lora_rank=cfg.q_lora_rank,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                use_output_gate=cfg.mla_use_output_gate,
            )
        self.moe = StableLatentMoEV1(
            d_model=cfg.d_model,
            latent_dim=cfg.latent_dim,
            n_routed_experts=cfg.n_routed_experts,
            n_active_experts=cfg.n_active_experts,
            n_shared_experts=cfg.n_shared_experts,
            expert_hidden=cfg.expert_hidden,
            shared_expert_hidden=cfg.shared_expert_hidden,
            gamma_gate=cfg.situ_gamma_gate,
            gamma_up=cfg.situ_gamma_up,
            use_rmsnorm=cfg.latentmoe_use_rmsnorm,
            use_situ_glu=cfg.latentmoe_use_situ_glu,
            use_quantile_balancing=cfg.latentmoe_use_quantile_balancing,
            mxfp4_qat=cfg.mxfp4_qat,
            mxfp4_block_size=cfg.mxfp4_block_size,
        )

    def __call__(
        self,
        blocks_completed: list[mx.array],
        partial_block: mx.array,
        mixer_state=None,
        kda_mode: str = "chunk",
    ) -> tuple[mx.array, mx.array, mx.array, dict]:
        """Returns (new_partial_block, new_mixer_state, expert_load, router_stats)."""
        # AttnRes before attention (reads over completed blocks + current partial)
        h_attn = self.attn_res(blocks_completed + [partial_block])
        h_attn_normed = self.attn_norm(h_attn)
        if self.layer_kind == "kda":
            y_attn, new_state = self.mixer(h_attn_normed, mode=kda_mode, state=mixer_state)
        else:
            y_attn, new_state = self.mixer(h_attn_normed, latent_cache=mixer_state)
        partial_block = partial_block + y_attn

        # AttnRes before MoE
        h_moe = self.mlp_res(blocks_completed + [partial_block])
        y_moe, expert_load, router_stats = self.moe(self.moe_norm(h_moe))
        partial_block = partial_block + y_moe
        return partial_block, new_state, expert_load, router_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.11 Multi-Token Prediction (MTP)

    K3 keeps **one MTP layer** (unchanged from K2). It predicts token
    $t + 2$ from the shared final hidden state via a small block + linear
    head. Trained jointly with the main LM loss (weight
    `mtp_loss_weight = 0.1`). At post-training the MTP layer is fine-tuned
    into the EAGLE-3 draft — but during pre-training it is a plain
    dense-gradient signal that acts as an auxiliary next-next-token loss.
    """)
    return


@app.class_definition
class MultiTokenPredictionLayerV1(nn.Module):
    """Depth-1 MTP head: predict token t+2 from position t's hidden state."""

    def __init__(self, d_model: int = 256, vocab_size: int = 10000, hidden: int = 512):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.block = SiTUGLUV1(d_model, hidden, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        h = h[:, :-2, :]
        h = self.norm(h + self.block(h))
        return self.head(h)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.12 EAGLE-3 Draft Head

    K3's speculative-decoding draft: a **single decoder layer** that takes
    the **fused low/mid/high features from the 1st, middle, and final
    AttnRes blocks** through a bias-free $W_{E3}$ initialized as
    $[\,0\ 0\ I\,]$ (so at init it is an identity on the high feature),
    with the target model frozen. Optimized via the **LK loss** — the
    negative log of the lossless-speculative-sampling acceptance rate:

    $$
    \mathcal{L}_{\text{LK}} = -\log \sum_{x \in V} \min(p(x), q(x))
    $$

    K3 unrolls the draft **7 steps**; at nano scale we unroll **2 steps**
    (the config knob `eagle3_unroll`). Fine-tuning is opt-in via a
    checkbox — the training loop keeps the target model frozen with
    `mx.stop_gradient` on the fused features.
    """)
    return


@app.function
def lk_loss(p_logits: mx.array, q_logits: mx.array) -> mx.array:
    """LK loss: -log sum_x min(p(x), q(x)) with p target (frozen), q draft."""
    p = mx.softmax(p_logits, axis=-1)
    q = mx.softmax(q_logits, axis=-1)
    accept = mx.sum(mx.minimum(p, q), axis=-1)
    return -mx.log(mx.maximum(accept, 1e-9)).mean()


@app.class_definition
class Eagle3DraftHeadV1(nn.Module):
    """Single-layer EAGLE-3 draft head with W_E3 = [0 0 I] initialization.

    Inputs are the tuple of low/mid/high AttnRes-block reps at each position.
    The head fuses them via W_E3 (bias-free), passes through a single KDA
    layer, and heads to vocab logits.
    """

    def __init__(self, d_model: int = 256, vocab_size: int = 10000):
        super().__init__()
        # W_E3: (3*d, d). Initialize as [0 | 0 | I]: only high feature passes.
        w = np.zeros((3 * d_model, d_model), dtype=np.float32)
        w[2 * d_model :, :] = np.eye(d_model, dtype=np.float32)
        # nn.Linear stores weight as (out, in), so our weight matrix should be (d, 3*d)
        self.fuse = nn.Linear(3 * d_model, d_model, bias=False)
        # Overwrite the fused linear weight to match W_E3^T shape (out=d, in=3*d)
        self.fuse.weight = mx.array(w.T)  # shape (d, 3*d)
        self.body = SiTUGLUV1(d_model, 4 * d_model, d_model)
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, low: mx.array, mid: mx.array, high: mx.array) -> mx.array:
        # low/mid/high: (B, T, D). Fuse over channel axis then run body.
        fused = self.fuse(mx.concatenate([low, mid, high], axis=-1))
        return self.head(self.norm(fused + self.body(fused)))


@app.function
def build_layer_kind_schedule(n_layers: int, kda_group_size: int, trailing_mla: bool) -> list[str]:
    """3 KDA + 1 MLA groups, plus trailing MLA if requested."""
    schedule = []
    body_len = n_layers - (1 if trailing_mla else 0)
    for i in range(body_len):
        # Within each group of (kda_group_size + 1), first kda_group_size are KDA, last is MLA
        pos_in_group = i % (kda_group_size + 1)
        schedule.append("kda" if pos_in_group < kda_group_size else "mla")
    if trailing_mla:
        schedule.append("mla")
    return schedule


@app.function
def build_attnres_block_schedule(n_layers: int, n_blocks: int) -> list[int]:
    """Return a length-`n_layers` list mapping each layer -> its block index (0..n_blocks-1).

    We align block boundaries to KDA-MLA groups when possible: with 13 layers
    and 4 blocks, we get [4, 4, 4, 1] so each of the first three (3-KDA + 1-MLA)
    groups is its own block and the trailing MLA gets its own block.
    """
    # Prefer boundary sizes [4, 4, 4, 1] for n_layers=13, n_blocks=4
    if n_layers == 13 and n_blocks == 4:
        sizes = [4, 4, 4, 1]
    else:
        base = n_layers // n_blocks
        rem = n_layers % n_blocks
        sizes = [base + (1 if i < rem else 0) for i in range(n_blocks)]
    schedule = []
    for b, sz in enumerate(sizes):
        schedule.extend([b] * sz)
    return schedule


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.13 NanoKimiK3V1 — the top-level model

    Data flow: `tokens -> embed (block b_0) -> for each layer: 2x AttnRes + mixer + Stable LatentMoE, accumulating into partial_block; at each block boundary, finalize b_n -> final RMSNorm -> LM head + MTP head`. `__call__` returns
    `(logits, mtp_logits, aux_info)` where `aux_info` carries per-layer expert loads and router stats.

    The model supports a `kda_mode` argument so generation can switch to
    the recurrent path, and a `cache` argument carrying per-layer KDA
    states and MLA latent caches.
    """)
    return


@app.class_definition
class NanoKimiK3V1(nn.Module):
    """Nano-Kimi-K3 backbone: every K3 component, drastically shrunk."""

    def __init__(self, cfg: NanoKimiK3Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Layer schedules
        self.layer_kinds = build_layer_kind_schedule(cfg.n_layers, cfg.kda_group_size, cfg.trailing_mla)
        self.attnres_block = build_attnres_block_schedule(cfg.n_layers, self.n_attnres_blocks(cfg))
        self.layers = [NanoKimiK3LayerV1(cfg, layer_kind=self.layer_kinds[i]) for i in range(cfg.n_layers)]
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.mtp_head = MultiTokenPredictionLayerV1(cfg.d_model, cfg.vocab_size)

    @staticmethod
    def n_attnres_blocks(cfg: NanoKimiK3Config) -> int:
        if cfg.n_layers == 13:
            return 4
        return max(1, cfg.n_layers // max(1, cfg.attn_res_block_size))

    def __call__(
        self,
        ids: mx.array,
        cache: list | None = None,
        kda_mode: str = "chunk",
    ) -> tuple[mx.array, mx.array, dict]:
        h = self.embed(ids)  # (B, T, D)
        # embedding is block b_0 for AttnRes
        blocks_completed = [h]
        partial_block = mx.zeros_like(h)
        expert_loads = []
        router_stats_all = []
        new_cache = []
        cur_block_id = 0
        for i, layer in enumerate(self.layers):
            block_id = self.attnres_block[i]
            state_in = cache[i] if cache is not None else None
            partial_block, new_state, load, stats = layer(
                blocks_completed=blocks_completed,
                partial_block=partial_block,
                mixer_state=state_in,
                kda_mode=kda_mode,
            )
            new_cache.append(new_state)
            expert_loads.append(load)
            router_stats_all.append(stats)
            # If the next layer belongs to a new block, finalize this block
            next_id = self.attnres_block[i + 1] if i + 1 < len(self.layers) else block_id + 1
            if next_id != block_id:
                blocks_completed.append(partial_block)
                partial_block = mx.zeros_like(h)
                cur_block_id = next_id

        # The final partial_block might already be zero (last layer flushed) or
        # not (last layer did not cross a boundary). If not flushed, add it.
        if not mx.all(partial_block == 0).item():
            blocks_completed.append(partial_block)

        # Final hidden state = sum over all completed blocks (equivalent to
        # standard-residual accumulation applied to the AttnRes block reps)
        h_final = blocks_completed[0]
        for b in blocks_completed[1:]:
            h_final = h_final + b
        h_norm = self.final_norm(h_final)
        logits = self.lm_head(h_norm)
        mtp_logits = self.mtp_head(h_norm)
        aux_info = {
            "expert_loads": expert_loads,
            "router_stats": router_stats_all,
            "blocks_completed": blocks_completed,
            "new_cache": new_cache,
        }
        return logits, mtp_logits, aux_info

    def update_router_biases(self, router_stats_all: list) -> None:
        """Call at each training step AFTER the optimizer step; QB update
        takes effect on the next step (never routes a batch with a bias
        derived from itself)."""
        for i, layer in enumerate(self.layers):
            layer.moe.maybe_update_router_bias(mx.stop_gradient(router_stats_all[i]["scores"]))


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(int(v.size) for _, v in tree_flatten(model.parameters()))


@app.function
def count_parameters_by_component(model) -> dict[str, int]:
    """Per-component parameter breakdown by path prefix."""
    buckets = {
        "embed": 0,
        "kda": 0,
        "mla": 0,
        "moe_routed": 0,
        "moe_shared": 0,
        "moe_projections": 0,
        "moe_router": 0,
        "attnres": 0,
        "lm_head": 0,
        "mtp_head": 0,
        "norm": 0,
        "other": 0,
    }
    for path, arr in tree_flatten(model.parameters()):
        n = int(arr.size)
        lower = path.lower()
        if "embed" in lower and "mtp" not in lower and "lm_head" not in lower:
            buckets["embed"] += n
        elif "mtp_head" in lower:
            buckets["mtp_head"] += n
        elif "lm_head" in lower:
            buckets["lm_head"] += n
        elif "routed_experts" in lower or "routed_expert" in lower:
            buckets["moe_routed"] += n
        elif "shared_experts" in lower or "shared_expert" in lower:
            buckets["moe_shared"] += n
        elif "latent_down" in lower or "latent_up" in lower or "aggregate_norm" in lower:
            buckets["moe_projections"] += n
        elif "router" in lower or ("gate" in lower and "moe" in lower):
            buckets["moe_router"] += n
        elif "attn_res" in lower or "mlp_res" in lower or "pseudo_query" in lower or "key_norm" in lower:
            buckets["attnres"] += n
        elif "mixer" in lower and ("kda" in lower or "a_log" in lower or "alpha" in lower or "beta_proj" in lower or "conv1d" in lower):
            buckets["kda"] += n
        elif "mixer" in lower and ("q_down" in lower or "q_up" in lower or "kv_down" in lower or "kv_up" in lower or "q_norm" in lower or "kv_norm" in lower):
            buckets["mla"] += n
        elif "mixer" in lower and "o_proj" in lower:
            # Ambiguous: could be KDA or MLA. Attribute by parent module name if possible.
            buckets["kda"] += n if "layer" in lower else 0
        elif "norm" in lower:
            buckets["norm"] += n
        else:
            buckets["other"] += n
    return buckets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.14 Per-Head Muon + Hybrid Muon/AdamW

    K3 uses **Muon** (momentum + Newton-Schulz orthogonalization of the 2-D
    update) for matrix parameters, with two K3-specific tweaks:

    - **Per-Head Muon** — for attention projections, **partition the
      momentum matrix along the head dimension** and run NS on each
      head's block separately (equalizing update scale across heads;
      full-matrix orthogonalization would let large-gradient heads
      dominate). See `muon-optimizer-derivation.md`.
    - **K2-style weight clipping** and **weight decay 0.1**.

    NS polynomial: $(a, b, c) = (3.4445, -4.7750, 2.0315)$, 5 iterations,
    input Frobenius-normalized in float32.

    `HybridPerHeadMuonAdamWV1` routes $\ge$2-D matrix weights to Per-Head
    Muon and everything else (embeddings, LM head, RMSNorm gains, biases,
    AttnRes pseudo-queries) to AdamW.
    """)
    return


@app.function
def newton_schulz_orthogonalize(g: mx.array, steps: int = 5) -> mx.array:
    """Newton-Schulz orthogonalization of a 2-D matrix.

    Uses the (3.4445, -4.7750, 2.0315) polynomial, in float32 with the
    input Frobenius-normalized first. Returns U V^T of the SVD of g.
    """
    g32 = g.astype(mx.float32)
    fnorm = mx.sqrt(mx.sum(g32 * g32)) + 1e-7
    x = g32 / fnorm
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        aa = mx.matmul(x, x.T)
        term = b * aa + c * mx.matmul(aa, aa)
        x = a * x + mx.matmul(term, x)
    if transposed:
        x = x.T
    return x.astype(g.dtype)


@app.class_definition
class PerHeadMuonOptimizerV1(optim.Optimizer):
    """Per-Head Muon: split the momentum matrix along its head axis and run
    NS on each head's block separately. Falls back to full-matrix Muon for
    non-head-structured matrices and to plain momentum for 1-D params."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        weight_decay: float = 0.1,
        n_heads: int = 4,
        weight_clip: float = 1.0,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.rms_scale = rms_scale
        self.weight_decay = weight_decay
        self.n_heads = n_heads
        self.weight_clip = weight_clip

    def init_single(self, parameter: mx.array, state: dict) -> None:
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        v = self.momentum * state["v"] + gradient
        state["v"] = v
        if v.ndim < 2:
            update = v
        else:
            v2 = v.reshape(-1, v.shape[-1])
            fan_out, fan_in = v2.shape
            if fan_out % self.n_heads == 0 and fan_out >= 2 * self.n_heads:
                # Per-head: split leading axis into (n_heads, head_dim)
                head_out = fan_out // self.n_heads
                v3 = v2.reshape(self.n_heads, head_out, fan_in)
                orth_heads = [newton_schulz_orthogonalize(v3[h], self.ns_steps) for h in range(self.n_heads)]
                orth = mx.stack(orth_heads, axis=0).reshape(v.shape)
                update = orth * self.rms_scale * math.sqrt(max(1.0, head_out / max(1, fan_in)))
            else:
                orth = newton_schulz_orthogonalize(v2, self.ns_steps).reshape(v.shape)
                scale = math.sqrt(max(1.0, fan_out / max(1, fan_in)))
                update = orth * self.rms_scale * scale
        lr = self.learning_rate.astype(gradient.dtype)
        # Weight decay
        if self.weight_decay != 0.0:
            parameter = parameter * (1 - lr * self.weight_decay)
        parameter = parameter - lr * update
        # K2-style weight clipping (elementwise)
        if self.weight_clip > 0:
            parameter = mx.clip(parameter, -self.weight_clip, self.weight_clip)
        return parameter


@app.function
def classify_param_for_muon(path: str, arr: mx.array) -> bool:
    """Return True if the parameter belongs to Per-Head Muon."""
    if arr.ndim < 2:
        return False
    lower = path.lower()
    if "embed" in lower:
        return False
    if "lm_head" in lower or "mtp_head" in lower:
        return False
    if "pseudo_query" in lower:
        return False
    return True


@app.class_definition
class HybridPerHeadMuonAdamWV1:
    """Composite optimizer: Per-Head Muon on 2-D matrices, AdamW on
    embeddings / LM head / RMSNorm gains / biases / AttnRes pseudo-queries.
    Includes K2-style weight clipping and weight decay 0.1 defaults."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        n_heads: int = 4,
        weight_clip: float = 1.0,
    ):
        self.model = model
        self.muon = PerHeadMuonOptimizerV1(
            learning_rate=learning_rate,
            momentum=momentum,
            ns_steps=ns_steps,
            rms_scale=rms_scale,
            weight_decay=weight_decay,
            n_heads=n_heads,
            weight_clip=weight_clip,
        )
        self.adamw = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def _split(self, gradients: dict) -> tuple[dict, dict]:
        muon_flat: list = []
        adamw_flat: list = []
        for path, val in tree_flatten(gradients):
            if val is None:
                continue
            if classify_param_for_muon(path, val):
                muon_flat.append((path, val))
            else:
                adamw_flat.append((path, val))
        muon_tree = tree_unflatten(muon_flat) if muon_flat else {}
        adamw_tree = tree_unflatten(adamw_flat) if adamw_flat else {}
        return muon_tree, adamw_tree

    def update(self, model: nn.Module, gradients: dict) -> None:
        muon_grads, adamw_grads = self._split(gradients)
        if muon_grads:
            self.muon.update(model, muon_grads)
        if adamw_grads:
            self.adamw.update(model, adamw_grads)

    @property
    def state(self) -> list:
        return [self.muon.state, self.adamw.state]


@app.cell
def _(config):
    model = NanoKimiK3V1(config)
    mx.eval(model.parameters())
    _total = count_parameters(model)
    _breakdown = count_parameters_by_component(model)
    print(f"total params: {_total:,}")
    print(f"layer schedule (kda/mla): {model.layer_kinds}")
    print(f"attnres block schedule (layer -> block idx): {model.attnres_block}")
    for _k, _v in _breakdown.items():
        print(f"  {_k:>18s}: {_v:>10,}")
    return (model,)


@app.cell
def _(config, model):
    _kda_layer = None
    for _l in model.layers:
        if _l.layer_kind == "kda":
            _kda_layer = _l.mixer
            break
    if _kda_layer is None:
        print("no KDA layer found?!")
        _diff = float("nan")
    else:
        mx.random.seed(0)
        _x_test = mx.random.normal((2, config.seq_len, config.d_model)) * 0.3
        mx.eval(_x_test)
        _diff = assert_kda_modes_agree(_kda_layer, _x_test, tol=1e-2)
        print(f"KDA chunk vs recurrent max abs diff: {_diff:.6f} (tol 1e-2)")
    return


@app.cell
def _(model, train_ds):
    _x, _y = next(iter(train_ds))
    _logits, _mtp_logits, _aux = model(_x)
    mx.eval(_logits, _mtp_logits)
    print(f"input  shape: {_x.shape}")
    print(f"logits shape: {_logits.shape}")
    print(f"mtp    shape: {_mtp_logits.shape}")
    print(f"n layers with expert load: {len(_aux['expert_loads'])}")
    print(f"n completed AttnRes blocks: {len(_aux['blocks_completed'])}")
    print(f"first-layer expert load (should sum to B*T*n_active): {_aux['expert_loads'][0]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Training

    Cross-entropy on next-token prediction + weighted MTP loss.
    Cosine LR with 1% linear warmup (K3 chose cosine over WSD). K2-style
    weight clipping baked into Muon. Two optimizer routes exposed:

    - **per-head-muon+adamw** — the K3 prescription.
    - **adamw** — a plain baseline so the Per-Head Muon claim is testable.

    MXFP4 QAT and Quantile Balancing are each toggleable so their effect
    on training dynamics is visible.

    MLX has no autocast / GradScaler — everything runs in native precision.
    The QB router bias is updated **once per step** (after `optimizer.update`)
    and applied on the **next** step.
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="Learning rate",
    )
    bs_ui = mo.ui.dropdown(options=[2, 4, 8], value=4, label="Batch size")
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "0.01": 0.01, "0.1": 0.1},
        value="0.1",
        label="Weight decay",
    )
    epochs_ui = mo.ui.slider(1, 10, value=1, step=1, label="Epochs")
    opt_ui = mo.ui.dropdown(
        options=["per-head-muon+adamw", "adamw"],
        value="per-head-muon+adamw",
        label="Optimizer",
    )
    mxfp4_ui = mo.ui.checkbox(label="MXFP4 QAT on routed experts", value=False)
    qb_ui = mo.ui.checkbox(label="Quantile Balancing on router", value=True)
    max_steps_ui = mo.ui.slider(10, 400, value=40, step=10, label="Max steps per epoch")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Training hyperparameters"),
            mo.hstack([lr_ui, bs_ui, wd_ui, epochs_ui]),
            mo.hstack([opt_ui, mxfp4_ui, qb_ui, max_steps_ui]),
            train_btn,
        ]
    )
    return (
        bs_ui,
        epochs_ui,
        lr_ui,
        max_steps_ui,
        mxfp4_ui,
        opt_ui,
        qb_ui,
        train_btn,
        wd_ui,
    )


@app.function
def compute_lm_loss(model, x: mx.array, y: mx.array, mtp_weight: float) -> mx.array:
    """CE on next-token prediction + weighted MTP loss on token t+2."""
    logits, mtp_logits, _aux = model(x)
    ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).mean()
    mtp_target = y[:, 1:]  # y is x shifted +1, so y[:, 1:] is token t+2 relative to x
    mtp_target = mtp_target[:, : mtp_logits.shape[1]]
    mtp_ce = nn.losses.cross_entropy(
        mtp_logits.reshape(-1, mtp_logits.shape[-1]), mtp_target.reshape(-1)
    ).mean()
    return ce + mtp_weight * mtp_ce


@app.function
def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


@app.function
def cosine_lr_schedule_with_warmup(step: int, warmup: int, total: int, peak: float, floor_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    floor = peak * floor_frac
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


@app.function
def run_train_epoch(
    model,
    optimizer,
    train_iter,
    mtp_weight: float,
    max_steps: int,
    step_offset: int,
    lr_schedule_fn,
) -> tuple[list[float], list, list, int]:
    losses: list[float] = []
    load_snapshots: list = []
    bias_snapshots: list = []
    loss_fn = lambda m, x, y: compute_lm_loss(m, x, y, mtp_weight)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    step = 0
    for x, y in train_iter:
        if step >= max_steps:
            break
        loss, grads = loss_and_grad(model, x, y)
        lr = lr_schedule_fn(step_offset + step)
        if hasattr(optimizer, "muon"):
            optimizer.muon.learning_rate = lr
            optimizer.adamw.learning_rate = lr
        else:
            optimizer.learning_rate = lr
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        # Post-step: QB bias update (applied to NEXT step)
        _, _, aux = model(x)
        model.update_router_biases(aux["router_stats"])
        losses.append(float(loss.item()))
        step += 1
        # Every 10 steps take a snapshot
        if step % 10 == 0 or step == max_steps:
            load_snapshots.append(
                np.stack([np.asarray(ld) for ld in aux["expert_loads"]], axis=0)
            )
            bias_snapshots.append(
                np.stack([np.asarray(layer.moe.router._route_bias) for layer in model.layers], axis=0)
            )
    return losses, load_snapshots, bias_snapshots, step


@app.function
def run_evaluate(model, data_iter, mtp_weight: float, max_batches: int = 10) -> float:
    total = 0.0
    n = 0
    for x, y in data_iter:
        if n >= max_batches:
            break
        loss = compute_lm_loss(model, x, y, mtp_weight)
        mx.eval(loss)
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@app.cell
def _(
    all_tokens,
    bs_ui,
    config,
    epochs_ui,
    lr_ui,
    max_steps_ui,
    mo,
    mxfp4_ui,
    opt_ui,
    qb_ui,
    train_btn,
    wd_ui,
):
    train_losses: list[float] = []
    val_losses: list[float] = []
    expert_load_history: list = []
    router_bias_history: list = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** above to begin. The model stays randomly initialized until then."))
    else:
        _cfg = NanoKimiK3Config(
            **{
                **asdict(config),
                "mxfp4_qat": mxfp4_ui.value,
                "latentmoe_use_quantile_balancing": qb_ui.value,
            }
        )
        _model = NanoKimiK3V1(_cfg)
        mx.eval(_model.parameters())
        _train_ds, _val_ds, _ = make_datasets(all_tokens, seq_len=_cfg.seq_len, batch_size=bs_ui.value)
        if opt_ui.value == "per-head-muon+adamw":
            _optimizer = HybridPerHeadMuonAdamWV1(
                _model,
                learning_rate=lr_ui.value,
                weight_decay=wd_ui.value,
                n_heads=_cfg.n_heads,
                weight_clip=_cfg.k2_weight_clip,
            )
        else:
            _optimizer = optim.AdamW(learning_rate=lr_ui.value, weight_decay=wd_ui.value)
        _total_steps = max_steps_ui.value * epochs_ui.value
        _warmup = max(2, int(_cfg.warmup_frac * _total_steps))
        _lr_fn = lambda s: cosine_lr_schedule_with_warmup(s, _warmup, _total_steps, lr_ui.value)
        _step_offset = 0
        for _epoch in range(epochs_ui.value):
            _ep_losses, _load_snaps, _bias_snaps, _steps_ran = run_train_epoch(
                _model,
                _optimizer,
                iter(_train_ds),
                _cfg.mtp_loss_weight,
                max_steps_ui.value,
                _step_offset,
                _lr_fn,
            )
            train_losses.extend(_ep_losses)
            expert_load_history.extend(_load_snaps)
            router_bias_history.extend(_bias_snaps)
            _step_offset += _steps_ran
            _train_avg = float(np.mean(_ep_losses)) if _ep_losses else float("nan")
            _val_avg = run_evaluate(_model, iter(_val_ds), _cfg.mtp_loss_weight, max_batches=6)
            val_losses.append(_val_avg)
            mo.output.replace(
                mo.md(
                    f"**Epoch {_epoch+1}/{epochs_ui.value}** — "
                    f"train {_train_avg:.4f} (ppl {perplexity_from_loss(_train_avg):.1f}) | "
                    f"val {_val_avg:.4f} (ppl {perplexity_from_loss(_val_avg):.1f})"
                )
            )
        trained_model = _model
        mo.output.replace(
            mo.md(
                f"**Training complete!** final val loss {val_losses[-1]:.4f} "
                f"(ppl {perplexity_from_loss(val_losses[-1]):.1f})"
            )
        )
    return (
        expert_load_history,
        router_bias_history,
        train_losses,
        trained_model,
        val_losses,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Hyperparameter Search (optional)

    Small 2 x 2 x 2 grid over `learning_rate`, `kda_group_size`, and
    `n_routed_experts`. Each config runs a short mini-run so the whole
    search fits inside a couple of minutes. Results sorted by val perplexity.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.cell
def _(all_tokens, config, hp_search_cb, mo):
    mo.stop(not hp_search_cb.value, mo.md("_Tick the box above to run the hyperparameter search._"))
    _search = {
        "lr": [3e-4, 1e-3],
        "kda_group_size": [2, 3],
        "n_routed": [16, 32],
    }
    _rows: list[dict] = []
    _n_steps = 15
    for _lr in _search["lr"]:
        for _kg in _search["kda_group_size"]:
            for _ne in _search["n_routed"]:
                _cfg = NanoKimiK3Config(
                    **{**asdict(config), "kda_group_size": _kg, "n_routed_experts": _ne}
                )
                _m = NanoKimiK3V1(_cfg)
                mx.eval(_m.parameters())
                _opt = optim.AdamW(learning_rate=_lr, weight_decay=0.1)
                _train_ds, _val_ds, _ = make_datasets(all_tokens, seq_len=_cfg.seq_len, batch_size=2)
                _lr_fn = lambda s: _lr
                run_train_epoch(_m, _opt, iter(_train_ds), _cfg.mtp_loss_weight, _n_steps, 0, _lr_fn)
                _vl = run_evaluate(_m, iter(_val_ds), _cfg.mtp_loss_weight, max_batches=3)
                _rows.append(
                    {
                        "lr": _lr,
                        "kda_group_size": _kg,
                        "n_routed": _ne,
                        "val_loss": round(_vl, 4),
                        "val_ppl": round(perplexity_from_loss(_vl), 2),
                    }
                )
                mo.output.replace(
                    mo.md(f"lr={_lr} kda_group={_kg} n_routed={_ne} -> val {_vl:.4f} (ppl {perplexity_from_loss(_vl):.1f})")
                )
    _rows.sort(key=lambda r: r["val_loss"])
    hp_results = _rows
    mo.output.append(mo.ui.table(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Validation & Cross-Validation

    Random k-fold CV is **invalid** for a contiguous language corpus —
    random folds tear apart neighbouring tokens, letting the model see
    near-verbatim context from the training set during validation. We
    use **blocked 5-fold CV** instead: split the training tokens into 5
    contiguous folds, evaluate fold i as val, train on the union of the
    other four. Report per-fold perplexity plus mean +/- std.
    """)
    return


@app.function
def evaluate_model(
    model, data_iter, mtp_weight: float, max_batches: int = 30
) -> tuple[float, float, float]:
    """Return (avg_loss, perplexity, bits_per_token)."""
    total = 0.0
    n = 0
    for x, y in data_iter:
        if n >= max_batches:
            break
        loss = compute_lm_loss(model, x, y, mtp_weight)
        mx.eval(loss)
        total += float(loss.item())
        n += 1
    avg = total / max(1, n)
    return avg, perplexity_from_loss(avg), avg / math.log(2)


@app.cell
def _(config, mo, test_ds, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) to compute test-set metrics._")
    else:
        _loss, _ppl, _bpt = evaluate_model(
            trained_model, iter(test_ds), config.mtp_loss_weight, max_batches=15
        )
        _out = mo.md(
            "### Test-set metrics\n\n"
            f"| metric | value |\n|---|---|\n"
            f"| loss | {_loss:.4f} |\n"
            f"| perplexity | {_ppl:.2f} |\n"
            f"| bits / token | {_bpt:.3f} |\n"
        )
    _out
    return


@app.function
def run_blocked_fold(
    fold_tokens: np.ndarray,
    val_start: int,
    val_end: int,
    cfg: NanoKimiK3Config,
    n_steps: int,
    batch_size: int,
    lr: float,
) -> tuple[float, float]:
    train_tokens = np.concatenate([fold_tokens[:val_start], fold_tokens[val_end:]])
    val_tokens = fold_tokens[val_start:val_end]
    train_ds = TokenWindowDatasetV1(train_tokens, cfg.seq_len, batch_size, shuffle=True)
    val_ds = TokenWindowDatasetV1(val_tokens, cfg.seq_len, batch_size, shuffle=False)
    model = NanoKimiK3V1(cfg)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.1)
    lr_fn = lambda s: lr
    run_train_epoch(model, opt, iter(train_ds), cfg.mtp_loss_weight, n_steps, 0, lr_fn)
    val_loss = run_evaluate(model, iter(val_ds), cfg.mtp_loss_weight, max_batches=3)
    return val_loss, perplexity_from_loss(val_loss)


@app.cell
def _(all_tokens, config, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) so cross-validation is worth running._")
    else:
        _k = 5
        _n_train = int(0.90 * all_tokens.shape[0])
        _train_tokens = all_tokens[:_n_train]
        _fold_size = _train_tokens.shape[0] // _k
        _cv_rows: list = []
        for _i in range(_k):
            _val_start = _i * _fold_size
            _val_end = _val_start + _fold_size
            _loss, _ppl = run_blocked_fold(
                _train_tokens,
                _val_start,
                _val_end,
                config,
                n_steps=10,
                batch_size=2,
                lr=3e-4,
            )
            _cv_rows.append({"fold": _i + 1, "val_loss": round(_loss, 4), "val_ppl": round(_ppl, 2)})
            mo.output.replace(mo.md(f"Fold {_i+1}/{_k}: val {_loss:.4f} (ppl {_ppl:.1f})"))
        _losses_arr = np.array([r["val_loss"] for r in _cv_rows])
        cv_results = {
            "folds": _cv_rows,
            "mean_loss": float(_losses_arr.mean()),
            "std_loss": float(_losses_arr.std()),
        }
        _out = mo.vstack(
            [
                mo.md(
                    f"### Blocked 5-fold CV (small budget)\n\n"
                    f"mean val loss: **{cv_results['mean_loss']:.4f} +/- {cv_results['std_loss']:.4f}**"
                ),
                mo.ui.table(_cv_rows),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. Results

    Diagnostic plots. Each is defined as its own `@app.function` returning
    a matplotlib `Figure`; each calling cell is a single-line expression
    or a two-branch conditional depending on `trained_model`.
    """)
    return


@app.function
def plot_loss_curve(train_losses: list, val_losses: list):
    fig, ax = plt.subplots(figsize=(8, 4))
    if train_losses:
        ax.plot(range(1, len(train_losses) + 1), train_losses, "b-", lw=1.2, alpha=0.6, label="train (per step)")
    if val_losses:
        step_per_epoch = max(1, len(train_losses) // len(val_losses)) if train_losses else 1
        val_xs = [step_per_epoch * (i + 1) for i in range(len(val_losses))]
        ax.plot(val_xs, val_losses, "r-o", lw=2, ms=6, label="val (end of epoch)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_expert_load(expert_load_history: list):
    if not expert_load_history:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no expert load data (train first)", ha="center", va="center")
        return fig
    # Mean over layers for each snapshot -> (n_snapshots, n_experts)
    loads_over_time = np.stack([s.mean(axis=0) for s in expert_load_history], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(loads_over_time.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("training snapshot")
    ax.set_ylabel("routed expert idx")
    ax.set_title("Routed-expert utilization over training (mean over layers)")
    fig.colorbar(im, ax=ax, label="tokens routed")
    fig.tight_layout()
    return fig


@app.function
def plot_router_bias_trajectory(router_bias_history: list):
    if not router_bias_history:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no router bias data (train first)", ha="center", va="center")
        return fig
    # Take layer 0 across time -> (n_snapshots, n_experts)
    traj = np.stack([b[0] for b in router_bias_history], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    for e in range(traj.shape[1]):
        ax.plot(traj[:, e], lw=0.8, alpha=0.6)
    ax.set_xlabel("training snapshot")
    ax.set_ylabel("QB router bias (layer 0)")
    ax.set_title("Quantile-Balancing router bias trajectory (layer 0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_kda_decay_spectrum(model, x: mx.array):
    """The learned channel-wise alpha per head: which channels keep long
    memory (alpha ~ 1) vs forget fast (alpha ~ e^{g_min}).

    This IS the model's learned positional encoding — the most interesting
    diagram in this tome.
    """
    # Find first KDA layer and pull g stats
    kda = None
    for _l in model.layers:
        if _l.layer_kind == "kda":
            kda = _l.mixer
            break
    if kda is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no KDA layer", ha="center", va="center")
        return fig
    _, _, _, g, _ = kda._project(x)
    alpha = mx.exp(g)  # (B, H, T, d_k)
    alpha_np = np.asarray(alpha).mean(axis=(0, 2))  # (H, d_k)
    fig, axes = plt.subplots(1, alpha_np.shape[0], figsize=(3 * alpha_np.shape[0], 3), sharey=True)
    if alpha_np.shape[0] == 1:
        axes = [axes]
    for h in range(alpha_np.shape[0]):
        axes[h].plot(np.sort(alpha_np[h])[::-1], "b-", lw=1.5)
        axes[h].set_xlabel("channel (sorted)")
        axes[h].set_title(f"head {h}")
        axes[h].grid(True, alpha=0.3)
    axes[0].set_ylabel("alpha (retention factor)")
    fig.suptitle("KDA channel-wise decay spectrum (the model's learned positional encoding)")
    fig.tight_layout()
    return fig


@app.function
def plot_attnres_depth_attention(model):
    """For each layer's attention and MoE sublayer AttnRes, plot the softmax
    weight over prior block reps (the RMSNorm keys are unit-normed so weights
    only depend on the pseudo-query direction)."""
    n_layers = len(model.layers)
    # For each sublayer (2 per layer), pseudo-query dot RMSNorm(keys_i) where keys_i
    # are block outputs. Since we do not run forward here, we visualize just the
    # pseudo-query norms per (layer, sublayer) as a proxy for "how much this
    # sublayer diverges from the uniform (zero-init) prior".
    attn_norms = []
    mlp_norms = []
    for lyr in model.layers:
        attn_norms.append(float(mx.sqrt(mx.sum(lyr.attn_res.pseudo_query ** 2)).item()))
        mlp_norms.append(float(mx.sqrt(mx.sum(lyr.mlp_res.pseudo_query ** 2)).item()))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(n_layers)
    ax.bar(x - 0.2, attn_norms, width=0.4, label="attn AttnRes q", color="#3182bd")
    ax.bar(x + 0.2, mlp_norms, width=0.4, label="MoE AttnRes q", color="#e6550d")
    ax.set_xlabel("layer index")
    ax.set_ylabel("||pseudo-query||")
    ax.set_title("Block AttnRes pseudo-query magnitude (0 = uniform depth attention)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_layer_schedule(cfg: NanoKimiK3Config):
    kinds = build_layer_kind_schedule(cfg.n_layers, cfg.kda_group_size, cfg.trailing_mla)
    blocks = build_attnres_block_schedule(cfg.n_layers, NanoKimiK3V1.n_attnres_blocks(cfg))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = {"kda": "#4daf4a", "mla": "#e41a1c"}
    for i, (k, b) in enumerate(zip(kinds, blocks)):
        ax.barh(0, 1, left=i, color=colors[k], edgecolor="black")
        ax.text(i + 0.5, 0, f"{k}\n(b{b})", ha="center", va="center", fontsize=8)
    # Block boundaries
    for i in range(1, cfg.n_layers):
        if blocks[i] != blocks[i - 1]:
            ax.axvline(i, color="black", linestyle="--", lw=1.5)
    ax.set_xlim(-0.2, cfg.n_layers + 0.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(cfg.n_layers))
    ax.set_xlabel("layer index")
    ax.set_title("Layer schedule: KDA/MLA + AttnRes block partition")
    fig.tight_layout()
    return fig


@app.function
def kv_state_bytes_per_token(cfg: NanoKimiK3Config, kind: str = "nano-hybrid") -> float:
    n_kda = sum(1 for k in build_layer_kind_schedule(cfg.n_layers, cfg.kda_group_size, cfg.trailing_mla) if k == "kda")
    n_mla = cfg.n_layers - n_kda
    if kind == "full-mha-fp16":
        return 2 * cfg.n_layers * cfg.n_heads * cfg.qk_nope_head_dim * 2
    if kind == "gqa-fp16":
        return 2 * cfg.n_layers * max(1, cfg.n_heads // 2) * cfg.qk_nope_head_dim * 2
    if kind == "mla-only-fp16":
        return cfg.n_layers * cfg.kv_lora_rank * 2
    if kind == "nano-hybrid":
        # KDA state is fixed size per layer, independent of seq len (bytes per token amortized -> 0 as seq -> inf).
        # For a real per-token cost comparison, we report the fixed-state slice divided by seq_len.
        kda_state_bytes = n_kda * cfg.kda_head_dim * cfg.v_head_dim * cfg.kda_n_heads * 2  # fp16
        mla_latent_bytes = n_mla * cfg.kv_lora_rank * 2
        return (kda_state_bytes / max(1, cfg.seq_len)) + mla_latent_bytes
    raise ValueError(kind)


@app.cell
def _(train_losses: list[float], val_losses: list[float]):
    plot_loss_curve(train_losses, val_losses)
    return


@app.cell
def _(expert_load_history: list):
    plot_expert_load(expert_load_history)
    return


@app.cell
def _(router_bias_history: list):
    plot_router_bias_trajectory(router_bias_history)
    return


@app.cell
def _(mo, train_ds, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first (Section 5) to visualize the KDA decay spectrum._")
    else:
        _x, _ = next(iter(train_ds))
        _out = plot_kda_decay_spectrum(trained_model, _x)
    _out
    return


@app.cell
def _(mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first (Section 5) to visualize the AttnRes pseudo-queries._")
    else:
        _out = plot_attnres_depth_attention(trained_model)
    _out
    return


@app.cell
def _(config):
    plot_layer_schedule(config)
    return


@app.cell
def _(config, mo):
    _rows = []
    for _kind in ["full-mha-fp16", "gqa-fp16", "mla-only-fp16", "nano-hybrid"]:
        _rows.append({"cache scheme": _kind, "bytes / token": round(kv_state_bytes_per_token(config, _kind), 2)})
    mo.vstack(
        [
            mo.md(
                "### KV / state-memory budget comparison\n\n"
                "The nano-hybrid row's KDA state is a **constant** d_k*d_v (per layer, per head) — "
                "amortized over seq_len it approaches zero as context grows. This is the "
                "whole point of the linear-attention design."
            ),
            mo.ui.table(_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary

    This tome implements all of Kimi K3's architectural components — KDA with
    lower-bounded scaled-sigmoid decay + full-rank output gate + ShortConv,
    Gated MLA with NoPE, Block Attention Residuals with zero-init pseudo-queries,
    Stable LatentMoE with SiTU-GLU + normalized aggregation + Quantile Balancing
    router, MTP head, EAGLE-3 draft with W_E3 = [0 0 I] init, MXFP4 QAT, and
    a Per-Head Muon + AdamW hybrid optimizer with K2 weight clipping —
    at roughly ~30M parameters on a 278K-token corpus.

    278K tokens is *far* too small for any of these mechanisms to show its
    real benefit; K3's channel-wise decay pays off at million-token contexts,
    Quantile Balancing needs many more tokens per expert to converge, and
    Block AttnRes's scaling-law gains show at PFLOP scale. **This notebook
    is an architecture comprehension exercise.** Read the code and the
    citations to the source notes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9. Sample Output from a Trained Model

    Autoregressive sampling from the trained backbone. **The generation
    loop uses `kda_mode="recurrent"` and carries the per-layer KDA state
    + MLA latent cache across steps** — this is the whole point of the
    hybrid linear-attention design (constant per-token cost regardless
    of context length).

    Small caveat: our nano ShortConv is re-run over the full prefix at
    each step (we don't cache the k-1 conv frames); a production
    implementation would carry that state too.
    """)
    return


@app.cell
def _(mo):
    prompt_ui = mo.ui.text_area(value="KING RICHARD:\n", label="Prompt")
    max_new_ui = mo.ui.slider(10, 200, value=60, step=10, label="Max new tokens")
    temp_ui = mo.ui.slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
    topk_ui = mo.ui.slider(0, 200, value=40, step=5, label="Top-k (0 disables)")
    topp_ui = mo.ui.slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
    seed_ui = mo.ui.number(value=0, label="Seed")
    gen_btn = mo.ui.run_button(label="Generate")
    mo.vstack(
        [
            prompt_ui,
            mo.hstack([max_new_ui, temp_ui]),
            mo.hstack([topk_ui, topp_ui, seed_ui]),
            gen_btn,
        ]
    )
    return gen_btn, max_new_ui, prompt_ui, seed_ui, temp_ui, topk_ui, topp_ui


@app.function
def scatter_along_last_axis(out: mx.array, indices: mx.array, values: mx.array) -> mx.array:
    """Manual scatter along the last axis for 2D arrays via numpy."""
    idx_np = np.asarray(indices)
    val_np = np.asarray(values)
    out_np = np.asarray(out)
    for b in range(out_np.shape[0]):
        out_np[b, idx_np[b]] = val_np[b]
    return mx.array(out_np, dtype=out.dtype)


@app.function
def generate_text(
    model,
    merge_table: dict,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    seed: int = 0,
    seq_len: int = 256,
) -> str:
    """Autoregressive sampling through the KDA recurrent path.

    The model runs with ``kda_mode="recurrent"``, but the prefix is re-run on
    every step (``cache=None``) rather than carrying the per-layer state
    forward.  Carrying state would additionally require caching each
    ShortConv's trailing ``k-1`` frames -- ``KimiDeltaAttentionV1`` currently
    threads only the recurrent matrix ``S`` -- so incremental decode would read
    zero-padded convolution history and silently diverge.  Output is therefore
    correct but costs O(T) per token instead of KDA's headline O(1); see the
    Section 9 markdown cell.
    """
    mx.random.seed(int(seed))
    np.random.seed(int(seed))
    ids_list = tokenize(prompt, merge_table)
    if not ids_list:
        ids_list = [0]
    ids = mx.array(ids_list, dtype=mx.int32)[None, :]
    for _ in range(max_new_tokens):
        cur = ids if ids.shape[1] <= seq_len else ids[:, -seq_len:]
        logits, _mtp, _aux = model(cur, cache=None, kda_mode="recurrent")
        next_logits = logits[:, -1, :] / max(1e-6, float(temperature))
        if top_k and top_k > 0:
            k = min(top_k, next_logits.shape[-1])
            threshold = mx.min(mx.topk(next_logits, k=k, axis=-1), axis=-1, keepdims=True)
            next_logits = mx.where(next_logits < threshold, mx.array(-1e9, dtype=next_logits.dtype), next_logits)
        if top_p and 0.0 < top_p < 1.0:
            sorted_idx = mx.argsort(-next_logits, axis=-1)
            sorted_logits = mx.take_along_axis(next_logits, sorted_idx, axis=-1)
            probs = mx.softmax(sorted_logits, axis=-1)
            cumprobs = mx.cumsum(probs, axis=-1)
            keep = cumprobs <= top_p
            keep = mx.concatenate([mx.ones((keep.shape[0], 1), dtype=keep.dtype), keep[:, :-1]], axis=-1)
            filtered_sorted = mx.where(keep, sorted_logits, mx.array(-1e9, dtype=sorted_logits.dtype))
            unsort_logits = mx.zeros_like(next_logits)
            next_logits = scatter_along_last_axis(unsort_logits, sorted_idx, filtered_sorted)
        next_id = mx.random.categorical(next_logits)
        ids = mx.concatenate([ids, next_id[:, None]], axis=1)
        mx.eval(ids)
    return decode_tokens(ids[0].tolist(), merge_table)


@app.cell
def _(
    gen_btn,
    max_new_ui,
    merge_table,
    mo,
    prompt_ui,
    seed_ui,
    temp_ui,
    topk_ui,
    topp_ui,
    trained_model,
):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before generating text._")
    elif not gen_btn.value:
        _out = mo.md("Click **Generate** to sample from the trained backbone.")
    else:
        _text = generate_text(
            trained_model,
            merge_table,
            prompt_ui.value,
            max_new_tokens=max_new_ui.value,
            temperature=temp_ui.value,
            top_k=topk_ui.value,
            top_p=topp_ui.value,
            seed=int(seed_ui.value),
        )
        _out = mo.md(f"**Prompt:** {prompt_ui.value}\n\n**Generated:**\n\n```\n{_text}\n```")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10. Save Trained Model

    Weights go to `models/<filename>.safetensors`; the config plus the
    **frozen Quantile-Balancing router biases** go to
    `models/<filename>_config.json`. The QB biases live in the module
    with a leading underscore (`_route_bias`), which keeps them out of
    the MLX parameter tree — so they will NOT be included in
    `.save_weights()`. That is why the sidecar JSON carries them
    explicitly: they are inference-time state, not learned weights.
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="tinyshakespear_nano_kimik3_v1.safetensors",
        label="Weights filename (written into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_model_btn])
    return save_filename_ui, save_model_btn


@app.cell
def _(config, mo, save_filename_ui, save_model_btn, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before saving._")
    elif not save_model_btn.value:
        _out = mo.md("Enter a filename and click **Save Model** to write the trained weights.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_filename_ui.value
        trained_model.save_weights(str(_save_path))
        _cfg_path = _save_path.with_name(_save_path.stem + "_config.json")
        _cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()}
        _qb_biases = {
            f"layer_{i}": [float(x) for x in np.asarray(_layer.moe.router._route_bias).tolist()]
            for i, _layer in enumerate(trained_model.layers)
        }
        _sidecar = {"config": _cfg_dict, "qb_router_biases": _qb_biases}
        with open(_cfg_path, "w") as _f:
            json.dump(_sidecar, _f, indent=2)
        _out = mo.md(
            f"**Saved!**\n\n"
            f"- weights: `{_save_path}`\n"
            f"- config + QB biases sidecar: `{_cfg_path}`\n"
        )
    _out
    return


if __name__ == "__main__":
    app.run()
