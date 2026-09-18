import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import math
    from dataclasses import dataclass, field, asdict
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
    # Nano-DeepSeek V4.1 on TinyShakespeare

    **Research goal** — train a *miniaturized* version of DeepSeek-V4.1-Flash
    (Sept 2026) on the TinyShakespeare corpus with the MLX framework, keeping
    **every** architectural component of the full-scale model in place and only
    reducing the *sizes*. The notebook doubles as an annotated reader for
    DeepSeek-V4.1-Flash: each module cell is preceded by a markdown cell that
    cites the paper and lays out the math.

    This notebook is an *architecture comprehension* exercise, not a scaling
    result. 278K training tokens is far too small a corpus for CSA2's
    KV-compression story or the aux-loss-free MoE balancing to show their
    real advantage. Read the *code* and the *notes* — the loss curve is a
    sanity check, not a benchmark.

    ## Component-by-component mapping to DeepSeek-V4.1-Flash

    | Nano component (this notebook) | V4.1-Flash counterpart | Reduction factor |
    |---|---|---|
    | vocab_size = 10K | 128K | 12.8x |
    | d_model = 256 | 5,120 | 20x |
    | n_layers = 8 (4+4) | 40 (20+20) | 5x |
    | n_heads = 4, head_dim = 64 | 64, head_dim = 512 | 16x heads, 8x dim |
    | kv_latent_dim = 64 | 512 | 8x |
    | seq_len = 256 | 1,048,576 | 4096x |
    | CSA2 compression m = 2 / 1 | same ratios | 1x |
    | Indexer top-k = 16 | 512 | 32x |
    | Sliding window = 64 | 128 | 2x |
    | Routed experts = 8, active = 2, shared = 1 | 384 routed, 6 active, 1 shared | 48x experts |
    | Expert hidden = 128 | 2,304 | 18x |
    | mHC streams = 4 | 4 | 1x |
    | Engram orders = {2, 3} | {2, 3, 4} | 1 order |
    | Engram buckets = 8,192 x 4 heads | ~16M x 8 heads | ~4000x |
    | MTP / DSpark depth = 1 | 1 (DSpark) | 1x |
    | FP4 QAT toggle | always on for KV / routed experts | -- |

    ## Section outline

    1. **Title & Research Goal** — this cell
    2. **Data Exploration** — Zipf, token-id histogram, merge lengths, corpus stats
    3. **Dataset Creation** — contiguous 90/5/5 split, window dataset, one batch
    4. **Model Definition** — every V4.1 component, with math, in dependency order
    5. **Training** — cosine LR, Hybrid Muon/AdamW vs plain AdamW, live progress
    6. **Hyperparameter Search** — small grid (opt-in via checkbox)
    7. **Validation & Cross-Validation** — blocked 5-fold CV (not random k-fold)
    8. **Results** — loss, expert load, indexer selection, mHC Birkhoff proof, KV budget
    9. **Text Generation** — interactive sampling from the trained backbone
    10. **Save Trained Model** — safetensors weights + JSON sidecar for reconstruction
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

    Returns pair->id, pair->rank, id->pair dicts (verbatim from the sibling
    nano_language_model.py notebook so both use identical decoding).
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
    # Derive vocab size instead of hardcoding
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
    lengths = []
    ranks = []
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

    for rank, (_l, _r, merged_id) in enumerate(
        sorted(
            (
                (l_r_id[0], l_r_id[1], merged_id)
                for (l_r_id, merged_id) in [((p[0], p[1]), i) for p, i in merge_table["pair_to_id"].items()]
            ),
            key=lambda tup: merge_table["pair_to_rank"][(tup[0], tup[1])],
        )
    ):
        lengths.append(token_len(merged_id))
        ranks.append(rank)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ranks, lengths, s=3, alpha=0.3, color="darkorange")
    ax.set_xlabel("merge rank")
    ax.set_ylabel("resulting token length (bytes)")
    ax.set_title("Merge rank vs. resulting token length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
    _byte_coverage = 100 * np.mean(all_tokens < 256)
    _bytes_per_tok = len(raw_text) / all_tokens.shape[0]
    _n_train = int(0.90 * all_tokens.shape[0])
    _n_val = int(0.05 * all_tokens.shape[0])
    _n_test = all_tokens.shape[0] - _n_train - _n_val
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
        batch_size: int = 32,
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
    batch_size: int = 32,
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
    train_ds, val_ds, test_ds = make_datasets(all_tokens, seq_len=256, batch_size=8)
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

    Every module below is a scaled-down implementation of a real
    DeepSeek-V4.1-Flash component. The build order follows *dependency
    order* — helpers first, leaves next, then composites, then the
    top-level model. Each class is preceded by a markdown cell that
    cites the source note and derives the math it implements.
    """)
    return


@app.class_definition
@dataclass
class NanoDeepSeekConfig:
    """All knobs of the nano-DeepSeek model in one place.

    Passed by value into every module `__init__` so nothing captures globals.
    Serialized to JSON alongside the model weights in Section 10 so the
    saved backbone can be reconstructed exactly.
    """

    vocab_size: int = 10000
    d_model: int = 256
    n_layers: int = 8
    n_encoder_layers: int = 4
    n_heads: int = 4
    head_dim: int = 64
    d_nope: int = 48
    d_rope: int = 16
    kv_latent_dim: int = 64
    q_latent_dim: int = 96
    seq_len: int = 256
    compress_m_encoder: int = 2
    compress_m_decoder: int = 1
    indexer_top_k_blocks: int = 16
    indexer_n_heads: int = 2
    indexer_dim: int = 32
    sliding_window: int = 64
    n_routed_experts: int = 8
    n_active_experts: int = 2
    n_shared_experts: int = 1
    expert_hidden: int = 128
    n_hc_streams: int = 4
    sinkhorn_iters: int = 5
    engram_orders: tuple = (2, 3)
    engram_n_buckets: int = 8192
    engram_n_heads: int = 4
    engram_dim: int = 32
    mtp_depth: int = 1
    fp4_qat: bool = False
    mtp_loss_weight: float = 0.1
    moe_aux_weight: float = 0.001
    moe_bias_lr: float = 1e-3
    encoder_modes: tuple = ("full", "reindex", "reuse", "reuse")
    decoder_modes: tuple = ("full", "reuse", "reindex", "reuse")


@app.cell
def _():
    config = NanoDeepSeekConfig()
    print(config)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1 FP4 Quantization-Aware Training

    DeepSeek-V4.1 stores the main KV cache and the indexer QK path in
    **E2M1 (FP4)** with a 1-byte E4M3 scale per 16 channels (MXFP4-style
    layout, NVFP4 minus the global scale). See
    `deepseek-v4-1-flash-kv-cache-compression.md`, section *"FP4 Main KV Cache"*.

    During training we do **fake quantization with a straight-through
    estimator** — round to the FP4 grid in the forward pass, pass the
    gradient through unchanged in the backward pass:

    $$
    \tilde x \;=\; x \;+\; \text{stop\_grad}\bigl(Q_{\text{FP4}}(x) - x\bigr)
    $$

    The scale per group of 16 channels is $\text{scale} = \text{absmax}(x_g) / 6$
    since the largest representable FP4 magnitude is 6. Our fake
    implementation uses uniform 4-bit quantization (16 levels) as a
    simple stand-in for the real E2M1 grid — the STE behaviour is
    identical and what matters here is the training dynamics.
    """)
    return


@app.function
def fake_quantize_fp4(x: mx.array, group_size: int = 16) -> mx.array:
    """Fake E2M1 quantize-dequantize with per-group absmax scaling and STE.

    Uniform 4-bit round-to-nearest is used as a simple, differentiable
    stand-in for the exact E2M1 grid. FP4 max magnitude is 6.
    """
    orig_shape = x.shape
    if x.shape[-1] % group_size != 0:
        # fall back to per-tensor scaling when last dim not divisible
        absmax = mx.max(mx.abs(x)) + 1e-6
        scale = absmax / 6.0
        q_int = mx.clip(mx.round(x / scale * (7.0 / 6.0)), -7, 7)
        x_q = q_int * (6.0 / 7.0) * scale
        return x + mx.stop_gradient(x_q - x)
    x_flat = x.reshape(-1, group_size)
    absmax = mx.max(mx.abs(x_flat), axis=-1, keepdims=True) + 1e-6
    scale = absmax / 6.0
    q_int = mx.clip(mx.round(x_flat / scale * (7.0 / 6.0)), -7, 7)
    x_q = q_int * (6.0 / 7.0) * scale
    x_q = x_q.reshape(orig_shape)
    return x + mx.stop_gradient(x_q - x)


@app.class_definition
class FP4QuantizerV1(nn.Module):
    """Toggleable fake-FP4 quantizer used on routed-expert weights and the
    indexer QK path (following V4.1's mixed-precision recipe).

    When `enabled = False`, the module is a no-op — useful as an ablation
    control in Section 6.
    """

    def __init__(self, enabled: bool = False, group_size: int = 16):
        super().__init__()
        self.enabled = enabled
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        if not self.enabled:
            return x
        return fake_quantize_fp4(x, self.group_size)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 Clamped SwiGLU FFN

    The "**SwiGLU Clamping**" stability mechanism from DeepSeek-V4 (retained
    in V4.1) prevents rare activation blow-ups during long training runs:

    $$
    \text{SwiGLU}(x) = W_o\bigl(\text{clip}(W_g x, -10, 10) \cdot \sigma_{\text{SiLU}}(W_g x) \cdot \min(W_u x, 10)\bigr)
    $$

    See `deepseek-v4-flash.md` — the gate branch is clipped to $[-10, 10]$
    and the up branch is capped from above at $10$. The clamps are gentle
    enough not to hurt training but firm enough to catch spikes.
    """)
    return


@app.class_definition
class ClampedSwiGLUFeedForwardV1(nn.Module):
    """SwiGLU with V4 stability clamps on the gate and up branches."""

    def __init__(
        self,
        d_model: int = 256,
        hidden: int = 512,
        clamp: float = 10.0,
    ):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.clamp = clamp

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.clip(self.gate(x), -self.clamp, self.clamp)
        u = mx.minimum(self.up(x), self.clamp)
        return self.down(nn.silu(g) * u)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 Sequence Compressor (CSA2 sequence axis)

    CSA2 pools every $m$ consecutive KV entries into one entry (spec:
    *"non-overlapping groups of $m$, drops the absolute positional
    embedding"* — see the CSA2 section of `deepseek-v4-1-flash-...md`).
    We use a **softmax-gated pool** with a *learned positional bias*
    across the $m$ intra-group positions:

    $$
    C_j = \sum_{p=0}^{m-1} \text{softmax}_p\bigl(W_g x_{jm+p} + b_p\bigr) \cdot x_{jm+p}
    $$

    For $m = 1$ (decoder), this is the identity (no compression).
    """)
    return


@app.class_definition
class SequenceCompressorV1(nn.Module):
    """Softmax-gated pooling of every m consecutive latent-KV entries."""

    def __init__(self, d_latent: int = 64, m: int = 2):
        super().__init__()
        self.d_latent = d_latent
        self.m = m
        if m > 1:
            self.gate = nn.Linear(d_latent, 1, bias=False)
            # learned positional bias over the m intra-group slots
            self.pos_bias = mx.zeros((m,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.m == 1:
            return x
        B, T, D = x.shape
        pad = (self.m - (T % self.m)) % self.m
        if pad > 0:
            x = mx.concatenate([x, mx.zeros((B, pad, D), dtype=x.dtype)], axis=1)
            T = T + pad
        n_groups = T // self.m
        x_grp = x.reshape(B, n_groups, self.m, D)
        logits = self.gate(x_grp).squeeze(-1) + self.pos_bias  # (B, n_groups, m)
        weights = mx.softmax(logits, axis=-1)[..., None]  # (B, n_groups, m, 1)
        return (weights * x_grp).sum(axis=-2)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.4 Latent KV Projection (MLA entry-size axis)

    Multi-head latent attention (MLA), from DeepSeek-V2/V3, attacks the
    **entry-size** axis of KV storage by keeping only a low-rank shared
    latent per position and up-projecting per head at attention time:

    $$
    c = \text{RMSNorm}(W_{DKV}\, h) \in \mathbb{R}^{d_{\text{latent}}}
    \qquad
    K_{\text{nope}}^{(h)} = c\, W_{UK}^{(h)}
    \qquad
    V^{(h)} = c\, W_{UV}^{(h)}
    $$

    RoPE dims are **decoupled**: the rotary portion is projected
    directly from $h$ (not from $c$) so that queries and keys can share
    the RoPE trigonometry independent of the compression. In V4.1 the
    main KV latent is 512-ch; we use 64-ch. See the "Three Multiplicative
    Axes" table and the MLA sub-section.
    """)
    return


@app.class_definition
class LatentKVProjectionV1(nn.Module):
    """MLA-style low-rank shared latent KV with decoupled RoPE dims.

    Splits each head into `d_nope` non-rotary channels (from the shared
    latent) and `d_rope` rotary channels (from the raw hidden state).
    Returns per-head K (concat of nope + rope) and V.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_nope: int = 48,
        d_rope: int = 16,
        kv_latent_dim: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_nope = d_nope
        self.d_rope = d_rope
        self.kv_latent_dim = kv_latent_dim
        self.head_dim = d_nope + d_rope
        self.down = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.latent_norm = nn.RMSNorm(kv_latent_dim)
        self.up_k = nn.Linear(kv_latent_dim, n_heads * d_nope, bias=False)
        self.up_v = nn.Linear(kv_latent_dim, n_heads * self.head_dim, bias=False)
        self.k_rope_proj = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.rope = nn.RoPE(d_rope)

    def compute_latent(self, source: mx.array) -> mx.array:
        return self.latent_norm(self.down(source))

    def project_from_latent(
        self, latent_compressed: mx.array, source_compressed: mx.array
    ) -> tuple[mx.array, mx.array]:
        """latent_compressed: (B, T_kv, kv_latent_dim); source_compressed same T_kv.

        Returns K, V with shape (B, n_heads, T_kv, head_dim).
        """
        B, T_kv, _ = latent_compressed.shape
        k_nope = self.up_k(latent_compressed).reshape(B, T_kv, self.n_heads, self.d_nope)
        v = self.up_v(latent_compressed).reshape(B, T_kv, self.n_heads, self.head_dim)
        k_rope = self.k_rope_proj(source_compressed).reshape(B, T_kv, self.n_heads, self.d_rope)
        # move heads to axis=1 for RoPE
        k_nope = k_nope.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        k_rope = k_rope.transpose(0, 2, 1, 3)
        k_rope = self.rope(k_rope)
        k = mx.concatenate([k_nope, k_rope], axis=-1)
        return k, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.5 Sinkhorn-Knopp projection

    mHC's residual mixing matrix must live on the **Birkhoff polytope**
    (doubly-stochastic square matrices). Given a matrix of log-alphas
    we iterate Sinkhorn-Knopp normalization to convergence:

    $$
    A^{(2k+1)}_{ij} = \frac{A^{(2k)}_{ij}}{\sum_{j'} A^{(2k)}_{ij'}}, \quad
    A^{(2k+2)}_{ij} = \frac{A^{(2k+1)}_{ij}}{\sum_{i'} A^{(2k+1)}_{i'j}}
    $$

    As $k \to \infty$ the result is doubly-stochastic (row and column
    sums all equal 1), which bounds its spectral norm at $1$ — critical
    for keeping the residual stream stable across depth. See
    `manifold-constrained-hyper-connections-mhc.md`.
    """)
    return


@app.function
def sinkhorn_knopp(log_alpha: mx.array, n_iters: int = 5) -> mx.array:
    """Sinkhorn-Knopp normalization producing a (near-)doubly-stochastic matrix.

    `log_alpha` is (..., n, n). We exponentiate then alternate row/column
    normalization for `n_iters` half-iterations (each is one row-then-column
    pair when done in log-space).
    """
    x = log_alpha
    for _ in range(n_iters):
        x = x - mx.logsumexp(x, axis=-1, keepdims=True)
        x = x - mx.logsumexp(x, axis=-2, keepdims=True)
    return mx.exp(x)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.6 Hierarchical Sparse Indexer

    The **Lightning / hierarchical indexer** scores compressed KV entries
    with a multi-head ReLU dot product, quantizes the QK path to FP4,
    and keeps the top-$k$ per query:

    $$
    s_{q,j} = \sum_{h=1}^{H_I} \text{ReLU}(Q^{(h)}_q \cdot K^{(h)}_j)
    $$

    The **shared candidate pool** trick makes deeper indexers O(1) in
    context length: the first Full layer emits a pool of ~16K candidates
    (top blocks by max index score) and every subsequent Reindex layer
    restricts its scoring to that pool. We use a nano version — top
    `indexer_top_k_blocks` positions per query at seq_len 256, and the
    pool is simply the union of all Full-layer selections. Real V4.1
    uses 2,048 blocks × 8 positions = 16,384 candidates.
    """)
    return


@app.class_definition
class HierarchicalSparseIndexerV1(nn.Module):
    """FP4-quantized multi-head ReLU indexer that returns a top-k boolean mask."""

    def __init__(
        self,
        d_model: int = 256,
        kv_latent_dim: int = 64,
        n_heads: int = 2,
        head_dim: int = 32,
        top_k: int = 16,
        fp4_qat: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_latent_dim, n_heads * head_dim, bias=False)
        self.qk_quant = FP4QuantizerV1(enabled=fp4_qat)

    def score(self, hidden: mx.array, latent_compressed: mx.array) -> mx.array:
        """Returns (B, T_q, T_kv) summed ReLU dot-product score."""
        B, T_q, _ = hidden.shape
        _, T_kv, _ = latent_compressed.shape
        q = self.q_proj(hidden).reshape(B, T_q, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(latent_compressed).reshape(B, T_kv, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.qk_quant(q)
        k = self.qk_quant(k)
        scores_per_head = nn.relu(mx.matmul(q, k.transpose(0, 1, 3, 2)))  # (B, H, T_q, T_kv)
        return scores_per_head.sum(axis=1)  # (B, T_q, T_kv)

    def select(
        self,
        scores: mx.array,
        candidate_pool: mx.array | None = None,
    ) -> mx.array:
        """Return a (B, T_q, T_kv) boolean mask of the top_k entries per query.

        If `candidate_pool` is a (T_kv,) boolean mask, scoring outside the
        pool is set to -inf so those positions can never be selected — this
        is the O(1)-in-context-length trick.
        """
        s = scores
        if candidate_pool is not None:
            s = mx.where(candidate_pool[None, None, :], s, mx.array(-1e9, dtype=s.dtype))
        k = min(self.top_k, s.shape[-1])
        threshold = mx.min(mx.topk(s, k=k, axis=-1), axis=-1, keepdims=True)
        return s >= threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.7 Sliding-Window Attention

    Every CSA2 layer keeps a layer-local sliding-window branch of width
    $w$ that stays layer-local even in decoder layers (V4.1 does not
    share SWA KV across layers — see the *CED* section note *"Sliding
    Window Attention Stays Layer-Local"*). We produce K, V from the
    layer's own hidden state and rely on the CSA2 mask to constrain
    attention to positions $[i - w + 1, i]$.
    """)
    return


@app.class_definition
class SlidingWindowAttentionV1(nn.Module):
    """Layer-local causal sliding-window KV projection (K/V only — attention
    is fused with the sparse global branch inside CSA2)."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        head_dim: int = 64,
        window: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window = window
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.rope = nn.RoPE(head_dim)

    def __call__(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        B, T, _ = hidden.shape
        k = self.k_proj(hidden).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.rope(k)
        return k, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.8 CSA2 mode schedule

    V4.1 statically labels each attention layer as **Full** / **Reindex**
    / **Reuse**. The paper's example (encoder $m=2$, decoder $m=1$):

    - Encoder groups of 6: `[Full, Reuse x5]` — main KV materialised
      only once per group.
    - Decoder groups of 4: `[Reindex, Reuse x3]` — sparse selection
      refreshed but KV is inherited from the encoder's last Full layer.

    In our nano notebook we default to `[Full, Reindex, Reuse, Reuse]`
    in the encoder and `[Full, Reuse, Reindex, Reuse]` in the decoder;
    the schedule is a `NanoDeepSeekConfig` field so the reader can
    experiment.
    """)
    return


@app.function
def build_csa2_mode_schedule(
    encoder_modes: tuple, decoder_modes: tuple
) -> list[str]:
    """Concatenate encoder + decoder mode lists into a single per-layer schedule.

    Validates that the first mode of each half is 'full' (otherwise there is
    no donor for the reuse/reindex layers).
    """
    if encoder_modes and encoder_modes[0] != "full":
        raise ValueError("encoder must start with a Full layer")
    if decoder_modes and decoder_modes[0] != "full":
        raise ValueError("decoder must start with a Full layer")
    for m in list(encoder_modes) + list(decoder_modes):
        if m not in ("full", "reindex", "reuse"):
            raise ValueError(f"invalid CSA2 mode: {m}")
    return list(encoder_modes) + list(decoder_modes)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.9 Compressed Sparse Attention 2 (CSA2)

    The core attention layer. Three static modes exchange information
    via a **shared state dict** that is carried through the encoder or
    decoder stack:

    | Mode | Main KV | Indexer K | Top-k indices |
    |---|---|---|---|
    | Full | computed here | projected from own main KV | computed here |
    | Reindex | reused from last Full | reused | recomputed with own idx-Q |
    | Reuse | reused from last Full | reused | reused from last idx-producing layer |

    The forward pass combines a **sparse global branch** (over the top-k
    of the compressed KV) with a **layer-local sliding-window branch**.
    We add a **learnable attention-sink logit** per head to the softmax
    denominator so a query may attend to "nothing" — this stabilises the
    softmax when a whole row is masked out and doubles as a numerical
    guard for otherwise fully-masked rows. See the CSA2 algorithm box.
    """)
    return


@app.class_definition
class CompressedSparseAttention2V1(nn.Module):
    """One CSA2 attention layer (Full / Reindex / Reuse).

    Reads and writes into `state` (a plain dict) so consecutive layers can
    share compressed KV and top-k selections. `encoder_hidden` is only used
    by decoder Full layers (CED: decoder KV projected from encoder output).
    """

    def __init__(
        self,
        cfg: NanoDeepSeekConfig,
        mode: str = "full",
        compress_m: int = 2,
    ):
        super().__init__()
        if mode not in ("full", "reindex", "reuse"):
            raise ValueError(f"invalid mode: {mode}")
        self.mode = mode
        self.compress_m = compress_m
        self.n_heads = cfg.n_heads
        self.d_nope = cfg.d_nope
        self.d_rope = cfg.d_rope
        self.head_dim = cfg.d_nope + cfg.d_rope
        self.window = cfg.sliding_window
        # Q path: MLA-style down/up with decoupled RoPE
        self.q_down = nn.Linear(cfg.d_model, cfg.q_latent_dim, bias=False)
        self.q_norm = nn.RMSNorm(cfg.q_latent_dim)
        self.q_up_nope = nn.Linear(cfg.q_latent_dim, cfg.n_heads * cfg.d_nope, bias=False)
        self.q_rope_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_rope, bias=False)
        self.q_rope = nn.RoPE(cfg.d_rope)
        # KV latent (only used when mode == "full")
        self.latent_kv = LatentKVProjectionV1(
            cfg.d_model, cfg.n_heads, cfg.d_nope, cfg.d_rope, cfg.kv_latent_dim
        )
        self.compressor = SequenceCompressorV1(cfg.kv_latent_dim, compress_m)
        self.source_compressor = SequenceCompressorV1(cfg.d_model, compress_m)
        # Sliding-window branch
        self.swa = SlidingWindowAttentionV1(cfg.d_model, cfg.n_heads, self.head_dim, cfg.sliding_window)
        # Indexer
        self.indexer = HierarchicalSparseIndexerV1(
            cfg.d_model,
            cfg.kv_latent_dim,
            cfg.indexer_n_heads,
            cfg.indexer_dim,
            cfg.indexer_top_k_blocks,
            cfg.fp4_qat,
        )
        # Attention sink logit (per head) — added to softmax denominator
        self.sink_logit = mx.zeros((cfg.n_heads,))
        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def _compute_q(self, hidden: mx.array) -> mx.array:
        B, T, _ = hidden.shape
        q_lat = self.q_norm(self.q_down(hidden))
        q_nope = self.q_up_nope(q_lat).reshape(B, T, self.n_heads, self.d_nope).transpose(0, 2, 1, 3)
        q_rope = (
            self.q_rope_proj(hidden)
            .reshape(B, T, self.n_heads, self.d_rope)
            .transpose(0, 2, 1, 3)
        )
        q_rope = self.q_rope(q_rope)
        return mx.concatenate([q_nope, q_rope], axis=-1)  # (B, H, T, head_dim)

    def _build_mask(self, T_q: int, T_g: int, T_swa: int) -> mx.array:
        """Base mask (T_q, T_g + T_swa). Uses -1e4 (finite) to avoid NaN when
        combined with the top-k mask."""
        NEG = -1e4
        q_pos = mx.arange(T_q)[:, None]
        g_blocks = mx.arange(T_g)[None, :]
        g_last_pos = g_blocks * self.compress_m + (self.compress_m - 1)
        global_mask = mx.where(g_last_pos <= q_pos, mx.array(0.0), mx.array(NEG))
        swa_pos = mx.arange(T_swa)[None, :]
        swa_causal = swa_pos <= q_pos
        swa_window = swa_pos > q_pos - self.window
        swa_mask = mx.where(swa_causal & swa_window, mx.array(0.0), mx.array(NEG))
        return mx.concatenate([global_mask, swa_mask], axis=-1)  # (T_q, T_g + T_swa)

    def __call__(
        self,
        hidden: mx.array,
        state: dict,
        encoder_hidden: mx.array | None = None,
    ) -> tuple[mx.array, dict]:
        B, T, _ = hidden.shape
        q = self._compute_q(hidden)  # (B, H, T, head_dim)
        k_swa, v_swa = self.swa(hidden)  # (B, H, T, head_dim)

        if self.mode == "full":
            source = encoder_hidden if encoder_hidden is not None else hidden
            latent = self.latent_kv.compute_latent(source)  # (B, T, kv_latent)
            latent_compressed = self.compressor(latent)
            source_compressed = self.source_compressor(source)
            k_global, v_global = self.latent_kv.project_from_latent(
                latent_compressed, source_compressed
            )
            state["latent_compressed"] = latent_compressed
            state["main_k"] = k_global
            state["main_v"] = v_global
            scores = self.indexer.score(hidden, latent_compressed)
            state["indexer_scores"] = scores
            state["topk_mask"] = self.indexer.select(scores, state.get("cand_pool"))
            # if this is the first Full layer in this stack, seed the candidate pool
            if "cand_pool" not in state:
                # union of top-k positions across queries: any position ever selected
                per_pos_selected = mx.any(state["topk_mask"], axis=1)  # (B, T_kv)
                # for the pool we take the union across the batch too — a static pool
                state["cand_pool"] = mx.any(per_pos_selected, axis=0)  # (T_kv,)
        elif self.mode == "reindex":
            scores = self.indexer.score(hidden, state["latent_compressed"])
            state["indexer_scores"] = scores
            state["topk_mask"] = self.indexer.select(scores, state.get("cand_pool"))
        else:  # reuse
            pass

        k_global = state["main_k"]
        v_global = state["main_v"]
        topk_mask = state["topk_mask"]  # (B, T, T_g) bool

        # Combined K/V: concat global (compressed) + swa (per-token)
        k_all = mx.concatenate([k_global, k_swa], axis=-2)  # (B, H, T_g + T, head_dim)
        v_all = mx.concatenate([v_global, v_swa], axis=-2)
        T_g = k_global.shape[-2]

        # Base mask: causal + window
        base_mask = self._build_mask(T, T_g, T).astype(hidden.dtype)  # (T, T_g + T)
        # Add top-k selection mask (global part only; swa is always allowed)
        NEG = mx.array(-1e4, dtype=hidden.dtype)
        topk_add = mx.where(topk_mask, mx.array(0.0, dtype=hidden.dtype), NEG)  # (B, T, T_g)
        swa_add = mx.zeros((B, T, T), dtype=hidden.dtype)
        add_mask = mx.concatenate([topk_add, swa_add], axis=-1)  # (B, T, T_g + T)
        full_mask = base_mask[None, None, :, :] + add_mask[:, None, :, :]  # (B, 1, T, T_g + T)

        # Scaled dot-product with attention-sink logit
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = mx.matmul(q, k_all.transpose(0, 1, 3, 2)) * scale
        logits = logits + full_mask
        sink = mx.broadcast_to(
            self.sink_logit.reshape(1, self.n_heads, 1, 1), (B, self.n_heads, T, 1)
        )
        logits_ext = mx.concatenate([logits, sink], axis=-1)
        attn = mx.softmax(logits_ext, axis=-1)
        attn_kv = attn[..., :-1]
        out = mx.matmul(attn_kv, v_all)  # (B, H, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out), state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.10 Manifold-Constrained Hyper Connections (mHC)

    mHC widens the residual to $n$ parallel streams and mixes them with
    matrices $A$ (input mix), $B$ (output broadcast), $C$ (residual mix):

    $$
    X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_{l-1} X_l)
    $$

    The **Single-Pass** form (V4.1) is the one shown above — $A$ is
    shifted by one block so residual update and coefficient prediction
    fuse. $C$ is projected onto the **Birkhoff polytope** (row- and
    column-stochastic) via Sinkhorn-Knopp so its spectral norm is $\le 1$
    and the residual stream stays stable across depth. See
    `manifold-constrained-hyper-connections-mhc.md`.

    In this nano implementation $A$, $B$, $C$ are learned static
    matrices (not predicted from the hidden state) — enough to
    demonstrate the doubly-stochastic constraint. The shift is realised
    by pre-computing the "next" $A$ inside `forward` and stashing it
    for the following block.
    """)
    return


@app.class_definition
class ManifoldConstrainedHyperConnectionV1(nn.Module):
    """mHC wrapper around a sub-block. Widens residual to n streams, mixes
    with a Sinkhorn-projected doubly-stochastic C matrix."""

    def __init__(
        self,
        d_model: int = 256,
        n_streams: int = 4,
        sinkhorn_iters: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        # Static learned mixing tensors (n_streams,) for A and B; (n_streams, n_streams) for C
        self.mhc_A = mx.ones((n_streams,)) / n_streams  # broadcast weight per stream for input
        self.mhc_B = mx.ones((n_streams,))              # output broadcast per stream
        self.mhc_C_log = mx.zeros((n_streams, n_streams))  # log-alpha of residual mix

    def widen(self, x: mx.array) -> mx.array:
        # Turn (B, T, D) into (B, T, n, D) by broadcasting
        B, T, D = x.shape
        return mx.broadcast_to(x[:, :, None, :], (B, T, self.n_streams, D))

    def contract(self, X: mx.array) -> mx.array:
        # Sum-pool across streams
        return X.sum(axis=-2)

    def get_C(self) -> mx.array:
        return sinkhorn_knopp(self.mhc_C_log, self.sinkhorn_iters)

    def __call__(self, X: mx.array, sub_block) -> mx.array:
        """X: (B, T, n_streams, D). Returns updated (B, T, n_streams, D).

        `sub_block` is a callable that takes a (B, T, D) tensor and returns
        one — the attention or MoE sublayer wrapped by this mHC.
        """
        # Input mixing (single-pass shifted A: static A here, used as-is)
        A = self.mhc_A  # (n,)
        mixed_in = (A.reshape(1, 1, self.n_streams, 1) * X).sum(axis=-2)  # (B, T, D)
        # Sub-block
        F = sub_block(mixed_in)  # (B, T, D)
        # Output broadcast + residual mix
        B_ = self.mhc_B.reshape(1, 1, self.n_streams, 1)
        C = self.get_C()  # (n, n), doubly stochastic
        # X_next[s] = sum_{s'} C[s, s'] * X[s'] + B[s] * F
        # Einsum: (B, T, s', D), (s, s') -> (B, T, s, D)
        residual_mixed = mx.einsum("btsd,ks->btkd", X, C)
        broadcast_F = B_ * F[:, :, None, :]
        return residual_mixed + broadcast_F


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.11 Engram conditional memory (hashed n-gram lookup)

    Engram gives the model **hashed n-gram lookup tables** that behave
    like a giant, cheap, non-parametric context memory. For each
    n-gram order $o \in \{2, 3\}$ and each head $h$ we deterministically
    hash the context tuple $(t_{i-o+1}, \dots, t_i)$ into a bucket and
    retrieve a small embedding.

    The retrieved embeddings are **gated by the current hidden state**
    so context that conflicts with what the transformer is currently
    computing is suppressed. See
    `engram-conditional-memory-scalable-lookup-sparsity.md`. V4.1 uses
    orders $\{2, 3, 4\}$ and ~16M-entry tables per head; we use a
    deterministic multiplicative hash and 8,192 buckets per head — a
    cheap toy version that shows the retrieval + gating dataflow.
    The short causal conv is omitted (V4.1 omits it too).
    """)
    return


@app.function
def hash_ngram_context(
    ids: mx.array, order: int, prime_a: int, prime_b: int, n_buckets: int
) -> mx.array:
    """Deterministic multiplicative hash of length-`order` n-gram suffixes.

    `ids` is (B, T) int32. For each position i we hash the tuple
    (ids[i-order+1], ..., ids[i]); positions with insufficient history use
    zero-padded tokens (still deterministic).
    """
    B, T = ids.shape
    h = mx.zeros((B, T), dtype=mx.int32)
    for k in range(order):
        if k == 0:
            shifted = ids
        else:
            pad = mx.zeros((B, k), dtype=ids.dtype)
            shifted = mx.concatenate([pad, ids[:, : T - k]], axis=1)
        h = (h * prime_a + shifted * prime_b) % n_buckets
    return h


@app.class_definition
class EngramMemoryV1(nn.Module):
    """Multi-order, multi-head hashed n-gram memory with context gating."""

    def __init__(
        self,
        d_model: int = 256,
        orders: tuple = (2, 3),
        n_buckets: int = 8192,
        n_heads: int = 4,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.orders = tuple(orders)
        self.n_buckets = n_buckets
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        # one table per (order, head)
        self.tables = [
            [nn.Embedding(n_buckets, embed_dim) for _ in range(n_heads)]
            for _ in orders
        ]
        # per-order per-head projection to d_model
        self.projs = [nn.Linear(embed_dim, d_model, bias=False) for _ in range(len(orders) * n_heads)]
        # context gate: hidden -> scalar per position
        self.gate = nn.Linear(d_model, 1)
        # per-head primes (fixed, chosen to comfortably fit in int32)
        self._prime_a = [1000003, 500009, 200003, 100019]
        self._prime_b = [999983, 611953, 300007, 150001]

    def __call__(self, ids: mx.array, hidden: mx.array) -> mx.array:
        acc = mx.zeros_like(hidden)
        idx = 0
        for o_i, order in enumerate(self.orders):
            for h in range(self.n_heads):
                pa = self._prime_a[h % len(self._prime_a)] + o_i
                pb = self._prime_b[h % len(self._prime_b)] + o_i
                buckets = hash_ngram_context(ids, order, pa, pb, self.n_buckets)
                emb = self.tables[o_i][h](buckets)  # (B, T, embed_dim)
                acc = acc + self.projs[idx](emb)
                idx += 1
        gate = mx.sigmoid(self.gate(hidden))  # (B, T, 1)
        return acc * gate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.12 DeepSeekMoE — shared + fine-grained routed experts

    DeepSeekMoE keeps a **shared expert** (always applied to every
    token) plus a large pool of **fine-grained routed experts** with
    top-K selection. V4.1 replaces the V3 sigmoid affinity with:

    $$
    \text{score}(x, e) = \sqrt{\text{softplus}(x \cdot W_g^{(e)})}
    $$

    and adds an **auxiliary-loss-free load-balancing bias**: a
    per-expert bias $b_e$ that is added to the affinity *only for the
    top-K selection* (not for the gate value), updated by
    $b_e \leftarrow b_e + \eta \cdot \text{sign}(\text{target\_load} - \text{actual\_load})$.
    We still keep a **mild sequence-wise balance loss** as a regulariser.
    See `deepseekmoe-ultimate-expert-specialization.md`.
    """)
    return


@app.class_definition
class DeepSeekExpertV1(nn.Module):
    """One fine-grained routed expert: a narrow ClampedSwiGLU FFN, with
    optional FP4 QAT on its input and weights."""

    def __init__(
        self,
        d_model: int = 256,
        hidden: int = 128,
        fp4_qat: bool = False,
    ):
        super().__init__()
        self.ffn = ClampedSwiGLUFeedForwardV1(d_model, hidden)
        self.quant = FP4QuantizerV1(enabled=fp4_qat)

    def __call__(self, x: mx.array) -> mx.array:
        return self.ffn(self.quant(x))


@app.class_definition
class DeepSeekMoEV1(nn.Module):
    """Shared-expert + fine-grained routed MoE with aux-loss-free bias
    balancing and a weak sequence-wise balance regulariser.

    Forward returns `(output, aux_loss, expert_load)`. `expert_load` is a
    (n_routed,) int array of routing counts, used both by the aux-loss-free
    balancing update (mutating `_route_bias` in-place) and by the training
    loop for the load-history plot.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_routed: int = 8,
        n_active: int = 2,
        n_shared: int = 1,
        expert_hidden: int = 128,
        fp4_qat: bool = False,
        bias_lr: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_routed = n_routed
        self.n_active = n_active
        self.n_shared = n_shared
        self.expert_hidden = expert_hidden
        self.bias_lr = bias_lr
        self.shared_experts = [
            ClampedSwiGLUFeedForwardV1(d_model, expert_hidden) for _ in range(n_shared)
        ]
        self.routed_experts = [
            DeepSeekExpertV1(d_model, expert_hidden, fp4_qat) for _ in range(n_routed)
        ]
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        # non-trainable routing bias (leading underscore keeps it out of the module's
        # parameter tree; we mutate it manually in the training loop)
        self._route_bias = mx.zeros((n_routed,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, T, D = x.shape
        # Shared experts (always on)
        shared_out = mx.zeros_like(x)
        for se in self.shared_experts:
            shared_out = shared_out + se(x)

        # Routed experts
        scores_raw = mx.sqrt(nn.softplus(self.gate(x)) + 1e-8)  # (B, T, E)
        scores_biased = scores_raw + mx.stop_gradient(self._route_bias)
        # top-k selection
        k = self.n_active
        threshold = mx.min(mx.topk(scores_biased, k=k, axis=-1), axis=-1, keepdims=True)
        selected = scores_biased >= threshold  # (B, T, E) bool
        # gate values from raw scores, normalized over selected
        gate_vals = mx.where(selected, scores_raw, mx.array(0.0, dtype=scores_raw.dtype))
        gate_norm = gate_vals / (gate_vals.sum(axis=-1, keepdims=True) + 1e-6)

        routed_out = mx.zeros_like(x)
        for e in range(self.n_routed):
            weight = gate_norm[..., e : e + 1]  # (B, T, 1)
            e_out = self.routed_experts[e](x)
            routed_out = routed_out + weight * e_out

        # Load stats (per expert, over the batch): count of times chosen
        expert_load = selected.astype(mx.float32).sum(axis=(0, 1))  # (E,)

        # Sequence-wise balance loss: entropy-like term (encourage uniform routing)
        avg_prob = mx.mean(nn.softmax(scores_raw, axis=-1), axis=(0, 1))  # (E,)
        avg_frac = expert_load / max(1, B * T)
        aux_loss = mx.sum(avg_prob * avg_frac) * float(self.n_routed)

        return shared_out + routed_out, aux_loss, expert_load

    def update_router_bias(self, expert_load: mx.array) -> None:
        """Auxiliary-loss-free bias update: nudge under-used experts up,
        over-used experts down, by ± bias_lr."""
        target = mx.mean(expert_load)
        signs = mx.sign(target - expert_load)
        self._route_bias = self._route_bias + self.bias_lr * signs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.13 NanoDeepSeekBlock — one Transformer block

    Pre-RMSNorm around each sublayer, mHC-wrapped residuals:

    ```
    X_{l+1} = mHC(X_l, CSA2 ∘ pre_norm)
    X_{l+2} = mHC(X_{l+1}, MoE ∘ pre_norm)
    ```

    The CSA2 sublayer reads/writes the shared state dict and the MoE
    sublayer returns an aux loss + per-expert load. Both are collected
    upstream by `CausalEncoderDecoderStackV1`.
    """)
    return


@app.class_definition
class NanoDeepSeekBlockV1(nn.Module):
    """One Transformer block: mHC-wrapped attention sublayer + mHC-wrapped
    MoE sublayer, both with pre-RMSNorm."""

    def __init__(
        self,
        cfg: NanoDeepSeekConfig,
        mode: str = "full",
        compress_m: int = 2,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.attn = CompressedSparseAttention2V1(cfg, mode=mode, compress_m=compress_m)
        self.attn_mhc = ManifoldConstrainedHyperConnectionV1(cfg.d_model, cfg.n_hc_streams, cfg.sinkhorn_iters)
        self.moe_norm = nn.RMSNorm(cfg.d_model)
        self.moe = DeepSeekMoEV1(
            cfg.d_model, cfg.n_routed_experts, cfg.n_active_experts,
            cfg.n_shared_experts, cfg.expert_hidden, cfg.fp4_qat, cfg.moe_bias_lr,
        )
        self.moe_mhc = ManifoldConstrainedHyperConnectionV1(cfg.d_model, cfg.n_hc_streams, cfg.sinkhorn_iters)

    def __call__(
        self,
        X: mx.array,
        state: dict,
        encoder_hidden: mx.array | None = None,
    ) -> tuple[mx.array, dict, mx.array, mx.array]:
        # Attention sublayer via mHC
        aux_holder: dict = {}

        def attn_fn(h: mx.array) -> mx.array:
            out, new_state = self.attn(self.attn_norm(h), state, encoder_hidden)
            # mutate state in-place is fine
            for k, v in new_state.items():
                state[k] = v
            return out

        X = self.attn_mhc(X, attn_fn)

        def moe_fn(h: mx.array) -> mx.array:
            out, aux, load = self.moe(self.moe_norm(h))
            aux_holder["aux"] = aux
            aux_holder["load"] = load
            return out

        X = self.moe_mhc(X, moe_fn)
        return X, state, aux_holder["aux"], aux_holder["load"]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.14 Causal Encoder-Decoder (CED) stack

    YOCO-derived split — see `yoco-you-only-cache-once-decoder-decoder.md`.
    The first half of the layers act as a **causal encoder**; the second
    half acts as a **decoder** whose **global** KV in Full-mode layers is
    derived from the *final encoder hidden state* $H_{L/2}$ via
    **layer-dependent** projections $W_l^{KV}, W_l^{Z}$ (which is why
    each CSA2 layer owns its own `LatentKVProjectionV1`). SWA stays
    layer-local. The encoder and decoder each carry their own CSA2
    shared state dict.
    """)
    return


@app.class_definition
class CausalEncoderDecoderStackV1(nn.Module):
    """Encoder half + decoder half of the CED architecture.

    Each half maintains its own CSA2 shared state during the forward pass.
    The decoder's Full-mode layers receive `encoder_hidden = H_{L/2}`.
    """

    def __init__(self, cfg: NanoDeepSeekConfig):
        super().__init__()
        self.cfg = cfg
        schedule = build_csa2_mode_schedule(cfg.encoder_modes, cfg.decoder_modes)
        self.schedule = schedule
        self.n_enc = cfg.n_encoder_layers
        self.n_dec = cfg.n_layers - cfg.n_encoder_layers
        assert len(cfg.encoder_modes) == self.n_enc, "encoder_modes length mismatch"
        assert len(cfg.decoder_modes) == self.n_dec, "decoder_modes length mismatch"
        self.encoder_blocks = [
            NanoDeepSeekBlockV1(cfg, mode=cfg.encoder_modes[i], compress_m=cfg.compress_m_encoder)
            for i in range(self.n_enc)
        ]
        self.decoder_blocks = [
            NanoDeepSeekBlockV1(cfg, mode=cfg.decoder_modes[i], compress_m=cfg.compress_m_decoder)
            for i in range(self.n_dec)
        ]

    def __call__(self, X: mx.array) -> tuple[mx.array, list, list]:
        """X: widened residual (B, T, n_streams, D). Returns (X, aux_losses, loads)."""
        enc_state: dict = {}
        aux_losses: list = []
        loads: list = []
        for blk in self.encoder_blocks:
            X, enc_state, aux, load = blk(X, enc_state, encoder_hidden=None)
            aux_losses.append(aux)
            loads.append(load)
        # Extract encoder hidden state (contract streams)
        enc_hidden = X.sum(axis=-2)  # (B, T, D)
        dec_state: dict = {}
        for blk in self.decoder_blocks:
            X, dec_state, aux, load = blk(X, dec_state, encoder_hidden=enc_hidden)
            aux_losses.append(aux)
            loads.append(load)
        return X, aux_losses, loads


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.15 Multi-Token Prediction head (kept for illustration)

    V3 used **MTP** (Multi-Token Prediction) — a small auxiliary head
    that predicts token $t + 2$ from the block state, densifying the
    training signal. V4.1 **drops MTP** in favour of **DSpark**, a
    confidence-scheduled speculative-decoding module trained *after*
    the backbone with gradients blocked from the backbone (see
    `dspark-confidence-scheduled-speculative-decoding.md`).

    We keep the cheap MTP form here because it plugs into the same
    training loop as the main LM loss and doubles as a lightweight
    auxiliary signal. In a production V4.1 build this cell would be
    deleted and a separate DSpark drafter would be trained post-hoc.
    """)
    return


@app.class_definition
class MultiTokenPredictionHeadV1(nn.Module):
    """Depth-1 MTP: predict token t+2 from the shared final hidden state
    via a small block + linear head."""

    def __init__(self, d_model: int = 256, vocab_size: int = 10000):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.block = ClampedSwiGLUFeedForwardV1(d_model, 4 * d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        # h: (B, T, D). Predict targets at t+2 using positions 0..T-3.
        h = self.norm(h[:, :-2, :] + self.block(h[:, :-2, :]))
        return self.head(h)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.16 NanoDeepSeekV1 — the top-level model

    Data flow: `tokens -> embedding + Engram injection -> mHC widen ->
    CED stack -> mHC contract -> final RMSNorm -> LM head (untied)
    -> MTP head`. `__call__` returns `(logits, mtp_logits, aux_losses_dict)`.
    """)
    return


@app.class_definition
class NanoDeepSeekV1(nn.Module):
    """Nano-DeepSeek V4.1: every component of V4.1-Flash, drastically shrunk."""

    def __init__(self, cfg: NanoDeepSeekConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.engram = EngramMemoryV1(
            cfg.d_model, cfg.engram_orders, cfg.engram_n_buckets,
            cfg.engram_n_heads, cfg.engram_dim,
        )
        self.ced = CausalEncoderDecoderStackV1(cfg)
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # untied
        self.mtp_head = MultiTokenPredictionHeadV1(cfg.d_model, cfg.vocab_size)
        self.n_hc_streams = cfg.n_hc_streams

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array, dict]:
        h = self.embed(ids)  # (B, T, D)
        h = h + self.engram(ids, h)
        # widen to n_streams for mHC
        B, T, D = h.shape
        X = mx.broadcast_to(h[:, :, None, :], (B, T, self.n_hc_streams, D))
        X, aux_losses, loads = self.ced(X)
        h = X.sum(axis=-2)  # contract streams
        h = self.final_norm(h)
        logits = self.lm_head(h)
        mtp_logits = self.mtp_head(h)
        aux_info = {
            "aux_losses": aux_losses,
            "expert_loads": loads,
        }
        return logits, mtp_logits, aux_info

    def update_moe_biases(self, loads: list) -> None:
        """Aux-loss-free routing bias update — called by the training loop
        once per step after `__call__`."""
        # Route bias updates should not run under gradient tracking
        idx = 0
        for blk in self.ced.encoder_blocks:
            blk.moe.update_router_bias(mx.stop_gradient(loads[idx]))
            idx += 1
        for blk in self.ced.decoder_blocks:
            blk.moe.update_router_bias(mx.stop_gradient(loads[idx]))
            idx += 1


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(int(v.size) for _, v in tree_flatten(model.parameters()))


@app.function
def count_parameters_by_component(model) -> dict[str, int]:
    """Per-component parameter breakdown by path prefix."""
    buckets = {
        "embed": 0,
        "engram": 0,
        "attention": 0,
        "moe": 0,
        "mhc": 0,
        "lm_head": 0,
        "mtp_head": 0,
        "norm": 0,
        "other": 0,
    }
    for path, arr in tree_flatten(model.parameters()):
        n = int(arr.size)
        lower = path.lower()
        if "embed" in lower and "engram" not in lower:
            buckets["embed"] += n
        elif "engram" in lower:
            buckets["engram"] += n
        elif "mtp_head" in lower:
            buckets["mtp_head"] += n
        elif "lm_head" in lower:
            buckets["lm_head"] += n
        elif "mhc" in lower or "hyper" in lower:
            buckets["mhc"] += n
        elif "moe" in lower or "expert" in lower or "gate" in lower:
            buckets["moe"] += n
        elif "attn" in lower or "csa" in lower or "kv" in lower or "swa" in lower or "indexer" in lower or "q_" in lower or "rope" in lower or "out_proj" in lower or "compressor" in lower:
            buckets["attention"] += n
        elif "norm" in lower:
            buckets["norm"] += n
        else:
            buckets["other"] += n
    return buckets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.17 Muon optimizer — Newton-Schulz orthogonalized momentum

    Muon: momentum + Newton-Schulz iterative orthogonalization of the
    2D gradient. The polynomial `X <- a X + b (X X^T) X + c (X X^T)^2 X`
    with `(a, b, c) = (3.4445, -4.7750, 2.0315)` converges to the
    orthogonal factor of the SVD in 5 iterations. See
    `muon-optimizer-derivation.md`.

    The update is scaled by $\sqrt{\text{fan\_out}/\text{fan\_in}}$ and
    an overall $\gamma \approx 0.2$ factor so the update RMS matches
    Adam's typical magnitude — otherwise Muon's orthogonal update is
    much larger than an Adam step and the learning rate needs recalibration.
    """)
    return


@app.function
def newton_schulz_orthogonalize(g: mx.array, steps: int = 5) -> mx.array:
    """Iterative orthogonalization of a 2D matrix.

    Uses the (3.4445, -4.7750, 2.0315) polynomial and runs in float32 with
    the input Frobenius-normalized first (numerical stability). Returns
    an approximation of U V^T of the SVD of g.
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
class MuonOptimizerV1(optim.Optimizer):
    """Muon: momentum SGD followed by Newton-Schulz orthogonalization of
    the 2D update. Falls back to plain momentum for 1D parameters."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.rms_scale = rms_scale
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict) -> None:
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        v = self.momentum * state["v"] + gradient
        state["v"] = v
        if v.ndim < 2:
            update = v
        else:
            # collapse any leading axes into the row dim so NS sees a matrix
            orig_shape = v.shape
            v2 = v.reshape(-1, v.shape[-1])
            g_orth = newton_schulz_orthogonalize(v2, self.ns_steps)
            fan_out, fan_in = v2.shape
            scale = math.sqrt(max(1.0, fan_out / max(1, fan_in)))
            update = g_orth.reshape(orig_shape) * self.rms_scale * scale
        lr = self.learning_rate.astype(gradient.dtype)
        if self.weight_decay != 0.0:
            parameter = parameter * (1 - lr * self.weight_decay)
        return parameter - lr * update


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.18 Hybrid Muon + AdamW router

    Per the DeepSeek / Moonlight prescription:

    - **Muon** for all $\ge$2-D linear weights (attention/MoE/mHC matrices);
    - **AdamW** for embeddings, output head, RMSNorm gains, biases,
      and any 1D static components.

    We partition the parameter tree by path substring at construction
    time and route each gradient sub-tree to its owner optimizer.
    """)
    return


@app.function
def classify_param_for_muon(path: str, arr: mx.array) -> bool:
    """Return True if the parameter belongs to the Muon optimizer."""
    if arr.ndim < 2:
        return False
    lower = path.lower()
    if "embed" in lower or "table" in lower:
        return False
    if "lm_head" in lower or "mtp_head" in lower:
        return False
    return True


@app.class_definition
class HybridMuonAdamWV1:
    """Composite optimizer that partitions the parameter tree by path and
    routes each sub-tree to Muon or AdamW."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
    ):
        self.model = model
        self.muon = MuonOptimizerV1(
            learning_rate=learning_rate,
            momentum=momentum,
            ns_steps=ns_steps,
            rms_scale=rms_scale,
            weight_decay=weight_decay,
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
    model = NanoDeepSeekV1(config)
    mx.eval(model.parameters())
    _total = count_parameters(model)
    _breakdown = count_parameters_by_component(model)
    print(f"total params: {_total:,}")
    for _k, _v in _breakdown.items():
        print(f"  {_k:>10s}: {_v:>10,}")
    return (model,)


@app.cell
def _(model, train_ds):
    _x, _y = next(iter(train_ds))
    _logits, _mtp_logits, _aux = model(_x)
    print(f"input  shape: {_x.shape}")
    print(f"logits shape: {_logits.shape}")
    print(f"mtp    shape: {_mtp_logits.shape}")
    print(f"n aux losses: {len(_aux['aux_losses'])}")
    print(f"first aux loss: {float(_aux['aux_losses'][0]):.6f}")
    print(f"expert load (block 0): {_aux['expert_loads'][0].tolist()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Training

    Cosine LR schedule with linear warmup, cross-entropy on next-token
    prediction plus a weighted MTP loss and a weighted MoE aux loss.
    Two optimizer routes are exposed:

    - **hybrid-muon-adamw** — the paper's prescription (Muon on 2D
      weights, AdamW on the rest).
    - **adamw** — a plain baseline so the Muon claim can be tested.

    MLX has no autocast / GradScaler — everything runs in native
    precision. FP4 QAT is exposed as a checkbox; when enabled, the
    routed experts and indexer QK path fake-quantize their inputs.
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4", label="Learning rate",
    )
    bs_ui = mo.ui.dropdown(options=[4, 8, 16, 32], value=8, label="Batch size")
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2},
        value="1e-4", label="Weight decay",
    )
    epochs_ui = mo.ui.slider(1, 20, value=1, step=1, label="Epochs")
    opt_ui = mo.ui.dropdown(
        options=["hybrid-muon-adamw", "adamw"],
        value="hybrid-muon-adamw", label="Optimizer",
    )
    fp4_ui = mo.ui.checkbox(label="FP4 QAT on routed experts + indexer", value=False)
    max_steps_ui = mo.ui.slider(20, 2000, value=200, step=20, label="Max steps per epoch")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack([
        mo.md("### Training hyperparameters"),
        mo.hstack([lr_ui, bs_ui, wd_ui, epochs_ui]),
        mo.hstack([opt_ui, fp4_ui, max_steps_ui]),
        train_btn,
    ])
    return (
        bs_ui,
        epochs_ui,
        fp4_ui,
        lr_ui,
        max_steps_ui,
        opt_ui,
        train_btn,
        wd_ui,
    )


@app.function
def compute_lm_loss(
    model,
    x: mx.array,
    y: mx.array,
    mtp_weight: float,
    moe_aux_weight: float,
) -> mx.array:
    """Cross-entropy on next-token + weighted MTP loss + weighted MoE aux loss."""
    logits, mtp_logits, aux_info = model(x)
    ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).mean()
    # MTP: predict token at t+2 from state at t
    mtp_target = y[:, 1:]  # y is x shifted by 1, so y[:, 1:] is token t+2 relative to x
    mtp_target = mtp_target[:, : mtp_logits.shape[1]]  # align lengths
    mtp_ce = nn.losses.cross_entropy(
        mtp_logits.reshape(-1, mtp_logits.shape[-1]), mtp_target.reshape(-1)
    ).mean()
    aux_sum = mx.zeros(())
    for a in aux_info["aux_losses"]:
        aux_sum = aux_sum + a
    aux_sum = aux_sum / max(1, len(aux_info["aux_losses"]))
    return ce + mtp_weight * mtp_ce + moe_aux_weight * aux_sum


@app.function
def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


@app.function
def cosine_lr_schedule(step: int, warmup: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


@app.function
def run_train_epoch(
    model,
    optimizer,
    train_iter,
    mtp_weight: float,
    moe_aux_weight: float,
    max_steps: int,
    step_offset: int,
    lr_schedule_fn,
) -> tuple[list[float], list[np.ndarray], int]:
    losses: list[float] = []
    loads_history: list = []
    loss_fn = lambda m, x, y: compute_lm_loss(m, x, y, mtp_weight, moe_aux_weight)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    step = 0
    for x, y in train_iter:
        if step >= max_steps:
            break
        loss, grads = loss_and_grad(model, x, y)
        # Update LR
        lr = lr_schedule_fn(step_offset + step)
        # apply to sub-optimizers if hybrid
        if hasattr(optimizer, "muon"):
            optimizer.muon.learning_rate = lr
            optimizer.adamw.learning_rate = lr
        else:
            optimizer.learning_rate = lr
        optimizer.update(model, grads)
        # After the update, do the aux-loss-free routing bias step
        # (this needs a fresh forward to get the load — but we can reuse from the loss forward via a hack:
        # simpler: skip in-loop update, do it at epoch end)
        mx.eval(model.parameters())
        losses.append(float(loss.item()))
        step += 1
    return losses, loads_history, step


@app.function
def run_evaluate(model, data_iter, mtp_weight: float, moe_aux_weight: float, max_batches: int = 50) -> float:
    total = 0.0
    n = 0
    for x, y in data_iter:
        if n >= max_batches:
            break
        loss = compute_lm_loss(model, x, y, mtp_weight, moe_aux_weight)
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
    fp4_ui,
    lr_ui,
    max_steps_ui,
    mo,
    opt_ui,
    train_btn,
    wd_ui,
):
    train_losses: list[float] = []
    val_losses: list[float] = []
    expert_load_history: list = []
    trained_model = None

    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** above to begin. The model stays randomly initialized until then."))
    else:
        _cfg = NanoDeepSeekConfig(**{**asdict(config), "fp4_qat": fp4_ui.value})
        _model = NanoDeepSeekV1(_cfg)
        mx.eval(_model.parameters())
        _train_ds, _val_ds, _ = make_datasets(all_tokens, seq_len=_cfg.seq_len, batch_size=bs_ui.value)
        if opt_ui.value == "hybrid-muon-adamw":
            _optimizer = HybridMuonAdamWV1(_model, learning_rate=lr_ui.value, weight_decay=wd_ui.value)
        else:
            _optimizer = optim.AdamW(learning_rate=lr_ui.value, weight_decay=wd_ui.value)
        _total_steps = max_steps_ui.value * epochs_ui.value
        _warmup = max(5, int(0.05 * _total_steps))
        _lr_fn = lambda s: cosine_lr_schedule(s, _warmup, _total_steps, lr_ui.value, lr_ui.value * 0.1)
        _step_offset = 0
        for _epoch in range(epochs_ui.value):
            _ep_losses, _ep_loads, _steps_ran = run_train_epoch(
                _model, _optimizer, iter(_train_ds),
                _cfg.mtp_loss_weight, _cfg.moe_aux_weight,
                max_steps_ui.value, _step_offset, _lr_fn,
            )
            train_losses.extend(_ep_losses)
            expert_load_history.extend(_ep_loads)
            # end-of-epoch: also snapshot MoE loads via a single forward
            _x, _y = next(iter(_val_ds))
            _, _, _aux_info = _model(_x)
            _snapshot = np.stack([np.array(load) for load in _aux_info["expert_loads"]], axis=0)
            expert_load_history.append(_snapshot)
            # aux-loss-free bias updates using the snapshot
            _model.update_moe_biases(_aux_info["expert_loads"])
            _step_offset += _steps_ran
            _train_avg = float(np.mean(_ep_losses)) if _ep_losses else float("nan")
            _val_avg = run_evaluate(
                _model, iter(_val_ds), _cfg.mtp_loss_weight, _cfg.moe_aux_weight, max_batches=10,
            )
            val_losses.append(_val_avg)
            mo.output.replace(mo.md(
                f"**Epoch {_epoch+1}/{epochs_ui.value}** — "
                f"train loss {_train_avg:.4f} (ppl {perplexity_from_loss(_train_avg):.1f}) | "
                f"val loss {_val_avg:.4f} (ppl {perplexity_from_loss(_val_avg):.1f})"
            ))
        trained_model = _model
        mo.output.replace(mo.md(
            f"**Training complete!** final val loss {val_losses[-1]:.4f} "
            f"(ppl {perplexity_from_loss(val_losses[-1]):.1f})"
        ))
    return expert_load_history, train_losses, trained_model, val_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Hyperparameter Search (optional)

    Small 2 × 2 × 2 grid over `learning_rate`, `n_routed_experts`, and
    `indexer_top_k_blocks`. Each config runs 2 short "mini-epochs"
    (few steps each) so the whole search fits inside a minute. Results
    are sorted by validation perplexity.
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
        "n_routed": [4, 8],
        "top_k": [8, 16],
    }
    _hp_results: list[dict] = []
    _n_steps = 30
    for _lr in _search["lr"]:
        for _ne in _search["n_routed"]:
            for _tk in _search["top_k"]:
                _cfg = NanoDeepSeekConfig(**{**asdict(config), "n_routed_experts": _ne, "indexer_top_k_blocks": _tk})
                _m = NanoDeepSeekV1(_cfg)
                mx.eval(_m.parameters())
                _opt = optim.AdamW(learning_rate=_lr, weight_decay=1e-4)
                _train_ds, _val_ds, _ = make_datasets(all_tokens, seq_len=_cfg.seq_len, batch_size=4)
                _lr_fn = lambda s: _lr
                run_train_epoch(_m, _opt, iter(_train_ds), _cfg.mtp_loss_weight, _cfg.moe_aux_weight, _n_steps, 0, _lr_fn)
                _vl = run_evaluate(_m, iter(_val_ds), _cfg.mtp_loss_weight, _cfg.moe_aux_weight, max_batches=5)
                _hp_results.append({
                    "lr": _lr, "n_routed": _ne, "top_k": _tk,
                    "val_loss": round(_vl, 4), "val_ppl": round(perplexity_from_loss(_vl), 2),
                })
                mo.output.replace(mo.md(
                    f"lr={_lr} n_routed={_ne} top_k={_tk} -> val {_vl:.4f} (ppl {perplexity_from_loss(_vl):.1f})"
                ))
    _hp_results.sort(key=lambda r: r["val_loss"])
    hp_results = _hp_results
    mo.output.append(mo.ui.table(_hp_results))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Validation & Cross-Validation

    Random k-fold cross-validation is **invalid** for a contiguous
    language corpus — random folds tear apart neighbouring tokens,
    letting the model see near-verbatim context from the training set
    during validation. Instead we use **blocked 5-fold CV**: split the
    training tokens into 5 contiguous folds, evaluate fold $i$ as val,
    train on the union of the other four. Report per-fold perplexity
    plus mean ± std.
    """)
    return


@app.function
def evaluate_model(model, data_iter, mtp_weight: float, moe_aux_weight: float, max_batches: int = 200) -> tuple[float, float, float]:
    """Return (avg_loss, perplexity, bits_per_token)."""
    total = 0.0
    n = 0
    for x, y in data_iter:
        if n >= max_batches:
            break
        loss = compute_lm_loss(model, x, y, mtp_weight, moe_aux_weight)
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
            trained_model, iter(test_ds), config.mtp_loss_weight, config.moe_aux_weight, max_batches=30,
        )
        _out = mo.md(
            f"### Test-set metrics\n\n"
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
    cfg: NanoDeepSeekConfig,
    n_steps: int,
    batch_size: int,
    lr: float,
) -> tuple[float, float]:
    train_tokens = np.concatenate([fold_tokens[:val_start], fold_tokens[val_end:]])
    val_tokens = fold_tokens[val_start:val_end]
    train_ds = TokenWindowDatasetV1(train_tokens, cfg.seq_len, batch_size, shuffle=True)
    val_ds = TokenWindowDatasetV1(val_tokens, cfg.seq_len, batch_size, shuffle=False)
    model = NanoDeepSeekV1(cfg)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    lr_fn = lambda s: lr
    run_train_epoch(model, opt, iter(train_ds), cfg.mtp_loss_weight, cfg.moe_aux_weight, n_steps, 0, lr_fn)
    val_loss = run_evaluate(model, iter(val_ds), cfg.mtp_loss_weight, cfg.moe_aux_weight, max_batches=5)
    return val_loss, perplexity_from_loss(val_loss)


@app.cell
def _(all_tokens, config, mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) so we know cross-validation is worth running._")
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
                _train_tokens, _val_start, _val_end,
                config, n_steps=20, batch_size=4, lr=3e-4,
            )
            _cv_rows.append({"fold": _i + 1, "val_loss": round(_loss, 4), "val_ppl": round(_ppl, 2)})
            mo.output.replace(mo.md(f"Fold {_i+1}/{_k}: val {_loss:.4f} (ppl {_ppl:.1f})"))
        _losses_arr = np.array([r["val_loss"] for r in _cv_rows])
        cv_results = {
            "folds": _cv_rows,
            "mean_loss": float(_losses_arr.mean()),
            "std_loss": float(_losses_arr.std()),
        }
        _out = mo.vstack([
            mo.md(f"### Blocked 5-fold CV (small budget)\n\nmean val loss: **{cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}**"),
            mo.ui.table(_cv_rows),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. Results

    Five diagnostic plots. Each is defined as its own `@app.function`
    returning a matplotlib `Figure`, and each calling cell is either
    one-line or a two-branch conditional depending on `trained_model`.
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
        ax.text(0.5, 0.5, "no expert load data", ha="center", va="center")
        return fig
    snapshots = [s for s in expert_load_history if isinstance(s, np.ndarray) and s.ndim == 2]
    if not snapshots:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no snapshot data", ha="center", va="center")
        return fig
    # Take mean over layers for each snapshot -> (n_snapshots, n_experts)
    loads_over_time = np.stack([s.mean(axis=0) for s in snapshots], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(loads_over_time.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("epoch snapshot")
    ax.set_ylabel("routed expert idx")
    ax.set_title("Routed-expert utilisation over training (mean over layers)")
    fig.colorbar(im, ax=ax, label="tokens routed")
    fig.tight_layout()
    return fig


@app.function
def plot_indexer_selection(model, x: mx.array):
    # Run one forward and collect the topk mask of the first Full CSA2 layer
    _logits, _mtp, _aux = model(x)
    # Re-run just the first encoder block to grab its state without polluting
    enc_state: dict = {}
    _h = model.embed(x) + model.engram(x, model.embed(x))
    B, T, D = _h.shape
    X = mx.broadcast_to(_h[:, :, None, :], (B, T, model.n_hc_streams, D))
    X, enc_state, _aux0, _load0 = model.ced.encoder_blocks[0](X, enc_state, encoder_hidden=None)
    mask = np.array(enc_state["topk_mask"][0])  # (T, T_g)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(mask.astype(float), aspect="auto", cmap="Greys", origin="lower")
    ax.set_xlabel("compressed KV block")
    ax.set_ylabel("query position")
    ax.set_title("Hierarchical Sparse Indexer — top-k selection (layer 0)")
    fig.tight_layout()
    return fig


@app.function
def plot_mhc_mixing_matrix(model):
    C = np.array(model.ced.encoder_blocks[0].attn_mhc.get_C())
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(C, cmap="viridis", vmin=0.0, vmax=1.0)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlabel(f"stream j  (col sums: {np.array2string(col_sums, precision=2)})")
    ax.set_ylabel(f"stream i  (row sums: {np.array2string(row_sums, precision=2)})")
    ax.set_title("mHC residual mixing C — Birkhoff (doubly stochastic) proof")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


@app.function
def plot_csa2_mode_schedule(cfg: NanoDeepSeekConfig):
    schedule = build_csa2_mode_schedule(cfg.encoder_modes, cfg.decoder_modes)
    n_enc = cfg.n_encoder_layers
    n = len(schedule)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = {"full": "#2c7fb8", "reindex": "#7fcdbb", "reuse": "#edf8b1"}
    for i, m in enumerate(schedule):
        ax.barh(0, 1, left=i, color=colors[m], edgecolor="black")
        ax.text(i + 0.5, 0, m, ha="center", va="center", fontsize=9)
    # KV donor arrows
    last_full_enc = 0
    for i in range(n_enc):
        if schedule[i] == "full":
            last_full_enc = i
        elif schedule[i] in ("reindex", "reuse"):
            ax.annotate("", xy=(i + 0.2, -0.4), xytext=(last_full_enc + 0.8, -0.4),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    last_full_dec = n_enc
    for i in range(n_enc, n):
        if schedule[i] == "full":
            last_full_dec = i
        elif schedule[i] in ("reindex", "reuse"):
            ax.annotate("", xy=(i + 0.2, -0.4), xytext=(last_full_dec + 0.8, -0.4),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.axvline(n_enc, color="red", linestyle="--", label=f"encoder/decoder split (l={n_enc})")
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.8, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(n))
    ax.set_xlabel("layer index")
    ax.set_title("CSA2 mode schedule and KV donor arrows")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


@app.cell
def _(train_losses: list[float], val_losses: list[float]):
    plot_loss_curve(train_losses, val_losses)
    return


@app.cell
def _(expert_load_history: list):
    plot_expert_load(expert_load_history)
    return


@app.cell
def _(mo, train_ds, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first (Section 5) to visualize the indexer's top-k selection._")
    else:
        _x, _y = next(iter(train_ds))
        _out = plot_indexer_selection(trained_model, _x)
    _out
    return


@app.cell
def _(mo, trained_model):
    if trained_model is None:
        _out = mo.md("_Train first (Section 5) to visualize the Sinkhorn'd C matrix._")
    else:
        _out = plot_mhc_mixing_matrix(trained_model)
    _out
    return


@app.cell
def _(config):
    plot_csa2_mode_schedule(config)
    return


@app.function
def kv_bytes_per_token(cfg: NanoDeepSeekConfig, kind: str = "nano-mla-fp4") -> float:
    """Approximate KV-cache bytes / token for various baselines."""
    n_enc_full = sum(1 for m in cfg.encoder_modes if m == "full")
    n_dec_full = sum(1 for m in cfg.decoder_modes if m == "full")
    head_dim = cfg.d_nope + cfg.d_rope
    if kind == "full-mha-fp16":
        return 2 * cfg.n_layers * cfg.n_heads * head_dim * 2
    if kind == "gqa-fp16":
        return 2 * cfg.n_layers * max(1, cfg.n_heads // 2) * head_dim * 2
    if kind == "mla-fp16":
        return cfg.n_layers * cfg.kv_latent_dim * 2
    if kind == "nano-mla-fp4":
        # only Full-mode CSA2 layers materialize distinct main KV
        main = (n_enc_full / cfg.compress_m_encoder + n_dec_full / cfg.compress_m_decoder) * cfg.kv_latent_dim * 0.5
        return main + n_enc_full / cfg.compress_m_encoder * cfg.indexer_dim * 0.5
    raise ValueError(kind)


@app.cell
def _(config, mo):
    _rows = []
    for _kind in ["full-mha-fp16", "gqa-fp16", "mla-fp16", "nano-mla-fp4"]:
        _rows.append({"cache scheme": _kind, "bytes / token": round(kv_bytes_per_token(config, _kind), 1)})
    mo.vstack([
        mo.md("### KV-cache budget comparison (this nano model's config)"),
        mo.ui.table(_rows),
        mo.md(
            "The nano-MLA-FP4 row shows the multiplicative effect of "
            "**entry-size (MLA latent) x sequence axis (m compression) x "
            "layer axis (only Full-mode layers materialize KV) x FP4 QAT** "
            "(the 0.5 factor = 4 bits / 8 bits vs FP16 = 0.25, times 2 for scale overhead)."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary

    This tome implements all of DeepSeek-V4.1-Flash's architectural
    components — MLA latent KV with decoupled RoPE, CSA2 with the
    Full/Reindex/Reuse mode schedule and shared candidate pool,
    sliding-window layer-local KV, learnable attention-sink logits,
    shared + fine-grained routed MoE with sqrt-softplus affinity and
    aux-loss-free load-balancing bias, hashed n-gram Engram memory
    with context gating, manifold-constrained hyper-connections with
    Sinkhorn-Knopp projection onto the Birkhoff polytope, an MTP head
    (kept for illustration in place of V4.1's DSpark), and a hybrid
    Muon + AdamW optimizer with Newton-Schulz orthogonalization —
    at roughly 15M parameters on a 278K-token corpus.

    278K tokens is *far too small* for any of these mechanisms to
    show its real benefit; the KV compression story lives in the
    1M-token regime, the aux-loss-free MoE balancing needs many more
    tokens per expert to converge, and Engram's win is in long-tail
    rare n-grams. This notebook is an *architecture comprehension*
    exercise. Read the code and the citations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9. Text Generation

    Autoregressive sampling from the trained backbone. Prefill re-runs
    the full CSA2 stack every step (no KV cache in this nano
    implementation — the machinery is defined, but generation-time
    cache reuse would require a separate refactor).
    """)
    return


@app.cell
def _(mo):
    prompt_ui = mo.ui.text_area(value="KING RICHARD:\n", label="Prompt")
    max_new_ui = mo.ui.slider(10, 200, value=80, step=10, label="Max new tokens")
    temp_ui = mo.ui.slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
    topk_ui = mo.ui.slider(0, 200, value=40, step=5, label="Top-k (0 = disabled)")
    topp_ui = mo.ui.slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p")
    seed_ui = mo.ui.number(value=0, label="Seed")
    gen_btn = mo.ui.run_button(label="Generate")
    mo.vstack([
        prompt_ui, mo.hstack([max_new_ui, temp_ui]),
        mo.hstack([topk_ui, topp_ui, seed_ui]), gen_btn,
    ])
    return gen_btn, max_new_ui, prompt_ui, seed_ui, temp_ui, topk_ui, topp_ui


@app.function
def generate_text(
    model,
    merge_table: dict,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    seed: int = 0,
    seq_len: int = 256,
) -> str:
    """Autoregressive sampling with temperature, top-k, top-p, seeded."""
    mx.random.seed(int(seed))
    np.random.seed(int(seed))
    ids_list = tokenize(prompt, merge_table)
    if not ids_list:
        ids_list = [0]
    ids = mx.array(ids_list, dtype=mx.int32)[None, :]
    for _ in range(max_new_tokens):
        # truncate to seq_len for the forward pass
        cur = ids if ids.shape[1] <= seq_len else ids[:, -seq_len:]
        logits, _mtp, _aux = model(cur)
        next_logits = logits[:, -1, :] / max(1e-6, float(temperature))
        # top-k
        if top_k and top_k > 0:
            k = min(top_k, next_logits.shape[-1])
            threshold = mx.min(mx.topk(next_logits, k=k, axis=-1), axis=-1, keepdims=True)
            next_logits = mx.where(next_logits < threshold, mx.array(-1e9, dtype=next_logits.dtype), next_logits)
        # top-p
        if top_p and 0.0 < top_p < 1.0:
            sorted_idx = mx.argsort(-next_logits, axis=-1)
            sorted_logits = mx.take_along_axis(next_logits, sorted_idx, axis=-1)
            probs = mx.softmax(sorted_logits, axis=-1)
            cumprobs = mx.cumsum(probs, axis=-1)
            keep = cumprobs <= top_p
            # always keep the top-1
            keep = mx.concatenate([mx.ones((keep.shape[0], 1), dtype=keep.dtype), keep[:, :-1]], axis=-1)
            filtered_sorted = mx.where(keep, sorted_logits, mx.array(-1e9, dtype=sorted_logits.dtype))
            # scatter back
            unsort_logits = mx.zeros_like(next_logits)
            unsort_logits = scatter_along_last_axis(unsort_logits, sorted_idx, filtered_sorted)
            next_logits = unsort_logits
        next_id = mx.random.categorical(next_logits)  # (1,)
        ids = mx.concatenate([ids, next_id[:, None]], axis=1)
        mx.eval(ids)
    return decode_tokens(ids[0].tolist(), merge_table)


@app.function
def scatter_along_last_axis(out: mx.array, indices: mx.array, values: mx.array) -> mx.array:
    """Manual scatter along the last axis for 2D arrays: out[b, indices[b, i]] = values[b, i].

    Falls back to numpy since mlx lacks a batched scatter primitive; only
    used at generation time on the vocab logits so the cost is negligible.
    """
    B, _ = out.shape
    idx_np = np.array(indices)
    val_np = np.array(values)
    out_np = np.array(out)
    for b in range(B):
        out_np[b, idx_np[b]] = val_np[b]
    return mx.array(out_np, dtype=out.dtype)


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
            trained_model, merge_table, prompt_ui.value,
            max_new_tokens=max_new_ui.value, temperature=temp_ui.value,
            top_k=topk_ui.value, top_p=topp_ui.value, seed=int(seed_ui.value),
        )
        _out = mo.md(f"**Prompt:** {prompt_ui.value}\n\n**Generated:**\n\n```\n{_text}\n```")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10. Save Trained Model

    Weights go to `models/<filename>.safetensors`; the
    `NanoDeepSeekConfig` fields go to `models/<filename>_config.json`
    so the model can be reconstructed exactly (`NanoDeepSeekV1(NanoDeepSeekConfig(**json))`).
    The `models/` directory is resolved relative to this notebook file
    so the same notebook works from any checkout location.
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="tinyshakespear_nano_deepseek_v1.safetensors",
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
        with open(_cfg_path, "w") as _f:
            json.dump(_cfg_dict, _f, indent=2)
        _out = mo.md(
            f"**Saved!**\n\n"
            f"- weights: `{_save_path}`\n"
            f"- config sidecar: `{_cfg_path}`\n"
        )
    _out
    return


if __name__ == "__main__":
    app.run()
