import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    import math
    from functools import partial

    import numpy as np

    import mlx
    import mlx.nn as nn
    from mlx.optimizers import optimizers
    import mlx.core as mx

    from tqdm import tqdm


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    data_dir = '../data/'
    merges_table_file = data_dir + 'tinyshakespear.json'
    training_token_file = data_dir + 'tinyshakespear_tokens.txt'
    training_file = data_dir + 'tinyshakespear.txt'
    return merges_table_file, training_file, training_token_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### tokenization
    """)
    return


@app.function
def load_merge_table(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    pair_to_id = {}
    pair_to_rank = {}
    id_to_pair = {}
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
def tokenize(text, merge_table):
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
        merged_tokens = []
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
def decode_tokens(tokens, merge_table):
    id_to_pair = merge_table["id_to_pair"]
    memo = {}

    def expand(token_id):
        if token_id < 256:
            return bytes([token_id])
        if token_id in memo:
            return memo[token_id]
        left, right = id_to_pair[token_id]
        expanded = expand(left) + expand(right)
        memo[token_id] = expanded
        return expanded

    byte_sequence = b"".join(expand(token_id) for token_id in tokens)
    return byte_sequence.decode("utf-8")


@app.cell
def _(merges_table_file):
    merge_table = load_merge_table(merges_table_file)
    return (merge_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### tokenizer tests
    """)
    return


@app.cell
def _(merge_table):
    _test_strings = [
        "First Citizen:\nBefore we proceed any further, hear me speak.",
        "You are all resolved rather to die than to famish?",
        "don't stop believing!",
    ]
    for _text in _test_strings:
        _tokens = tokenize(_text, merge_table)
        _decoded = decode_tokens(_tokens, merge_table)
        assert _decoded == _text, f"round-trip mismatch: {_decoded!r} != {_text!r}"
        print(f"OK round-trip ({len(_text)} chars -> {len(_tokens)} tokens): {_text!r}")
    return


@app.cell
def _(merge_table, training_file, training_token_file):
    with open(training_file, "r") as _f:
        _raw_text = _f.read()

    with open(training_token_file, "r") as _f:
        _reference_tokens = [int(_tok) for _tok in _f.read().split()]

    _num_check = 200
    _prefix_tokens = _reference_tokens[:_num_check]
    _decoded_prefix = decode_tokens(_prefix_tokens, merge_table)

    assert _raw_text.startswith(_decoded_prefix), "decoded prefix does not match reference corpus"
    print(f"OK: decoded first {_num_check} reference tokens matches corpus prefix:")
    print(repr(_decoded_prefix))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### preparing training dataset
    """)
    return


@app.class_definition
class TokenWindowDatasetV1:
    def __init__(
        self,
        tokens: list[int],
        window_size: int = 1024,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.batch_size = batch_size
        self._tokens = np.asarray(tokens, dtype=np.int32)
        self._num_windows = max(0, len(self._tokens) - window_size + 1)
        # persisted Generator: each __iter__ call advances its state, so every
        # epoch gets a different shuffle while the whole run stays reproducible
        # from one constructor seed (re-creating Generator(seed) per call would
        # replay the identical order every epoch instead)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._num_windows // self.batch_size

    def __iter__(self):
        starts = self._rng.permutation(self._num_windows)
        offsets = np.arange(self.window_size)
        num_batches = self._num_windows // self.batch_size
        for b in range(num_batches):
            batch_starts = starts[b * self.batch_size : (b + 1) * self.batch_size]
            window_idx = batch_starts[:, None] + offsets[None, :]
            batch_np = self._tokens[window_idx]
            yield mx.array(batch_np, dtype=mx.int32)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### model
    """)
    return


@app.class_definition
class GPTV1(nn.Module):
    def __init__(self, token_sz, em_sz, attn_hz, attn_lz):
        super().__init__()
        self.embedding = nn.Embedding(token_sz, em_sz)
        self.pos_encoder = nn.SinusoidalPositionalEncoding(em_sz)
        self.tencoder = nn.TransformerEncoder(attn_lz, em_sz, attn_hz)
        self.linear = nn.Linear(em_sz, token_sz)

    def __call__(self, tokens):
        em = self.embedding(tokens)
        # pos_encoder expects sequence positions, not token ids
        positions = mx.arange(tokens.shape[-1])
        pem = self.pos_encoder(positions)
        em = em + pem
        em = self.tencoder(em, 'causal')
        em = self.linear(em)
        return em


@app.class_definition
class MultiHeadAttentionV2(nn.Module):
    def __init__(self, dims, nh, qidim = None, kidim = None, vidim = None, vdim = None, vodim = None):
        super().__init__()
        if dims % nh != 0:
            raise ValueError("dims % nh != 0")
        query_input_dims = qidim or dims
        key_input_dims = kidim or dims
        value_input_dims = vidim or dims
        value_dims = vdim or dims
        value_output_dims = vodim or dims
        self.nh = nh
        self.query_proj = nn.Linear(query_input_dims, dims, bias=False)
        self.key_proj = nn.Linear(key_input_dims, dims, bias=False)
        self.value_proj = nn.Linear(value_input_dims, value_dims, bias=False)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=False)
        # RoPE must rotate each head's own head_dim-wide slice independently.
        # Rotating the full concatenated `dims` vector before splitting into
        # heads is wrong: mlx's RoPE pairs feature i with feature i + dims/2,
        # so for nh=2 that pairs head 0's raw projection directly with head
        # 1's, entangling the two heads' pre-projection values together
        # instead of giving each head its own independent rotary encoding.
        self.rope = nn.RoPE(dims // nh)

    def __call__(self, query, key, value, mask=None):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        nh = self.nh
        query = mx.unflatten(query, -1, (nh, -1)).transpose(0, 2, 1, 3)
        key = mx.unflatten(key, -1, (nh, -1)).transpose(0, 2, 1, 3)
        value = mx.unflatten(value, -1, (nh, -1)).transpose(0, 2, 1, 3)
        query = self.rope(query)
        key = self.rope(key)
        scale = math.sqrt(1. / query.shape[-1])
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(output)


@app.class_definition
class TransformerEncoderLayerV2(nn.Module):
    def __init__(self, dims, nh, mdim = None, dropout = 0., activation = nn.SiLU):
        super().__init__()
        mlp_dims = mdim or dims * 4
        self.ln = nn.LayerNorm(dims)
        self.attention = MultiHeadAttentionV2(dims, nh)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, mlp_dims),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims, dims),
        )

    def __call__(self, x, mask):
        y = self.ln(x)
        y = self.attention(y, y, y, mask)
        y = self.dropout(y)
        x = x + y
        y = self.mlp(x)
        y = y + x
        return y


@app.class_definition
class TransformerEncoderV2(nn.Module):
    def __init__(self, n, dims, nh, mdim = None, dropout = 0., activation=nn.SiLU):
        super().__init__()
        self.layers = [TransformerEncoderLayerV2(dims, nh, mdim, dropout, activation) for _ in range(n)]
        self.ln = nn.LayerNorm(dims)
    
    def __call__(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.ln(x)


@app.class_definition
class GPTV2(nn.Module):
    def __init__(self, token_sz, em_sz, attn_hz, attn_lz):
        super().__init__()
        self.embedding = nn.Embedding(token_sz, em_sz)
        self.pos_encoder = nn.SinusoidalPositionalEncoding(em_sz)
        self.tencoder = TransformerEncoderV2(attn_lz, em_sz, attn_hz)
        self.linear = nn.Linear(em_sz, token_sz)

    def __call__(self, tokens):
        em = self.embedding(tokens)
        # pos_encoder expects sequence positions, not token ids
        positions = mx.arange(tokens.shape[-1])
        pem = self.pos_encoder(positions)
        em = em + pem
        em = self.tencoder(em, 'causal')
        em = self.linear(em)
        return em


@app.cell
def _():
    ### training
    return


@app.function
def round_up_to_multiple(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return value + (multiple - value % multiple)


@app.cell
def _(mo):
    learning_rate_ui = mo.ui.slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate")
    weight_decay_ui = mo.ui.slider(0.0, 1e-2, value=1e-4, step=1e-4, label="Weight decay")
    batch_size_ui = mo.ui.slider(8, 512, value=256, step=8, label="Batch size")
    window_size_ui = mo.ui.slider(16, 512, value=128, step=16, label="Window size (tokens)")
    token_size_ui = mo.ui.slider(256, 50_000, value=10_000, step=256, label="Token vocab size")
    epochs_ui = mo.ui.slider(1, 100, value=20, step=1, label="Epochs")
    embed_dim_ui = mo.ui.slider(8, 512, value=8, step=8, label="Embedding size")
    num_heads_ui = mo.ui.slider(1, 16, value=4, step=1, label="Attention heads")
    num_layers_ui = mo.ui.slider(1, 12, value=1, step=1, label="Transformer layers")
    train_button = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md(
                "### Training hyperparameters\n"
                "`Token vocab size` must be at least as large as the tokenizer's "
                "true vocab (base 256 bytes + BPE merges) or training/generation "
                "will index out of range."
            ),
            mo.hstack([learning_rate_ui, weight_decay_ui, batch_size_ui, window_size_ui]),
            mo.hstack([token_size_ui, epochs_ui, embed_dim_ui, num_heads_ui, num_layers_ui]),
            train_button,
        ]
    )
    return (
        batch_size_ui,
        embed_dim_ui,
        epochs_ui,
        learning_rate_ui,
        num_heads_ui,
        num_layers_ui,
        token_size_ui,
        train_button,
        weight_decay_ui,
        window_size_ui,
    )


@app.cell
def _(batch_size_ui, training_token_file, window_size_ui):
    with open(training_token_file, 'r') as f:
        token_str = f.read()

    tokens = [int(tok) for tok in token_str.split()]

    train_dataset = TokenWindowDatasetV1(
        tokens, window_size=window_size_ui.value, batch_size=batch_size_ui.value
    )

    print(f"dataset size {len(train_dataset)}")
    return (train_dataset,)


@app.cell
def _(embed_dim_ui, num_heads_ui, num_layers_ui, token_size_ui):
    # embed_dim must be divisible by num_heads for multi-head attention;
    # the two sliders move independently, so round up rather than assert.
    effective_embed_dim = round_up_to_multiple(embed_dim_ui.value, num_heads_ui.value)
    model = GPTV2(token_size_ui.value, effective_embed_dim, num_heads_ui.value, num_layers_ui.value)
    print(
        f"model: vocab={token_size_ui.value} embed_dim={effective_embed_dim} "
        f"heads={num_heads_ui.value} layers={num_layers_ui.value}"
    )
    return (model,)


@app.cell
def _(learning_rate_ui, weight_decay_ui):
    optimizer = optimizers.AdamW(learning_rate_ui.value, weight_decay=weight_decay_ui.value)
    return (optimizer,)


@app.function
def compute_loss(model, x, y):
    out = model(x)
    return mx.mean(nn.losses.cross_entropy(out, y))


@app.cell
def _(model, optimizer):
    # state declares which arrays the compiled step is allowed to mutate;
    # without it mx.compile freezes model/optimizer arrays at trace time
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grad = nn.value_and_grad(model, compute_loss)(model, x, y)
        optimizer.update(model, grad)
        return loss

    return state, train_step


@app.function
def train_epoch(model, train_iter, train_step, state):
    model.train(True)
    losses = 0.
    count = 0
    for batch in tqdm(train_iter):
        x = batch[:,:-1]
        y = batch[:,1:]
        loss = train_step(x, y)
        mx.eval(state)
        losses += loss.item()
        count += 1
    return losses / max(count, 1)


@app.function
def train(model, train_iter, train_step, state, epochs):
    for epoch in range(epochs):
        l = train_epoch(model, train_iter, train_step, state)
        print(f"{epoch}: {l}")


@app.cell
def _(epochs_ui, mo, model, state, train_button, train_dataset, train_step):
    mo.stop(
        not train_button.value,
        mo.md("Click **Train** above to start training. The model is randomly initialized until then."),
    )
    train(model, train_dataset, train_step, state, epochs_ui.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### sampling
    """)
    return


@app.function
def generate(model, merge_table, prompt, max_new_tokens=200, temperature=1.0, top_k=None, seed=None):
    model.train(False)
    if seed is not None:
        mx.random.seed(seed)

    ids = mx.array(tokenize(prompt, merge_table), dtype=mx.int32)[None, :]

    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :] / temperature
        if top_k is not None:
            threshold = mx.min(mx.topk(logits, top_k, axis=-1), axis=-1, keepdims=True)
            logits = mx.where(logits < threshold, -float("inf"), logits)
        next_id = mx.random.categorical(logits)
        ids = mx.concatenate([ids, next_id[:, None]], axis=1)
        mx.eval(ids)

    return decode_tokens(ids[0].tolist(), merge_table)


@app.cell
def _(mo):
    prompt_input = mo.ui.text_area(value="ROMEO:", label="Prompt")
    max_new_tokens_input = mo.ui.slider(10, 500, value=200, step=10, label="Max new tokens")
    temperature_input = mo.ui.slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
    top_k_input = mo.ui.slider(0, 100, value=40, step=5, label="Top-k (0 = disabled)")
    generate_button = mo.ui.run_button(label="Generate")
    mo.vstack([prompt_input, max_new_tokens_input, temperature_input, top_k_input, generate_button])
    return (
        generate_button,
        max_new_tokens_input,
        prompt_input,
        temperature_input,
        top_k_input,
    )


@app.cell
def _(
    generate_button,
    max_new_tokens_input,
    merge_table,
    mo,
    model,
    prompt_input,
    temperature_input,
    top_k_input,
):
    mo.stop(not generate_button.value, mo.md("Click **Generate** to sample from the trained model."))

    _top_k = top_k_input.value if top_k_input.value > 0 else None
    generated_text = generate(
        model,
        merge_table,
        prompt_input.value,
        max_new_tokens=max_new_tokens_input.value,
        temperature=temperature_input.value,
        top_k=_top_k,
    )
    mo.md(f"**Prompt:** {prompt_input.value}\n\n**Generated:**\n\n```\n{generated_text}\n```")
    return


@app.cell
def _():
    ### save model
    return


@app.cell
def _():
    model_filename = 'models/tinyshakespear_transformer_v1.safetensors'
    return (model_filename,)


@app.cell
def _(model, model_filename):
    model.save_weights(model_filename)
    return


if __name__ == "__main__":
    app.run()
