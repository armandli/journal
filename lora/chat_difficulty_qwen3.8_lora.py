import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import math
    import random
    import re
    import shlex
    import shutil
    import subprocess
    import time
    import types
    from pathlib import Path

    import altair as alt
    import matplotlib.pyplot as plt
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    import polars as pl
    from mlx.utils import tree_flatten, tree_unflatten


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # QLoRA on Qwen3.8-27B-4bit for Chat Difficulty Prediction

    **Goal.** Fine-tune the 4-bit quantized `mlx-community/Qwen3.8-27B-4bit` to predict the
    **difficulty rating** of a chat prompt. Dataset: [`agentlans/chat-difficulty`](https://huggingface.co/datasets/agentlans/chat-difficulty)
    — ~100k multilingual human prompts labelled with an integer difficulty rating from 1 to 10.
    Framework: [`mlx-lm`](https://github.com/ml-explore/mlx-lm) on Apple Silicon.

    **LoRA vs QLoRA.** Standard LoRA attaches small low-rank adapters (`A @ B`) to the frozen
    base-model linear layers; only the adapters are trained. **QLoRA** is the same technique
    applied over a quantised (here 4-bit) frozen base: training adapters while the backbone
    stays quantised. Because we load `Qwen3.8-27B-4bit`, every run here is QLoRA — the
    matmul reaches full precision only inside the small `A/B` factors and when de-quantising
    the base weights on-the-fly for the forward pass.

    **Task framing.** Given one user prompt, the model must output a single integer in
    `[1, 10]`. We format each example as a chat with a fixed system prompt and an assistant
    answer that is just the stringified rating; `mlx-lm` ingests the `{"messages": [...]}`
    format directly via `ChatDataset`.

    **Environment.** This notebook uses the project venv at `env/` (Python 3.14, mlx-lm
    0.32.2). The alternative `env312/` is a Python 3.12 fallback — not needed here.

    **Outline.**

    1. Title & Research Goal (this cell)
    2. Data Exploration
    3. Dataset Creation (chat-formatted `train.jsonl` / `valid.jsonl` / `test.jsonl`)
    4. Model Definition (load Qwen3.8-27B-4bit, apply LoRA layers)
    5. Training (configurable LoRA params; in-notebook via `mlx_lm.tuner.train`, or CLI)
    6. Hyperparameter Search (optional; expensive on 27B)
    7. Evaluation: Vanilla vs LoRA (the key comparison requested)
    8. Results
    9. Save Trained Model (adapters + optional fuse via `mlx_lm.fuse`)
    10. Load the Trained Model & Rate New Prompts (inference on the trained task)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Paths & Config
    """)
    return


@app.function
def resolve_data_dir() -> Path:
    """Return `/Users/<user>/data/chat-difficulty` resolved from this notebook's location.

    Notebook lives at `<repo>/lora/<file>.py`, so parent.parent.parent is the user's
    home-adjacent root. We put raw + formatted data there to keep the git repo clean.
    """
    base = Path(__file__).resolve().parent.parent.parent / "data" / "chat-difficulty"
    base.mkdir(parents=True, exist_ok=True)
    return base


@app.function
def resolve_models_dir() -> Path:
    """Return `<repo>/models` — adapters, fused checkpoints, base-model snapshots all go here."""
    base = Path(__file__).resolve().parent.parent / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


@app.cell
def _(mo):
    data_dir = resolve_data_dir()
    models_dir = resolve_models_dir()
    mo.md(
        f"""
        - **Data directory:** `{data_dir}`
        - **Models directory:** `{models_dir}`

        Both are created on first run; nothing here lives inside the git tree.
        """
    )
    return data_dir, models_dir


@app.cell
def _():
    DATASET_ID = "agentlans/chat-difficulty"
    BASE_MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
    SYSTEM_PROMPT = (
        "You are a difficulty rater. Given a chat prompt, respond with a single integer "
        "between 1 (very easy) and 10 (very hard) that reflects how difficult it is to "
        "answer well. Reply with ONLY the integer, no words, no punctuation."
    )
    RATING_MIN = 1
    RATING_MAX = 10
    return BASE_MODEL_ID, DATASET_ID, RATING_MAX, RATING_MIN, SYSTEM_PROMPT


@app.cell
def _(BASE_MODEL_ID, DATASET_ID, mo):
    mo.md(
        f"""
        - Base model repo: `{BASE_MODEL_ID}` (4-bit quantised ⇒ **QLoRA**)
        - Dataset repo: `{DATASET_ID}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Data Exploration
    """)
    return


@app.function
def download_raw_dataset(dataset_id: str, cache_dir: Path) -> "pl.DataFrame":
    """Download the HF dataset once and return a polars DataFrame of the single train split."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, cache_dir=str(cache_dir / "hf_cache"))
    train = ds["train"]
    rows = train.to_list()
    for r in rows:
        convo = r.get("conversations") or []
        user_turns = [t["value"] for t in convo if t.get("from") in ("human", "user")]
        r["user_prompt"] = user_turns[0] if user_turns else ""
        r["prompt_chars"] = len(r["user_prompt"])
    df = pl.DataFrame(
        rows,
        schema={
            "id": pl.Int64,
            "conversations": pl.Object,
            "rating": pl.Int64,
            "language": pl.String,
            "source": pl.String,
            "user_prompt": pl.String,
            "prompt_chars": pl.Int64,
        },
        strict=False,
    )
    return df


@app.cell
def _(DATASET_ID, data_dir, mo):
    load_data_btn = mo.ui.run_button(label="Download / Load dataset", kind="primary")
    mo.vstack(
        [
            mo.md(
                f"""
                Click to download `{DATASET_ID}` into `{data_dir}` (via the `datasets`
                library — ~60 MB; cached across runs).
                """
            ),
            load_data_btn,
        ]
    )
    return (load_data_btn,)


@app.cell
def _(DATASET_ID, data_dir, load_data_btn, mo):
    raw_df: "pl.DataFrame | None" = None
    if not load_data_btn.value:
        mo.output.replace(mo.md("_Click **Download / Load dataset** above to begin._"))
    else:
        raw_df = download_raw_dataset(DATASET_ID, data_dir)
        mo.output.replace(
            mo.md(f"Loaded **{len(raw_df):,}** rows into memory from `{DATASET_ID}`.")
        )
    return (raw_df,)


@app.function
def summarize_schema(df: "pl.DataFrame") -> str:
    rows = []
    for name, dt in df.schema.items():
        rows.append(f"| `{name}` | `{dt}` |")
    return "\n".join(["| column | dtype |", "|---|---|", *rows])


@app.cell
def _(mo, raw_df: "pl.DataFrame | None"):
    if raw_df is None:
        _out = mo.md("_Load the dataset first._")
    else:
        _out = mo.md(
            f"""
            ### Schema

            {summarize_schema(raw_df.drop("conversations"))}

            `conversations` is a `List[Struct{{from, value}}]` of 1 user turn and 1
            assistant turn. We only use the first user turn (`user_prompt`) and the
            `rating` label.
            """
        )
    _out
    return


@app.cell
def _(mo, raw_df: "pl.DataFrame | None"):
    if raw_df is None:
        _out = mo.md("_Load the dataset first._")
    else:
        sample_tbl = (
            raw_df.select(
                pl.col("id"),
                pl.col("rating"),
                pl.col("language"),
                pl.col("source"),
                pl.col("prompt_chars"),
                pl.col("user_prompt").str.slice(0, 240).alias("user_prompt_preview"),
            )
            .head(5)
            .to_pandas()
        )
        _out = mo.vstack(
            [mo.md("### 5 sample rows"), mo.ui.table(sample_tbl, selection=None)]
        )
    _out
    return


@app.function
def plot_rating_distribution(df: "pl.DataFrame") -> alt.Chart:
    agg = (
        df.group_by("rating")
        .agg(pl.len().alias("count"))
        .sort("rating")
    )
    return (
        alt.Chart(agg.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("rating:O", title="Difficulty rating"),
            y=alt.Y("count:Q", title="# examples"),
            tooltip=["rating", "count"],
        )
        .properties(title="Difficulty rating distribution", width=500, height=280)
    )


@app.function
def plot_language_distribution(df: "pl.DataFrame", top_k: int = 10) -> alt.Chart:
    agg = (
        df.group_by("language")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_k)
    )
    return (
        alt.Chart(agg.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="# examples"),
            y=alt.Y("language:N", sort="-x", title="Language (ISO)"),
            tooltip=["language", "count"],
        )
        .properties(title=f"Top {top_k} languages", width=500, height=280)
    )


@app.function
def plot_prompt_length_distribution(df: "pl.DataFrame") -> alt.Chart:
    clipped = df.with_columns(
        pl.col("prompt_chars").clip(0, 8000).alias("prompt_chars_clipped")
    )
    return (
        alt.Chart(clipped.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("prompt_chars_clipped:Q", bin=alt.Bin(maxbins=40), title="Prompt length (chars, clipped at 8k)"),
            y=alt.Y("count():Q", title="# examples"),
            tooltip=[alt.Tooltip("count():Q")],
        )
        .properties(title="Prompt length distribution (chars)", width=500, height=280)
    )


@app.cell
def _(mo, raw_df: "pl.DataFrame | None"):
    if raw_df is None:
        _out = mo.md("_Load the dataset first._")
    else:
        _out = mo.ui.altair_chart(plot_rating_distribution(raw_df))
    _out
    return


@app.cell
def _(mo, raw_df: "pl.DataFrame | None"):
    if raw_df is None:
        _out = mo.md("_Load the dataset first._")
    else:
        _out = mo.ui.altair_chart(plot_language_distribution(raw_df))
    _out
    return


@app.cell
def _(mo, raw_df: "pl.DataFrame | None"):
    if raw_df is None:
        _out = mo.md("_Load the dataset first._")
    else:
        _out = mo.ui.altair_chart(plot_prompt_length_distribution(raw_df))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Dataset Creation — chat-formatted JSONL for `mlx-lm`
    """)
    return


@app.cell
def _(mo):
    max_train_ui = mo.ui.number(value=4000, start=100, label="Max train examples")
    max_valid_ui = mo.ui.number(value=500, start=50, label="Max valid examples")
    max_test_ui = mo.ui.number(value=500, start=50, label="Max test examples")
    max_chars_ui = mo.ui.number(
        value=2000,
        start=200,
        label="Clip prompts longer than N chars (helps max_seq_length)",
    )
    split_seed_ui = mo.ui.number(value=42, label="Split seed")
    build_jsonl_btn = mo.ui.run_button(label="Write train/valid/test JSONL", kind="primary")
    mo.vstack(
        [
            mo.md(
                """
                Chat-difficulty ships one big `train` split, so we build train / valid / test
                ourselves with a stratified-ish random split (seeded). Each example becomes:

                ```json
                {"messages": [{"role":"system","content":"..."},
                              {"role":"user","content":"<prompt>"},
                              {"role":"assistant","content":"<rating>"}]}
                ```

                `max_chars` clips extremely long prompts; this keeps `max_seq_length` in training
                tractable without having to drop examples. The subsample caps keep the smoke test
                feasible on 27B — raise them when you want to run the full job.
                """
            ),
            mo.hstack([max_train_ui, max_valid_ui, max_test_ui]),
            mo.hstack([max_chars_ui, split_seed_ui]),
            build_jsonl_btn,
        ]
    )
    return (
        build_jsonl_btn,
        max_chars_ui,
        max_test_ui,
        max_train_ui,
        max_valid_ui,
        split_seed_ui,
    )


@app.function
def format_as_chat_messages(
    user_prompt: str,
    rating: int,
    system_prompt: str,
    max_chars: int,
) -> dict:
    if max_chars and len(user_prompt) > max_chars:
        user_prompt = user_prompt[:max_chars]
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(int(rating))},
        ]
    }


@app.function
def split_indices(
    n: int,
    seed: int,
    train_n: int,
    valid_n: int,
    test_n: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    train = idx[:train_n]
    valid = idx[train_n : train_n + valid_n]
    test = idx[train_n + valid_n : train_n + valid_n + test_n]
    return train, valid, test


@app.function
def write_split_jsonl(
    df: "pl.DataFrame",
    out_path: Path,
    system_prompt: str,
    indices: list[int],
    max_chars: int,
) -> int:
    user_prompts = df.get_column("user_prompt").to_list()
    ratings = df.get_column("rating").to_list()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for i in indices:
            prompt = user_prompts[i]
            if not prompt:
                continue
            msg = format_as_chat_messages(prompt, ratings[i], system_prompt, max_chars)
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            n += 1
    return n


@app.cell
def _(
    SYSTEM_PROMPT,
    build_jsonl_btn,
    data_dir,
    max_chars_ui,
    max_test_ui,
    max_train_ui,
    max_valid_ui,
    mo,
    raw_df: "pl.DataFrame | None",
    split_seed_ui,
):
    jsonl_paths: dict[str, Path] | None = None
    jsonl_counts: dict[str, int] | None = None
    if raw_df is None:
        mo.output.replace(mo.md("_Load the dataset first._"))
    elif not build_jsonl_btn.value:
        mo.output.replace(
            mo.md("Choose split sizes and click **Write train/valid/test JSONL**.")
        )
    else:
        _tr, _va, _te = split_indices(
            n=len(raw_df),
            seed=int(split_seed_ui.value),
            train_n=int(max_train_ui.value),
            valid_n=int(max_valid_ui.value),
            test_n=int(max_test_ui.value),
        )
        _paths = {
            "train": data_dir / "train.jsonl",
            "valid": data_dir / "valid.jsonl",
            "test": data_dir / "test.jsonl",
        }
        _counts = {}
        for _name, _indices in zip(("train", "valid", "test"), (_tr, _va, _te)):
            _counts[_name] = write_split_jsonl(
                raw_df,
                _paths[_name],
                SYSTEM_PROMPT,
                _indices,
                int(max_chars_ui.value),
            )
        jsonl_paths = _paths
        jsonl_counts = _counts
        mo.output.replace(
            mo.md(
                "**JSONL written:**\n\n"
                + "\n".join(
                    f"- `{k}`: {v} examples → `{_paths[k]}`" for k, v in _counts.items()
                )
            )
        )
    return (jsonl_paths,)


@app.function
def read_first_jsonl_record(path: Path) -> dict:
    with open(path) as f:
        return json.loads(f.readline())


@app.cell
def _(jsonl_paths: dict[str, Path] | None, mo):
    if jsonl_paths is None:
        _out = mo.md("_Build JSONL first._")
    else:
        _rec = read_first_jsonl_record(jsonl_paths["train"])
        _out = mo.vstack(
            [
                mo.md("### Example formatted record (first train row)"),
                mo.ui.table(
                    [
                        {"role": m["role"], "content": m["content"][:500]}
                        for m in _rec["messages"]
                    ],
                    selection=None,
                ),
            ]
        )
    _out
    return


@app.function
def jsonl_token_length_stats(
    path: Path, tokenizer, max_rows: int = 500
) -> dict:
    lengths: list[int] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rec = json.loads(line)
            toks = tokenizer.apply_chat_template(rec["messages"], return_dict=False)
            lengths.append(len(toks))
    arr = np.array(lengths, dtype=np.int64)
    return {
        "n_sampled": int(len(arr)),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Model Definition — load Qwen3.8-27B-4bit + LoRA
    """)
    return


@app.cell
def _(BASE_MODEL_ID, mo):
    base_model_repo_ui = mo.ui.text(
        value=BASE_MODEL_ID,
        label="Base model (HF repo id or local path)",
        full_width=True,
    )
    local_base_path_ui = mo.ui.text(
        value="",
        label="Local base-model snapshot (optional)",
        full_width=True,
    )
    download_base_btn = mo.ui.run_button(label="Snapshot base model to models/", kind="primary")
    load_base_btn = mo.ui.run_button(label="Load base model + tokenizer", kind="primary")
    mo.vstack(
        [
            mo.md(
                """
                **Base model.** You can either (a) load straight from the HF hub id — mlx-lm's
                default — or (b) pre-snapshot the weights into `models/` and point `local` at
                that path. Pre-snapshotting is recommended because the 27B 4-bit shards are
                ~15 GB and you only want to download them once.

                `Load base model + tokenizer` is heavy (~15 GB resident, several seconds); it is
                gated behind the run button.
                """
            ),
            base_model_repo_ui,
            local_base_path_ui,
            mo.hstack([download_base_btn, load_base_btn]),
        ]
    )
    return (
        base_model_repo_ui,
        download_base_btn,
        load_base_btn,
        local_base_path_ui,
    )


@app.function
def snapshot_base_model(repo_id: str, dest: Path) -> Path:
    """Download the full mlx-community weights into `dest` and return the path."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    return Path(local)


@app.cell
def _(base_model_repo_ui, download_base_btn, mo, models_dir):
    snapshot_path: Path | None = None
    if not download_base_btn.value:
        mo.output.replace(
            mo.md(
                "Optional: click **Snapshot base model to models/** to pre-download "
                "the 4-bit weights before you load them."
            )
        )
    else:
        _dest = models_dir / base_model_repo_ui.value.replace("/", "__")
        snapshot_path = snapshot_base_model(base_model_repo_ui.value, _dest)
        mo.output.replace(mo.md(f"Snapshotted to `{snapshot_path}`."))
    return (snapshot_path,)


@app.function
def load_base_model_and_tokenizer(repo_or_path: str):
    """Thin wrapper around `mlx_lm.load`. Returns `(model, tokenizer)`."""
    from mlx_lm import load as mlx_lm_load

    model, tokenizer = mlx_lm_load(repo_or_path)
    return model, tokenizer


@app.function
def get_total_params(model) -> int:
    try:
        from mlx_lm.utils import get_total_parameters

        return int(get_total_parameters(model))
    except Exception:
        return int(sum(v.size for _, v in tree_flatten(model.parameters())))


@app.function
def get_trainable_params(model) -> int:
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


@app.cell
def _(
    base_model_repo_ui,
    load_base_btn,
    local_base_path_ui,
    mo,
    snapshot_path: Path | None,
):
    base_model = None
    tokenizer = None
    model_source = None
    if not load_base_btn.value:
        mo.output.replace(mo.md("_Click **Load base model + tokenizer** to continue._"))
    else:
        if local_base_path_ui.value.strip():
            model_source = local_base_path_ui.value.strip()
        elif snapshot_path is not None:
            model_source = str(snapshot_path)
        else:
            model_source = base_model_repo_ui.value
        base_model, tokenizer = load_base_model_and_tokenizer(model_source)
        mo.output.replace(
            mo.md(
                f"**Loaded** `{model_source}`. "
                f"Total params: `{get_total_params(base_model):,}`."
            )
        )
    return base_model, model_source, tokenizer


@app.function
def summarize_architecture(model) -> str:
    try:
        n_layers = len(model.layers)
    except Exception:
        n_layers = -1
    cfg = getattr(model, "args", None) or getattr(model, "config", None)
    hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
    heads = getattr(cfg, "num_attention_heads", None) if cfg is not None else None
    kv_heads = getattr(cfg, "num_key_value_heads", None) if cfg is not None else None
    vocab = getattr(cfg, "vocab_size", None) if cfg is not None else None
    quant_bits = None
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            quant_bits = int(getattr(m, "bits", 4))
            break
    return (
        "| field | value |\n|---|---|\n"
        f"| layers (`model.layers`) | {n_layers} |\n"
        f"| hidden size | {hidden} |\n"
        f"| attention heads | {heads} |\n"
        f"| KV heads | {kv_heads} |\n"
        f"| vocab size | {vocab} |\n"
        f"| quantization bits | {quant_bits} |\n"
    )


@app.cell
def _(base_model, mo):
    if base_model is None:
        _out = mo.md("_Load the base model first._")
    else:
        _out = mo.md("### Architecture summary\n\n" + summarize_architecture(base_model))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### LoRA / DoRA configuration
    """)
    return


@app.cell
def _(mo):
    fine_tune_type_ui = mo.ui.dropdown(
        options=["lora", "dora"],
        value="lora",
        label="Fine-tune type",
    )
    num_lora_layers_ui = mo.ui.number(value=8, start=1, label="# LoRA layers (from top)")
    rank_ui = mo.ui.dropdown(
        options={"4": 4, "8": 8, "16": 16, "32": 32, "64": 64},
        value="8",
        label="LoRA rank",
    )
    scale_ui = mo.ui.dropdown(
        options={"4.0": 4.0, "10.0": 10.0, "20.0": 20.0, "32.0": 32.0},
        value="20.0",
        label="LoRA scale (α)",
    )
    dropout_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "0.05": 0.05, "0.1": 0.1},
        value="0.0",
        label="LoRA dropout",
    )
    keys_text_ui = mo.ui.text(
        value="",
        label="Target keys (comma-sep; blank = auto-detect all linear)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.hstack([fine_tune_type_ui, num_lora_layers_ui, rank_ui]),
            mo.hstack([scale_ui, dropout_ui]),
            keys_text_ui,
        ]
    )
    return (
        dropout_ui,
        fine_tune_type_ui,
        keys_text_ui,
        num_lora_layers_ui,
        rank_ui,
        scale_ui,
    )


@app.function
def build_lora_config(
    rank: int,
    scale: float,
    dropout: float,
    keys_text: str,
) -> dict:
    cfg = {"rank": int(rank), "scale": float(scale), "dropout": float(dropout)}
    if keys_text.strip():
        cfg["keys"] = [k.strip() for k in keys_text.split(",") if k.strip()]
    return cfg


@app.function
def apply_lora_to_model(
    model,
    num_layers: int,
    lora_config: dict,
    use_dora: bool,
) -> None:
    """Mutates `model` in place: freezes everything, replaces N layers' linears with LoRA/DoRA."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(
        model=model,
        num_layers=int(num_layers),
        config=lora_config,
        use_dora=use_dora,
    )


@app.cell
def _(mo):
    apply_lora_btn = mo.ui.run_button(label="Apply LoRA layers to loaded model", kind="primary")
    mo.vstack(
        [
            mo.md(
                "Applies LoRA to the already-loaded base model in place. If you re-apply "
                "with different settings, reload the base model first."
            ),
            apply_lora_btn,
        ]
    )
    return (apply_lora_btn,)


@app.cell
def _(
    apply_lora_btn,
    base_model,
    dropout_ui,
    fine_tune_type_ui,
    keys_text_ui,
    mo,
    num_lora_layers_ui,
    rank_ui,
    scale_ui,
):
    lora_applied = False
    lora_config_used: dict | None = None
    if base_model is None:
        mo.output.replace(mo.md("_Load the base model first._"))
    elif not apply_lora_btn.value:
        mo.output.replace(mo.md("_Click **Apply LoRA layers to loaded model** to attach adapters._"))
    else:
        lora_config_used = build_lora_config(
            rank=rank_ui.value,
            scale=scale_ui.value,
            dropout=dropout_ui.value,
            keys_text=keys_text_ui.value,
        )
        apply_lora_to_model(
            model=base_model,
            num_layers=int(num_lora_layers_ui.value),
            lora_config=lora_config_used,
            use_dora=(fine_tune_type_ui.value == "dora"),
        )
        lora_applied = True
        _trainable = get_trainable_params(base_model)
        _total = get_total_params(base_model)
        _pct = (_trainable * 100.0) / max(_total, 1)
        mo.output.replace(
            mo.md(
                f"**{fine_tune_type_ui.value.upper()} applied.** Trainable: "
                f"`{_trainable:,}` ({_pct:.3f}% of `{_total:,}`). "
                f"Config: `{lora_config_used}`."
            )
        )
    return lora_applied, lora_config_used


@app.cell
def _(jsonl_paths: dict[str, Path] | None, mo, tokenizer):
    if tokenizer is None or jsonl_paths is None:
        _out = mo.md("_Load model and build JSONL first; token stats skipped._")
    else:
        _stats = {k: jsonl_token_length_stats(v, tokenizer) for k, v in jsonl_paths.items()}
        _rows = [{"split": s, **st} for s, st in _stats.items()]
        _out = mo.vstack(
            [mo.md("### Token-length stats per split (first 500 rows each)"), mo.ui.table(_rows, selection=None)]
        )
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Training
    """)
    return


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"5e-6": 5e-6, "1e-5": 1e-5, "2e-5": 2e-5, "5e-5": 5e-5, "1e-4": 1e-4},
        value="1e-5",
        label="Learning rate",
    )
    batch_size_ui = mo.ui.dropdown(
        options=[1, 2, 4, 8],
        value=1,
        label="Batch size",
    )
    iters_ui = mo.ui.slider(50, 2000, value=200, step=50, label="Iterations")
    max_seq_len_ui = mo.ui.dropdown(
        options=[512, 1024, 2048, 4096],
        value=1024,
        label="Max seq length",
    )
    steps_per_eval_ui = mo.ui.number(value=50, start=5, label="Steps per eval")
    save_every_ui = mo.ui.number(value=100, start=10, label="Save every N steps")
    grad_checkpoint_ui = mo.ui.checkbox(value=True, label="Gradient checkpointing")
    grad_accum_ui = mo.ui.number(value=1, start=1, label="Grad accumulation steps")
    seed_ui = mo.ui.number(value=0, label="Training seed")
    mask_prompt_ui = mo.ui.checkbox(value=True, label="Mask prompt in loss (train on completion only)")
    adapter_name_ui = mo.ui.text(
        value="chat_difficulty_qwen3.8_27b_lora_v1",
        label="Adapter folder name (saved into models/)",
        full_width=True,
    )
    train_btn = mo.ui.run_button(label="Train", kind="primary")
    mo.vstack(
        [
            mo.md("### Training hyperparameters"),
            mo.hstack([lr_ui, batch_size_ui, iters_ui]),
            mo.hstack([max_seq_len_ui, steps_per_eval_ui, save_every_ui]),
            mo.hstack([grad_checkpoint_ui, grad_accum_ui, seed_ui]),
            mo.hstack([mask_prompt_ui, adapter_name_ui]),
            train_btn,
        ]
    )
    return (
        adapter_name_ui,
        batch_size_ui,
        grad_accum_ui,
        grad_checkpoint_ui,
        iters_ui,
        lr_ui,
        mask_prompt_ui,
        max_seq_len_ui,
        save_every_ui,
        seed_ui,
        steps_per_eval_ui,
        train_btn,
    )


@app.function
def build_cli_command(
    model_source: str,
    data_dir: Path,
    adapter_dir: Path,
    fine_tune_type: str,
    lora_config: dict,
    num_layers: int,
    lr: float,
    batch_size: int,
    iters: int,
    max_seq_length: int,
    steps_per_eval: int,
    save_every: int,
    grad_checkpoint: bool,
    grad_accum: int,
    seed: int,
    mask_prompt: bool,
) -> str:
    """Build the equivalent `mlx_lm.lora` CLI invocation for the current UI settings."""
    parts = [
        "mlx_lm.lora",
        "--train",
        f"--model {shlex.quote(model_source)}",
        f"--data {shlex.quote(str(data_dir))}",
        f"--adapter-path {shlex.quote(str(adapter_dir))}",
        f"--fine-tune-type {fine_tune_type}",
        f"--num-layers {num_layers}",
        f"--batch-size {batch_size}",
        f"--iters {iters}",
        f"--learning-rate {lr}",
        f"--max-seq-length {max_seq_length}",
        f"--steps-per-eval {steps_per_eval}",
        f"--save-every {save_every}",
        f"--grad-accumulation-steps {grad_accum}",
        f"--seed {seed}",
    ]
    if grad_checkpoint:
        parts.append("--grad-checkpoint")
    if mask_prompt:
        parts.append("--mask-prompt")
    cmd = " \\\n  ".join(parts)
    cfg_hint = (
        f"\n\n# Note: rank={lora_config['rank']}, scale={lora_config['scale']}, "
        f"dropout={lora_config['dropout']} need to go into a YAML config "
        f"(see `mlx_lm.lora --help`) because the CLI reads lora_parameters from YAML only."
    )
    return cmd + cfg_hint


@app.cell
def _(
    adapter_name_ui,
    base_model_repo_ui,
    batch_size_ui,
    data_dir,
    fine_tune_type_ui,
    grad_accum_ui,
    grad_checkpoint_ui,
    iters_ui,
    local_base_path_ui,
    lora_config_used: dict | None,
    lr_ui,
    mask_prompt_ui,
    max_seq_len_ui,
    mo,
    model_source,
    models_dir,
    num_lora_layers_ui,
    save_every_ui,
    seed_ui,
    steps_per_eval_ui,
):
    _effective_source = (
        model_source
        or local_base_path_ui.value.strip()
        or base_model_repo_ui.value
    )
    cli_command = build_cli_command(
        model_source=_effective_source,
        data_dir=data_dir,
        adapter_dir=models_dir / adapter_name_ui.value,
        fine_tune_type=fine_tune_type_ui.value,
        lora_config=lora_config_used
        or {"rank": 8, "scale": 20.0, "dropout": 0.0},
        num_layers=int(num_lora_layers_ui.value),
        lr=float(lr_ui.value),
        batch_size=int(batch_size_ui.value),
        iters=int(iters_ui.value),
        max_seq_length=int(max_seq_len_ui.value),
        steps_per_eval=int(steps_per_eval_ui.value),
        save_every=int(save_every_ui.value),
        grad_checkpoint=bool(grad_checkpoint_ui.value),
        grad_accum=int(grad_accum_ui.value),
        seed=int(seed_ui.value),
        mask_prompt=bool(mask_prompt_ui.value),
    )
    mo.vstack(
        [
            mo.md("### Equivalent CLI (run from the project root)"),
            mo.md(f"```bash\n{cli_command}\n```"),
            mo.md(
                "Prefer the CLI for long jobs — the notebook training cell exists for small "
                "runs and smoke tests. The CLI writes exactly the same adapter files that "
                "Section 9 and the evaluation cells load."
            ),
        ]
    )
    return


@app.class_definition
class LossHistoryCallbackV1:
    """Collects train/val loss history in memory; swaps `mo.output.replace` for live progress."""

    def __init__(self, output_sink=None):
        self.train_history: list[dict] = []
        self.val_history: list[dict] = []
        self.output_sink = output_sink

    def on_train_loss_report(self, info: dict) -> None:
        self.train_history.append(dict(info))
        if self.output_sink is not None:
            self.output_sink(
                f"iter {info.get('iteration')} — train_loss={info.get('train_loss'):.4f} "
                f"tok/s={info.get('tokens_per_second', 0.0):.1f} "
                f"peak_mem={info.get('peak_memory', 0.0):.2f} GB"
            )

    def on_val_loss_report(self, info: dict) -> None:
        self.val_history.append(dict(info))
        if self.output_sink is not None:
            self.output_sink(
                f"iter {info.get('iteration')} — val_loss={info.get('val_loss'):.4f}"
            )


@app.function
def load_jsonl_as_mlx_datasets(data_dir: Path, tokenizer, mask_prompt: bool):
    """Load local {train,valid,test}.jsonl with mlx-lm's dataset loader."""
    from mlx_lm.tuner.datasets import load_local_dataset

    ns = types.SimpleNamespace(mask_prompt=mask_prompt)
    return load_local_dataset(data_dir, tokenizer, ns)


@app.function
def run_lora_training(
    model,
    tokenizer,
    data_dir: Path,
    adapter_dir: Path,
    lr: float,
    batch_size: int,
    iters: int,
    max_seq_length: int,
    steps_per_eval: int,
    save_every: int,
    grad_checkpoint: bool,
    grad_accum: int,
    seed: int,
    mask_prompt: bool,
    output_sink,
) -> "LossHistoryCallbackV1":
    """Call mlx-lm's `train` directly with a loss-history callback. Writes adapters to `adapter_dir`."""
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train

    mx.random.seed(int(seed))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"
    train_set, valid_set, _ = load_jsonl_as_mlx_datasets(data_dir, tokenizer, mask_prompt)
    args = TrainingArgs(
        batch_size=int(batch_size),
        iters=int(iters),
        val_batches=10,
        steps_per_report=max(1, int(iters) // 20),
        steps_per_eval=int(steps_per_eval),
        steps_per_save=int(save_every),
        max_seq_length=int(max_seq_length),
        adapter_file=str(adapter_file),
        grad_checkpoint=bool(grad_checkpoint),
        grad_accumulation_steps=int(grad_accum),
    )
    optimizer = optim.AdamW(learning_rate=float(lr))
    callback = LossHistoryCallbackV1(output_sink=output_sink)
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=args,
        training_callback=callback,
    )
    return callback


@app.function
def write_adapter_config(
    adapter_dir: Path,
    model_source: str,
    data_dir: Path,
    fine_tune_type: str,
    lora_config: dict,
    num_layers: int,
    batch_size: int,
    iters: int,
    lr: float,
    max_seq_length: int,
    mask_prompt: bool,
    seed: int,
) -> Path:
    """mlx-lm's load_adapters() reads adapter_config.json from the adapter directory."""
    cfg = {
        "model": model_source,
        "data": str(data_dir),
        "fine_tune_type": fine_tune_type,
        "lora_parameters": lora_config,
        "num_layers": int(num_layers),
        "batch_size": int(batch_size),
        "iters": int(iters),
        "learning_rate": float(lr),
        "max_seq_length": int(max_seq_length),
        "mask_prompt": bool(mask_prompt),
        "seed": int(seed),
    }
    out = adapter_dir / "adapter_config.json"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    return out


@app.cell
def _(
    adapter_name_ui,
    base_model,
    batch_size_ui,
    data_dir,
    fine_tune_type_ui,
    grad_accum_ui,
    grad_checkpoint_ui,
    iters_ui,
    jsonl_paths: dict[str, Path] | None,
    lora_applied,
    lora_config_used: dict | None,
    lr_ui,
    mask_prompt_ui,
    max_seq_len_ui,
    mo,
    model_source,
    models_dir,
    num_lora_layers_ui,
    save_every_ui,
    seed_ui,
    steps_per_eval_ui,
    tokenizer,
    train_btn,
):
    train_losses: list[dict] = []
    val_losses: list[dict] = []
    trained_adapter_dir: Path | None = None
    if not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to start the LoRA training loop."))
    elif base_model is None or tokenizer is None:
        mo.output.replace(mo.md("_Load base model + tokenizer first._"))
    elif not lora_applied or lora_config_used is None:
        mo.output.replace(mo.md("_Apply LoRA layers first._"))
    elif jsonl_paths is None:
        mo.output.replace(mo.md("_Build train/valid/test JSONL first._"))
    else:
        _progress_lines: list[str] = []

        def _sink(line: str) -> None:
            _progress_lines.append(line)
            mo.output.replace(
                mo.md("### Training progress\n\n```\n" + "\n".join(_progress_lines[-30:]) + "\n```")
            )

        trained_adapter_dir = models_dir / adapter_name_ui.value
        write_adapter_config(
            adapter_dir=trained_adapter_dir,
            model_source=model_source or "",
            data_dir=data_dir,
            fine_tune_type=fine_tune_type_ui.value,
            lora_config=lora_config_used,
            num_layers=int(num_lora_layers_ui.value),
            batch_size=int(batch_size_ui.value),
            iters=int(iters_ui.value),
            lr=float(lr_ui.value),
            max_seq_length=int(max_seq_len_ui.value),
            mask_prompt=bool(mask_prompt_ui.value),
            seed=int(seed_ui.value),
        )
        _callback = run_lora_training(
            model=base_model,
            tokenizer=tokenizer,
            data_dir=data_dir,
            adapter_dir=trained_adapter_dir,
            lr=float(lr_ui.value),
            batch_size=int(batch_size_ui.value),
            iters=int(iters_ui.value),
            max_seq_length=int(max_seq_len_ui.value),
            steps_per_eval=int(steps_per_eval_ui.value),
            save_every=int(save_every_ui.value),
            grad_checkpoint=bool(grad_checkpoint_ui.value),
            grad_accum=int(grad_accum_ui.value),
            seed=int(seed_ui.value),
            mask_prompt=bool(mask_prompt_ui.value),
            output_sink=_sink,
        )
        train_losses = _callback.train_history
        val_losses = _callback.val_history
        mo.output.replace(
            mo.md(
                f"**Training complete.** {len(train_losses)} train-loss reports / "
                f"{len(val_losses)} val-loss reports. Adapters saved to "
                f"`{trained_adapter_dir / 'adapters.safetensors'}`."
            )
        )
    return train_losses, trained_adapter_dir, val_losses


@app.cell
def _(mo):
    mo.md(r"""
    ### Load an already-trained adapter (e.g. produced by the CLI)
    """)
    return


@app.cell
def _(mo, models_dir):
    load_adapter_path_ui = mo.ui.text(
        value=str(models_dir / "chat_difficulty_qwen3.8_27b_lora_v1"),
        label="Adapter directory to load",
        full_width=True,
    )
    load_adapter_btn = mo.ui.run_button(label="Load adapters into current model", kind="primary")
    mo.vstack(
        [
            mo.md(
                "If you trained via `mlx_lm.lora` on the command line (or you ran training "
                "earlier), point this at the adapter folder to inject the weights into the "
                "currently loaded base model."
            ),
            load_adapter_path_ui,
            load_adapter_btn,
        ]
    )
    return load_adapter_btn, load_adapter_path_ui


@app.function
def load_adapters_into_model(model, adapter_dir: Path) -> None:
    from mlx_lm.tuner.utils import load_adapters

    load_adapters(model, str(adapter_dir))


@app.cell
def _(base_model, load_adapter_btn, load_adapter_path_ui, mo):
    loaded_adapter_dir: Path | None = None
    if base_model is None:
        mo.output.replace(mo.md("_Load the base model first._"))
    elif not load_adapter_btn.value:
        mo.output.replace(mo.md("Enter an adapter path and click **Load adapters**."))
    else:
        loaded_adapter_dir = Path(load_adapter_path_ui.value)
        if not loaded_adapter_dir.exists():
            mo.output.replace(
                mo.md(f"**Not found:** `{loaded_adapter_dir}`. Check the path.")
            )
            loaded_adapter_dir = None
        else:
            load_adapters_into_model(base_model, loaded_adapter_dir)
            mo.output.replace(
                mo.md(
                    f"**Adapters merged** from `{loaded_adapter_dir}`. "
                    f"Trainable params: `{get_trainable_params(base_model):,}`."
                )
            )
    return (loaded_adapter_dir,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Hyperparameter Search (Optional)
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.cell
def _(mo):
    hp_ranks_ui = mo.ui.text(value="4,8,16", label="Ranks (comma-sep)")
    hp_lrs_ui = mo.ui.text(value="5e-6,1e-5,5e-5", label="Learning rates (comma-sep)")
    hp_iters_ui = mo.ui.slider(20, 500, value=60, step=20, label="Iters per cell")
    hp_run_btn = mo.ui.run_button(label="Run HP sweep", kind="primary")
    mo.vstack(
        [
            mo.md(
                "**Warning:** each cell loads the base model, applies LoRA, trains for "
                "`iters_per_cell` steps, and runs validation. On 27B that is ~5–10 minutes per cell."
            ),
            mo.hstack([hp_ranks_ui, hp_lrs_ui, hp_iters_ui]),
            hp_run_btn,
        ]
    )
    return hp_iters_ui, hp_lrs_ui, hp_ranks_ui, hp_run_btn


@app.function
def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


@app.function
def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


@app.function
def run_hp_cell(
    model_source: str,
    data_dir: Path,
    tmp_dir: Path,
    rank: int,
    lr: float,
    iters: int,
    num_layers: int,
    scale: float,
    dropout: float,
    batch_size: int,
    max_seq_length: int,
    mask_prompt: bool,
    grad_checkpoint: bool,
    seed: int,
) -> dict:
    """One HP cell: fresh model load → LoRA → short train → final val loss."""
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tok = mlx_lm_load(model_source)
    model.freeze()
    linear_to_lora_layers(
        model=model,
        num_layers=int(num_layers),
        config={"rank": int(rank), "scale": float(scale), "dropout": float(dropout)},
        use_dora=False,
    )
    train_set, valid_set, _ = load_jsonl_as_mlx_datasets(data_dir, tok, mask_prompt)
    cell_dir = tmp_dir / f"rank{rank}_lr{lr:g}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = cell_dir / "adapters.safetensors"
    args = TrainingArgs(
        batch_size=int(batch_size),
        iters=int(iters),
        val_batches=5,
        steps_per_report=max(1, int(iters) // 10),
        steps_per_eval=int(iters),
        steps_per_save=int(iters),
        max_seq_length=int(max_seq_length),
        adapter_file=str(adapter_file),
        grad_checkpoint=bool(grad_checkpoint),
    )
    mx.random.seed(seed)
    train(
        model=model,
        optimizer=optim.AdamW(learning_rate=float(lr)),
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=args,
    )
    val_loss = evaluate(
        model=model,
        dataset=CacheDataset(valid_set),
        batch_size=int(batch_size),
        num_batches=5,
        max_seq_length=int(max_seq_length),
    )
    return {
        "rank": int(rank),
        "lr": float(lr),
        "iters": int(iters),
        "val_loss": round(float(val_loss), 4),
        "val_ppl": round(math.exp(float(val_loss)), 4),
    }


@app.cell
def _(
    base_model_repo_ui,
    batch_size_ui,
    data_dir,
    grad_checkpoint_ui,
    hp_iters_ui,
    hp_lrs_ui,
    hp_ranks_ui,
    hp_run_btn,
    hp_search_cb,
    jsonl_paths: dict[str, Path] | None,
    local_base_path_ui,
    mask_prompt_ui,
    max_seq_len_ui,
    mo,
    model_source,
    models_dir,
    num_lora_layers_ui,
    scale_ui,
    seed_ui,
):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable the checkbox above to run hyperparameter search._"),
    )
    hp_results: list[dict] = []
    if jsonl_paths is None:
        mo.output.replace(mo.md("_Build train/valid/test JSONL first._"))
    elif not hp_run_btn.value:
        mo.output.replace(mo.md("Click **Run HP sweep** to start."))
    else:
        _ranks = parse_int_list(hp_ranks_ui.value)
        _lrs = parse_float_list(hp_lrs_ui.value)
        _source = (
            model_source
            or local_base_path_ui.value.strip()
            or base_model_repo_ui.value
        )
        _tmp = models_dir / "_hp_search"
        for _r in _ranks:
            for _lr in _lrs:
                _res = run_hp_cell(
                    model_source=_source,
                    data_dir=data_dir,
                    tmp_dir=_tmp,
                    rank=_r,
                    lr=_lr,
                    iters=int(hp_iters_ui.value),
                    num_layers=int(num_lora_layers_ui.value),
                    scale=float(scale_ui.value),
                    dropout=0.0,
                    batch_size=int(batch_size_ui.value),
                    max_seq_length=int(max_seq_len_ui.value),
                    mask_prompt=bool(mask_prompt_ui.value),
                    grad_checkpoint=bool(grad_checkpoint_ui.value),
                    seed=int(seed_ui.value),
                )
                hp_results.append(_res)
                mo.output.replace(
                    mo.md(
                        f"rank={_r}, lr={_lr:g} → val_loss={_res['val_loss']}, "
                        f"val_ppl={_res['val_ppl']}"
                    )
                )
        hp_results.sort(key=lambda r: r["val_loss"])
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("### HP sweep results (sorted by val_loss)"),
                    mo.ui.table(hp_results, selection=None),
                ]
            )
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Evaluation: Vanilla vs LoRA

    We parse the first integer out of each generation (robust to Qwen's "answer: 7" chatter
    when enable_thinking fails). Numeric metrics (this is a 1–10 regression framed as
    classification): **MAE**, **RMSE**, **Pearson / Spearman**, **accuracy** (exact match
    on the integer), **parse rate** (fraction of outputs we could parse), **test-set loss**
    (next-token CE on the assistant target).

    We do **not** run k-fold CV — on a 27B QLoRA setup each fold is multi-hour. Instead we
    report bootstrap 95% CIs on the metrics computed over the test set (resampling test
    examples with replacement).
    """)
    return


@app.cell
def _(mo):
    eval_n_ui = mo.ui.number(value=100, start=10, label="# test examples to score")
    eval_max_tokens_ui = mo.ui.number(value=12, start=1, label="Max generated tokens")
    bootstrap_n_ui = mo.ui.number(value=200, start=50, label="Bootstrap resamples")
    enable_thinking_ui = mo.ui.checkbox(value=False, label="Enable Qwen thinking mode")
    eval_btn = mo.ui.run_button(label="Run Vanilla vs LoRA eval", kind="primary")
    mo.vstack(
        [
            mo.hstack([eval_n_ui, eval_max_tokens_ui, bootstrap_n_ui]),
            enable_thinking_ui,
            eval_btn,
        ]
    )
    return (
        bootstrap_n_ui,
        enable_thinking_ui,
        eval_btn,
        eval_max_tokens_ui,
        eval_n_ui,
    )


@app.function
def parse_rating(text: str, lo: int, hi: int) -> int | None:
    """Return the first standalone integer in [lo, hi] inside `text`, else None."""
    if not text:
        return None
    for m in re.finditer(r"-?\d+", text):
        try:
            v = int(m.group())
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None


@app.function
def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


@app.function
def generate_rating_for_prompt(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
    enable_thinking: bool,
) -> str:
    from mlx_lm import generate as mlx_lm_generate

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
    return mlx_lm_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=int(max_tokens),
        verbose=False,
    )


@app.function
def evaluate_model_on_test(
    model,
    tokenizer,
    test_records: list[dict],
    system_prompt: str,
    max_tokens: int,
    rating_lo: int,
    rating_hi: int,
    enable_thinking: bool,
) -> list[dict]:
    out: list[dict] = []
    for rec in test_records:
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        true = next(m["content"] for m in msgs if m["role"] == "assistant")
        try:
            true_i = int(true)
        except Exception:
            continue
        gen = generate_rating_for_prompt(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        pred = parse_rating(gen, rating_lo, rating_hi)
        out.append(
            {
                "user_prompt": user,
                "true": true_i,
                "generated_text": gen,
                "pred": pred,
            }
        )
    return out


@app.function
def compute_scalar_metrics(predictions: list[dict]) -> dict:
    """MAE / RMSE / correlation / accuracy / parse rate over a list of {true, pred}."""
    from scipy.stats import pearsonr, spearmanr

    parsed = [p for p in predictions if p["pred"] is not None]
    parse_rate = len(parsed) / max(len(predictions), 1)
    if not parsed:
        return {
            "n": len(predictions),
            "parse_rate": 0.0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "accuracy": None,
            "off_by_1_acc": None,
        }
    y_true = np.array([p["true"] for p in parsed], dtype=np.float64)
    y_pred = np.array([p["pred"] for p in parsed], dtype=np.float64)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    acc = float(np.mean(y_pred == y_true))
    off1 = float(np.mean(np.abs(diff) <= 1))
    pear = float(pearsonr(y_true, y_pred).statistic) if len(parsed) > 1 else float("nan")
    spear = float(spearmanr(y_true, y_pred).statistic) if len(parsed) > 1 else float("nan")
    return {
        "n": len(predictions),
        "parse_rate": round(parse_rate, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson": round(pear, 4),
        "spearman": round(spear, 4),
        "accuracy": round(acc, 4),
        "off_by_1_acc": round(off1, 4),
    }


@app.function
def bootstrap_metric_ci(
    predictions: list[dict],
    metric_name: str,
    n_boot: int,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    parsed = [p for p in predictions if p["pred"] is not None]
    if not parsed:
        return (float("nan"), float("nan"))
    y_true = np.array([p["true"] for p in parsed], dtype=np.float64)
    y_pred = np.array([p["pred"] for p in parsed], dtype=np.float64)
    n = len(parsed)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        diff = yp - yt
        if metric_name == "mae":
            vals.append(float(np.mean(np.abs(diff))))
        elif metric_name == "rmse":
            vals.append(float(np.sqrt(np.mean(diff ** 2))))
        elif metric_name == "accuracy":
            vals.append(float(np.mean(yp == yt)))
        elif metric_name == "off_by_1_acc":
            vals.append(float(np.mean(np.abs(diff) <= 1)))
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return (round(lo, 4), round(hi, 4))


@app.function
def compute_testset_loss(
    model,
    tokenizer,
    data_dir: Path,
    batch_size: int,
    max_seq_length: int,
    mask_prompt: bool,
) -> float:
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import evaluate

    _, _, test_set = load_jsonl_as_mlx_datasets(data_dir, tokenizer, mask_prompt)
    return float(
        evaluate(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=int(batch_size),
            num_batches=max(1, len(test_set) // max(int(batch_size), 1)),
            max_seq_length=int(max_seq_length),
        )
    )


@app.function
def remove_lora_from_model(model) -> None:
    """Reverse `linear_to_lora_layers` so we can measure the vanilla base's behavior."""
    from mlx_lm.tuner.utils import remove_lora_layers

    remove_lora_layers(model)


@app.cell
def _(
    RATING_MAX,
    RATING_MIN,
    SYSTEM_PROMPT,
    base_model,
    data_dir,
    enable_thinking_ui,
    eval_btn,
    eval_max_tokens_ui,
    eval_n_ui,
    jsonl_paths: dict[str, Path] | None,
    loaded_adapter_dir: Path | None,
    lora_applied,
    mask_prompt_ui,
    max_seq_len_ui,
    mo,
    tokenizer,
    trained_adapter_dir: Path | None,
):
    eval_predictions_lora: list[dict] = []
    eval_predictions_vanilla: list[dict] = []
    eval_test_loss_lora: float | None = None
    eval_test_loss_vanilla: float | None = None
    eval_source_adapter: Path | None = None
    if not eval_btn.value:
        mo.output.replace(mo.md("Click **Run Vanilla vs LoRA eval** to begin."))
    elif base_model is None or tokenizer is None:
        mo.output.replace(mo.md("_Load base model + tokenizer first._"))
    elif jsonl_paths is None:
        mo.output.replace(mo.md("_Build train/valid/test JSONL first._"))
    else:
        eval_source_adapter = trained_adapter_dir or loaded_adapter_dir
        _test_records = read_jsonl(jsonl_paths["test"])[: int(eval_n_ui.value)]
        if lora_applied or loaded_adapter_dir is not None:
            mo.output.replace(mo.md("Evaluating **LoRA-adapted** model..."))
            eval_predictions_lora = evaluate_model_on_test(
                model=base_model,
                tokenizer=tokenizer,
                test_records=_test_records,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=int(eval_max_tokens_ui.value),
                rating_lo=RATING_MIN,
                rating_hi=RATING_MAX,
                enable_thinking=bool(enable_thinking_ui.value),
            )
            eval_test_loss_lora = compute_testset_loss(
                model=base_model,
                tokenizer=tokenizer,
                data_dir=data_dir,
                batch_size=1,
                max_seq_length=int(max_seq_len_ui.value),
                mask_prompt=bool(mask_prompt_ui.value),
            )
            remove_lora_from_model(base_model)
            mo.output.replace(mo.md("Evaluating **vanilla** base model..."))
        else:
            mo.output.replace(mo.md("Evaluating **vanilla** base model (no LoRA applied)..."))
        eval_predictions_vanilla = evaluate_model_on_test(
            model=base_model,
            tokenizer=tokenizer,
            test_records=_test_records,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=int(eval_max_tokens_ui.value),
            rating_lo=RATING_MIN,
            rating_hi=RATING_MAX,
            enable_thinking=bool(enable_thinking_ui.value),
        )
        eval_test_loss_vanilla = compute_testset_loss(
            model=base_model,
            tokenizer=tokenizer,
            data_dir=data_dir,
            batch_size=1,
            max_seq_length=int(max_seq_len_ui.value),
            mask_prompt=bool(mask_prompt_ui.value),
        )
        mo.output.replace(
            mo.md(
                f"Evaluation complete. Vanilla: {len(eval_predictions_vanilla)} preds. "
                f"LoRA: {len(eval_predictions_lora)} preds."
            )
        )
    return (
        eval_predictions_lora,
        eval_predictions_vanilla,
        eval_source_adapter,
        eval_test_loss_lora,
        eval_test_loss_vanilla,
    )


@app.function
def build_comparison_table(
    vanilla_preds: list[dict],
    lora_preds: list[dict],
    test_loss_vanilla: float | None,
    test_loss_lora: float | None,
    n_boot: int,
) -> list[dict]:
    def row(name: str, preds: list[dict], tloss: float | None) -> dict:
        m = compute_scalar_metrics(preds)
        mae_ci = bootstrap_metric_ci(preds, "mae", n_boot) if preds else (None, None)
        acc_ci = bootstrap_metric_ci(preds, "accuracy", n_boot) if preds else (None, None)
        return {
            "model": name,
            "n": m["n"],
            "parse_rate": m["parse_rate"],
            "accuracy": m["accuracy"],
            "accuracy_95ci": f"[{acc_ci[0]}, {acc_ci[1]}]" if preds else "n/a",
            "off_by_1_acc": m["off_by_1_acc"],
            "mae": m["mae"],
            "mae_95ci": f"[{mae_ci[0]}, {mae_ci[1]}]" if preds else "n/a",
            "rmse": m["rmse"],
            "pearson": m["pearson"],
            "spearman": m["spearman"],
            "test_set_ce_loss": round(tloss, 4) if tloss is not None else None,
            "test_set_ppl": round(math.exp(tloss), 4) if tloss is not None else None,
        }

    rows = []
    if vanilla_preds:
        rows.append(row("vanilla (no LoRA)", vanilla_preds, test_loss_vanilla))
    if lora_preds:
        rows.append(row("LoRA-adapted", lora_preds, test_loss_lora))
    return rows


@app.cell
def _(
    bootstrap_n_ui,
    eval_predictions_lora: list[dict],
    eval_predictions_vanilla: list[dict],
    eval_test_loss_lora: float | None,
    eval_test_loss_vanilla: float | None,
    mo,
):
    if not eval_predictions_vanilla and not eval_predictions_lora:
        _out = mo.md("_Run the eval first._")
    else:
        _rows = build_comparison_table(
            vanilla_preds=eval_predictions_vanilla,
            lora_preds=eval_predictions_lora,
            test_loss_vanilla=eval_test_loss_vanilla,
            test_loss_lora=eval_test_loss_lora,
            n_boot=int(bootstrap_n_ui.value),
        )
        _out = mo.vstack(
            [
                mo.md("### Vanilla vs LoRA — scalar metrics (with 95% bootstrap CI)"),
                mo.ui.table(_rows, selection=None),
            ]
        )
    _out
    return


@app.function
def build_example_prediction_rows(
    vanilla_preds: list[dict],
    lora_preds: list[dict],
    n_show: int = 20,
) -> list[dict]:
    """Zip the first `n_show` pairs; both lists are in test-record order."""
    rows = []
    for i, (v, l) in enumerate(zip(vanilla_preds, lora_preds)):
        rows.append(
            {
                "i": i,
                "true": v["true"],
                "vanilla_pred": v["pred"],
                "vanilla_gen": (v["generated_text"] or "").replace("\n", " ")[:60],
                "lora_pred": l["pred"],
                "lora_gen": (l["generated_text"] or "").replace("\n", " ")[:60],
                "prompt_preview": v["user_prompt"][:160],
            }
        )
        if len(rows) >= n_show:
            break
    return rows


@app.cell
def _(
    eval_predictions_lora: list[dict],
    eval_predictions_vanilla: list[dict],
    mo,
):
    if not eval_predictions_vanilla:
        _out = mo.md("_Run eval first._")
    else:
        _rows = build_example_prediction_rows(
            eval_predictions_vanilla, eval_predictions_lora
        )
        _out = mo.vstack(
            [mo.md("### Example predictions (first 20)"), mo.ui.table(_rows, selection=None)]
        )
    _out
    return


@app.function
def plot_predicted_vs_true(
    vanilla_preds: list[dict], lora_preds: list[dict]
) -> alt.Chart:
    rows = []
    for p in vanilla_preds:
        if p["pred"] is None:
            continue
        rows.append({"model": "vanilla", "true": p["true"], "pred": p["pred"]})
    for p in lora_preds:
        if p["pred"] is None:
            continue
        rows.append({"model": "LoRA", "true": p["true"], "pred": p["pred"]})
    df = pl.DataFrame(rows)
    return (
        alt.Chart(df.to_pandas())
        .mark_circle(opacity=0.5, size=70)
        .encode(
            x=alt.X("true:O", title="True rating"),
            y=alt.Y("pred:Q", title="Predicted rating", scale=alt.Scale(domain=[0, 11])),
            color=alt.Color("model:N"),
            tooltip=["model", "true", "pred"],
        )
        .properties(title="Predicted vs. true rating", width=500, height=360)
    )


@app.cell
def _(
    eval_predictions_lora: list[dict],
    eval_predictions_vanilla: list[dict],
    mo,
):
    if not eval_predictions_vanilla and not eval_predictions_lora:
        _out = mo.md("_Run eval first._")
    else:
        _out = mo.ui.altair_chart(
            plot_predicted_vs_true(eval_predictions_vanilla, eval_predictions_lora)
        )
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Results
    """)
    return


@app.function
def plot_training_curves(train_losses: list[dict], val_losses: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    if train_losses:
        iters = [h.get("iteration", i) for i, h in enumerate(train_losses)]
        tl = [h.get("train_loss") for h in train_losses]
        ax.plot(iters, tl, "b-o", lw=2, ms=4, label="train")
    if val_losses:
        iters = [h.get("iteration", i) for i, h in enumerate(val_losses)]
        vl = [h.get("val_loss") for h in val_losses]
        ax.plot(iters, vl, "r-s", lw=2, ms=4, label="val")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses: list[dict], val_losses: list[dict]):
    if not train_losses and not val_losses:
        _out = mo.md("_Train (or load a run with history) first._")
    else:
        _out = plot_training_curves(train_losses, val_losses)
    _out
    return


@app.function
def build_results_summary_md(
    vanilla_preds: list[dict],
    lora_preds: list[dict],
    test_loss_vanilla: float | None,
    test_loss_lora: float | None,
    n_boot: int,
) -> str:
    if not vanilla_preds and not lora_preds:
        return "_No evaluation data yet._"
    rows = build_comparison_table(
        vanilla_preds, lora_preds, test_loss_vanilla, test_loss_lora, n_boot
    )
    by_name = {r["model"]: r for r in rows}
    v = by_name.get("vanilla (no LoRA)")
    l = by_name.get("LoRA-adapted")
    lines = ["### Summary"]
    if v is not None:
        lines.append(
            f"- **Vanilla:** acc={v['accuracy']}, mae={v['mae']}, spearman={v['spearman']}, "
            f"test_ppl={v['test_set_ppl']}"
        )
    if l is not None:
        lines.append(
            f"- **LoRA:**    acc={l['accuracy']}, mae={l['mae']}, spearman={l['spearman']}, "
            f"test_ppl={l['test_set_ppl']}"
        )
    if v is not None and l is not None and v["mae"] is not None and l["mae"] is not None:
        delta_mae = round(v["mae"] - l["mae"], 4)
        delta_acc = round((l["accuracy"] or 0) - (v["accuracy"] or 0), 4)
        lines.append("")
        lines.append(
            f"**Δ (LoRA improvement over vanilla):** MAE drop = {delta_mae}, "
            f"accuracy lift = {delta_acc}."
        )
        if delta_mae > 0 or delta_acc > 0:
            lines.append(
                "A positive MAE drop and/or positive accuracy lift means LoRA helped. "
                "Confidence intervals above should be inspected before claiming significance."
            )
        else:
            lines.append(
                "No improvement observed on this test slice — try more LoRA layers, higher rank, "
                "longer training, or `mask_prompt=True` to focus the loss on the single-token answer."
            )
    return "\n".join(lines)


@app.cell
def _(
    bootstrap_n_ui,
    eval_predictions_lora: list[dict],
    eval_predictions_vanilla: list[dict],
    eval_test_loss_lora: float | None,
    eval_test_loss_vanilla: float | None,
    mo,
):
    mo.md(
        build_results_summary_md(
            vanilla_preds=eval_predictions_vanilla,
            lora_preds=eval_predictions_lora,
            test_loss_vanilla=eval_test_loss_vanilla,
            test_loss_lora=eval_test_loss_lora,
            n_boot=int(bootstrap_n_ui.value),
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Save Trained Model
    """)
    return


@app.cell
def _(mo):
    save_name_ui = mo.ui.text(
        value="chat_difficulty_qwen3.8_27b_lora_v1",
        label="Save name (used for adapter + fused folders in models/)",
        full_width=True,
    )
    fuse_cb_ui = mo.ui.checkbox(value=False, label="Also fuse adapters into the base model")
    dequantize_cb_ui = mo.ui.checkbox(value=False, label="De-quantise fused model (bf16 weights)")
    save_model_btn = mo.ui.run_button(label="Save Model", kind="primary")
    mo.vstack(
        [
            mo.md(
                "Writes `adapters.safetensors` + `adapter_config.json` into "
                "`models/<name>_adapters/`, which is directly usable by `mlx_lm.generate "
                "--adapter-path`. Fusing (optional) produces a standalone model in "
                "`models/<name>_fused/` via `mlx_lm.fuse`. The base model is 4-bit; by default "
                "the fused model stays 4-bit (dequantize=False)."
            ),
            save_name_ui,
            mo.hstack([fuse_cb_ui, dequantize_cb_ui]),
            save_model_btn,
        ]
    )
    return dequantize_cb_ui, fuse_cb_ui, save_model_btn, save_name_ui


@app.function
def save_adapters_to_dir(model, src_adapter_dir: Path, dest_dir: Path) -> Path:
    """Write adapter weights + copy adapter_config.json to a fresh folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(dest_dir / "adapters.safetensors"), adapter_weights)
    src_cfg = src_adapter_dir / "adapter_config.json"
    if src_cfg.exists():
        shutil.copyfile(src_cfg, dest_dir / "adapter_config.json")
    return dest_dir


@app.function
def fuse_adapters(
    base_source: str,
    adapter_dir: Path,
    dest_dir: Path,
    dequantize: bool,
) -> Path:
    """Mirror of mlx_lm.fuse.main() but as a function call (no CLI argparse)."""
    from mlx_lm.utils import dequantize_model, load, save

    model, tok, cfg = load(
        base_source, adapter_path=str(adapter_dir), return_config=True
    )
    fused = [
        (n, m.fuse(dequantize=dequantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused:
        model.update_modules(tree_unflatten(fused))
    if dequantize:
        model = dequantize_model(model)
        cfg.pop("quantization", None)
        cfg.pop("quantization_config", None)
    dest_dir.mkdir(parents=True, exist_ok=True)
    save(dest_dir, base_source, model, tok, cfg, donate_model=False)
    return dest_dir


@app.function
def build_fuse_cli(
    base_source: str,
    adapter_dir: Path,
    dest_dir: Path,
    dequantize: bool,
) -> str:
    parts = [
        "mlx_lm.fuse",
        f"--model {shlex.quote(base_source)}",
        f"--adapter-path {shlex.quote(str(adapter_dir))}",
        f"--save-path {shlex.quote(str(dest_dir))}",
    ]
    if dequantize:
        parts.append("--dequantize")
    return " \\\n  ".join(parts)


@app.cell
def _(
    base_model,
    base_model_repo_ui,
    dequantize_cb_ui,
    eval_source_adapter: Path | None,
    fuse_cb_ui,
    loaded_adapter_dir: Path | None,
    local_base_path_ui,
    mo,
    model_source,
    models_dir,
    save_model_btn,
    save_name_ui,
    trained_adapter_dir: Path | None,
):
    _effective_adapter_src = (
        trained_adapter_dir or loaded_adapter_dir or eval_source_adapter
    )
    if base_model is None:
        _out = mo.md("_Load the base model first._")
    elif _effective_adapter_src is None:
        _out = mo.md(
            "_No trained or loaded adapters in this session. Train in Section 5 or "
            "point Section 5's adapter-loader at an existing folder._"
        )
    elif not save_model_btn.value:
        _out = mo.md(
            "Enter a name and click **Save Model** to copy adapters into "
            "`models/<name>_adapters/` (and optionally fuse)."
        )
    else:
        _adapter_dest = models_dir / f"{save_name_ui.value}_adapters"
        save_adapters_to_dir(base_model, _effective_adapter_src, _adapter_dest)
        _lines = [f"Adapters saved to `{_adapter_dest}`."]
        _base_source = (
            model_source
            or local_base_path_ui.value.strip()
            or base_model_repo_ui.value
        )
        _fused_dest = models_dir / f"{save_name_ui.value}_fused"
        _fuse_cli = build_fuse_cli(
            _base_source, _adapter_dest, _fused_dest, bool(dequantize_cb_ui.value)
        )
        _lines.append("\n**Equivalent fuse CLI:**")
        _lines.append(f"```bash\n{_fuse_cli}\n```")
        if fuse_cb_ui.value:
            fuse_adapters(
                base_source=_base_source,
                adapter_dir=_adapter_dest,
                dest_dir=_fused_dest,
                dequantize=bool(dequantize_cb_ui.value),
            )
            _lines.append(f"Fused model saved to `{_fused_dest}`.")
            if not dequantize_cb_ui.value:
                _lines.append(
                    "_Note: fused model remains 4-bit quantised. Check "
                    "**De-quantise fused model** if you need a bf16 checkpoint._"
                )
        _out = mo.md("\n\n".join(_lines))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Load the Trained Model & Rate New Prompts

    This section is self-contained: it loads the saved rater from disk in a fresh process
    (no training-session state needed) and uses it on the trained task: rating a chat
    prompt's difficulty from 1 to 10.

    Two ways to load it:

    - **Base model + adapters**: `mlx_lm.load(base, adapter_path=<name>_adapters)`. The
      base model gets wrapped with LoRA layers and the saved adapter weights are loaded
      into them. The adapter folder is small (MBs), so this is the usual way to ship it.
    - **Fused model**: `mlx_lm.load(<name>_fused)`. The adapters are already merged into
      the weights, so this is a standalone checkpoint with no base model needed.

    The prompt has to be formatted the same way as in training (same system prompt, chat
    template, thinking mode off), or the model won't reply with just the integer.
    Loading here creates a second copy of the 27B model next to the one from Section 4
    (about 15 GB more at 4-bit).
    """)
    return


@app.cell
def _(BASE_MODEL_ID, mo, models_dir):
    infer_mode_ui = mo.ui.radio(
        options=["Base model + adapters", "Fused model"],
        value="Base model + adapters",
        label="Load mode",
    )
    infer_base_ui = mo.ui.text(
        value=BASE_MODEL_ID,
        label="Base model (HF repo id or local path)",
        full_width=True,
    )
    infer_adapter_ui = mo.ui.text(
        value=str(models_dir / "chat_difficulty_qwen3.8_27b_lora_v1_adapters"),
        label="Adapter directory (Section 9 output)",
        full_width=True,
    )
    infer_fused_ui = mo.ui.text(
        value=str(models_dir / "chat_difficulty_qwen3.8_27b_lora_v1_fused"),
        label="Fused model directory (Section 9 output)",
        full_width=True,
    )
    load_infer_btn = mo.ui.run_button(label="Load fine-tuned rater", kind="primary")
    mo.vstack(
        [infer_mode_ui, infer_base_ui, infer_adapter_ui, infer_fused_ui, load_infer_btn]
    )
    return (
        infer_adapter_ui,
        infer_base_ui,
        infer_fused_ui,
        infer_mode_ui,
        load_infer_btn,
    )


@app.function
def resolve_rater_source(
    mode: str,
    base_source: str,
    adapter_dir: str,
    fused_dir: str,
) -> tuple[str, str | None]:
    """Return `(model_source, adapter_path)` for `mlx_lm.load`; raise if files are missing."""
    if mode == "Fused model":
        fused_path = Path(fused_dir)
        if not (fused_path / "config.json").exists():
            raise FileNotFoundError(f"No fused model (config.json) in {fused_path}")
        return str(fused_path), None
    adapter_path = Path(adapter_dir)
    for required in ("adapters.safetensors", "adapter_config.json"):
        if not (adapter_path / required).exists():
            raise FileNotFoundError(f"Missing {required} in {adapter_path}")
    return base_source, str(adapter_path)


@app.function
def load_finetuned_rater(model_source: str, adapter_path: str | None):
    """Load a LoRA-adapted (or fused) model for inference. Returns `(model, tokenizer)`."""
    from mlx_lm import load as mlx_lm_load

    model, tokenizer = mlx_lm_load(model_source, adapter_path=adapter_path)
    model.eval()
    return model, tokenizer


@app.cell
def _(
    infer_adapter_ui,
    infer_base_ui,
    infer_fused_ui,
    infer_mode_ui,
    load_infer_btn,
    mo,
):
    rater_model = None
    rater_tokenizer = None
    rater_source: tuple[str, str | None] | None = None
    if not load_infer_btn.value:
        mo.output.replace(mo.md("_Click **Load fine-tuned rater** to load it from disk._"))
    else:
        try:
            rater_source = resolve_rater_source(
                infer_mode_ui.value,
                infer_base_ui.value.strip(),
                infer_adapter_ui.value.strip(),
                infer_fused_ui.value.strip(),
            )
        except FileNotFoundError as err:
            mo.output.replace(mo.md(f"**Cannot load:** {err}. Save the model in Section 9 first."))
        else:
            rater_model, rater_tokenizer = load_finetuned_rater(*rater_source)
            mo.output.replace(
                mo.md(
                    f"**Loaded** `{rater_source[0]}`"
                    + (f" with adapters from `{rater_source[1]}`." if rater_source[1] else ".")
                )
            )
    return rater_model, rater_tokenizer


@app.function
def split_prompts(text: str, separator: str = "---") -> list[str]:
    """Split a text block into prompts on lines that consist only of `separator`."""
    blocks = re.split(rf"(?m)^\s*{re.escape(separator)}\s*$", text)
    return [b.strip() for b in blocks if b.strip()]


@app.function
def rate_prompts(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str,
    max_tokens: int,
    rating_lo: int,
    rating_hi: int,
    enable_thinking: bool = False,
) -> list[dict]:
    """Rate each prompt; returns rows with the raw generation and the parsed rating (or None)."""
    rows = []
    for prompt in prompts:
        raw = generate_rating_for_prompt(
            model, tokenizer, prompt, system_prompt, max_tokens, enable_thinking
        )
        rows.append(
            {
                "prompt": prompt,
                "raw_output": raw.strip(),
                "rating": parse_rating(raw, rating_lo, rating_hi),
            }
        )
    return rows


@app.cell
def _(mo):
    infer_prompts_ui = mo.ui.text_area(
        value=(
            "What is 2 + 2?\n"
            "---\n"
            "Write a haiku about autumn leaves.\n"
            "---\n"
            "Explain how to implement a lock-free concurrent hash map in C++ and prove "
            "its linearizability.\n"
            "---\n"
            "Derive the evidence lower bound for a variational autoencoder and explain "
            "why the reparameterization trick gives low-variance gradients."
        ),
        label="Prompts to rate (separate prompts with a line containing only ---)",
        full_width=True,
        rows=12,
    )
    infer_max_tokens_ui = mo.ui.number(value=12, start=1, label="Max generated tokens")
    infer_thinking_ui = mo.ui.checkbox(value=False, label="Enable Qwen thinking mode")
    rate_btn = mo.ui.run_button(label="Rate prompts", kind="primary")
    mo.vstack(
        [infer_prompts_ui, mo.hstack([infer_max_tokens_ui, infer_thinking_ui]), rate_btn]
    )
    return infer_max_tokens_ui, infer_prompts_ui, infer_thinking_ui, rate_btn


@app.cell
def _(
    RATING_MAX,
    RATING_MIN,
    SYSTEM_PROMPT,
    infer_max_tokens_ui,
    infer_prompts_ui,
    infer_thinking_ui,
    mo,
    rate_btn,
    rater_model,
    rater_tokenizer,
):
    rated_prompts: list[dict] = []
    if rater_model is None:
        _out = mo.md("_Load the fine-tuned rater above first._")
    elif not rate_btn.value:
        _out = mo.md("Enter prompts and click **Rate prompts**.")
    else:
        rated_prompts = rate_prompts(
            rater_model,
            rater_tokenizer,
            split_prompts(infer_prompts_ui.value),
            SYSTEM_PROMPT,
            int(infer_max_tokens_ui.value),
            RATING_MIN,
            RATING_MAX,
            bool(infer_thinking_ui.value),
        )
        _out = mo.vstack(
            [
                mo.md("### Predicted difficulty"),
                mo.ui.table(pl.DataFrame(rated_prompts), selection=None),
            ]
        )
    _out
    return


@app.function
def build_generate_cli(
    model_source: str,
    adapter_path: str | None,
    system_prompt: str,
    max_tokens: int,
    user_prompt: str = "<your prompt here>",
) -> str:
    parts = ["mlx_lm.generate", f"--model {shlex.quote(model_source)}"]
    if adapter_path:
        parts.append(f"--adapter-path {shlex.quote(adapter_path)}")
    parts += [
        f"--system-prompt {shlex.quote(system_prompt)}",
        f"--chat-template-config {shlex.quote(json.dumps({'enable_thinking': False}))}",
        "--temp 0.0",
        f"--max-tokens {int(max_tokens)}",
        f"--prompt {shlex.quote(user_prompt)}",
    ]
    return " \\\n  ".join(parts)


@app.function
def build_python_usage_snippet(
    model_source: str,
    adapter_path: str | None,
    system_prompt: str,
    max_tokens: int,
) -> str:
    adapter_arg = f", adapter_path={adapter_path!r}" if adapter_path else ""
    return (
        "import re\n"
        "from mlx_lm import load, generate\n\n"
        f"model, tokenizer = load({model_source!r}{adapter_arg})\n"
        f"SYSTEM_PROMPT = {system_prompt!r}\n\n"
        "def rate(user_prompt: str) -> int | None:\n"
        "    messages = [\n"
        '        {"role": "system", "content": SYSTEM_PROMPT},\n'
        '        {"role": "user", "content": user_prompt},\n'
        "    ]\n"
        "    prompt = tokenizer.apply_chat_template(\n"
        "        messages, add_generation_prompt=True, enable_thinking=False\n"
        "    )\n"
        f"    out = generate(model, tokenizer, prompt=prompt, max_tokens={int(max_tokens)})\n"
        '    m = re.search(r"\\b(10|[1-9])\\b", out)\n'
        "    return int(m.group()) if m else None\n\n"
        'print(rate("Explain the Banach fixed-point theorem."))\n'
    )


@app.cell
def _(
    SYSTEM_PROMPT,
    infer_adapter_ui,
    infer_base_ui,
    infer_fused_ui,
    infer_max_tokens_ui,
    infer_mode_ui,
    mo,
):
    if infer_mode_ui.value == "Fused model":
        usage_source, usage_adapter = infer_fused_ui.value.strip(), None
    else:
        usage_source, usage_adapter = (
            infer_base_ui.value.strip(),
            infer_adapter_ui.value.strip(),
        )
    usage_cli = build_generate_cli(
        usage_source, usage_adapter, SYSTEM_PROMPT, int(infer_max_tokens_ui.value)
    )
    usage_py = build_python_usage_snippet(
        usage_source, usage_adapter, SYSTEM_PROMPT, int(infer_max_tokens_ui.value)
    )
    mo.md(
        "### Using the rater outside this notebook\n\n"
        "**Command line** (run from the project root with `env/` active):\n\n"
        f"```bash\n{usage_cli}\n```\n\n"
        "**Python script:**\n\n"
        f"```python\n{usage_py}```"
    )
    return


if __name__ == "__main__":
    app.run()
