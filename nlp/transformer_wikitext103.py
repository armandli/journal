import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import hashlib
    import json
    import math
    import os
    import random
    import re
    import shutil
    import subprocess
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import dataclass, asdict, field
    from pathlib import Path
    from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import datasets as hf_datasets


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPT Language Model on WikiText-103 (PyTorch, CUDA, Flash Attention, RoPE)

    ## Research goal

    Train a decoder-only Transformer (GPT-style) language model on the WikiText-103
    corpus, using the **binary byte-level BPE** tokenizer (custom executables
    `build_bpe` / `tokenize_bpe` / `decode_bpe` / `table_bpe`, with a pre-built
    merge table at `~/data/wikitext103_mergetable.json`, vocab = 10,000).
    The architecture uses **RMSNorm**, **rotary positional embedding (RoPE)**,
    **SwiGLU MLP**, **weight-tied embeddings**, and **causal self-attention** routed
    through PyTorch's **FlashAttention-2 SDPA backend** for both forward and backward.
    Training uses **bfloat16 autocast** mixed precision, fused AdamW, gradient
    accumulation, cosine decay with linear warmup, and gradient clipping.

    ## What is *not* used, and why

    - **`torchtext` is NOT used**. Its final release (0.18.0) fails to load against
      `torch 2.10` (missing `libtorchtext.so` ABI), its WikiText-103 mirror URL is
      dead, and its published WikiText-103 is the `<unk>`-tokenized variant, which
      does not match the merge table (trained on the *raw* variant). Instead we
      read the byte-identical raw text via the HuggingFace `datasets` copy of
      `wikitext / wikitext-103-raw-v1` (already saved to `~/data/wikitext103-raw`).
    - **`flash_attn` (the pip package) is NOT installed and NOT used**. On this GPU
      (RTX 5080 Laptop, sm_120), the PyTorch built-in `F.scaled_dot_product_attention`
      under `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` runs
      FlashAttention-2 fwd+bwd in bf16 at ~1.9 ms per (8, 12, 1024, 64) block,
      with `is_causal=True` and dropout. That is the "flash attention" this notebook
      uses.
    - **`torch.compile` is NOT used**. Inductor requires `Python.h`, which is not
      installed in this environment (no `python3.12-dev`).

    ## Section outline

    1. Title & research goal (this cell)
    2. Data exploration — HF-datasets raw WikiText-103, article-length histogram,
       tokenizer merge-table sample
    3. Dataset creation — byte-level BPE tokenization + memmap cache under
       `~/data/wikitext103/`, contiguous-window sampler
    4. Model definition — `RotaryPositionalEmbeddingV1`, `RMSNorm`, `SwiGLUFeedForwardV1`,
       `CausalSelfAttentionV1` (Flash SDPA + RoPE + KV cache), `GPTBlockV1`,
       `GPTLanguageModelV1`; sanity check with flash-only forward
    5. Training — presets (tiny/small/base/medium), fractional epochs, AMP (bf16),
       fused AdamW, grad accumulation, cosine schedule, live progress
    6. Optional hyperparameter search — lr × preset grid, short token budget
    7. Validation & 5-fold cross-validation — test loss (nats/token), token PPL,
       bits-per-byte, word PPL; 5-fold CV on the train stream (behind its own gate)
    8. Results — loss curves, LR schedule, model comparison table, summary
    9. Sampling from the trained model — KV-cached autoregressive generate,
       temperature/top-k/top-p
    10. Save the trained model — `.pt` state dict + `.json` config sidecar
    11. Load a saved model & sample — no training required; independent UI
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Runtime — seed, device, precision
    """)
    return


@app.function
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.function
def make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device.type if device.type == "cpu" else "cpu")
    g.manual_seed(int(seed))
    return g


@app.function
def build_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


@app.function
def select_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.bfloat16


@app.cell
def _(mo):
    seed_ui = mo.ui.number(value=1337, label="Seed", start=0, stop=2**31 - 1)
    seed_ui
    return (seed_ui,)


@app.cell
def _(mo, seed_ui):
    device = build_device()
    mo.stop(
        device.type != "cuda",
        mo.md(
            "**CUDA is not available.** This notebook is written for CUDA training on "
            "an RTX 5080 (sm_120). Nothing else in the notebook will run until a CUDA "
            "device is visible to PyTorch."
        ),
    )
    torch.set_float32_matmul_precision("high")
    amp_dtype = select_amp_dtype(device)
    set_seed(int(seed_ui.value))
    _bf16_ok = torch.cuda.is_bf16_supported()
    mo.md(
        f"""
    | Setting | Value |
    |---|---|
    | Device | `{device}` ({torch.cuda.get_device_name(0)}) |
    | Compute capability | `sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}` |
    | AMP dtype | `{amp_dtype}` |
    | bf16 supported | `{_bf16_ok}` |
    | TF32 fp32 matmul | `high` |
    | Seed | `{int(seed_ui.value)}` |
    """
    )
    return amp_dtype, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data Exploration — WikiText-103 (raw)

    The dataset lives on disk under `~/data/wikitext103-raw` as a
    `datasets.DatasetDict.save_to_disk` of `wikitext / wikitext-103-raw-v1` with
    three splits: `train` (1,801,350 rows), `validation` (3,760 rows), `test`
    (4,358 rows). Each row is one line **including** its trailing `\n`; blank
    lines are empty rows; article titles look like `" = Title = \n"`, section
    headings like `" = = Section = = \n"`. The raw text of a split is exactly
    `"\".join(rows)`.

    If the on-disk copy is missing we transparently download the same corpus via
    `datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` and
    persist it there.
    """)
    return


@app.function
def default_wikitext_dir() -> Path:
    return Path.home() / "data" / "wikitext103-raw"


@app.function
def load_or_download_wikitext(local_dir: Path) -> "hf_datasets.DatasetDict":
    if local_dir.exists():
        return hf_datasets.load_from_disk(str(local_dir))
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    ds = hf_datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    ds.save_to_disk(str(local_dir))
    return ds


@app.function
def concat_split_text(ds_split: "hf_datasets.Dataset") -> str:
    return "".join(ds_split["text"])


@app.function
def compute_split_stats(ds_split: "hf_datasets.Dataset") -> Dict[str, int]:
    text = concat_split_text(ds_split)
    return {
        "rows": len(ds_split),
        "bytes": len(text.encode("utf-8")),
        "words": len(text.split()),
    }


@app.function
def extract_article_lengths(ds_split: "hf_datasets.Dataset", max_articles: int = 5000) -> List[int]:
    title_pat = re.compile(r"^ = [^=].* = \n$")
    lengths: List[int] = []
    current = 0
    for row in ds_split["text"]:
        if title_pat.match(row):
            if current > 0:
                lengths.append(current)
                if len(lengths) >= max_articles:
                    break
            current = 0
        current += len(row)
    if current > 0 and len(lengths) < max_articles:
        lengths.append(current)
    return lengths


@app.cell
def _():
    wikitext_dir = default_wikitext_dir()
    wikitext_ds = load_or_download_wikitext(wikitext_dir)
    return (wikitext_ds,)


@app.cell
def _(mo, wikitext_ds):
    train_stats = compute_split_stats(wikitext_ds["train"])
    val_stats = compute_split_stats(wikitext_ds["validation"])
    test_stats = compute_split_stats(wikitext_ds["test"])
    mo.md(
        f"""
    ### Split sizes

    | Split | Rows | UTF-8 bytes | Whitespace words |
    |---|---:|---:|---:|
    | train | {train_stats["rows"]:,} | {train_stats["bytes"]:,} | {train_stats["words"]:,} |
    | validation | {val_stats["rows"]:,} | {val_stats["bytes"]:,} | {val_stats["words"]:,} |
    | test | {test_stats["rows"]:,} | {test_stats["bytes"]:,} | {test_stats["words"]:,} |
    """
    )
    return test_stats, train_stats, val_stats


@app.function
def plot_article_length_histogram(lengths: List[int], title: str = "Article length (bytes)"):
    lengths_np = np.asarray(lengths, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(np.clip(lengths_np, 0, 40000), bins=60, color="steelblue", alpha=0.85)
    ax.set_xlabel("Bytes (clipped at 40,000 for display)")
    ax.set_ylabel("Number of articles")
    ax.set_title(f"{title}  |  n={len(lengths_np):,}  median={int(np.median(lengths_np)):,}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(wikitext_ds):
    plot_article_length_histogram(extract_article_lengths(wikitext_ds["train"]), title="Train article length (bytes)")
    return


@app.function
def plot_split_row_counts(stats_by_split: Dict[str, Dict[str, int]]):
    labels = list(stats_by_split.keys())
    rows = [stats_by_split[k]["rows"] for k in labels]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.bar(labels, rows, color=["steelblue", "seagreen", "goldenrod"])
    ax.set_ylabel("Row count (log)")
    ax.set_yscale("log")
    ax.set_title("Rows per split (log scale)")
    for bar, r in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2.0, r, f"{r:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


@app.cell
def _(test_stats, train_stats, val_stats):
    plot_split_row_counts({"train": train_stats, "validation": val_stats, "test": test_stats})
    return


@app.function
def default_merge_table_path() -> Path:
    return Path.home() / "data" / "wikitext103_mergetable.json"


@app.function
def read_merge_table_vocab_size(merge_table_path: Path) -> int:
    with open(merge_table_path, "r") as f:
        payload = json.load(f)
    n_merges = len(payload["merges"])
    return 256 + n_merges


@app.function
def resolve_executable(name: str) -> Path:
    candidate = shutil.which(name)
    if candidate is not None:
        p = Path(candidate)
        if os.access(p, os.X_OK):
            return p
    fallback = Path.home() / "bin" / name
    if fallback.exists() and os.access(fallback, os.X_OK):
        return fallback
    raise FileNotFoundError(
        f"Executable {name!r} not found on PATH or at {fallback}. Ensure the BPE "
        f"tools are installed (they normally live in ~/bin)."
    )


@app.function
def sample_learned_tokens(merge_table_path: Path, num_samples: int = 24) -> List[Tuple[int, str]]:
    exe = resolve_executable("table_bpe")
    result = subprocess.run(
        [str(exe), "-t", str(merge_table_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    entries: List[Tuple[int, str]] = []
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        head, tail = line.split(":", 1)
        head = head.strip()
        try:
            tid = int(head)
        except ValueError:
            continue
        entries.append((tid, tail))
    if not entries:
        return []
    step = max(1, len(entries) // num_samples)
    return entries[::step][:num_samples]


@app.cell
def _(mo):
    merge_table_path = default_merge_table_path()
    tokenizer_vocab_size = read_merge_table_vocab_size(merge_table_path)
    _sample_learned = sample_learned_tokens(merge_table_path, num_samples=24)
    _rows = "\n".join(f"| `{tid}` | `{text!r}` |" for tid, text in _sample_learned)
    mo.md(
        f"""
    ### Tokenizer merge table

    - Path: `{merge_table_path}`
    - Learned merges: {tokenizer_vocab_size - 256:,}
    - Tokenizer vocab size (byte base + merges): **{tokenizer_vocab_size:,}**
    - No special tokens (no BOS / EOS / PAD): the corpus is treated as one
      continuous byte stream.

    Sample of learned tokens (every ~1/24th of the learned range):

    | id | decodes to |
    |---:|---|
    {_rows}
    """
    )
    return merge_table_path, tokenizer_vocab_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Dataset Creation — tokenize once, memmap thereafter

    The `tokenize_bpe` executable is single-threaded (~0.6 MB/s), and it uses
    roughly 14 bytes of RAM per input byte. A single call on the full 546 MB
    train split would take ~15 minutes and about 7.6 GB of RAM. To stay well
    under this machine's 15 GB budget we:

    1. Stream the split's rows through a **shard writer** that cuts new shards at
       article-title boundaries and targets ~32 MB per shard.
    2. Tokenize the shards **in parallel** via
       `concurrent.futures.ThreadPoolExecutor(max_workers=8)` invoking
       `subprocess.run(["tokenize_bpe", ...], check=True)` on each shard file.
    3. Concatenate the shard id files **in shard order**, parse them with
       `np.array(text.split(), dtype=np.int64)`, verify `max_id < 65536`,
       cast to `uint16`, and write to `~/data/wikitext103/{split}.bin`.
    4. Cache a `meta.json` sidecar (merge-table SHA, token counts, raw byte counts,
       whitespace-word counts). Subsequent notebook runs load from the memmap in
       O(1).

    We never call the tokenizer on an in-memory string longer than one shard.
    """)
    return


@app.function
def default_token_cache_dir() -> Path:
    return Path.home() / "data" / "wikitext103"


@app.function
def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@app.function
def shard_split_text(
    ds_split: "hf_datasets.Dataset",
    shard_dir: Path,
    target_shard_bytes: int = 32 * 1024 * 1024,
) -> List[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    title_pat = re.compile(r"^ = [^=].* = \n$")
    shard_paths: List[Path] = []
    buf: List[str] = []
    buf_bytes = 0
    shard_idx = 0

    def _flush() -> None:
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        shard_path = shard_dir / f"shard_{shard_idx:04d}.txt"
        with open(shard_path, "w", encoding="utf-8") as f:
            f.write("".join(buf))
        shard_paths.append(shard_path)
        shard_idx += 1
        buf = []
        buf_bytes = 0

    for row in ds_split["text"]:
        row_bytes = len(row.encode("utf-8"))
        if buf_bytes >= target_shard_bytes and title_pat.match(row):
            _flush()
        buf.append(row)
        buf_bytes += row_bytes
    _flush()
    return shard_paths


@app.function
def tokenize_shard(shard_path: Path, merge_table_path: Path, ids_path: Path) -> str:
    exe = resolve_executable("tokenize_bpe")
    result = subprocess.run(
        [
            str(exe),
            "-t",
            str(merge_table_path),
            "-i",
            str(shard_path),
            "-o",
            str(ids_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@app.function
def parse_ids_file(ids_path: Path) -> np.ndarray:
    with open(ids_path, "r") as f:
        text = f.read()
    arr = np.array(text.split(), dtype=np.int64)
    if arr.size > 0 and int(arr.max()) >= (1 << 16):
        raise RuntimeError(
            f"Token id {int(arr.max())} in {ids_path} exceeds uint16 range; "
            f"the merge table has more than 65,536 tokens."
        )
    return arr.astype(np.uint16, copy=False)


@app.function
def tokenize_split_to_bin(
    ds_split: "hf_datasets.Dataset",
    merge_table_path: Path,
    out_bin_path: Path,
    workers: int = 8,
    target_shard_bytes: int = 32 * 1024 * 1024,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="bpe_shards_") as tmp:
        tmp_dir = Path(tmp)
        shard_paths = shard_split_text(ds_split, tmp_dir / "raw", target_shard_bytes)
        ids_paths = [tmp_dir / "ids" / f"{p.stem}.ids" for p in shard_paths]
        (tmp_dir / "ids").mkdir(parents=True, exist_ok=True)

        def _work(pair: Tuple[Path, Path]) -> None:
            src, dst = pair
            tokenize_shard(src, merge_table_path, dst)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_work, list(zip(shard_paths, ids_paths))))

        arrays = [parse_ids_file(p) for p in ids_paths]
        total_tokens = int(sum(a.size for a in arrays))

        out_bin_path.parent.mkdir(parents=True, exist_ok=True)
        merged = np.memmap(str(out_bin_path), dtype=np.uint16, mode="w+", shape=(total_tokens,))
        cursor = 0
        for arr in arrays:
            merged[cursor : cursor + arr.size] = arr
            cursor += arr.size
        merged.flush()
        del merged

    return {"tokens": total_tokens}


@app.function
def ensure_token_cache(
    hf_ds: "hf_datasets.DatasetDict",
    merge_table_path: Path,
    cache_dir: Path,
    workers: int = 8,
    target_shard_bytes: int = 32 * 1024 * 1024,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    merge_sha = hash_file_sha256(merge_table_path)
    vocab_size = 256 + len(json.load(open(merge_table_path, "r"))["merges"])
    splits = ("train", "validation", "test")
    bin_paths = {s: cache_dir / f"{s}.bin" for s in splits}
    meta_ok = False
    if meta_path.exists():
        with open(meta_path, "r") as f:
            existing = json.load(f)
        if (
            existing.get("merge_table_sha256") == merge_sha
            and existing.get("tokenizer_vocab_size") == vocab_size
            and all(bin_paths[s].exists() for s in splits)
            and all(existing.get("token_counts", {}).get(s, -1) > 0 for s in splits)
        ):
            meta_ok = True
    if meta_ok:
        with open(meta_path, "r") as f:
            return json.load(f)

    token_counts: Dict[str, int] = {}
    byte_counts: Dict[str, int] = {}
    word_counts: Dict[str, int] = {}
    walltime: Dict[str, float] = {}
    for split in splits:
        raw_bytes = sum(len(row.encode("utf-8")) for row in hf_ds[split]["text"])
        raw_text = concat_split_text(hf_ds[split])
        byte_counts[split] = raw_bytes
        word_counts[split] = len(raw_text.split())
        del raw_text
        start = time.perf_counter()
        info = tokenize_split_to_bin(
            hf_ds[split],
            merge_table_path,
            bin_paths[split],
            workers=workers,
            target_shard_bytes=target_shard_bytes,
        )
        walltime[split] = time.perf_counter() - start
        token_counts[split] = info["tokens"]

    meta = {
        "merge_table_path": str(merge_table_path),
        "merge_table_sha256": merge_sha,
        "tokenizer_vocab_size": vocab_size,
        "token_counts": token_counts,
        "byte_counts": byte_counts,
        "word_counts": word_counts,
        "walltime_s": walltime,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


@app.function
def load_split_memmap(cache_dir: Path, split: str) -> np.ndarray:
    return np.memmap(str(cache_dir / f"{split}.bin"), dtype=np.uint16, mode="r")


@app.cell
def _(merge_table_path, wikitext_ds):
    token_cache_dir = default_token_cache_dir()
    cache_meta = ensure_token_cache(
        wikitext_ds,
        merge_table_path,
        token_cache_dir,
        workers=8,
        target_shard_bytes=32 * 1024 * 1024,
    )
    train_ids = load_split_memmap(token_cache_dir, "train")
    val_ids = load_split_memmap(token_cache_dir, "validation")
    test_ids = load_split_memmap(token_cache_dir, "test")
    return cache_meta, test_ids, train_ids, val_ids


@app.cell
def _(cache_meta, mo, test_ids, train_ids, val_ids):
    _tc = cache_meta["token_counts"]
    _bc = cache_meta["byte_counts"]
    _wc = cache_meta["word_counts"]
    _wt = cache_meta["walltime_s"]
    mo.md(
        f"""
    ### Token cache

    Cached at `{Path.home() / "data" / "wikitext103"}`.

    | Split | Tokens | Bytes | Words | Bytes/token | Tokenize wall time |
    |---|---:|---:|---:|---:|---:|
    | train | {_tc["train"]:,} | {_bc["train"]:,} | {_wc["train"]:,} | {_bc["train"] / max(_tc["train"], 1):.2f} | {_wt["train"]:.1f} s |
    | validation | {_tc["validation"]:,} | {_bc["validation"]:,} | {_wc["validation"]:,} | {_bc["validation"] / max(_tc["validation"], 1):.2f} | {_wt["validation"]:.1f} s |
    | test | {_tc["test"]:,} | {_bc["test"]:,} | {_wc["test"]:,} | {_bc["test"] / max(_tc["test"], 1):.2f} | {_wt["test"]:.1f} s |

    Loaded as `np.memmap(..., dtype=uint16, mode="r")` in
    `train_ids` / `val_ids` / `test_ids` — shapes: `{train_ids.shape}`,
    `{val_ids.shape}`, `{test_ids.shape}`.
    """
    )
    return


@app.class_definition
class BinaryBPETokenizerV1:
    """Thin Python wrapper around the user's binary-BPE executables.

    Instances are cheap to build — they only resolve executable paths and read
    the vocab size once. Every method spawns a subprocess and passes data
    through the filesystem, matching the tools' file-only I/O contract.
    """

    def __init__(
        self,
        merge_table_path: Path = default_merge_table_path(),
        tokenize_executable: str = "tokenize_bpe",
        decode_executable: str = "decode_bpe",
    ):
        self.merge_table_path = Path(merge_table_path)
        self.tokenize_exe = resolve_executable(tokenize_executable)
        self.decode_exe = resolve_executable(decode_executable)
        with open(self.merge_table_path, "r") as f:
            payload = json.load(f)
        self._vocab_size = 256 + len(payload["merges"])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode_file(self, src_path: Path, dst_path: Path) -> None:
        subprocess.run(
            [
                str(self.tokenize_exe),
                "-t",
                str(self.merge_table_path),
                "-i",
                str(src_path),
                "-o",
                str(dst_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def encode(self, text: str) -> List[int]:
        if text == "":
            return []
        with tempfile.TemporaryDirectory(prefix="bpe_enc_") as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / "in.txt"
            dst = tmp_dir / "out.ids"
            with open(src, "w", encoding="utf-8") as f:
                f.write(text)
            self.encode_file(src, dst)
            with open(dst, "r") as f:
                blob = f.read()
        return [int(t) for t in blob.split()] if blob.strip() else []

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        with tempfile.TemporaryDirectory(prefix="bpe_dec_") as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / "in.ids"
            dst = tmp_dir / "out.bin"
            with open(src, "w") as f:
                f.write(" ".join(str(int(i)) for i in ids))
            subprocess.run(
                [
                    str(self.decode_exe),
                    "-t",
                    str(self.merge_table_path),
                    "-i",
                    str(src),
                    "-o",
                    str(dst),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with open(dst, "rb") as f:
                raw = f.read()
        return raw.decode("utf-8", errors="replace")


@app.cell
def _(merge_table_path):
    tokenizer = BinaryBPETokenizerV1(merge_table_path=merge_table_path)
    return (tokenizer,)


@app.function
def sample_contiguous_windows(
    memmap_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = memmap_ids.shape[0] - context_length - 1
    if max_start <= 0:
        raise ValueError(
            f"Split has {memmap_ids.shape[0]} tokens; need > context_length+1 = {context_length + 1}."
        )
    starts = rng.integers(0, max_start + 1, size=batch_size, dtype=np.int64)
    x = np.empty((batch_size, context_length), dtype=np.int64)
    y = np.empty((batch_size, context_length), dtype=np.int64)
    for i, s in enumerate(starts):
        chunk = memmap_ids[s : s + context_length + 1].astype(np.int64, copy=False)
        x[i] = chunk[:-1]
        y[i] = chunk[1:]
    return torch.from_numpy(x), torch.from_numpy(y)


@app.function
def iter_sequential_windows(
    memmap_ids: np.ndarray,
    batch_size: int,
    context_length: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    n = memmap_ids.shape[0]
    step = context_length
    starts = list(range(0, n - context_length - 1, step))
    for i in range(0, len(starts), batch_size):
        block = starts[i : i + batch_size]
        b = len(block)
        x = np.empty((b, context_length), dtype=np.int64)
        y = np.empty((b, context_length), dtype=np.int64)
        for j, s in enumerate(block):
            chunk = memmap_ids[s : s + context_length + 1].astype(np.int64, copy=False)
            x[j] = chunk[:-1]
            y[j] = chunk[1:]
        yield torch.from_numpy(x), torch.from_numpy(y)


@app.cell
def _(mo, tokenizer, train_ids):
    _rng = np.random.default_rng(0)
    _x, _y = sample_contiguous_windows(train_ids, batch_size=4, context_length=128, rng=_rng)
    _preview_ids = _x[0, :64].tolist()
    _preview_text = tokenizer.decode(_preview_ids)
    mo.md(
        f"""
    ### Batch preview

    - Random 4×128 batch (contiguous windows):
      shape `{tuple(_x.shape)}`, dtype `{_x.dtype}`, id range `[{int(_x.min())}, {int(_x.max())}]`
    - Decoded first 64 tokens of the first window:

    ```
    {_preview_text}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Model Definition — decoder-only GPT with RoPE + Flash SDPA

    Every module is single-file and version-suffixed. The building blocks are:

    | Component | Class | Notes |
    |---|---|---|
    | Rotary positional embedding | `RotaryPositionalEmbeddingV1` | precomputed `cos`/`sin` cache in fp32 |
    | Causal self-attention | `CausalSelfAttentionV1` | QKV projection, RoPE on q/k, `F.scaled_dot_product_attention` under `sdpa_kernel([...])`, optional KV cache |
    | Feed-forward | `SwiGLUFeedForwardV1` | SwiGLU with hidden = round-multiple-of-64(8/3·d) |
    | Transformer block | `GPTBlockV1` | pre-`RMSNorm` on attn + FFN residual |
    | Top-level | `GPTLanguageModelV1` | token embedding, block stack, final `RMSNorm`, tied `lm_head` |
    | Config | `GPTConfigV1` | dataclass; drives `asdict` sidecar for save/load |

    RoPE is applied to `q` and `k` in fp32 and cast back to the incoming dtype so
    the SDPA kernel enters in bf16. `is_causal=True` gives the top-left aligned
    causal mask during prefill; during single-token decode with a filled KV cache
    (q_len = 1, k_len = past + 1) we pass `is_causal=False` and rely on the query
    already attending only to positions ≤ current.
    """)
    return


@app.class_definition
@dataclass
class GPTConfigV1:
    vocab_size: int = 10_048
    context_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff_multiple: int = 64
    d_ff_ratio: float = 8.0 / 3.0
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.context_length <= 1:
            raise ValueError("context_length must be > 1.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        raw = self.d_model * self.d_ff_ratio
        rounded = int(math.ceil(raw / self.d_ff_multiple) * self.d_ff_multiple)
        return max(rounded, self.d_ff_multiple)


@app.function
def pad_vocab_to_multiple(tokenizer_vocab: int, multiple: int = 64) -> int:
    return int(math.ceil(tokenizer_vocab / multiple) * multiple)


@app.function
def gpt_presets() -> Dict[str, Dict[str, int]]:
    return {
        "tiny": {"n_layers": 6, "d_model": 384, "n_heads": 6},
        "small": {"n_layers": 8, "d_model": 512, "n_heads": 8},
        "base": {"n_layers": 12, "d_model": 768, "n_heads": 12},
        "medium": {"n_layers": 16, "d_model": 1024, "n_heads": 16},
    }


@app.function
def make_gpt_config(
    tokenizer_vocab_size: int,
    preset: str = "base",
    context_length: int = 1024,
    dropout: float = 0.0,
    tie_embeddings: bool = True,
) -> GPTConfigV1:
    presets = gpt_presets()
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {list(presets)}.")
    fields_ = dict(presets[preset])
    padded = pad_vocab_to_multiple(tokenizer_vocab_size, multiple=64)
    return GPTConfigV1(
        vocab_size=padded,
        context_length=context_length,
        n_layers=fields_["n_layers"],
        n_heads=fields_["n_heads"],
        d_model=fields_["d_model"],
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


@app.function
def build_rope_cache(
    context_length: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(context_length, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


@app.function
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


@app.function
def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    out = x_f * cos + rotate_half(x_f) * sin
    return out.to(dtype=orig_dtype)


@app.class_definition
class RotaryPositionalEmbeddingV1(nn.Module):
    def __init__(self, context_length: int = 1024, head_dim: int = 64, base: float = 10000.0):
        super().__init__()
        cos, sin = build_rope_cache(context_length, head_dim, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.cos.shape[0]:
            raise ValueError(
                f"Positions {offset}..{offset + seq_len - 1} exceed the RoPE cache "
                f"(context_length={self.cos.shape[0]})."
            )
        return (
            self.cos[offset : offset + seq_len],
            self.sin[offset : offset + seq_len],
        )


@app.class_definition
class CausalSelfAttentionV1(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        dropout: float = 0.0,
        context_length: int = 1024,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // self.n_heads
        self.d_model = int(d_model)
        self.dropout_p = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbeddingV1(context_length, self.head_dim, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        sdpa_backends: Tuple[SDPBackend, ...] = (SDPBackend.FLASH_ATTENTION,),
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        offset = 0 if kv_cache is None else int(kv_cache.get("length", 0))
        cos, sin = self.rope(t, offset=offset)
        cos = cos.view(1, 1, t, self.head_dim).to(q.device)
        sin = sin.view(1, 1, t, self.head_dim).to(q.device)
        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)

        if kv_cache is not None and "k" in kv_cache:
            past_len = int(kv_cache["length"])
            if past_len > 0 and t > 1:
                raise ValueError(
                    "Multi-token input on a non-empty KV cache is unsupported: SDPA's is_causal mask "
                    "is top-left aligned when q_len != k_len. Prefill once, then decode one token at a time."
                )
            kv_cache["k"][:, :, past_len : past_len + t] = k
            kv_cache["v"][:, :, past_len : past_len + t] = v
            k_full = kv_cache["k"][:, :, : past_len + t]
            v_full = kv_cache["v"][:, :, : past_len + t]
            kv_cache["length"] = past_len + t
            is_causal = past_len == 0 and t > 1
        else:
            k_full = k
            v_full = v
            is_causal = t > 1

        dropout_p = self.dropout_p if self.training else 0.0
        with sdpa_kernel(list(sdpa_backends)):
            out = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.proj(out)


@app.class_definition
class SwiGLUFeedForwardV1(nn.Module):
    def __init__(self, d_model: int = 768, d_ff: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


@app.class_definition
class GPTBlockV1(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.0,
        context_length: int = 1024,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttentionV1(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            context_length=context_length,
            rope_base=rope_base,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLUFeedForwardV1(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        sdpa_backends: Tuple[SDPBackend, ...] = (SDPBackend.FLASH_ATTENTION,),
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), sdpa_backends=sdpa_backends, kv_cache=kv_cache)
        x = x + self.mlp(self.norm2(x))
        return x


@app.class_definition
class GPTLanguageModelV1(nn.Module):
    def __init__(self, config: GPTConfigV1):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                GPTBlockV1(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    context_length=config.context_length,
                    rope_base=config.rope_base,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm_final = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.init_std
        residual_scale = 1.0 / math.sqrt(2.0 * max(self.config.n_layers, 1))
        for name, p in self.named_parameters():
            if p.dim() >= 2 and (
                "qkv.weight" in name
                or "w_gate.weight" in name
                or "w_up.weight" in name
                or "token_embedding.weight" in name
            ):
                nn.init.normal_(p, mean=0.0, std=std)
            elif p.dim() >= 2 and ("attn.proj.weight" in name or "w_down.weight" in name):
                nn.init.normal_(p, mean=0.0, std=std * residual_scale)
            elif p.dim() >= 2 and "lm_head.weight" in name and not self.config.tie_embeddings:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(
        self,
        idx: torch.Tensor,
        sdpa_backends: Tuple[SDPBackend, ...] = (SDPBackend.FLASH_ATTENTION,),
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        h = self.dropout(self.token_embedding(idx))
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            h = block(h, sdpa_backends=sdpa_backends, kv_cache=cache)
        h = self.norm_final(h)
        return self.lm_head(h)

    def make_kv_caches(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Dict[str, torch.Tensor]]:
        caches: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.config.n_layers):
            k = torch.zeros(
                batch_size,
                self.config.n_heads,
                max_length,
                self.config.head_dim,
                device=device,
                dtype=dtype,
            )
            v = torch.zeros_like(k)
            caches.append({"k": k, "v": v, "length": 0})
        return caches


@app.function
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@app.function
def compute_lm_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    tokenizer_vocab_size: int,
    sdpa_backends: Tuple[SDPBackend, ...] = (SDPBackend.FLASH_ATTENTION,),
) -> torch.Tensor:
    logits = model(x, sdpa_backends=sdpa_backends)
    logits_use = logits[..., :tokenizer_vocab_size]
    return F.cross_entropy(
        logits_use.reshape(-1, tokenizer_vocab_size).float(),
        y.reshape(-1),
    )


@app.function
def split_parameters_for_weight_decay(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> List[Dict[str, Any]]:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    seen: set = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


@app.function
def linear_warmup_cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos_out = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cos_out)


@app.cell
def _(device, mo, tokenizer_vocab_size):
    default_config = make_gpt_config(tokenizer_vocab_size, preset="base", context_length=1024)
    _demo_model = GPTLanguageModelV1(default_config).to(device)
    default_param_count = count_parameters(_demo_model)
    _kv_dtype_est = "bf16"
    mo.md(
        f"""
    ### Default preset (`base`)

    - `vocab_size` (padded from {tokenizer_vocab_size}): `{default_config.vocab_size}`
    - `context_length`: `{default_config.context_length}`
    - `n_layers`: `{default_config.n_layers}`, `n_heads`: `{default_config.n_heads}`, `d_model`: `{default_config.d_model}` (`head_dim={default_config.head_dim}`)
    - `d_ff` (SwiGLU): `{default_config.d_ff}` (~8/3 · d_model rounded up to a multiple of 64)
    - Weight tying: `{default_config.tie_embeddings}`
    - **Trainable parameters: {default_param_count:,}**
    """
    )
    del _demo_model
    torch.cuda.empty_cache()
    return (default_config,)


@app.function
def flash_sanity_forward(
    config: GPTConfigV1,
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_size: int = 2,
    seq_len: int = 32,
) -> Tuple[bool, str, Tuple[int, ...]]:
    model = GPTLanguageModelV1(config).to(device)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    try:
        with torch.autocast(device_type=device.type, dtype=amp_dtype), torch.no_grad():
            out = model(idx, sdpa_backends=(SDPBackend.FLASH_ATTENTION,))
        shape = tuple(out.shape)
        del model, idx, out
        torch.cuda.empty_cache()
        return True, "", shape
    except RuntimeError as err:
        del model, idx
        torch.cuda.empty_cache()
        return False, str(err).splitlines()[0], (0,)


@app.cell
def _(amp_dtype, default_config, device, mo):
    _ok, _err, _shape = flash_sanity_forward(default_config, device, amp_dtype)
    if _ok:
        _out = mo.md(f"**Flash SDPA sanity forward:** OK — output shape `{_shape}` under bf16 autocast + `FLASH_ATTENTION`.")
    else:
        _out = mo.md(f"**Flash SDPA sanity forward:** FAILED — `{_err}`")
    _out
    return


@app.function
def kv_cache_equivalence(
    config: GPTConfigV1,
    device: torch.device,
    prompt_len: int = 12,
    decode_len: int = 6,
    seed: int = 0,
) -> Tuple[bool, float]:
    torch.manual_seed(seed)
    model = GPTLanguageModelV1(config).to(device).eval()
    backends = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)
    x = torch.randint(0, config.vocab_size, (1, prompt_len + decode_len), device=device)
    with torch.no_grad():
        full_logits = model(x, sdpa_backends=backends).float()
    caches = model.make_kv_caches(1, prompt_len + decode_len, device=device, dtype=torch.float32)
    with torch.no_grad():
        prefill_logits = model(
            x[:, :prompt_len],
            sdpa_backends=backends,
            kv_caches=caches,
        ).float()
        step_logits = [prefill_logits[:, -1, :]]
        for i in range(decode_len):
            step = model(
                x[:, prompt_len + i : prompt_len + i + 1],
                sdpa_backends=backends,
                kv_caches=caches,
            ).float()
            step_logits.append(step[:, -1, :])
    incremental = torch.stack(step_logits, dim=1)
    reference = full_logits[:, prompt_len - 1 :, :]
    diff = (incremental - reference).abs().max().item()
    ok = diff < 1e-3
    del model, x, full_logits, caches, prefill_logits, step_logits, incremental, reference
    torch.cuda.empty_cache()
    return ok, float(diff)


@app.cell
def _(device, mo, tokenizer_vocab_size):
    _tiny_config = make_gpt_config(tokenizer_vocab_size, preset="tiny", context_length=64)
    _ok, _diff = kv_cache_equivalence(_tiny_config, device, prompt_len=12, decode_len=6)
    _label = "PASSED" if _ok else "FAILED"
    mo.md(
        f"**KV-cache equivalence (tiny cfg, fp32, efficient+math backends):** "
        f"{_label} — max |Δlogits| = `{_diff:.3e}` (tolerance `1e-3`)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Training

    ### Controls

    - **Preset**: tiny / small / base / medium — sets `n_layers × d_model × n_heads`.
    - **Learning rate**: dropdown; default `6e-4`.
    - **Weight decay**: dropdown; default `0.1` (applied only to ≥ 2-D weight matrices).
    - **Micro-batch**: dropdown `[8, 16, 32]`.
    - **Gradient accumulation**: dropdown `[1, 2, 4, 8]`.
    - **Epochs**: slider `0.05 – 3.00` (step `0.05`, fractional pass over train tokens),
      converted to an optimizer-step count based on the effective batch.
    - **Warmup fraction**: dropdown.
    - **Val batches during training**: dropdown; number of fixed validation windows used
      at each periodic-eval milestone.
    - Click **Train** to start.

    Every training helper uses `torch.autocast(device_type=device.type, dtype=amp_dtype)`
    around the forward pass + loss, and a `GradScaler` that is a no-op under bf16
    (`enabled=(amp_dtype == torch.float16)`) so the same code path also works if the
    user selects fp16.
    """)
    return


@app.cell
def _(mo):
    preset_ui = mo.ui.dropdown(
        options=["tiny", "small", "base", "medium"],
        value="base",
        label="Preset",
    )
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "6e-4": 6e-4, "1e-3": 1e-3},
        value="6e-4",
        label="Learning Rate",
    )
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "0.01": 0.01, "0.05": 0.05, "0.1": 0.1},
        value="0.1",
        label="Weight Decay",
    )
    micro_bs_ui = mo.ui.dropdown(options=[8, 16, 32], value=16, label="Micro-batch")
    grad_accum_ui = mo.ui.dropdown(options=[1, 2, 4, 8], value=4, label="Grad Accum")
    epochs_ui = mo.ui.slider(0.05, 3.0, value=1.0, step=0.05, label="Epochs (fraction of train tokens)")
    warmup_frac_ui = mo.ui.dropdown(
        options={"0.005": 0.005, "0.01": 0.01, "0.03": 0.03, "0.05": 0.05},
        value="0.01",
        label="Warmup Fraction",
    )
    val_batches_ui = mo.ui.dropdown(options=[8, 16, 32, 64], value=16, label="Val batches per check")
    eval_every_ui = mo.ui.dropdown(options=[50, 100, 200, 500], value=100, label="Eval every N steps")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Hyperparameters"),
            mo.hstack([preset_ui, lr_ui, wd_ui]),
            mo.hstack([micro_bs_ui, grad_accum_ui, epochs_ui]),
            mo.hstack([warmup_frac_ui, val_batches_ui, eval_every_ui]),
            train_btn,
        ]
    )
    return (
        epochs_ui,
        eval_every_ui,
        grad_accum_ui,
        lr_ui,
        micro_bs_ui,
        preset_ui,
        train_btn,
        val_batches_ui,
        warmup_frac_ui,
        wd_ui,
    )


@app.function
def build_gpt_model_and_optimizer(
    tokenizer_vocab_size: int,
    preset: str,
    context_length: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    dropout: float = 0.0,
) -> Tuple[GPTLanguageModelV1, torch.optim.Optimizer]:
    config = make_gpt_config(
        tokenizer_vocab_size,
        preset=preset,
        context_length=context_length,
        dropout=dropout,
    )
    model = GPTLanguageModelV1(config).to(device)
    param_groups = split_parameters_for_weight_decay(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    return model, optimizer


@app.function
def move_batch_to_device(
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
    return x, y


@app.function
def run_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    train_ids: np.ndarray,
    rng: np.random.Generator,
    micro_batch: int,
    context_length: int,
    grad_accum: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    tokenizer_vocab_size: int,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for _ in range(grad_accum):
        x, y = sample_contiguous_windows(train_ids, micro_batch, context_length, rng)
        x, y = move_batch_to_device((x, y), device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            loss = compute_lm_loss(
                model, x, y, tokenizer_vocab_size, sdpa_backends=(SDPBackend.FLASH_ATTENTION,)
            )
            scaled = loss / grad_accum
        scaler.scale(scaled).backward()
        total_loss += float(loss.detach())
    if grad_clip > 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return total_loss / grad_accum


@app.function
def evaluate_windows(
    model: nn.Module,
    ids: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    tokenizer_vocab_size: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0
    with torch.no_grad():
        for x, y in iter_sequential_windows(ids, batch_size, context_length):
            x, y = move_batch_to_device((x, y), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(
                    x, sdpa_backends=(SDPBackend.FLASH_ATTENTION,)
                )[..., :tokenizer_vocab_size]
                loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer_vocab_size).float(),
                    y.reshape(-1),
                    reduction="sum",
                )
            total_nll += float(loss.detach())
            total_tokens += int(y.numel())
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "loss": mean_nll,
        "perplexity": float(math.exp(mean_nll)),
        "total_nll": float(total_nll),
        "total_tokens": int(total_tokens),
        "n_batches": int(n_batches),
    }


@app.function
def compute_num_steps(
    train_tokens: int,
    tokens_per_optim_step: int,
    epochs_fraction: float,
) -> int:
    if tokens_per_optim_step <= 0:
        return 0
    return max(1, int(round(train_tokens * epochs_fraction / tokens_per_optim_step)))


@app.function
def train_gpt_language_model(
    tokenizer_vocab_size: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    preset: str,
    context_length: int,
    lr: float,
    weight_decay: float,
    micro_batch: int,
    grad_accum: int,
    num_steps: int,
    warmup_steps: int,
    eval_every: int,
    val_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    model, optimizer = build_gpt_model_and_optimizer(
        tokenizer_vocab_size=tokenizer_vocab_size,
        preset=preset,
        context_length=context_length,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))
    rng = np.random.default_rng(seed)
    train_losses: List[float] = []
    val_history: List[Dict[str, float]] = []
    lr_history: List[float] = []

    tokens_per_step = micro_batch * grad_accum * context_length
    start_time = time.perf_counter()
    for step in range(num_steps):
        scale = linear_warmup_cosine_schedule(step, num_steps, warmup_steps)
        current_lr = lr * scale
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        loss = run_train_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_ids=train_ids,
            rng=rng,
            micro_batch=micro_batch,
            context_length=context_length,
            grad_accum=grad_accum,
            device=device,
            amp_dtype=amp_dtype,
            tokenizer_vocab_size=tokenizer_vocab_size,
        )
        train_losses.append(loss)
        lr_history.append(current_lr)

        elapsed = time.perf_counter() - start_time
        tok_per_s = tokens_per_step * (step + 1) / max(elapsed, 1e-6)
        latest_val: Optional[Dict[str, float]] = None
        if (step + 1) % eval_every == 0 or (step + 1) == num_steps:
            val_stats = evaluate_windows(
                model=model,
                ids=val_ids,
                batch_size=micro_batch,
                context_length=context_length,
                device=device,
                amp_dtype=amp_dtype,
                tokenizer_vocab_size=tokenizer_vocab_size,
                max_batches=val_batches,
            )
            val_history.append({"step": step + 1, **val_stats})
            latest_val = val_stats

        if progress_callback is not None:
            progress_callback(
                {
                    "step": step + 1,
                    "num_steps": num_steps,
                    "loss": loss,
                    "lr": current_lr,
                    "tokens_per_s": tok_per_s,
                    "elapsed_s": elapsed,
                    "latest_val": latest_val,
                }
            )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_history": val_history,
        "lr_history": lr_history,
        "tokens_seen": tokens_per_step * num_steps,
        "elapsed_s": time.perf_counter() - start_time,
        "config": model.config,
    }


@app.cell
def _(
    amp_dtype,
    device,
    epochs_ui,
    eval_every_ui,
    grad_accum_ui,
    lr_ui,
    micro_bs_ui,
    mo,
    preset_ui,
    seed_ui,
    tokenizer_vocab_size,
    train_btn,
    train_ids,
    val_batches_ui,
    val_ids,
    warmup_frac_ui,
    wd_ui,
):
    train_losses: List[float] = []
    val_history: List[Dict[str, float]] = []
    lr_history: List[float] = []
    trained_model: Optional[GPTLanguageModelV1] = None
    train_summary: Dict[str, Any] = {}

    if not train_btn.value:
        mo.output.replace(
            mo.md("Choose hyperparameters and click **Train** to begin. "
                  "Nothing is trained until you click.")
        )
    else:
        _context_length = 1024
        _tokens_per_step = int(micro_bs_ui.value) * int(grad_accum_ui.value) * _context_length
        _num_steps = compute_num_steps(
            train_tokens=int(train_ids.shape[0]),
            tokens_per_optim_step=_tokens_per_step,
            epochs_fraction=float(epochs_ui.value),
        )
        _warmup_steps = max(1, int(round(_num_steps * float(warmup_frac_ui.value))))
        mo.output.replace(
            mo.md(
                f"Starting training — preset `{preset_ui.value}`, "
                f"{_num_steps:,} steps × {_tokens_per_step:,} tokens/step "
                f"= {_num_steps * _tokens_per_step:,} tokens."
            )
        )

        def _cb(info: Dict[str, Any]) -> None:
            eta_s = info["elapsed_s"] * (info["num_steps"] / max(info["step"], 1) - 1.0)
            latest_val = info.get("latest_val")
            val_str = (
                f" | val_loss: {latest_val['loss']:.4f}, ppl: {latest_val['perplexity']:.2f}"
                if latest_val is not None
                else ""
            )
            mo.output.replace(
                mo.md(
                    f"**step {info['step']}/{info['num_steps']}** "
                    f"— loss: {info['loss']:.4f} "
                    f"| lr: {info['lr']:.2e} "
                    f"| toks/s: {info['tokens_per_s']:,.0f} "
                    f"| elapsed: {info['elapsed_s']:.1f}s "
                    f"| ETA: {eta_s:.1f}s{val_str}"
                )
            )

        _result = train_gpt_language_model(
            tokenizer_vocab_size=int(tokenizer_vocab_size),
            train_ids=train_ids,
            val_ids=val_ids,
            preset=str(preset_ui.value),
            context_length=_context_length,
            lr=float(lr_ui.value),
            weight_decay=float(wd_ui.value),
            micro_batch=int(micro_bs_ui.value),
            grad_accum=int(grad_accum_ui.value),
            num_steps=_num_steps,
            warmup_steps=_warmup_steps,
            eval_every=int(eval_every_ui.value),
            val_batches=int(val_batches_ui.value),
            device=device,
            amp_dtype=amp_dtype,
            seed=int(seed_ui.value),
        )
        trained_model = _result["model"]
        train_losses = _result["train_losses"]
        val_history = _result["val_history"]
        lr_history = _result["lr_history"]
        train_summary = {
            "preset": str(preset_ui.value),
            "num_steps": _num_steps,
            "tokens_seen": _result["tokens_seen"],
            "elapsed_s": _result["elapsed_s"],
            "final_train_loss": float(train_losses[-1]) if train_losses else float("nan"),
            "final_val_loss": float(val_history[-1]["loss"]) if val_history else float("nan"),
            "final_val_ppl": float(val_history[-1]["perplexity"]) if val_history else float("nan"),
        }
        mo.output.replace(
            mo.md(
                f"**Training complete.** {train_summary['num_steps']} steps, "
                f"{train_summary['tokens_seen']:,} tokens, "
                f"{train_summary['elapsed_s']:.1f} s.  "
                f"Final train loss: **{train_summary['final_train_loss']:.4f}**, "
                f"val loss: **{train_summary['final_val_loss']:.4f}** "
                f"(ppl {train_summary['final_val_ppl']:.2f})."
            )
        )
    return lr_history, train_losses, train_summary, trained_model, val_history


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Hyperparameter Search (optional)

    A short grid over `lr × preset` on the `tiny` / `small` presets with a fixed
    500-step budget per configuration. The whole grid runs in a few minutes on the
    RTX 5080. Results are sorted by validation loss.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_search(
    tokenizer_vocab_size: int,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    lrs: List[float],
    presets: List[str],
    num_steps: int = 500,
    micro_batch: int = 8,
    grad_accum: int = 2,
    context_length: int = 512,
    val_batches: int = 8,
    weight_decay: float = 0.1,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for preset in presets:
        for lr in lrs:
            outcome = train_gpt_language_model(
                tokenizer_vocab_size=tokenizer_vocab_size,
                train_ids=train_ids,
                val_ids=val_ids,
                preset=preset,
                context_length=context_length,
                lr=lr,
                weight_decay=weight_decay,
                micro_batch=micro_batch,
                grad_accum=grad_accum,
                num_steps=num_steps,
                warmup_steps=max(1, num_steps // 20),
                eval_every=num_steps,
                val_batches=val_batches,
                device=device,
                amp_dtype=amp_dtype,
                seed=seed,
            )
            final_val = outcome["val_history"][-1] if outcome["val_history"] else {"loss": float("nan"), "perplexity": float("nan")}
            results.append(
                {
                    "preset": preset,
                    "lr": lr,
                    "val_loss": round(float(final_val["loss"]), 4),
                    "val_ppl": round(float(final_val["perplexity"]), 3),
                    "final_train_loss": round(float(outcome["train_losses"][-1]), 4) if outcome["train_losses"] else float("nan"),
                    "elapsed_s": round(float(outcome["elapsed_s"]), 1),
                }
            )
            del outcome
            torch.cuda.empty_cache()
    results.sort(key=lambda r: r["val_loss"])
    return results


@app.cell
def _(
    amp_dtype,
    device,
    hp_search_cb,
    mo,
    seed_ui,
    tokenizer_vocab_size,
    train_ids,
    val_ids,
):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    hp_results = run_hp_search(
        tokenizer_vocab_size=int(tokenizer_vocab_size),
        train_ids=train_ids,
        val_ids=val_ids,
        device=device,
        amp_dtype=amp_dtype,
        seed=int(seed_ui.value),
        lrs=[3e-4, 6e-4, 1e-3],
        presets=["tiny", "small"],
        num_steps=500,
        micro_batch=8,
        grad_accum=2,
        context_length=512,
        val_batches=8,
    )
    mo.ui.table(hp_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Validation & Cross-Validation

    ### Test-set metrics

    The trained model is evaluated on the WikiText-103 test split via a
    sequential non-overlapping window sweep at `context_length`. We report:

    - **Test loss** (mean cross-entropy in *nats/token*)
    - **Token perplexity** = exp(loss)
    - **Bits-per-byte** = total NLL / (ln 2 · UTF-8 bytes of the split)
      — a tokenizer-agnostic language-modeling metric
    - **Word perplexity** = exp(total NLL / whitespace word count of the split)
      — for rough comparison with published WikiText-103 numbers

    ### 5-fold cross-validation

    5 contiguous folds of the train token stream. For each fold we train a fresh
    small-preset model for a short fixed budget on the other 4 folds and evaluate
    on the held-out fold. Gated behind its own checkbox because it trains 5 models.
    """)
    return


@app.function
def evaluate_model(
    model: nn.Module,
    ids: np.ndarray,
    context_length: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    tokenizer_vocab_size: int,
    micro_batch: int = 8,
) -> Dict[str, float]:
    return evaluate_windows(
        model=model,
        ids=ids,
        batch_size=micro_batch,
        context_length=context_length,
        device=device,
        amp_dtype=amp_dtype,
        tokenizer_vocab_size=tokenizer_vocab_size,
        max_batches=None,
    )


@app.function
def compute_lm_metrics(
    eval_stats: Dict[str, float],
    raw_bytes: int,
    raw_words: int,
) -> Dict[str, float]:
    total_nll = float(eval_stats["total_nll"])
    total_tokens = int(eval_stats["total_tokens"])
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "loss_nats_per_token": mean_nll,
        "token_perplexity": float(math.exp(mean_nll)),
        "bits_per_byte": total_nll / (math.log(2.0) * max(raw_bytes, 1)),
        "word_perplexity": float(math.exp(total_nll / max(raw_words, 1))),
        "eval_tokens": total_tokens,
    }


@app.cell
def _(
    amp_dtype,
    cache_meta,
    device,
    mo,
    test_ids,
    tokenizer_vocab_size,
    trained_model: Optional[GPTLanguageModelV1],
):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before test-set evaluation._")
    else:
        _stats = evaluate_model(
            model=trained_model,
            ids=test_ids,
            context_length=trained_model.config.context_length,
            device=device,
            amp_dtype=amp_dtype,
            tokenizer_vocab_size=int(tokenizer_vocab_size),
            micro_batch=8,
        )
        _metrics = compute_lm_metrics(
            _stats,
            raw_bytes=int(cache_meta["byte_counts"]["test"]),
            raw_words=int(cache_meta["word_counts"]["test"]),
        )
        _out = mo.md(
            f"""
    ### Test-set metrics

    | Metric | Value |
    |---|---:|
    | Test loss (nats/token) | `{_metrics["loss_nats_per_token"]:.4f}` |
    | Token perplexity | `{_metrics["token_perplexity"]:.3f}` |
    | Bits per byte | `{_metrics["bits_per_byte"]:.4f}` |
    | Word perplexity | `{_metrics["word_perplexity"]:.3f}` |
    | Eval tokens seen | `{_metrics["eval_tokens"]:,}` |
    """
        )
    _out
    return


@app.cell
def _(mo):
    cv_cb = mo.ui.checkbox(label="Enable 5-fold Cross-Validation (trains 5 models)", value=False)
    cv_cb
    return (cv_cb,)


@app.function
def make_train_stream_folds(train_ids: np.ndarray, k: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = int(train_ids.shape[0])
    fold_size = n // k
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        val_slice = np.asarray(train_ids[val_start:val_end])
        train_slice = np.concatenate(
            [np.asarray(train_ids[:val_start]), np.asarray(train_ids[val_end:])]
        )
        folds.append((train_slice, val_slice))
    return folds


@app.function
def run_cross_validation(
    tokenizer_vocab_size: int,
    train_ids: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    k: int = 5,
    preset: str = "tiny",
    context_length: int = 256,
    num_steps: int = 300,
    micro_batch: int = 8,
    grad_accum: int = 2,
) -> Dict[str, Any]:
    folds = make_train_stream_folds(train_ids, k=k)
    per_fold: List[Dict[str, float]] = []
    for fold_idx, (fold_train, fold_val) in enumerate(folds):
        outcome = train_gpt_language_model(
            tokenizer_vocab_size=tokenizer_vocab_size,
            train_ids=fold_train,
            val_ids=fold_val,
            preset=preset,
            context_length=context_length,
            lr=6e-4,
            weight_decay=0.1,
            micro_batch=micro_batch,
            grad_accum=grad_accum,
            num_steps=num_steps,
            warmup_steps=max(1, num_steps // 20),
            eval_every=num_steps,
            val_batches=8,
            device=device,
            amp_dtype=amp_dtype,
            seed=seed + fold_idx,
        )
        final_val = outcome["val_history"][-1] if outcome["val_history"] else {"loss": float("nan"), "perplexity": float("nan")}
        per_fold.append(
            {
                "fold": fold_idx + 1,
                "val_loss": round(float(final_val["loss"]), 4),
                "val_ppl": round(float(final_val["perplexity"]), 3),
                "final_train_loss": round(float(outcome["train_losses"][-1]), 4) if outcome["train_losses"] else float("nan"),
            }
        )
        del outcome
        torch.cuda.empty_cache()
    losses = np.array([r["val_loss"] for r in per_fold], dtype=np.float64)
    return {
        "folds": per_fold,
        "mean_loss": float(losses.mean()),
        "std_loss": float(losses.std()),
        "mean_ppl": float(math.exp(losses.mean())),
    }


@app.cell
def _(amp_dtype, cv_cb, device, mo, seed_ui, tokenizer_vocab_size, train_ids):
    mo.stop(
        not cv_cb.value,
        mo.md("_Enable 5-fold CV above to run this section (it trains 5 models)._"),
    )
    cv_result = run_cross_validation(
        tokenizer_vocab_size=int(tokenizer_vocab_size),
        train_ids=train_ids,
        device=device,
        amp_dtype=amp_dtype,
        seed=int(seed_ui.value),
        k=5,
        preset="tiny",
        context_length=256,
        num_steps=300,
        micro_batch=8,
        grad_accum=2,
    )
    mo.vstack(
        [
            mo.md(
                f"**5-fold CV — val loss:** {cv_result['mean_loss']:.4f} ± {cv_result['std_loss']:.4f} "
                f"(implied PPL {cv_result['mean_ppl']:.2f})"
            ),
            mo.ui.table(cv_result["folds"]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Results
    """)
    return


@app.function
def plot_loss_curves(
    train_losses: List[float],
    val_history: List[Dict[str, float]],
    smooth_window: int = 25,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    if train_losses:
        steps = np.arange(1, len(train_losses) + 1)
        ax.plot(steps, train_losses, color="steelblue", alpha=0.3, lw=1, label="train (raw)")
        if len(train_losses) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(train_losses, kernel, mode="valid")
            smooth_steps = steps[smooth_window - 1 :]
            ax.plot(smooth_steps, smoothed, color="steelblue", lw=2, label=f"train (SMA {smooth_window})")
    if val_history:
        val_steps = [v["step"] for v in val_history]
        val_losses = [v["loss"] for v in val_history]
        ax.plot(val_steps, val_losses, "r-s", lw=2, ms=4, label="validation")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss (nats/token)")
    ax.set_title("Training & validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses: List[float], val_history: List[Dict[str, float]]):
    if not train_losses:
        _out = mo.md("_Train the model first (Section 5) to see loss curves._")
    else:
        _out = plot_loss_curves(train_losses, val_history)
    _out
    return


@app.function
def plot_lr_schedule(lr_history: List[float]):
    fig, ax = plt.subplots(figsize=(9, 3))
    if lr_history:
        ax.plot(np.arange(1, len(lr_history) + 1), lr_history, color="darkorange", lw=2)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedule (warmup + cosine decay)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(lr_history: List[float], mo):
    if not lr_history:
        _out = mo.md("_Train the model first (Section 5) to see the LR schedule._")
    else:
        _out = plot_lr_schedule(lr_history)
    _out
    return


@app.function
def build_variants_table(
    train_summary: Dict[str, Any],
    hp_results: Optional[List[Dict[str, Any]]] = None,
    cv_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if train_summary:
        rows.append(
            {
                "source": "main train",
                "preset": train_summary.get("preset", ""),
                "val_loss": round(float(train_summary.get("final_val_loss", float("nan"))), 4),
                "val_ppl": round(float(train_summary.get("final_val_ppl", float("nan"))), 3),
                "notes": f"{train_summary.get('num_steps', 0):,} steps, {train_summary.get('tokens_seen', 0):,} tokens",
            }
        )
    if hp_results:
        for r in hp_results[:5]:
            rows.append(
                {
                    "source": "hp search",
                    "preset": r["preset"],
                    "val_loss": r["val_loss"],
                    "val_ppl": r["val_ppl"],
                    "notes": f"lr={r['lr']:.1e}, {r['elapsed_s']}s",
                }
            )
    if cv_result:
        rows.append(
            {
                "source": "5-fold CV mean",
                "preset": "tiny",
                "val_loss": round(float(cv_result["mean_loss"]), 4),
                "val_ppl": round(float(cv_result["mean_ppl"]), 3),
                "notes": f"± {cv_result['std_loss']:.4f}",
            }
        )
    return rows


@app.cell
def _(mo, train_summary: Dict[str, Any]):
    _hp = globals().get("hp_results", None)
    _cv = globals().get("cv_result", None)
    _rows = build_variants_table(train_summary or {}, _hp, _cv)
    if not _rows:
        _out = mo.md("_Train (Section 5), optionally run HP search / CV, to populate this table._")
    else:
        _out = mo.ui.table(_rows)
    _out
    return


@app.cell
def _(mo, train_summary: Dict[str, Any]):
    if not train_summary:
        _out = mo.md("_Train the model first (Section 5) to see the summary._")
    else:
        _out = mo.md(
            f"""
    ### Training summary

    - Preset: **{train_summary.get('preset', '')}**
    - Optimizer steps: **{train_summary.get('num_steps', 0):,}**
    - Tokens seen: **{train_summary.get('tokens_seen', 0):,}**
    - Wall time: **{train_summary.get('elapsed_s', 0):.1f} s**
    - Final train loss: **{train_summary.get('final_train_loss', float('nan')):.4f}**
    - Final validation loss: **{train_summary.get('final_val_loss', float('nan')):.4f}** (ppl {train_summary.get('final_val_ppl', float('nan')):.3f})
    """
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Sampling from the Trained Model

    KV-cached autoregressive sampling: the prompt is prefilled with `is_causal=True`
    on empty caches, then tokens are appended one at a time (`is_causal=False`,
    `q_len=1`). Positions passed to RoPE are offset by the current cache length.
    Logits are masked at ids ≥ tokenizer vocab size before sampling, so padded
    vocab slots (indices ≥ {tokenizer_vocab_size}) never appear in the output.
    """)
    return


@app.cell
def _(mo):
    sample_prompt_ui = mo.ui.text_area(
        value="The history of language modeling",
        label="Prompt",
    )
    sample_max_tokens_ui = mo.ui.slider(1, 512, value=128, step=1, label="Max new tokens")
    sample_temperature_ui = mo.ui.slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
    sample_topk_ui = mo.ui.slider(0, 500, value=50, step=1, label="top-k (0 = off)")
    sample_topp_ui = mo.ui.slider(0.0, 1.0, value=0.95, step=0.01, label="top-p")
    sample_seed_ui = mo.ui.number(value=42, label="Seed", start=0, stop=2**31 - 1)
    generate_btn = mo.ui.run_button(label="Generate")
    mo.vstack(
        [
            sample_prompt_ui,
            mo.hstack([sample_max_tokens_ui, sample_temperature_ui]),
            mo.hstack([sample_topk_ui, sample_topp_ui, sample_seed_ui]),
            generate_btn,
        ]
    )
    return (
        generate_btn,
        sample_max_tokens_ui,
        sample_prompt_ui,
        sample_seed_ui,
        sample_temperature_ui,
        sample_topk_ui,
        sample_topp_ui,
    )


@app.function
def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_k and top_k > 0:
        top_vals, _ = torch.topk(logits, top_k)
        kth = top_vals[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = cumulative - probs > top_p
        cutoff[..., 0] = False
        sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


@app.function
def generate(
    model: nn.Module,
    prompt_ids: List[int],
    max_new_tokens: int,
    tokenizer_vocab_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    sdpa_backends: Tuple[SDPBackend, ...] = (SDPBackend.FLASH_ATTENTION,),
) -> List[int]:
    was_training = model.training
    model.eval()
    ctx_len = int(model.config.context_length)
    max_new_tokens = max(0, min(int(max_new_tokens), ctx_len - 1))
    output: List[int] = list(prompt_ids) if prompt_ids else [0]
    # RoPE only covers ctx_len positions: condition on the prompt tail so prompt + new <= ctx_len
    context_ids = output[-(ctx_len - max_new_tokens) :]
    total_len = len(context_ids) + max_new_tokens
    caches = model.make_kv_caches(
        batch_size=1,
        max_length=total_len,
        device=device,
        dtype=amp_dtype if device.type == "cuda" else torch.float32,
    )
    tokens = torch.tensor([context_ids], device=device, dtype=torch.long)
    generator = torch.Generator(device=device.type if device.type == "cpu" else "cpu")
    generator.manual_seed(int(seed))
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
            prefill_logits = model(tokens, sdpa_backends=sdpa_backends, kv_caches=caches)
            last_logits = prefill_logits[:, -1, :tokenizer_vocab_size].float()
            for _ in range(max_new_tokens):
                if temperature > 0.0:
                    scaled = last_logits / temperature
                    scaled = apply_top_k_top_p(scaled, top_k=top_k, top_p=top_p)
                    probs = torch.softmax(scaled, dim=-1)
                    probs_cpu = probs.detach().to("cpu")
                    next_id = int(torch.multinomial(probs_cpu, num_samples=1, generator=generator).item())
                else:
                    next_id = int(last_logits.argmax(dim=-1).item())
                output.append(next_id)
                next_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
                logits_step = model(next_tok, sdpa_backends=sdpa_backends, kv_caches=caches)
                last_logits = logits_step[:, -1, :tokenizer_vocab_size].float()
    finally:
        if was_training:
            model.train()
    return output


@app.cell
def _(
    amp_dtype,
    device,
    generate_btn,
    mo,
    sample_max_tokens_ui,
    sample_prompt_ui,
    sample_seed_ui,
    sample_temperature_ui,
    sample_topk_ui,
    sample_topp_ui,
    tokenizer,
    tokenizer_vocab_size,
    trained_model: Optional[GPTLanguageModelV1],
):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) or load a saved model (Section 11) before sampling._")
    elif not generate_btn.value:
        _out = mo.md("Enter a prompt and click **Generate**.")
    else:
        _prompt_ids = tokenizer.encode(str(sample_prompt_ui.value))
        _out_ids = generate(
            model=trained_model,
            prompt_ids=_prompt_ids,
            max_new_tokens=int(sample_max_tokens_ui.value),
            tokenizer_vocab_size=int(tokenizer_vocab_size),
            device=device,
            amp_dtype=amp_dtype,
            temperature=float(sample_temperature_ui.value),
            top_k=int(sample_topk_ui.value),
            top_p=float(sample_topp_ui.value),
            seed=int(sample_seed_ui.value),
        )
        _generated = tokenizer.decode(_out_ids)
        _new_only = tokenizer.decode(_out_ids[len(_prompt_ids):])
        _out = mo.md(
            f"""
    **Prompt tokens:** {len(_prompt_ids)}, **new tokens:** {int(sample_max_tokens_ui.value)}

    **Generated (prompt + continuation):**

    ```
    {_generated}
    ```

    **Continuation only:**

    ```
    {_new_only}
    ```
    """
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Save the Trained Model

    Writes a two-file checkpoint under `<repo>/models/`:

    - `<name>.pt` — `model.state_dict()`
    - `<name>.json` — sidecar containing the model config (`asdict(model.config)`),
      the merge-table path + tokenizer vocab size, and a short training summary
      (tokens seen, final val loss, elapsed wall time).

    Both files are needed by Section 11 to reload the model without training.
    """)
    return


@app.function
def save_gpt_model_and_config(
    model: GPTLanguageModelV1,
    models_dir: Path,
    filename: str,
    tokenizer_vocab_size: int,
    merge_table_path: Path,
    training_summary: Dict[str, Any],
) -> Tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / filename
    config_path = weights_path.with_suffix(".json")
    torch.save(model.state_dict(), weights_path)
    sidecar = {
        "model_class": "GPTLanguageModelV1",
        "config": asdict(model.config),
        "tokenizer_vocab_size": int(tokenizer_vocab_size),
        "merge_table_path": str(merge_table_path),
        "training_summary": training_summary,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(config_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return weights_path, config_path


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="wikitext103_gpt_v1.pt",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_btn])
    return save_btn, save_filename_ui


@app.cell
def _(
    merge_table_path,
    mo,
    save_btn,
    save_filename_ui,
    tokenizer_vocab_size,
    train_summary: Dict[str, Any],
    trained_model: Optional[GPTLanguageModelV1],
):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before saving._")
    elif not save_btn.value:
        _out = mo.md("Enter a filename and click **Save Model** to write the trained weights to `models/`.")
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _fname = str(save_filename_ui.value).strip() or "wikitext103_gpt_v1.pt"
        _weights_path, _config_path = save_gpt_model_and_config(
            model=trained_model,
            models_dir=_models_dir,
            filename=_fname,
            tokenizer_vocab_size=int(tokenizer_vocab_size),
            merge_table_path=merge_table_path,
            training_summary=train_summary or {},
        )
        _out = mo.md(
            f"""
    **Saved.**

    - weights: `{_weights_path}`
    - config:  `{_config_path}`
    """
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Load a Saved Model & Sample

    This section is fully independent of Section 5's training run. It scans
    `<repo>/models/*.pt` for files that have a matching `.json` sidecar,
    rebuilds `GPTLanguageModelV1` from that sidecar, loads the state dict,
    and offers its own sampling UI (distinct variable names from Section 9).
    """)
    return


@app.function
def list_saved_checkpoints(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    return sorted(
        p for p in models_dir.glob("*.pt")
        if p.with_suffix(".json").exists()
    )


@app.function
def load_gpt_model_from_disk(
    weights_path: Path,
    device: torch.device,
) -> Tuple[GPTLanguageModelV1, Dict[str, Any]]:
    config_path = weights_path.with_suffix(".json")
    with open(config_path, "r") as f:
        sidecar = json.load(f)
    config = GPTConfigV1(**sidecar["config"])
    model = GPTLanguageModelV1(config).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, sidecar


@app.cell
def _(mo):
    _models_dir = Path(__file__).resolve().parent.parent / "models"
    checkpoint_options = list_saved_checkpoints(_models_dir)
    if not checkpoint_options:
        checkpoint_dropdown = mo.ui.dropdown(options=["<no checkpoints found>"], value="<no checkpoints found>", label="Checkpoint")
    else:
        checkpoint_dropdown = mo.ui.dropdown(
            options={p.name: str(p) for p in checkpoint_options},
            value=checkpoint_options[0].name,
            label="Checkpoint",
        )
    load_btn = mo.ui.run_button(label="Load Model")
    mo.vstack([checkpoint_dropdown, load_btn])
    return checkpoint_dropdown, load_btn


@app.cell
def _(
    amp_dtype,
    checkpoint_dropdown,
    device,
    load_btn,
    mo,
    tokenizer_vocab_size,
    val_ids,
):
    loaded_model: Optional[GPTLanguageModelV1] = None
    loaded_sidecar: Dict[str, Any] = {}
    loaded_val_metrics: Dict[str, float] = {}

    if not load_btn.value:
        mo.output.replace(
            mo.md("Choose a checkpoint and click **Load Model**. Nothing is loaded until you click.")
        )
    else:
        _ckpt = str(checkpoint_dropdown.value)
        if not _ckpt or "no checkpoints" in _ckpt:
            mo.output.replace(mo.md("_No checkpoints available. Save one via Section 10 first._"))
        else:
            _weights_path = Path(_ckpt)
            _model, _sidecar = load_gpt_model_from_disk(_weights_path, device)
            loaded_model = _model
            loaded_sidecar = _sidecar
            _val = evaluate_windows(
                model=_model,
                ids=val_ids,
                batch_size=8,
                context_length=_model.config.context_length,
                device=device,
                amp_dtype=amp_dtype,
                tokenizer_vocab_size=int(tokenizer_vocab_size),
                max_batches=16,
            )
            loaded_val_metrics = _val
            _params = count_parameters(_model)
            mo.output.replace(
                mo.md(
                    f"""
    **Loaded.** `{_weights_path.name}`

    | Field | Value |
    |---|---|
    | Parameters (trainable) | `{_params:,}` |
    | Config | `{_sidecar.get("config", {})}` |
    | Quick val loss (16 batches × 8 × ctx) | `{_val["loss"]:.4f}` (ppl `{_val["perplexity"]:.3f}`) |
    | Merge table | `{_sidecar.get("merge_table_path", "")}` |
    | Training summary at save-time | `{_sidecar.get("training_summary", {})}` |
    """
                )
            )
    return (loaded_model,)


@app.cell
def _(mo):
    loaded_prompt_ui = mo.ui.text_area(
        value="In the beginning",
        label="Prompt (loaded model)",
    )
    loaded_max_tokens_ui = mo.ui.slider(1, 512, value=128, step=1, label="Max new tokens")
    loaded_temperature_ui = mo.ui.slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature")
    loaded_topk_ui = mo.ui.slider(0, 500, value=50, step=1, label="top-k (0 = off)")
    loaded_topp_ui = mo.ui.slider(0.0, 1.0, value=0.95, step=0.01, label="top-p")
    loaded_seed_ui = mo.ui.number(value=7, label="Seed", start=0, stop=2**31 - 1)
    loaded_generate_btn = mo.ui.run_button(label="Generate (loaded)")
    mo.vstack(
        [
            loaded_prompt_ui,
            mo.hstack([loaded_max_tokens_ui, loaded_temperature_ui]),
            mo.hstack([loaded_topk_ui, loaded_topp_ui, loaded_seed_ui]),
            loaded_generate_btn,
        ]
    )
    return (
        loaded_generate_btn,
        loaded_max_tokens_ui,
        loaded_prompt_ui,
        loaded_seed_ui,
        loaded_temperature_ui,
        loaded_topk_ui,
        loaded_topp_ui,
    )


@app.cell
def _(
    amp_dtype,
    device,
    loaded_generate_btn,
    loaded_max_tokens_ui,
    loaded_model: Optional[GPTLanguageModelV1],
    loaded_prompt_ui,
    loaded_seed_ui,
    loaded_temperature_ui,
    loaded_topk_ui,
    loaded_topp_ui,
    mo,
    tokenizer,
    tokenizer_vocab_size,
):
    if loaded_model is None:
        _out = mo.md("_Load a checkpoint above to sample from it._")
    elif not loaded_generate_btn.value:
        _out = mo.md("Enter a prompt and click **Generate (loaded)**.")
    else:
        _prompt_ids = tokenizer.encode(str(loaded_prompt_ui.value))
        _out_ids = generate(
            model=loaded_model,
            prompt_ids=_prompt_ids,
            max_new_tokens=int(loaded_max_tokens_ui.value),
            tokenizer_vocab_size=int(tokenizer_vocab_size),
            device=device,
            amp_dtype=amp_dtype,
            temperature=float(loaded_temperature_ui.value),
            top_k=int(loaded_topk_ui.value),
            top_p=float(loaded_topp_ui.value),
            seed=int(loaded_seed_ui.value),
        )
        _generated = tokenizer.decode(_out_ids)
        _new_only = tokenizer.decode(_out_ids[len(_prompt_ids):])
        _out = mo.md(
            f"""
    **Prompt tokens:** {len(_prompt_ids)}, **new tokens:** {int(loaded_max_tokens_ui.value)}

    **Generated (prompt + continuation):**

    ```
    {_generated}
    ```

    **Continuation only:**

    ```
    {_new_only}
    ```
    """
        )
    _out
    return


if __name__ == "__main__":
    app.run()
