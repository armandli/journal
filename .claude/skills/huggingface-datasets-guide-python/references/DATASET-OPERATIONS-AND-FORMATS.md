# Dataset/DatasetDict Operations, Formats, and Export

## `map()` — the main transform primitive

```python
ds = ds.map(lambda example: {"text": "Review: " + example["text"]})
ds = ds.map(lambda batch: tokenizer(batch["text"]), batched=True)          # batched, much faster
ds = ds.map(add_prefix, num_proc=4)                                          # parallel workers
```

- `batched=False` (default): function takes/returns one example dict at a
  time.
- `batched=True`: function takes/returns a batch dict of lists
  (`{"text": [...]}`), `batch_size` controls how many rows per call
  (default 1000). Always prefer this for vectorizable work (tokenization,
  resizing, resampling) — the difference in throughput is large.
- `with_indices=True` / `with_rank=True`: adds `idx`/`rank` args to your
  function signature.
- `remove_columns=[...]`: drop original columns after mapping (common when
  a tokenizer replaces `text` with `input_ids`/`attention_mask`).
- Results are **cached to disk** keyed by a fingerprint of the transform —
  rerunning identical code reuses the cache instead of recomputing. Async
  functions are supported and run with internal concurrency.
- Returning `None` from the map function leaves the dataset unchanged for
  that call; returning nothing at all is the identity transform if no
  function is given.

## `filter()`

```python
ds = ds.filter(lambda x: x["label"] == 1)
ds = ds.filter(lambda batch: [l == 1 for l in batch["label"]], batched=True)
```

Same `batched`/`with_indices`/`input_columns` shape as `map()`, but the
function returns a bool (or list of bools if batched) instead of a
replacement row.

## Selection, sorting, shuffling

```python
ds = ds.select(range(100))              # contiguous ranges are cheap (slice); scattered indices are slower
ds = ds.sort("label")                    # ascending; reverse=True or a list per-column for multi-column sort
ds = ds.shuffle(seed=42)                  # reproducible with a seed
ds = ds.select_columns(["text", "label"])
ds = ds.rename_column("label", "labels")
ds = ds.remove_columns(["unused_col"])
ds = ds.class_encode_column("label")      # turn a string/int column into ClassLabel automatically
```

**Shuffle performance note:** `shuffle()` builds an indices mapping, which
after the first shuffle makes row access noticeably slower (non-contiguous
reads) — up to 10x. Call `ds.flatten_indices()` afterward to rewrite the
dataset contiguously on disk if you'll access it repeatedly, or switch to
`IterableDataset` and use its shuffle-buffer-based `.shuffle(seed=..., buffer_size=...)`
which stays fast (shuffles shard order + a bounded in-memory buffer, no full
materialization).

## Splitting and combining

```python
split_ds = ds.train_test_split(test_size=0.1)         # -> DatasetDict{"train": ..., "test": ...}
shard = ds.shard(num_shards=4, index=0)                 # 1/4 of the dataset, for distributed workers

from datasets import concatenate_datasets, interleave_datasets
combined = concatenate_datasets([ds1, ds2])              # stack rows (same features required)
mixed = interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3], seed=42)   # sample-mix multiple datasets
```

## Format conversion — `set_format` / `with_format`

```python
ds.set_format(type="torch", columns=["input_ids", "labels"])   # in-place; formats on-the-fly at __getitem__
ds2 = ds.with_format("torch")                                    # returns a NEW Dataset instead of mutating
ds.reset_format()                                                  # back to plain python objects, all columns
```

`type` is one of `None` (python objects, default), `"numpy"`, `"torch"`,
`"tensorflow"`, `"jax"`, `"arrow"`, `"pandas"`, `"polars"`. Formatting is
applied lazily on `__getitem__`, not eagerly — cheap to change repeatedly.
`with_format(..., device=...)` places PyTorch tensors directly on a GPU:

```python
ds = ds.with_format("torch", device="cuda")
```

For arbitrary on-the-fly transforms (e.g. image augmentation) instead of a
fixed dtype conversion, use `set_transform()`/`with_transform()` — same
in-place-vs-new-object distinction as `set_format`/`with_format`, but the
argument is a callable operating on a batch dict rather than a format name.
Prefer these over `map()` for things you want re-applied every epoch (like
random augmentation), since `map()` caches its output once.

## N-dimensional / fixed-shape arrays with PyTorch format

Variable-shape nested lists become lists of tensors (not stacked) under
`with_format("torch")`. To get a single stacked tensor per batch, declare the
column as an `Array2D`/`Array3D`/etc. feature with an explicit shape:

```python
from datasets import Dataset, Features, Array2D

features = Features({"data": Array2D(shape=(2, 2), dtype="int32")})
ds = Dataset.from_dict({"data": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}, features=features)
ds = ds.with_format("torch")
ds[:2]["data"].shape   # (2, 2, 2) — properly stacked
```

## Streaming (`IterableDataset`) operations

```python
ds = load_dataset("dataset_name", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)   # shard-order shuffle + fixed-size buffer, not a full shuffle
ds = ds.skip(1000)
small = ds.take(10)

from torch.utils.data import DataLoader
loader = DataLoader(ds.with_format("torch"), num_workers=4)   # each worker streams a disjoint subset of shards
```

Convert a regular `Dataset` to a shardable `IterableDataset` for this pattern:
`ds.to_iterable_dataset(num_shards=128)`.

## Saving and exporting

```python
ds.save_to_disk("path/to/dir")                 # Arrow format, reloadable with load_from_disk
ds.push_to_hub("username/my-dataset")           # upload to the Hugging Face Hub

ds.to_csv("out.csv")
ds.to_json("out.jsonl")
ds.to_parquet("out.parquet")
ds.to_pandas()
ds.to_dict()
ds.to_sql("table_name", con="sqlite:///out.db")
```

## PyTorch `DataLoader` integration

```python
from torch.utils.data import DataLoader

ds = ds.with_format("torch")
loader = DataLoader(ds, batch_size=8, shuffle=True)   # map-style Dataset works directly with DataLoader
```

`Dataset` is a wrapper around an Arrow table, so `with_format("torch")` gives
zero-copy reads into `torch.Tensor` wherever possible — prefer this over
manually converting rows to tensors in a custom `collate_fn` when the data
is already tensor-shaped.
