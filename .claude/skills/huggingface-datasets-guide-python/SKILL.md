---
name: huggingface-datasets-guide-python
description: Write and debug Python code using the Hugging Face `datasets` library — loading standard datasets from the Hub, loading custom datasets from local files/folders/GitHub/SQL, and processing audio, vision, text, and tabular data with the Features system. Use when the user asks to "load a dataset with huggingface datasets", "use datasets.load_dataset", "load a custom dataset", "load an audio/image/text dataset", or writes code importing `datasets`. Do NOT use for the `transformers` model/tokenizer library itself (use huggingface-transformers-guide-python) or for pandas/polars-only data work with no `datasets` involvement.
argument-hint: "[task or description of what to implement]"
---

# Hugging Face `datasets` Python Guide

`datasets` loads and processes data for audio/vision/text/tabular ML tasks,
backed by Apache Arrow for fast zero-copy access and streaming. Everything
goes through one entry point: `load_dataset()`.

**No more loading scripts / `trust_remote_code`.** Older tutorials describe
Python "loading scripts" that shipped arbitrary code with a dataset repo —
current `datasets` no longer uses that mechanism for the built-in path;
loading is driven by file format + repository layout (CSV/JSON/Parquet/
folder structure), not executable scripts. Don't reach for
`trust_remote_code` here — that's a `transformers` concept, not a
`datasets` one.

## The one function: `load_dataset()`

```python
from datasets import load_dataset

# From the Hub
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
ds = load_dataset("nyu-mll/glue", "mrpc", split="train")   # "mrpc" = config/subset name

# From local/remote files — first arg names the FORMAT, not a Hub repo
ds = load_dataset("csv", data_files="my_file.csv")
ds = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})
ds = load_dataset("parquet", data_files="hf://datasets/org/name/data.parquet")

# Streaming (no download — iterate lazily)
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
```

`path` is either a Hub repo id, a local directory, or one of the builder
names: `"csv"`, `"json"`, `"parquet"`, `"arrow"`, `"text"`, `"xml"`,
`"hdf5"`, `"webdataset"`, `"imagefolder"`, `"audiofolder"`,
`"videofolder"` — combined with `data_files=`/`data_dir=` pointing at your
actual data. Full parameter reference (`data_files`, `data_dir`, `split`,
`streaming`, `revision`, `num_proc`, split slicing syntax like
`"train[:10%]"`, offline mode) and every local/remote/custom-loading path
(including loading straight from a GitHub raw URL or a SQL database):
[references/LOADING-AND-CUSTOM-DATASETS.md](references/LOADING-AND-CUSTOM-DATASETS.md).

## Loading YOUR data (not a Hub dataset)

```python
from datasets import Dataset

Dataset.from_dict({"text": [...], "label": [...]})       # in-memory dict
Dataset.from_list([{"a": 1}, {"a": 2}])                    # list of records
Dataset.from_pandas(df)                                    # pandas DataFrame
Dataset.from_generator(my_gen)                              # lazy, for big/streamed data
Dataset.from_sql("SELECT * FROM table", con="sqlite:///db.sqlite")

# A folder of raw files (GitHub-cloned repo, downloaded archive, etc.)
load_dataset("imagefolder", data_dir="/path/to/images")     # class-per-subfolder + optional metadata.csv
load_dataset("audiofolder", data_dir="/path/to/audio")

# A file hosted on GitHub (or anywhere else on the web) — just pass the raw URL
load_dataset("csv", data_files="https://raw.githubusercontent.com/org/repo/main/data.csv")
```

"Loading from GitHub" isn't a special API — it's the general remote-file
path: point `data_files=` at a `raw.githubusercontent.com` URL (or clone the
repo locally and load the files from disk). See
[references/LOADING-AND-CUSTOM-DATASETS.md](references/LOADING-AND-CUSTOM-DATASETS.md)
for the full custom-dataset playbook.

## By domain

| Domain | Feature type | Folder builder | Details |
|---|---|---|---|
| Audio | `Audio` | `audiofolder` | [references/AUDIO-AND-VISION.md](references/AUDIO-AND-VISION.md) |
| Vision | `Image` | `imagefolder` | [references/AUDIO-AND-VISION.md](references/AUDIO-AND-VISION.md) |
| Text | `Value("string")` | `text`/`xml`/`json` | [references/TEXT-AND-TABULAR.md](references/TEXT-AND-TABULAR.md) |
| Tabular | `Value(dtype)` per column | `csv`/`parquet`/`hdf5`/SQL | [references/TEXT-AND-TABULAR.md](references/TEXT-AND-TABULAR.md) |

Every domain shares the same `Dataset`/`DatasetDict` object and the same
`Features` type system — see
[references/FEATURES-REFERENCE.md](references/FEATURES-REFERENCE.md) for
`Value`, `ClassLabel`, `List`/`LargeList` (prefer these over the legacy
`Sequence`), `Array2D`-`Array5D`, `Translation`, and `Json` (for genuinely
mixed-type fields).

## Core Dataset/DatasetDict operations

```python
ds = ds.map(tokenize_fn, batched=True)             # transform, cached to disk
ds = ds.filter(lambda x: x["label"] == 1)
ds = ds.select(range(100))                          # index subset
ds = ds.sort("label")
ds = ds.shuffle(seed=42)
ds = ds.train_test_split(test_size=0.1)              # -> DatasetDict{train, test}

ds = ds.with_format("torch")                          # or "numpy"/"pandas"/"tensorflow"/"jax"/"polars"
ds.set_format(type="torch", columns=["input_ids", "labels"])   # in-place, on-the-fly
```

`map`/`filter` are cached — rerunning the same transform reuses the cached
result instead of recomputing. Use `batched=True` (default `batch_size=1000`)
for anything vectorizable (tokenization, resizing) — it's substantially
faster than row-at-a-time. Full operation catalogue (`shard`, `concatenate_datasets`,
`interleave_datasets`, streaming-specific `IterableDataset` methods,
`save_to_disk`/`push_to_hub`/`to_csv`/`to_parquet`):
[references/DATASET-OPERATIONS-AND-FORMATS.md](references/DATASET-OPERATIONS-AND-FORMATS.md).

## Testing

After writing a loading/processing pipeline, actually call `load_dataset(...)`
(or build the `Dataset` from local files) and inspect `ds.features`, `ds[0]`,
and `len(ds)` — feature types are inferred from the data and silently
diverge from what you expect (e.g. a numeric column read as `string`, or a
`ClassLabel` not applied), and this only surfaces by actually loading, not
by reading the code.

## References

- [references/LOADING-AND-CUSTOM-DATASETS.md](references/LOADING-AND-CUSTOM-DATASETS.md) — `load_dataset()` full reference, Hub/local/remote/GitHub/SQL loading, streaming, split slicing, offline mode
- [references/DATASET-OPERATIONS-AND-FORMATS.md](references/DATASET-OPERATIONS-AND-FORMATS.md) — map/filter/select/sort/shuffle/train_test_split/shard/concatenate, format conversion, saving/exporting
- [references/FEATURES-REFERENCE.md](references/FEATURES-REFERENCE.md) — the `Features` type system: `Value`, `ClassLabel`, `List`/`LargeList`/`Sequence`, `Array2D`-`5D`, `Json`
- [references/AUDIO-AND-VISION.md](references/AUDIO-AND-VISION.md) — `Audio`/`Image` features, `AudioFolder`/`ImageFolder`, resampling, transforms, decoding performance
- [references/TEXT-AND-TABULAR.md](references/TEXT-AND-TABULAR.md) — text loading + tokenizer integration + label alignment; CSV/Parquet/pandas/SQL/HDF5 tabular loading

## External Docs

- Full docs: https://huggingface.co/docs/datasets/index

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "huggingface-datasets-guide-python"
```
