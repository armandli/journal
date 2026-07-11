# Loading Datasets — Hub, Local, Remote, Custom

## `load_dataset()` parameters

```python
load_dataset(
    path,             # Hub repo id, local directory, or builder name ("csv", "json", ...)
    name=None,        # dataset "config"/subset name, e.g. "mrpc" for glue
    data_dir=None,    # restrict to a subfolder (Hub repo or local/builder path)
    data_files=None,  # str, list, or {"split": path_or_glob_or_url} mapping
    split=None,       # e.g. "train", "train[:10%]", "train+test" — None returns a DatasetDict of all splits
    cache_dir=None,   # defaults to ~/.cache/huggingface/datasets
    features=None,    # explicit Features to override type inference
    revision=None,    # branch/tag/commit for Hub repos
    token=None,       # auth token for private/gated repos
    streaming=False,  # True -> IterableDataset/IterableDatasetDict, no download
    num_proc=None,    # parallel workers for downloading/preparing multi-shard datasets
)
```

Returns a `Dataset` if `split` is given, otherwise a `DatasetDict` with one
entry per split.

## From the Hugging Face Hub

```python
ds = load_dataset("lhoestq/demo1")                              # infers format from repo contents
ds = load_dataset("nyu-mll/glue", "sst2", split="train")         # "sst2" = dataset config
ds = load_dataset("lhoestq/custom_squad", revision="main")       # tag/branch/commit

# Map specific Hub files to specific splits
ds = load_dataset("namespace/dataset", data_files={"train": "train.csv", "test": "test.csv"})

# Restrict to a subset of files (glob pattern) or a directory within the repo
ds = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
ds = load_dataset("allenai/c4", data_dir="en")

# A directory inside a Hugging Face Storage Bucket
ds = load_dataset("buckets/username/bucket_name/my_dataset", split="train")
```

**Warning:** if you don't specify `data_files`/`data_dir` on a Hub repo with
many files, `load_dataset()` returns *all* of them — for huge repos (e.g. a
13TB corpus) this can take a very long time. Narrow it down explicitly.

## Local and remote files by format

All of these accept `data_files` as a single path, a list, a glob pattern,
or a `{"split": path}` dict — and the exact same call works with an
`https://` URL (including a raw GitHub URL) instead of a local path.

```python
load_dataset("csv", data_files="my_file.csv")
load_dataset("csv", data_files=["a.csv", "b.csv", "c.csv"])
load_dataset("json", data_files="my_file.json")                 # JSON Lines preferred (one object per line)
load_dataset("json", data_files="my_file.json", field="data")   # nested: {"data": [...]}
load_dataset("parquet", data_files={"train": "train.parquet", "test": "test.parquet"})
load_dataset("arrow", data_files={"train": "train.arrow"})
load_dataset("hdf5", data_files="data.h5")                        # tabular-structured HDF5 only
load_dataset("text", data_files="my_file.txt")                    # one example per line, by default
```

### Loading from GitHub (or any other remote host)

There's no dedicated "GitHub loader" — a GitHub-hosted file is just a remote
file. Point `data_files=` at the raw URL:

```python
base = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/"
ds = load_dataset("csv", data_files=base + "us-states.csv")

# Multiple splits from different URLs
ds = load_dataset("json", data_files={
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}, field="data")
```

Also valid: `hf://datasets/org/name/file.parquet` (Hub-hosted files/buckets)
and plain `https://` URLs pointing at zipped/gzipped archives (transparently
decompressed; formats beyond zip/gzip like rar/xz are not supported for
streaming). For a full repo rather than one file, `git clone` it and load
from the local checkout with the patterns above, or push the data to a Hub
dataset repo and load it by repo id instead.

### SQL databases

```python
from datasets import Dataset

ds = Dataset.from_sql("table_name", con="sqlite:///my.db")                       # whole table
ds = Dataset.from_sql("SELECT * FROM table WHERE length(text) > 100", con="sqlite:///my.db")  # query
```

Works with any SQLAlchemy-style connection URI (SQLite, PostgreSQL, etc.).

### Other formats
- **WebDataset** (TAR-archive shards, good for big image/audio datasets): `load_dataset("webdataset", data_files={"train": "path/*.tar"}, streaming=True)`.
- **Lance** (multimodal lakehouse table format): `load_dataset("lance_repo_or_path", streaming=True)`.
- **Folder builders** (`imagefolder`, `audiofolder`, `videofolder`) — see [AUDIO-AND-VISION.md](AUDIO-AND-VISION.md).

## In-memory data (no file at all)

```python
from datasets import Dataset

Dataset.from_dict({"a": [1, 2, 3]})
Dataset.from_list([{"a": 1}, {"a": 2}, {"a": 3}])
Dataset.from_pandas(df)

def gen():
    for i in range(1, 4):
        yield {"a": i}
Dataset.from_generator(gen)                      # memory-efficient, works for data > RAM

# Sharded generator (each worker gets a subset — good with DataLoader num_workers)
from datasets import IterableDataset
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"line": line}
ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": [f"data{i}.txt" for i in range(32)]})
```

## Streaming (`streaming=True`)

Returns an `IterableDataset`/`IterableDatasetDict` instead of a `Dataset` —
nothing is downloaded upfront; data streams as you iterate. Use this for
datasets too large to fit on disk, or to explore a few samples quickly:

```python
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
next(iter(ds))

for row in ds.take(3):
    print(row)
```

Parquet sources support column pruning and predicate pushdown even while
streaming (much faster than downloading everything first):

```python
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True, columns=["url", "date"])
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True, filters=[("language_score", ">=", 0.99)])
```

`IterableDataset` trades random access for fast, low-memory iteration — you
cannot index into it (`ds[5]`), only iterate. See
[about_mapstyle_vs_iterable](https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable)
for the full tradeoff if unsure which to use. `DATASET-OPERATIONS-AND-FORMATS.md`
covers `IterableDataset`-specific methods (`.shuffle(buffer_size=...)`, `.skip`, `.take`).

## Split slicing

```python
load_dataset("dataset_name", split="train+test")           # concatenate two splits
load_dataset("dataset_name", split="train[10:20]")          # rows 10-19
load_dataset("dataset_name", split="train[:10%]")            # first 10%
load_dataset("dataset_name", split="train[:10%]+train[-80%:]")   # combine slices

# k-fold cross-validation splits
val_ds = load_dataset("dataset_name", split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
train_ds = load_dataset("dataset_name", split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])
```

Percent boundaries round to the nearest integer by default, which can make
some slices contain more rows than others; pass
`rounding="pct1_dropremainder"` (via `datasets.ReadInstruction`, or the
`"train[50%:52%](pct1_dropremainder)"` string suffix) for exactly equal-sized
percentage slices, at the cost of possibly dropping trailing examples.

## Offline mode

Once a Hub dataset has been downloaded once, it's cached — set
`HF_HUB_OFFLINE=1` to skip network calls entirely and load straight from
cache (avoids waiting for a download attempt to time out with no
connection).

## Overriding inferred features

Local files get their column types auto-inferred by Arrow, which doesn't
always match what you want (e.g. a label column you want as `ClassLabel`,
not a bare int/string):

```python
from datasets import Features, Value, ClassLabel

class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
features = Features({"text": Value("string"), "label": ClassLabel(names=class_names)})

dataset = load_dataset("csv", data_files=file_dict, delimiter=";", column_names=["text", "label"], features=features)
```

## Loading a previously saved dataset

```python
from datasets import load_from_disk
ds = load_from_disk("path/to/dataset/directory")   # written by Dataset.save_to_disk(); also supports s3:// etc via fsspec
```

Distinct from `Dataset.from_file("data.arrow")`, which memory-maps a raw
Arrow file directly without going through the cache-preparation step
(saves disk space, but only works for the raw Arrow streaming format, not
Arrow IPC/Feather V2).
