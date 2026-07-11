# Text and Tabular Datasets

## Text

### Loading

```python
from datasets import load_dataset

# One example per line, by default
dataset = load_dataset("text", data_files={"train": ["a.txt", "b.txt"], "test": "test.txt"})
dataset = load_dataset("text", data_dir="path/to/text/dataset")

# Sample by paragraph or whole document instead of by line
dataset = load_dataset("text", data_files="file.txt", sample_by="paragraph")
dataset = load_dataset("text", data_files="file.txt", sample_by="document")

# XML — equivalent to "text" with sample_by="document"
dataset = load_dataset("xml", data_files={"train": ["a.xml", "b.xml"]})

# Remote text file
dataset = load_dataset("text", data_files="https://huggingface.co/datasets/.../train.txt")
```

Most real text datasets are JSON Lines rather than plain `.txt` — use
`load_dataset("json", data_files=...)` for those (see
[LOADING-AND-CUSTOM-DATASETS.md](LOADING-AND-CUSTOM-DATASETS.md)); the
`"text"` builder is specifically for unstructured raw text files.

### Feature types for text columns

- `Value("string")` for free text.
- `ClassLabel(names=[...])` for a label column — see
  [FEATURES-REFERENCE.md](FEATURES-REFERENCE.md). Use
  `dataset.class_encode_column("label")` to convert an existing int/string
  label column into a `ClassLabel` automatically.

### Tokenizing with `map()`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
# adds input_ids / token_type_ids / attention_mask columns
```

`batched=True` is essential here — tokenizing one example at a time is much
slower. The tokenizer output gets converted to a PyArrow-compatible format
automatically, but returning tensors as NumPy directly
(`return_tensors="np"`) is faster since NumPy arrays map natively to Arrow:

```python
dataset = dataset.map(lambda examples: tokenizer(examples["text"], return_tensors="np"), batched=True)
```

After tokenizing, set the format for your training framework:
```python
dataset = dataset.select_columns(["input_ids", "token_type_ids", "attention_mask", "labels"])
dataset = dataset.with_format("torch")
```

### Aligning labels to a model's expected mapping

Datasets and models don't always agree on which int maps to which class
name (common for NLI datasets like MNLI, where sources order
entailment/neutral/contradiction differently):

```python
label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}   # the mapping YOUR model expects
mnli_aligned = mnli.align_labels_with_mapping(label2id, "label")
```

## Tabular

A tabular dataset is anything row/column-shaped: CSV, Parquet, HDF5, SQL
tables, or pandas DataFrames. All of these produce a normal `Dataset` with
one `Value(dtype)` feature per column (types auto-inferred by Arrow, or set
explicitly via `features=`).

### CSV

```python
dataset = load_dataset("csv", data_files="my_file.csv")
dataset = load_dataset("csv", data_files=["a.csv", "b.csv", "c.csv"])
dataset = load_dataset("csv", data_files={"train": ["t1.csv", "t2.csv"], "test": "test.csv"})

base_url = "https://huggingface.co/datasets/lhoestq/demo1/resolve/main/data/"
dataset = load_dataset("csv", data_files={"train": base_url + "train.csv", "test": base_url + "test.csv"})

# zipped CSVs are transparently extracted
dataset = load_dataset("csv", data_files={"train": "https://domain.org/train_data.zip"})
```

### Pandas DataFrames

```python
from datasets import Dataset
import pandas as pd

df = pd.read_csv("https://huggingface.co/datasets/imodels/credit-card/raw/main/train.csv")
dataset = Dataset.from_pandas(df)
train_ds = Dataset.from_pandas(train_df, split="train")
```

If the resulting features look wrong, specify `features=` explicitly — a
`pandas.Series` doesn't always carry enough type info for Arrow to infer
correctly (e.g. an empty or all-`None`/`NaN` column gets typed as `null`).

### HDF5

```python
dataset = load_dataset("hdf5", data_files="data.h5")
```
Assumes a "tabular" HDF5 layout: every dataset in the file has the same
number of rows along its first dimension.

### SQL databases

```python
from datasets import Dataset

# Whole table
ds = Dataset.from_sql("states", con="sqlite:///us_covid_data.db")

# Arbitrary query — supports joins across multiple tables
ds = Dataset.from_sql('SELECT * FROM states WHERE state="California";', con="sqlite:///us_covid_data.db")

ds.filter(lambda x: x["cases"] > 10000)   # normal Dataset ops apply after loading
```

The connection string (`con=`) is a standard SQLAlchemy-style URI — differs
per database dialect (SQLite: `sqlite:///path.db`; PostgreSQL, MySQL, etc.
have their own URI forms). Any database driver SQLAlchemy supports works
here, not just SQLite.

### Exporting back to tabular formats

```python
dataset.to_csv("out.csv")
dataset.to_parquet("out.parquet")
dataset.to_pandas()
dataset.to_sql("table_name", con="sqlite:///out.db")
```
