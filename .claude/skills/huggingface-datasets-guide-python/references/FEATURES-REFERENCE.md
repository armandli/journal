# The `Features` System

`Features` is the typed schema of a dataset: `dict[column_name, column_type]`.
It's the backbone that determines serialization, casting behavior, and what
you get back when you index a row.

```python
from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")
dataset.features
# {'idx': Value('int32'),
#  'label': ClassLabel(names=['not_equivalent', 'equivalent']),
#  'sentence1': Value('string'),
#  'sentence2': Value('string')}
```

## `Value` — scalars

Wraps standard Arrow scalar types: `"bool"`, `"int8"`/`"int16"`/`"int32"`/
`"int64"` (and unsigned variants), `"float16"`/`"float32"`/`"float64"`,
`"string"`, `"binary"`, and date/time/timestamp variants. This is the
feature type for any plain column (numbers, strings, booleans).

```python
Features({"text": Value("string"), "score": Value("float32")})
```

## `ClassLabel` — categorical labels stored as ints

```python
from datasets import ClassLabel

class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
ClassLabel(names=class_names)
```

Labels are stored as integers internally; `ClassLabel.int2str()` and
`ClassLabel.str2int()` convert between the integer and the name. Use
`dataset.class_encode_column("label")` to convert an existing string/int
column into a proper `ClassLabel` automatically (infers the class list from
observed values). Use `align_labels_with_mapping()` when a dataset's label
ids don't match the mapping your model/checkpoint expects — common for NLI
datasets where different sources order `entailment`/`neutral`/`contradiction`
differently:

```python
label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
mnli_aligned = mnli.align_labels_with_mapping(label2id, "label")
```

## `List` / `LargeList` — lists of a feature (prefer over `Sequence`)

```python
from datasets import List

Features({"answers": {"text": List(Value("string")), "answer_start": List(Value("int32"))}})
```

- **`List(feature, length=-1)`** — backed by `pyarrow.ListType` (32-bit
  offsets), optionally fixed-length.
- **`LargeList(feature)`** — same idea with 64-bit offsets, for very large
  lists.
- **`Sequence(feature, length=-1)`** — the older/legacy name. Its one
  special behavior: a `Sequence` of a `dict` feature auto-converts to a
  `dict` of lists (kept for TensorFlow-Datasets compatibility). If you don't
  specifically need that dict-of-lists behavior, prefer `List`/`LargeList`.

Nested nested fields (dict-valued columns) can be flattened into independent
top-level columns with `dataset.flatten()`.

## `Array2D` … `Array5D` — fixed or partially-dynamic shape tensors

```python
from datasets import Array2D, Array3D

Features({"a": Array2D(shape=(1, 3), dtype="int32")})
Features({"a": Array3D(shape=(None, 5, 2), dtype="int32")})   # first dim dynamic (e.g. variable sequence length)
```

Use these instead of nested `List`s when you want columns to convert to a
single stacked tensor (rather than a list of ragged tensors) under
`with_format("torch"/"numpy")` — see
[DATASET-OPERATIONS-AND-FORMATS.md](DATASET-OPERATIONS-AND-FORMATS.md) for
the concrete stacking example.

## `Audio` and `Image` — media features

Covered in depth in [AUDIO-AND-VISION.md](AUDIO-AND-VISION.md). Quick shape:
- `Audio(sampling_rate=...)` — decodes to a torchcodec `AudioDecoder` on
  access; `decode=False` gives raw path/bytes instead.
- `Image(mode=...)` — decodes to `PIL.Image`; `decode=False` gives raw
  path/bytes instead.

## `Translation` — parallel-text convenience type

For datasets with the same text in multiple languages (e.g. machine
translation corpora) — a dict-like feature keyed by language code.

## `Json` — escape hatch for genuinely mixed/unstructured fields

Arrow (and therefore `datasets`) expects every value in a column to share a
type, and every dict in a column to share the same keys/value-types.
Non-conforming data either errors or gets its missing fields silently filled
with `None`. Use `Json()` to store a field as opaque, mixed-type JSON instead:

```python
from datasets import Features, Json

# This raises pyarrow.lib.ArrowInvalid — int/str/dict mixed in one column:
Dataset.from_dict({"a": [0, "foo", {"subfield": "bar"}]})

# This works and preserves the original mixed values:
features = Features({"a": Json()})
ds = Dataset.from_dict({"a": [0, "foo", {"subfield": "bar"}]}, features=features)
list(ds["a"])   # [0, "foo", {"subfield": "bar"}]
```

Also useful for lists of dicts with heterogeneous keys (e.g. tool-calling
message logs) to avoid `None`-filling mismatched keys:

```python
Features({"a": List(Json())})
```

Or skip specifying `features=` manually and pass `on_mixed_types="use_json"`
directly to `Dataset.from_dict(...)`/`from_list(...)` to auto-apply `Json()`
wherever a mixed-type field is detected.

## Casting between feature types

```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))   # change one column's feature/config
dataset = dataset.cast(new_features)                                   # change the whole schema at once
```

`cast_column` is the standard way to trigger resampling (`Audio`) or mode
conversion (`Image(mode="RGB")`) — the underlying file isn't touched; the
new sampling rate/mode is applied lazily whenever you access the row.
