---
name: mlx-data-guide-python
description: Write and debug Python data-loading pipelines using Apple's mlx-data module (mlx.data), the companion data library to MLX. Covers Buffer/Stream construction, common built-in datasets, HuggingFace dataset integration, tokenizing with CharTrie/Tokenizer, and sample transforms (batch, shuffle, prefetch, pad, image/audio loading). Use when the user asks to "use mlx-data", "load a dataset with mlx.data", "build an mlx-data pipeline", "wrap a HuggingFace dataset for MLX", or "tokenize with mlx-data". Do NOT use for MLX core arrays/nn/training (use mlx-guide-python), or for PyTorch DataLoader / tf.data pipelines.
argument-hint: "[task or description of the data pipeline to build]"
---

# mlx-data (mlx.data) Python Guide

`mlx.data` is Apple's data-loading library that pairs with MLX. It builds lazy,
composable pipelines out of three concepts:

- **Sample** — a `dict[str, array-like]`. Scalars, numpy arrays, and (byte)strings
  are all valid values; scalars are auto-cast to scalar arrays.
- **Buffer** — an *indexable*, known-length container of samples (like a list).
  Supports `shuffle()`, `perm()`, random access, and `Buffer.ordered_prefetch()`.
- **Stream** — a *potentially infinite iterator* of samples (like a generator).
  No random access, but supports nesting (e.g. read lines out of files named
  inside samples) and non-deterministic `Stream.prefetch()`.

Buffers and Streams share the same chainable transform API (`batch`,
`key_transform`, `sample_transform`, `shuffle`, image/audio/pad ops, etc.) —
see [references/BUFFERS-AND-STREAMS.md](references/BUFFERS-AND-STREAMS.md).
Go `Buffer -> .to_stream()` at the point you want prefetching/batching for
training; keep things a `Buffer` as long as you need shuffling or random access.

## Install

```bash
pip install mlx-data
```

## Canonical pipeline (MNIST)

```python
import mlx.data as dx
from mlx.data.datasets import load_mnist

mnist_train = load_mnist(train=True)   # -> Buffer(size=60000, keys={'image', 'label'})

mnist_mlp = (
    mnist_train
    .shuffle()                                                    # Buffer op: random permutation
    .to_stream()                                                   # Buffer -> Stream
    .key_transform("image", lambda x: x.astype("float32").reshape(-1) / 255)
    .batch(32)
    .prefetch(4, 2)                                                # 4 batches ahead, 2 threads
)

for batch in mnist_mlp:
    x, y = batch["image"], batch["label"]   # numpy arrays; wrap with mx.array(...) for MLX
```

Streams are stateful iterators: call `stream.reset()` to restart one (e.g. at
the start of each training epoch).

## GIL note

Python callables passed to `key_transform`/`sample_transform` still run under
the GIL — one sample tokenizes/transforms at a time. Prefer numpy-vectorized
work in those callables, or push tokenization into `mlx.data.core.CharTrie`
(true multicore, no GIL) — see
[references/TOKENIZING.md](references/TOKENIZING.md). Don't optimize this
prematurely; only chase GIL overhead once it's confirmed to be the bottleneck.

## Loading a common dataset

```python
from mlx.data.datasets import load_mnist, load_cifar10, load_wikitext_lines

mnist = load_mnist(train=True)                     # Buffer
cifar = load_cifar10(train=True)                   # Buffer
wiki = load_wikitext_lines(split="train")          # Stream of text lines
```

Full list of loaders (MNIST, Fashion-MNIST, CIFAR-10/100, ImageNet,
LibriSpeech, LibriTTS-R, WikiText, SpeechCommands, image folders) and their
signatures: [references/DATASETS.md](references/DATASETS.md).

## Wrapping a HuggingFace dataset

`mlx.data` has no direct HF loader — convert the HF dataset to a list of
numpy-friendly dicts, then hand it to `buffer_from_vector`:

```python
import numpy as np
import mlx.data as dx
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")

def to_dicts(split):
    return [{"image": np.array(img).copy(), "label": label}
            for label, img in zip(split["label"], split["image"])]

def to_stream(split, shuffle=False):
    buf = dx.buffer_from_vector(to_dicts(split))
    if shuffle:
        buf = buf.shuffle()
    return (buf.to_stream()
               .key_transform("image", lambda x: x.astype("float32") / 255)
               .batch(32)
               .prefetch(prefetch_size=8, num_threads=4))

train_stream = to_stream(ds["train"], shuffle=True)
train_stream.reset()
for batch in train_stream:
    ...
```

Details and caveats (PIL image conversion, `reset()` requirement):
[references/DATASETS.md](references/DATASETS.md).

## Tokenizing

Avoid pure-Python tokenizers in `key_transform` (GIL-bound, one sample at a
time). Use `mlx.data.core.CharTrie` + `Tokenizer` instead:

```python
from mlx.data.core import CharTrie, Tokenizer
from mlx.data.tokenizer_helpers import read_trie_from_spm

trie, weights = read_trie_from_spm("path/to/spm/model")   # or read_trie_from_vocab(vocab_txt)
tokenizer = Tokenizer(trie, trie_key_scores=weights)

dset = (
    wiki_stream
    .tokenize("line", trie, output_key="tokens")
    .filter_key("tokens")
    .prefetch(512, 8)
    .batch(128, dim=dict(tokens=0))
    .sliding_window("tokens", 1025, 1025)
    .batch(32)
)
```

Full API (`CharTrie`, `Tokenizer.tokenize_shortest/_rand`, vocab/SPM helpers):
[references/TOKENIZING.md](references/TOKENIZING.md).

## References

- [references/DATASETS.md](references/DATASETS.md) — every `mlx.data.datasets.load_*` loader, plus the HuggingFace conversion recipe in full
- [references/BUFFERS-AND-STREAMS.md](references/BUFFERS-AND-STREAMS.md) — Buffer/Stream factory methods, buffer-only and stream-only ops, and the full shared transform API (batch, filter, image, I/O, pad, shape, tokenize, `*_if` conditionals)
- [references/TOKENIZING.md](references/TOKENIZING.md) — CharTrie, Tokenizer, SentencePiece/vocab-file helpers
- [references/FEATURES-AND-MISC.md](references/FEATURES-AND-MISC.md) — `mfsc` audio feature extraction and `AWSFileFetcher` for remote (S3) file loading

## Testing pipelines

After writing a pipeline, actually iterate it (`next(stream)` or a short `for`
loop over a couple of batches) and print shapes/dtypes before wiring it into a
training loop — mlx-data ops are lazy, so shape/type errors only surface on
first access, not at pipeline-construction time.

## External Docs

- Full docs: https://ml-explore.github.io/mlx-data/build/html/index.html

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "mlx-data-guide-python"
```
