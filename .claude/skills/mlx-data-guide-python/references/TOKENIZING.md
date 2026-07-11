# Tokenizing with mlx-data

Plain Python tokenizers work fine inside `Buffer.key_transform()`, but they run
under the GIL — only one sample tokenizes at a time. `mlx.data.core.CharTrie`
+ `Tokenizer` implement tokenization in C++ and take full advantage of a
multicore system, so prefer them for anything throughput-sensitive.

## Building a trie manually

```python
from mlx.data.core import CharTrie, Tokenizer

trie = CharTrie()
for t in b"a quick brown fox jumped over the lazy dog".split():
    trie.insert(t)
trie.insert(b" ")

tokenizer = Tokenizer(trie)
print(tokenizer.tokenize_shortest(b"a quick brown fox jumped over the lazy dog"))
# [0, 9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8]

# Add individual characters too, so anything can be tokenized (falls back to chars)
import string
for l in string.ascii_letters:
    trie.insert(bytes(l, "utf-8"))

print(tokenizer.tokenize_shortest(b"This is a quick example"))
```

## Building a trie from a vocab file or SentencePiece model

```python
from mlx.data.tokenizer_helpers import read_trie_from_vocab, read_trie_from_spm

# One token per line, plain text vocab file
trie = read_trie_from_vocab("/path/to/vocab.txt")   # -> CharTrie

# From a SentencePiece model (or a vocab+scores file extracted from one)
trie, weights = read_trie_from_spm("path/to/spm/model")   # -> (CharTrie, list[float])
tokenizer = Tokenizer(trie, trie_key_scores=weights)
tokenizer.tokenize_shortest(b"This is some more text to tokenize")
```

`read_trie_from_spm` needs the `sentencepiece` package installed to read a
raw `.model` file directly; if you instead export just the vocabulary and
scores, it can be read without that dependency. SentencePiece models are
almost always BPE, and while a `CharTrie` + shortest-path tokenization gives
the smallest token count / highest likelihood, it can diverge slightly from
true BPE merges — if you need bit-exact BPE, use the corresponding BPE loader
(`read_bpe_from_spm` / `mlx.data.core.BPETokenizer`) rather than `CharTrie`.

## `CharTrie` API

| Method | Signature | Behavior |
|---|---|---|
| `__init__` | `(self)` | Empty trie. |
| `insert` | `(self, token: bytes[, id])` | Insert a token (creating it if new). |
| `search` | `(self, token: bytes)` | Look up a token; returns the node or `None`. |
| `key` | `(self, id)` | The `id`-th token as a list of characters. |
| `key_bytes` | `(self, id)` | The `id`-th token as `bytes`. |
| `key_string` | `(self, id)` | The `id`-th token as a `str`. |
| `num_keys` | `(self)` | How many tokens/nodes are in the trie. |
| `root` | `(self)` | The trie's root node. |

## `Tokenizer` API

```python
Tokenizer(trie: CharTrie, ignore_unk: bool = False, trie_key_scores: list[float] = [])
```
- `ignore_unk` — if `False` (default), content that can't be tokenized raises;
  if `True`, it's silently skipped.
- `trie_key_scores` — one weight per trie node; defaults to uniform weight 1
  per node, so `tokenize_shortest` minimizes raw token count. Supply
  SentencePiece log-likelihood weights (from `read_trie_from_spm`) to
  approximate real BPE behavior.

| Method | Behavior |
|---|---|
| `tokenize(self, input)` | Returns the full graph of valid tokenizations (not just one path). |
| `tokenize_shortest(self, input)` | The single tokenization minimizing total `trie_key_scores` — the "best"/canonical tokenization. |
| `tokenize_rand(self, input)` | A tokenization chosen randomly among all valid ones — useful as tokenization-level data augmentation (subword regularization). |

## Using `tokenize` inside a pipeline

`Buffer.tokenize` / `Stream.tokenize` (the shared op, see
[BUFFERS-AND-STREAMS.md](BUFFERS-AND-STREAMS.md#tokenization-op)) applies a
trie to an array field directly in the pipeline, no per-sample Python call
needed:

```python
from mlx.data.datasets import load_wikitext_lines
from mlx.data.tokenizer_helpers import read_trie_from_vocab

wiki = load_wikitext_lines(split="train")
trie = read_trie_from_vocab("/path/to/vocab.txt")

pipeline = (
    wiki
    .tokenize("line", trie, output_key="tokens")   # mode defaults to Shortest
    .filter_key("tokens")
    .prefetch(512, 8)
    .batch(128, dim=dict(tokens=0))
    .sliding_window("tokens", 1025, 1025)
    .shape("tokens", "tokens_length", 0)
    .batch(32)
    .prefetch(2, 1)
)
```

`mode` accepts `mlx.data.core.TokenizeMode.shortest` (default) or `.rand`,
mirroring `Tokenizer.tokenize_shortest` / `tokenize_rand`. `ignore_unk=True`
drops untokenizable content instead of raising.
