# Pipeline Components

Every component is assigned as an attribute on a `Tokenizer` instance and
can be swapped independently. `Sequence([...])` (available for normalizers
and pre-tokenizers) chains several together in order.

## Normalizers (`tokenizers.normalizers`)

| Normalizer | Effect |
|---|---|
| `NFC` / `NFD` / `NFKC` / `NFKD` | Unicode normalization forms — pick one consistently; mixing forms across train/inference silently changes token boundaries for accented/composed characters. |
| `Lowercase` | Lowercase all text. |
| `StripAccents` | Remove combining accent marks (run after `NFD`, which decomposes accented chars into base+accent first). |
| `Replace(pattern, content)` | Regex/string substitution. |
| `BertNormalizer` | Bundles BERT's original recipe: lowercase, strip accents, clean text, handle CJK characters — use when replicating BERT-style preprocessing exactly. |
| `Sequence([...])` | Chain multiple normalizers. |

```python
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase

normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.normalizer = normalizer
```

## Pre-tokenizers (`tokenizers.pre_tokenizers`)

| Pre-tokenizer | Effect | Typical pairing |
|---|---|---|
| `Whitespace` | Split on whitespace and punctuation via regex. | Generic BPE/WordLevel. |
| `WhitespaceSplit` | Split on whitespace only (keeps punctuation attached to words). | Custom pipelines needing coarser splits. |
| `Punctuation` | Split off punctuation as isolated tokens. | Combine via `Sequence` with `Whitespace`/`WhitespaceSplit`. |
| `ByteLevel` | Map every byte to a printable unicode char, split on that — guarantees no `[UNK]` ever occurs since every byte sequence is representable. | GPT-2/RoBERTa-style BPE. Pair with `decoders.ByteLevel` and (usually) a `ByteLevel` post-processor for correct offsets/trimming. |
| `Metaspace` | Replace spaces with a visible marker (`▁` by default) and split on it — SentencePiece-style, keeps whitespace information inside tokens themselves. | Unigram (ALBERT/XLNet/T5-style) and SentencePiece-BPE tokenizers. Pair with `decoders.Metaspace`. |
| `Digits` | Split digits from surrounding text, optionally one-digit-per-token (`individual_digits=True`) — helps arithmetic-sensitive models avoid inconsistent number tokenization. | Combine via `Sequence`. |
| `Sequence([...])` | Chain multiple pre-tokenizers. | e.g. `Sequence([Whitespace(), Digits(individual_digits=True)])`. |

## Models + Trainers (`tokenizers.models` / `tokenizers.trainers`)

Each model has exactly one matching trainer — this is the only pipeline
stage that learns anything from data.

| Model | Trainer | Algorithm | Used by |
|---|---|---|---|
| `BPE` | `BpeTrainer` | Iteratively merges the most frequent adjacent symbol pair. | GPT-2, RoBERTa. |
| `WordPiece` | `WordPieceTrainer` | Like BPE but merges by likelihood gain rather than raw frequency; subword continuation pieces marked with `##`. | BERT, DistilBERT. |
| `WordLevel` | `WordLevelTrainer` | No subword splitting — whole words only, unknown words become `[UNK]`. | Simple/legacy pipelines; rarely a good choice for modern open-vocabulary text. |
| `Unigram` | `UnigramTrainer` | Starts from a large candidate vocab and prunes to maximize corpus likelihood. | ALBERT, XLNet, T5, most SentencePiece-based models. |

```python
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)
```

`special_tokens` passed to the trainer are guaranteed slots in the final
vocab (never merged/split) — always include every special token the
downstream model needs *before* training, not after.

## Post-processors (`tokenizers.processors`)

Add special tokens and build `type_ids` (segment ids) around the raw
model output — the model stage itself never adds `[CLS]`/`[SEP]`/etc.

```python
from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
```

`$A`/`$B` mark where each input sequence's tokens go; the `:1` suffix marks
tokens/segments belonging to `type_ids` group 1 (second sentence) —
everything unmarked defaults to group 0. `TemplateProcessing` is general
enough to replicate `BertProcessing` and `RobertaProcessing` (both still
exist as narrower convenience classes for their respective exact recipes).

## Decoders (`tokenizers.decoders`)

Must match the pre-tokenizer, or `tokenizer.decode(ids)` produces garbled
spacing (e.g. missing spaces, stray `##`/`▁` markers left in the output).

| Decoder | Pairs with |
|---|---|
| `decoders.WordPiece` | `WordPiece` model — strips `##` continuation markers and re-joins. |
| `decoders.ByteLevel` | `ByteLevel` pre-tokenizer — inverts the byte-to-unicode mapping. |
| `decoders.Metaspace` | `Metaspace` pre-tokenizer — turns `▁` markers back into spaces. |
| `decoders.BPEDecoder` | Plain `BPE` with a custom suffix/prefix convention (older char-BPE style). |

## Matched recipes

**GPT-2-style BPE** (no normalizer; whitespace is meaningful):
```python
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
```

**BERT-style WordPiece**:
```python
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = BertNormalizer()
tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[...])
tokenizer.decoder = decoders.WordPiece()
```

**SentencePiece-style Unigram** (T5/ALBERT/XLNet family):
```python
tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Metaspace()
tokenizer.decoder = decoders.Metaspace()
```

## Convenience wrappers (`tokenizers.implementations`)

`ByteLevelBPETokenizer`, `BertWordPieceTokenizer`,
`SentencePieceBPETokenizer`, `SentencePieceUnigramTokenizer`,
`CharBPETokenizer` pre-assemble the recipes above into single classes with
a simpler `.train(files=..., vocab_size=..., special_tokens=[...])` call —
reach for one of these first for a standard recipe, and drop to the
manual `Tokenizer(...)` component assembly above only when a recipe needs
customizing (e.g. a nonstandard special-token template, a domain-specific
normalizer).
