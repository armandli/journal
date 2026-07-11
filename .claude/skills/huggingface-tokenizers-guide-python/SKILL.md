---
name: huggingface-tokenizers-guide-python
description: Write and debug Python code using the Hugging Face `tokenizers` library — the low-level, Rust-backed tokenization engine underneath `transformers`. Covers the Tokenizer pipeline (normalizer, pre-tokenizer, model, post-processor, decoder), training a tokenizer from scratch (BPE/WordPiece/WordLevel/Unigram), the Encoding object (ids, offsets, word_ids, attention_mask, padding/truncation), bridging into `transformers` via `PreTrainedTokenizerFast`/`AutoTokenizer`, and tokenizer setup for LM training stages (pre-training vocab design, SFT special-token/chat-template and prompt-masking, RL/RLHF generation padding). Use when the user asks to "train a tokenizer", "build a custom tokenizer", "use the tokenizers library", "add special tokens to a tokenizer", or writes code importing `tokenizers`. Do NOT use for `AutoTokenizer`-only usage with no custom pipeline/training involved (use huggingface-transformers-guide-python) or for `datasets`/`diffusers` work without tokenization.
argument-hint: "[task or description of what to implement]"
---

# Hugging Face Tokenizers Python Guide

`tokenizers` is the Rust-backed library that implements the actual
tokenization pipeline; `transformers`' `AutoTokenizer` / `PreTrainedTokenizerFast`
wraps a `tokenizers.Tokenizer` instance and adds framework glue (chat
templates, `Trainer` integration, Python-side convenience methods). Reach
for this skill's content whenever the task is building/training/customizing
the pipeline itself, not just calling an existing `AutoTokenizer`.

## The pipeline

A `Tokenizer` is five swappable components applied in order:

| Stage | Role | Common choices |
|---|---|---|
| Normalizer | Clean raw text (unicode form, casing, accents) before splitting. | `NFD`, `NFKC`, `Lowercase`, `StripAccents`, `BertNormalizer`, `Sequence([...])` |
| Pre-tokenizer | Split text into word-ish chunks before the subword algorithm runs. | `Whitespace`, `ByteLevel` (GPT-2 style), `Metaspace` (SentencePiece style), `Digits`, `Sequence([...])` |
| Model | The subword algorithm — turns pre-tokenized chunks into subword tokens/ids. This is the only stage that's *trained*. | `BPE`, `WordPiece`, `WordLevel`, `Unigram` |
| Post-processor | Add special tokens / build segment (type) ids for single or paired sequences. | `TemplateProcessing`, `BertProcessing`, `RobertaProcessing` |
| Decoder | Invert the pre-tokenizer's splitting when turning ids back into a string. Must match the pre-tokenizer or round-tripping breaks. | `WordPiece`, `ByteLevel`, `Metaspace`, `BPEDecoder` |

Full component catalogue, matching pre-tokenizer/decoder pairs, and
ready-to-copy recipes (GPT-2-style BPE, BERT-style WordPiece,
SentencePiece-style Unigram):
[references/PIPELINE-COMPONENTS.md](references/PIPELINE-COMPONENTS.md).

## Training a tokenizer from scratch

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

For data already in memory (e.g. a `datasets.Dataset`), use
`tokenizer.train_from_iterator(iterator, trainer=trainer)` instead of
writing text files to disk first. Vocab size and the special-tokens list
must be decided **before** training — changing them later means retraining,
not patching.

## Encoding text

```python
tokenizer = Tokenizer.from_file("tokenizer.json")
output = tokenizer.encode("Hello, y'all! How are you?")
output.tokens        # ['Hello', ',', 'y', "'", 'all', '!', ...]
output.ids            # matching integer ids
output.offsets        # (start, end) char span per token in the original string
```

`Encoding` also carries `attention_mask`, `special_tokens_mask`,
`word_ids()` (map each token back to its source word, `None` for special
tokens), and `overflowing` (extra windows when truncation uses a `stride`).
Padding/truncation configuration, batch encoding, pair sequences (sentence
A/B), and saving/loading:
[references/ENCODING-AND-IO.md](references/ENCODING-AND-IO.md).

## Using a trained tokenizer with `transformers`

```python
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
)
fast_tokenizer.save_pretrained("my-tokenizer")   # writes tokenizer.json + tokenizer_config.json + special_tokens_map.json
```

The saved directory loads back with plain `AutoTokenizer.from_pretrained("my-tokenizer")` —
from that point on it behaves like any Hub tokenizer. Special-token kwargs
passed here must match the special tokens actually present in the trained
vocabulary, or lookups (`pad_token_id`, etc.) silently resolve to the wrong
id. Round-trip details and adding tokens to an *existing* pretrained
tokenizer: [references/TRANSFORMERS-INTEGRATION.md](references/TRANSFORMERS-INTEGRATION.md).

## Tokenizer setup across LM training stages

The tokenizer decisions that matter differ by stage:

- **Pre-training**: vocab size and special-token set are locked in at
  tokenizer-training time and must match the model's embedding table size
  exactly.
- **Supervised fine-tuning (SFT)**: normally reuse the base model's
  tokenizer unchanged, but add chat/role special tokens
  (`add_special_tokens`) and resize the model's embedding table to match;
  mask prompt tokens out of the loss using offsets, not naive string
  splitting.
- **RL/RLHF (PPO/DPO/GRPO-style)**: `padding_side` must be `"left"` for
  batched generation; causal LMs frequently have no `pad_token` and need
  one assigned or added before batching.

Concrete code for each stage: [references/LM-TRAINING-WORKFLOWS.md](references/LM-TRAINING-WORKFLOWS.md).

## Testing

After building or training a tokenizer, round-trip a real string through it
and inspect the result — `tokenizer.decode(tokenizer.encode(text).ids)`
should reconstruct `text` (whitespace differences aside for some
pre-tokenizers). Check `output.tokens` on a sentence containing the
characters your domain actually uses (punctuation, digits, non-ASCII/emoji,
code snippets, whatever's in the real corpus) — an `[UNK]`-heavy or
byte-fragmented output signals a normalizer/pre-tokenizer mismatch with the
data, not a training-data-size problem. When bridging into
`PreTrainedTokenizerFast`, verify `tokenizer.pad_token_id`,
`tokenizer.cls_token_id`, etc. resolve to real (non-`None`) ids before
handing the tokenizer to a training loop.

## References

- [references/PIPELINE-COMPONENTS.md](references/PIPELINE-COMPONENTS.md) — normalizers, pre-tokenizers, models + trainers, post-processors, decoders, matched recipes
- [references/ENCODING-AND-IO.md](references/ENCODING-AND-IO.md) — `Encoding` object, padding/truncation, pair sequences, batch encode, saving/loading, `train_from_iterator`
- [references/TRANSFORMERS-INTEGRATION.md](references/TRANSFORMERS-INTEGRATION.md) — `PreTrainedTokenizerFast`, `AutoTokenizer` round-trip, adding tokens to an existing tokenizer, chat templates
- [references/LM-TRAINING-WORKFLOWS.md](references/LM-TRAINING-WORKFLOWS.md) — pre-training vocab design, SFT special tokens + prompt-loss-masking, RL/RLHF generation padding

## External Docs

- Full docs: https://huggingface.co/docs/tokenizers/index

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "huggingface-tokenizers-guide-python"
```
