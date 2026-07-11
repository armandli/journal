# Encoding and I/O

## The `Encoding` object

`tokenizer.encode(...)` returns an `Encoding` with these attributes:

| Attribute | Contents |
|---|---|
| `.ids` | Integer token ids — what a model actually consumes. |
| `.tokens` | String form of each token (for debugging/inspection). |
| `.offsets` | `(start, end)` character span per token in the *original* input string — the key to mapping model outputs (e.g. a QA answer span, an NER label) back to source text. |
| `.attention_mask` | 1 for real tokens, 0 for padding. |
| `.type_ids` | Segment id per token (0 for sequence A, 1 for sequence B) — set by the post-processor's `pair` template. |
| `.special_tokens_mask` | 1 where a token was inserted by the post-processor rather than coming from the input text. |
| `.word_ids()` | Maps each token index back to its source word index (`None` for special tokens) — use this, not string splitting, to align per-word labels (NER, POS) to subword tokens. |
| `.sequence_ids()` | Which input sequence (0, 1, or `None` for special tokens) each token belongs to. |
| `.overflowing` | List of additional `Encoding`s produced when truncation with `stride` splits long input into overlapping windows. |

```python
output = tokenizer.encode("Hello, y'all!")
for token, (start, end) in zip(output.tokens, output.offsets):
    print(token, "->", "Hello, y'all!"[start:end])
```

## Batch encoding

```python
outputs = tokenizer.encode_batch(["Hello, y'all!", "How are you?"])
```

Always prefer `encode_batch` over a Python loop of `encode` calls — the
Rust implementation parallelizes across the batch.

## Pair sequences

```python
output = tokenizer.encode("What is the capital of France?", "Paris is the capital of France.")
```

Produces one `Encoding` covering both sequences with `type_ids` marking
which tokens belong to which — used for tasks like QA (question + context)
or NSP-style pretraining (sentence A + sentence B). Requires a
post-processor with a `pair` template (see
[PIPELINE-COMPONENTS.md](PIPELINE-COMPONENTS.md)); without one, `type_ids`
won't distinguish the two sequences correctly.

## Pre-split ("pre-tokenized") input

```python
output = tokenizer.encode(["This", "is", "pre-split", "."], is_pretokenized=True)
```

Skips the pre-tokenizer stage and encodes each given word directly with
the model — needed when word boundaries are already fixed externally (e.g.
token-classification datasets that ship word-level annotations).

## Padding and truncation

```python
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=None)
tokenizer.enable_truncation(max_length=512, stride=0, strategy="longest_first", direction="right")
```

- `length=None` pads each batch to its own longest sequence; set an int to
  pad every batch to a fixed length instead (needed for some
  static-shape/compiled/TPU pipelines).
- `strategy`: `"longest_first"` (default, trims whichever sequence in a
  pair is currently longer), `"only_first"`, `"only_second"` — matters only
  for pair encoding.
- `stride > 0` keeps `stride` tokens of overlap between a truncated
  sequence and its continuation window, populating `.overflowing` — use for
  long-document tasks (long-context QA) where a single window would lose
  information at the cut point.
- `direction`: truncate from `"right"` (default) or `"left"`. For causal-LM
  generation contexts, padding side is a separate, equally important
  setting — see
  [LM-TRAINING-WORKFLOWS.md](LM-TRAINING-WORKFLOWS.md) for why RL/RLHF
  sampling needs left-padded batches.

Disable either with `tokenizer.no_padding()` / `tokenizer.no_truncation()`.

## Adding tokens

```python
tokenizer.add_special_tokens(["<|user|>", "<|assistant|>"])   # never split/merged, always a single token
tokenizer.add_tokens(["supercalifragilisticexpialidocious"])   # ordinary vocab addition, still subject to future retraining behavior
```

Both extend the vocabulary and return the number of tokens actually added
(0 if a token already existed). After adding tokens to a tokenizer that
already backs a model, the model's embedding table must be resized to
match (see [TRANSFORMERS-INTEGRATION.md](TRANSFORMERS-INTEGRATION.md)).

## Saving and loading

```python
tokenizer.save("tokenizer.json")               # full pipeline + trained vocab, single file
tokenizer = Tokenizer.from_file("tokenizer.json")

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")   # fetch an existing Hub tokenizer.json directly, no `transformers` needed
```

## Training from an in-memory iterator

```python
def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))
```

Avoids writing the corpus to disk as plain text files first — use this
directly against a `datasets.Dataset` (see `huggingface-datasets-guide-python`)
rather than exporting to `.txt` and calling `tokenizer.train(files=...)`.
