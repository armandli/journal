# Tokenizer Setup Across LM Training Stages

The `tokenizers`/`transformers` API surface stays the same throughout, but
which knobs matter — and which mistakes are costly — changes by stage.

## Pre-training: designing the tokenizer itself

This is the one stage where you actually train a *new* tokenizer (see
[PIPELINE-COMPONENTS.md](PIPELINE-COMPONENTS.md) and
[ENCODING-AND-IO.md](ENCODING-AND-IO.md)). Decisions that are expensive to
change later because they're baked into the model's embedding table:

- **Vocab size** — must equal the model config's vocab size
  (`vocab_size` in the model config, the embedding table's first
  dimension). Pick it before writing any model code, not after.
- **Special-token set** — include every token the architecture needs
  (`[PAD]`/`[BOS]`/`[EOS]`/`[UNK]`/task-specific tokens) in the trainer's
  `special_tokens=[...]` list at training time; retrofitting a special
  token later means either wasting a real vocab slot via `add_tokens` or
  retraining the tokenizer (and re-tokenizing the whole pre-training
  corpus) from scratch.
- **Pre-tokenizer/decoder pairing correctness** — verify round-trip
  (`decode(encode(x).ids) == x`, modulo whitespace normalization) on
  representative corpus text *before* launching a multi-day pre-training
  run; a mismatched pair silently degrades every downstream sample rather
  than raising an error.

```python
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
trainer = BpeTrainer(vocab_size=50257, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(batch_iterator(pretrain_dataset), trainer=trainer)
```

## Supervised fine-tuning (SFT): reuse the tokenizer, extend the format

Retraining the tokenizer for SFT is almost never right — it would
invalidate the base model's learned embeddings for every existing token.
Instead:

1. **Add format-specific special tokens** (chat role markers, tool-call
   delimiters) to the *existing* pretrained tokenizer via
   `add_special_tokens`, then `model.resize_token_embeddings(len(tokenizer))`
   — see [TRANSFORMERS-INTEGRATION.md](TRANSFORMERS-INTEGRATION.md).
2. **Format conversations with `apply_chat_template`** rather than
   hand-concatenating strings — it places role tokens exactly where the
   template (and thus the model's expectations) define them.
3. **Mask the prompt out of the loss.** Tokenize the full
   prompt+completion text once (not the prompt and completion separately —
   a subword merge can span the boundary, so tokenizing them separately and
   concatenating ids can produce a *different* token sequence than
   tokenizing the whole string at once). Find the completion's start
   character offset in the original string, then use `output.offsets` to
   find the first token whose span starts at or after that offset, and set
   `labels[:boundary_token_idx] = -100` for that example:

```python
full_text = prompt + completion
encoding = tokenizer(full_text, return_offsets_mapping=True)
completion_char_start = len(prompt)
boundary = next(i for i, (start, end) in enumerate(encoding["offset_mapping"]) if start >= completion_char_start)

labels = list(encoding["input_ids"])
labels[:boundary] = [-100] * boundary   # -100 is the ignore_index PyTorch's cross-entropy loss skips
```

## RL / RLHF (PPO, DPO, GRPO-style): generation-time tokenizer settings

These stages repeatedly *generate* from the policy model in batches, which
imposes different requirements than the single-forward-pass SFT case:

- **Left padding for generation**: `tokenizer.padding_side = "left"`.
  Causal-LM generation appends new tokens at the sequence end; with
  right-padding, batched prompts of different lengths would have padding
  tokens sitting *before* the generation point for shorter sequences,
  corrupting position ids/attention for those rows. (Switch back to
  right-padding, the default, for any subsequent non-generation forward
  pass, e.g. computing log-probs over a fixed full sequence.)
- **`pad_token` often doesn't exist** on base causal LMs (GPT-2, Llama).
  Either reuse an existing token —
  `tokenizer.pad_token = tokenizer.eos_token` (simplest, but means padding
  and end-of-sequence are indistinguishable in `input_ids` alone — rely on
  `attention_mask`, not token identity, wherever the distinction matters) —
  or add a genuine new pad token and resize embeddings as in the SFT
  section, if the training code needs the two concepts kept separate.
- **Decode generations with `skip_special_tokens=True`** when extracting
  the completion text to score with a reward model or judge — otherwise
  padding/eos tokens leak into the text being scored.
- **Tokenizer consistency between policy and reward/reference models.**
  If the reward model is a *different* checkpoint from the policy, confirm
  it uses a compatible tokenizer (same vocab, same special tokens) before
  assuming token-level alignment between the two — mismatched vocabularies
  make token-level reward shaping (as opposed to sequence-level scoring)
  meaningless.
