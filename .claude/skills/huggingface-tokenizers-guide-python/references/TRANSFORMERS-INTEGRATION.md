# Integration with `transformers`

## Wrapping a trained `Tokenizer` as a `PreTrainedTokenizerFast`

```python
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,        # a tokenizers.Tokenizer instance, in memory
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```

Equivalently, load directly from a saved `tokenizer.json` without keeping
the `Tokenizer` object around:

```python
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json", unk_token="[UNK]", pad_token="[PAD]", ...)
```

The special-token kwargs here are metadata `transformers` uses for
`tokenizer.pad_token_id`, `tokenizer.cls_token_id`, etc. — they must name
tokens that actually exist in the trained vocabulary (typically the same
strings passed as `special_tokens=[...]` to the `Trainer`). A mismatch
doesn't raise an error; the corresponding `*_token_id` property silently
resolves to `None`, which then surfaces later as a confusing
padding/collation failure.

## Save / reload round trip

```python
fast_tokenizer.save_pretrained("my-tokenizer")
```

Writes `tokenizer.json` (the full `tokenizers` pipeline), plus
`tokenizer_config.json` and `special_tokens_map.json` (the `transformers`-side
metadata). From then on, treat the directory like any Hub repo:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("my-tokenizer")
tok.push_to_hub("my-username/my-tokenizer")
```

## Adding tokens to an existing pretrained tokenizer

Far more common in practice than training from scratch — e.g. adding
chat-format tokens to an off-the-shelf base model before SFT:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model.resize_token_embeddings(len(tokenizer))   # required whenever num_added > 0
```

Skipping `resize_token_embeddings` after adding tokens leaves the model
with an embedding table smaller than the tokenizer's vocab — any batch
containing a newly added token's id then indexes out of bounds and crashes
(or silently corrupts memory, depending on backend) at the embedding
lookup.

## Chat templates

`transformers` tokenizers carry a Jinja `chat_template` (from
`tokenizer_config.json`) that formats a list of role-tagged messages into a
single token sequence with the right special/role tokens interleaved:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
```

When building a *new* tokenizer for a *new* chat format from scratch (not
just reusing an existing base model's), set
`tokenizer.chat_template = "<jinja template string>"` on the
`PreTrainedTokenizerFast` before saving, or the chat template that ships
with the underlying model's original tokenizer (if any) gets used instead
— possibly referencing role tokens your new vocabulary doesn't contain.
