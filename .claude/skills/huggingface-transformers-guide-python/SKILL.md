---
name: huggingface-transformers-guide-python
description: Write and debug Python code using the Hugging Face transformers library — loading pretrained/custom models and configs from the Hub (Auto classes), running inference with pipeline(), tokenizing/processing multimodal inputs, fine-tuning with Trainer/TrainingArguments, and text generation with generate()/GenerationConfig. Use when the user asks to "load a model from huggingface", "use AutoModel/AutoTokenizer", "fine-tune with Trainer", "use a huggingface pipeline", "generate text with a transformers model", or writes code importing `transformers`. Do NOT use for other HF libraries used alone (datasets, accelerate, peft, diffusers) without transformers, or for non-HF model-loading code.
argument-hint: "[task or description of what to implement]"
---

# Hugging Face Transformers Python Guide

**This library ships a new major version (v5) with renamed kwargs — verify
against installed version before trusting older tutorials/training data.**
Notably in current docs:
- `AutoModel.from_pretrained(..., dtype="auto")` — **not** `torch_dtype=` (the
  old kwarg still loads but `dtype` is now the documented, canonical name).
- `Trainer(..., processing_class=tokenizer)` — **not** `tokenizer=tokenizer`.
- `TrainingArguments(eval_strategy=...)` — **not** `evaluation_strategy=`.
- Tokenizer backends were renamed internally to `PythonBackend` /
  `TokenizersBackend` / `SentencePieceBackend`; the public names
  `PreTrainedTokenizer` / `PreTrainedTokenizerFast` still work as the
  documented aliases.
- Adapters (LoRA/IA3/AdaLoRA via PEFT) can now be attached directly to any
  `PreTrainedModel` with `model.add_adapter(peft_config)` — no need to wrap in
  a separate `PeftModel` for the common case.

## The three base classes

Every pretrained model is built from three classes, all loaded via
`from_pretrained(...)` from a Hub repo id or local directory:

| Class | Role |
|---|---|
| `PreTrainedConfig` | Model hyperparameters (hidden size, num layers, vocab size, ...). |
| `PreTrainedModel` | The architecture + weights. Returns raw hidden states unless it's a task-specific `*For*` head. |
| Preprocessor (tokenizer / image processor / feature extractor / processor) | Converts raw input (text/image/audio/multimodal) into model tensors. |

## Loading a model (the Auto-class pattern)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    dtype="auto",          # load weights in their saved dtype instead of fp32 (avoids 2x memory)
    device_map="auto",     # let accelerate place weights on the fastest available device(s)
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=30)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

Always set `dtype="auto"` and `device_map="auto"` unless you have a specific
reason not to — omitting `dtype` silently doubles memory usage for any model
saved in fp16/bf16 (PyTorch defaults to fp32). Full Auto-class catalogue,
`from_pretrained` kwargs (`attn_implementation`, `quantization_config`,
`tp_plan`, `trust_remote_code`, etc.), `save_pretrained`/`push_to_hub`, and
how custom architectures register into the Auto API:
[references/AUTO-CLASSES-AND-LOADING.md](references/AUTO-CLASSES-AND-LOADING.md).

## Quick inference: `pipeline()`

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cuda")
pipe("The secret to baking a good cake is ", max_length=50)
```

Covers 25+ tasks (text/audio/vision/multimodal) with one interface — full
task list and batching/dataset-iteration patterns:
[references/PIPELINE-API.md](references/PIPELINE-API.md).

## Fine-tuning: `Trainer` + `TrainingArguments`

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes").map(lambda x: tokenizer(x["text"]), batched=True)

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    eval_strategy="epoch",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,             # NOT `tokenizer=`
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
trainer.push_to_hub()
```

Full `TrainingArguments` option groups (mixed precision, gradient
accumulation/checkpointing, `torch.compile`, optimizers, hub push strategy),
callbacks, data collators, and the PEFT/`add_adapter()` integration:
[references/TRAINER-AND-TRAINING.md](references/TRAINER-AND-TRAINING.md).

## Tokenizers and multimodal processors

```python
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(["text one", "text two"], return_tensors="pt", padding=True, truncation=True)
tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
```

Encode/decode, padding/truncation, special/extra special tokens, chat
templates, and multimodal `ProcessorMixin`/`AutoImageProcessor`/
`AutoFeatureExtractor`:
[references/TOKENIZERS-AND-PROCESSORS.md](references/TOKENIZERS-AND-PROCESSORS.md).

## Text generation

```python
model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50)
```

`max_new_tokens` defaults effectively to 20 if unset — always set it
explicitly. `generate()`/`GenerationConfig`, decoding strategies (greedy,
sampling, beam search), and common pitfalls (padding side, prompt format):
[references/GENERATION.md](references/GENERATION.md).

## Testing

After wiring model/tokenizer/pipeline code, actually run it against a real
(possibly small/local) checkpoint and inspect the output — shape mismatches,
dtype/device placement errors, and wrong chat-template formatting are the
most common silent failures and won't show up from reading the code alone.
Prefer a small public checkpoint (e.g. `distilbert/distilbert-base-uncased`,
`hf-internal-testing/*` tiny models) for quick smoke tests before pointing at
a large model.

## References

- [references/AUTO-CLASSES-AND-LOADING.md](references/AUTO-CLASSES-AND-LOADING.md) — Auto class catalogue, `from_pretrained`/`save_pretrained`/`push_to_hub`, custom model registration, `ModelOutput`
- [references/PIPELINE-API.md](references/PIPELINE-API.md) — `pipeline()` full task list, batching/streaming
- [references/TOKENIZERS-AND-PROCESSORS.md](references/TOKENIZERS-AND-PROCESSORS.md) — tokenizer API, chat templates, image/feature/video processors
- [references/TRAINER-AND-TRAINING.md](references/TRAINER-AND-TRAINING.md) — `Trainer`/`TrainingArguments`, callbacks, data collators, PEFT
- [references/GENERATION.md](references/GENERATION.md) — `generate()`, `GenerationConfig`, decoding strategies, pitfalls

## External Docs

- Full docs: https://huggingface.co/docs/transformers/index

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "huggingface-transformers-guide-python"
```
