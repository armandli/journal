# Trainer and Fine-Tuning

## Standard fine-tuning workflow

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(batch["horoscope"], truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)   # dynamic padding per-batch

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,                       # prefer bf16 over fp16 on Ampere+ GPUs
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",           # NOT evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,     # requires eval_strategy to be set
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,      # NOT tokenizer=
    data_collator=data_collator,
)

trainer.train()
trainer.push_to_hub()                # uploads weights, generation config, tokenizer, model config
```

`Trainer` requires almost nothing beyond a model and dataset — it handles
batching, shuffling, padding-via-collator, the forward/backward/step loop,
logging, evaluation, and checkpointing. `output_dir` is the one required
`TrainingArguments` field in practice (everything else has a default).

## `Trainer(...)` constructor arguments worth knowing

| Arg | Notes |
|---|---|
| `model` | A `PreTrainedModel` or plain `nn.Module`. If omitted, pass `model_init` instead (a zero/one-arg callable — used for hyperparameter search, gets a fresh model each `train()` call). |
| `args` | A `TrainingArguments` instance. Defaults to `output_dir="tmp_trainer"` if omitted. |
| `train_dataset` / `eval_dataset` | `torch.utils.data.Dataset`, `IterableDataset`, or a 🤗 `datasets.Dataset`. `eval_dataset` can be a `dict[str, Dataset]` to evaluate multiple sets at once (metric names get the dict key prepended). Columns not accepted by `model.forward()` are auto-dropped for plain `Dataset`s. |
| `processing_class` | The tokenizer/image processor/feature extractor/processor — used to build the default data collator and saved alongside checkpoints so training can resume cleanly. This is the renamed `tokenizer=` argument. |
| `data_collator` | Defaults to `default_data_collator` with no `processing_class`, or `DataCollatorWithPadding` if one is provided. Override for custom batch assembly (e.g. `DataCollatorForLanguageModeling` for causal/masked LM). |
| `compute_metrics` | `Callable[[EvalPrediction], dict]` for evaluation metrics. If `TrainingArguments(batch_eval_metrics=True)`, the function additionally takes a `compute_result: bool` arg, called at the last batch to compute the global summary. |
| `callbacks` | List of `TrainerCallback` instances/classes to hook into the loop; use `trainer.remove_callback(...)` to drop a default one. |
| `optimizers` | `(optimizer, scheduler)` tuple; defaults to AdamW + linear-warmup scheduler from `args`. |
| `compute_loss_func` | Custom loss given raw outputs, labels, and batch item count — for anything beyond the default "first output element is the loss" convention. |

## `TrainingArguments` — grouped by concern

**Duration/batch size:** `per_device_train_batch_size` (default 8, this ×
device count = global batch size), `num_train_epochs` (default 3.0),
`max_steps` (overrides epochs if positive).

**Learning rate:** `learning_rate` (default 5e-5), `lr_scheduler_type`
(`"linear"`, `"cosine"`, `"constant"`, `"constant_with_warmup"`),
`warmup_steps` (int) or `warmup_ratio` (float 0-1, fraction of total steps).

**Optimizer:** `optim` (default `"adamw_torch"`, or `"adamw_torch_fused"` on
torch≥2.8; also `"adafactor"`, `"adamw_8bit"` via bitsandbytes),
`weight_decay`, `adam_beta1`/`adam_beta2`/`adam_epsilon`.

**Regularization/stability:** `gradient_accumulation_steps` (effective batch
= `per_device_train_batch_size × num_devices × gradient_accumulation_steps`
— note logging/eval/save cadence is counted in these accumulated steps, not
raw forward passes), `max_grad_norm` (default 1.0), `label_smoothing_factor`.

**Mixed precision:** `bf16=True` (preferred on Ampere+), `fp16=True`
(older hardware), `bf16_full_eval`/`fp16_full_eval` for full (not mixed)
precision at eval time, `tf32` (Ampere+ matmul speedup).

**Memory/speed:** `gradient_checkpointing=True` (trade ~20% slower training
for much lower activation memory), `torch_compile=True` (+
`torch_compile_backend`/`torch_compile_mode`), `use_liger_kernel=True`
(Liger Kernel fused ops — ~20% multi-GPU throughput, ~60% less memory; Llama/
Mistral/Mixtral/Gemma only as of now), `auto_find_batch_size=True`,
`torch_empty_cache_steps` (periodic cache clearing to avoid OOM at ~10%
speed cost).

**Evaluation/saving:** `eval_strategy` (`"no"`/`"steps"`/`"epoch"`, default
`"no"` — **this replaced `evaluation_strategy`**), `eval_steps`,
`save_strategy`, `save_steps` (default 500), `save_total_limit`,
`load_best_model_at_end` (requires `eval_strategy` set; when `True`,
`save_strategy` must match `eval_strategy` unless it's `"best"`),
`metric_for_best_model`, `greater_is_better`.

**Hub:** `push_to_hub=True`, `hub_model_id`, `hub_strategy` (default
`"every_save"`), `hub_private_repo`, `hub_token`.

**Logging:** `logging_steps` (default 500), `report_to` (default `"none"` —
set to `"tensorboard"`/`"wandb"`/`list` to enable trackers).

**Distributed:** `ddp_backend`, `ddp_find_unused_parameters`, `fsdp`,
`fsdp_config`, `deepspeed` (path or dict), `parallelism_config`.

**Reproducibility:** `seed` (default 42), `data_seed`, `full_determinism`.

Also see `Seq2SeqTrainer`/`Seq2SeqTrainingArguments` for encoder-decoder
models — same interface plus generation-specific args (`predict_with_generate`,
`generation_config`) for computing eval metrics on generated text.

## PEFT integration — fine-tune adapters instead of the full model

Any `PreTrainedModel` can host adapters directly (no separate `PeftModel`
wrapper needed for LoRA/IA3/AdaLoRA):

```python
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=32, lora_dropout=0.1,
    # target_modules=["q_proj", "k_proj"],   # only needed for uncommon architectures
    # modules_to_save=["lm_head"],            # fully fine-tune these layers too
)
model.add_adapter(lora_config, adapter_name="my_adapter")
```

Then pass the adapted model straight to `Trainer` — only parameters with
`requires_grad=True` (the adapter) get updated, since the base model is
frozen. Checkpoints save only `adapter_model.safetensors` +
`adapter_config.json` (small, fast to save/resume). Requires `peft>=0.19.1`.
Prompt-based PEFT methods (prompt tuning, prefix tuning) still require using
the `peft` library's own `PeftModel` wrapper directly — `add_adapter()` only
covers the non-prompt-learning methods.

## Callbacks and data collators

- **Callbacks** (`TrainerCallback` subclasses) hook into training-loop events
  (step start/end, epoch end, save, log, evaluate) for things like early
  stopping or custom logging. Pass instances via `Trainer(callbacks=[...])`;
  remove a default one with `trainer.remove_callback(cls_or_instance)`.
- **Data collators** assemble a list of dataset samples into a padded batch
  tensor. `DataCollatorWithPadding` (classification/generic) and
  `DataCollatorForLanguageModeling` (causal/masked LM, set `mlm=False` for
  causal) cover the common cases; write a custom collator (a callable taking
  a list of examples, returning a dict of batched tensors) for anything more
  specialized (e.g. dynamic sequence-to-sequence label padding).
