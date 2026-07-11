# Text Generation: `generate()` and `GenerationConfig`

## Basic usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", dtype="auto", device_map="auto")

model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=30)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

`generate()` is available on any model exposing `GenerationMixin` (check
`model.can_generate()`). Settings come from the model's
`generation_config.json` on the Hub (inspect via `model.generation_config`,
which only shows values overriding the library default) — override per-call
by passing kwargs directly to `generate()`, or build a whole
`GenerationConfig` for reuse.

## Common options

| Option | Type | Effect |
|---|---|---|
| `max_new_tokens` | int | **Always set this explicitly.** Without it, effectively defaults to a small value (historically 20) unless the model's own `GenerationConfig` overrides it — a frequent source of "why did generation stop so early" bugs. |
| `do_sample` | bool | `False` (default) = greedy/deterministic. `True` = sample from the probability distribution — use for creative/chat use cases. |
| `temperature` | float | Only with `do_sample=True`. High (>0.8) = more random/creative; low (<0.4) = more focused/deterministic. |
| `num_beams` | int | `>1` enables beam search (keeps multiple candidate sequences, picks highest overall-probability one). Good for input-grounded tasks (translation, summarization, ASR); combine with `do_sample=True` for beam sampling. |
| `top_k` | int | Sampling restricted to the k highest-probability tokens. |
| `top_p` | float | Nucleus sampling — restricted to the smallest token set whose cumulative probability ≥ p. |
| `repetition_penalty` | float | `>1.0` discourages the model from repeating itself; increase if you see loops. |
| `eos_token_id` | int or list[int] | Token(s) that stop generation. Usually fine at the default. |

```python
model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
model.generate(**inputs, max_new_tokens=50, num_beams=4, do_sample=True)   # beam sampling
```

## Decoding strategies in detail

- **Greedy search** (default): picks the single most likely next token each
  step. Fast, deterministic, but degrades on long outputs (repetition).
- **Sampling** (`do_sample=True, num_beams=1`): draws from the actual
  probability distribution — more diverse/creative, non-deterministic.
- **Beam search** (`num_beams>1`): tracks several candidate continuations in
  parallel, returns the one with highest overall sequence probability at the
  end. Best for tasks where the output should closely track the input
  (translation, image captioning, ASR) rather than be creative.

## `GenerationConfig` — save/reuse a decoding recipe

```python
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
)
generation_config.save_pretrained("my_account/my_model", push_to_hub=True)

# Multiple named configs in one repo (e.g. one for creative writing, one for translation):
generation_config.save_pretrained("/tmp", config_file_name="translation_generation_config.json")
loaded = GenerationConfig.from_pretrained("/tmp", config_file_name="translation_generation_config.json")
outputs = model.generate(**inputs, generation_config=loaded)
```

`generate()` also accepts `logits_processor` (custom `LogitsProcessor`
instances to manipulate next-token probabilities) and `stopping_criteria`
(custom `StoppingCriteria`) for advanced control, plus a `custom_generate`
flag to load an entirely custom decoding loop shared on the Hub.

## Common pitfalls

### Output length
Without `max_new_tokens`, output is often much shorter than expected
(historically capped at 20 new tokens by the library default). Always set it
explicitly rather than relying on defaults.

### Padding side for batched generation
```python
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token   # if the tokenizer has no dedicated pad token
```
Causal LMs are trained left-to-right and were never trained to continue
after a padding token — **right-padding a batch silently produces garbage
continuations** on the shorter sequences. Set `padding_side="left"` (either
at `from_pretrained` time or via `tokenizer.padding_side = "left"`) whenever
batching multiple prompts of different lengths into one `generate()` call.

### Prompt format
Chat/instruction-tuned models expect a specific prompt structure (control
tokens around user/assistant turns). Passing a raw string instead of using
`tokenizer.apply_chat_template(...)` frequently produces noticeably worse
completions even though nothing errors — see
[TOKENIZERS-AND-PROCESSORS.md](TOKENIZERS-AND-PROCESSORS.md) for the chat
template workflow.

### Quantization for large models
```python
from transformers import BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)
```
Install `bitsandbytes` and use `quantization_config` to fit larger models
into limited GPU memory before generation; see the Quantization docs for
other backends (GPTQ, AWQ, etc.) beyond bitsandbytes.
