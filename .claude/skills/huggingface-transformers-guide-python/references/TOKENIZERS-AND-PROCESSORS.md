# Tokenizers and Multimodal Processors

## Loading

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```

Always prefer `AutoTokenizer.from_pretrained(...)` over a model-specific
tokenizer class — it resolves the correct tokenizer (and correct backend)
automatically. Most tokenizers resolve to a fast, Rust-backed implementation.

**Naming note (v5):** internally, tokenizer backends were renamed
`PythonBackend` (pure-Python), `TokenizersBackend` (Rust-based, fast — most
common), and `SentencePieceBackend`. The familiar public names
`PreTrainedTokenizer` and `PreTrainedTokenizerFast` remain the documented
aliases for `PythonBackend`/`TokenizersBackend` respectively — you'll mostly
never need to reference the backend names directly.

## Encode / decode

```python
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
# {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

tokenizer.encode("Sphinx of black quartz, judge my vow.")   # -> list[int], input_ids only

tokenizer.decode(output_ids)                                  # preserves exact spacing
tokenizer.decode(output_ids, skip_special_tokens=True)         # strip <bos>/<eos>/etc.
tokenizer.batch_decode(batch_of_output_ids, skip_special_tokens=True)
```

## Batch processing, padding, truncation

```python
tokenizer(
    ["Sphinx of black quartz, judge my vow.", "Pack my box with five dozen liquor jugs."],
    return_tensors="pt",
    padding=True,        # pad to the longest sequence in the batch
    truncation=True,     # truncate sequences longer than max_length/model max
    max_length=512,
)
```

- `padding=True` pads to the longest sequence in the current batch;
  `padding="max_length"` + `max_length=N` pads every sequence to a fixed
  size. The attention mask marks padded positions as `0` so the model
  ignores them.
- For **generation with decoder-only (causal) LMs**, set
  `tokenizer.padding_side = "left"` (or pass `padding_side="left"` to
  `from_pretrained`) — these models weren't trained to continue from padding
  tokens, so right-padding silently produces garbage continuations. This
  usually also requires setting `tokenizer.pad_token = tokenizer.eos_token`
  if the tokenizer has no dedicated pad token.

## Special and extra-special tokens

```python
tokenizer.encode("...")           # special tokens (bos/eos/etc.) inserted automatically
tokenizer.decode(ids)              # '<bos>...'

# Register additional named special tokens (common for multimodal placeholders)
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    extra_special_tokens={"image_token": "<image>"},
)
tokenizer.image_token, tokenizer.image_token_id
```

Multimodal tokenizers (e.g. loaded from a vision-language model) expose their
special tokens as direct attributes (`tokenizer.image_token_id`, etc.) for
easy access without hardcoding ids.

## Chat templates

Chat models are trained on a specific control-token format
(`[INST]`/`[/INST]`, `<|user|>`/`<|assistant|>`, etc.) that differs per model
family — `apply_chat_template` applies the model's own template so you don't
have to hardcode it:

```python
messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
outputs = model.generate(tokenized_chat, max_new_tokens=128)
tokenizer.decode(outputs[0])
```

- `tokenize=False` returns the formatted string instead of token ids — useful
  for inspection, but if you tokenize it yourself afterward, pass
  `add_special_tokens=False` to avoid double-inserting BOS/EOS (the template
  already includes them).
- `add_generation_prompt=True` appends the tokens that signal "assistant,
  your turn" (e.g. `<|assistant|>`) — needed when you want the model to
  respond next, not when formatting a complete finished conversation.
- Roles are conventionally `system` (instructions, usually first),
  `user`, and `assistant`.

## Multimodal processors (`ProcessorMixin`)

Multimodal models need more than a tokenizer — e.g. a vision-language model
needs both an image processor and a tokenizer. `ProcessorMixin` bundles them:

```python
from transformers import AutoProcessor
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text="answer en Where is the cat standing?", images=image, return_tensors="pt")
```

`AutoProcessor` is the recommended entry point (infers the right processor
for the model); model-specific processor classes also work
(`WhisperProcessor.from_pretrained(...)`). You can access the underlying
tokenizer/image-processor/feature-extractor separately if needed —
`AutoProcessor` just saves you from wiring them together yourself.

## Image processors

```python
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
inputs = image_processor(image, return_tensors="pt")   # -> pixel_values tensor
```

Image processors handle resize/center-crop/normalize/rescale to match what
the model was pretrained on. Since v5 they use a **backend-based
architecture**:
- `TorchvisionBackend` (default when torchvision is installed) — GPU-capable,
  up to ~33x faster than PIL for batched tensor inputs. All models support
  it; newer models support *only* this backend.
- `PilBackend` — CPU-only, only available for some older models, useful when
  you need to reproduce a model's original numeric outputs exactly.

```python
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", backend="torchvision")
# or backend="pil" (older models only)
```

Check `image_processor.backend` to see which one is active. A handful of
older models (Chameleon, Flava, Idefics3, SmolVLM) default to PIL depending
on your installed torchvision version, due to interpolation-mode support
differences — pass `backend="torchvision"` explicitly to force it once your
torchvision is new enough.

## Feature extractors and video processors

`AutoFeatureExtractor` (audio models) and `AutoVideoProcessor` (video models)
follow the same `from_pretrained()` / callable pattern as image processors —
converting raw audio arrays or video frames into the tensors a specific
pretrained model expects (sampling rate, frame count, normalization, etc.).
