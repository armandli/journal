# The `pipeline()` API

`pipeline()` is a factory function that wraps preprocessing (tokenizer/
image processor/feature extractor/processor), the model, and postprocessing
into one callable.

```python
from transformers import pipeline

pipe = pipeline("text-classification")
pipe("This restaurant is awesome")
# [{'label': 'POSITIVE', 'score': 0.9998743534088135}]

pipe(["This restaurant is awesome", "This restaurant is awful"])   # batched call
```

Pick a specific model instead of the task default:

```python
pipe = pipeline(model="FacebookAI/roberta-large-mnli")   # task inferred from the model
```

Or pass explicit components (useful when you already loaded a specific model/tokenizer):

```python
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
```

## Key `pipeline()` kwargs

| Kwarg | Purpose |
|---|---|
| `task` | See task table below. |
| `model` | Repo id or a `PreTrainedModel` instance. Defaults to the task's default model if omitted. |
| `tokenizer` / `feature_extractor` / `image_processor` / `processor` | Explicit preprocessing components; don't specify all of them at once — let `pipeline()` infer the rest from `model`. |
| `device` | `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal int. |
| `device_map="auto"` | Sent through to the model's `from_pretrained` — don't combine with `device`. |
| `dtype` | Sent through to the model's `from_pretrained` (e.g. `"auto"`, `torch.float16`). |
| `trust_remote_code=True` | Same caveat as `from_pretrained` — only for repos you trust. |
| `model_kwargs` | Dict forwarded to the underlying model's `from_pretrained(**model_kwargs)`. |
| `use_fast` | Prefer a fast (Rust-backed) tokenizer when available (default `True`). |

## Full task list

**Audio:** `"audio-classification"`, `"automatic-speech-recognition"`,
`"text-to-audio"` (alias `"text-to-speech"`),
`"zero-shot-audio-classification"`.

**Computer vision:** `"depth-estimation"`, `"image-classification"`,
`"image-feature-extraction"`, `"image-segmentation"`, `"image-text-to-text"`,
`"keypoint-matching"`, `"object-detection"`, `"video-classification"`,
`"zero-shot-image-classification"`, `"zero-shot-object-detection"`.

**NLP:** `"fill-mask"`, `"table-question-answering"`,
`"text-classification"` (alias `"sentiment-analysis"`), `"text-generation"`,
`"token-classification"` (alias `"ner"`), `"zero-shot-classification"`.

**Multimodal:** `"document-question-answering"`, `"feature-extraction"`,
`"mask-generation"`.

```python
from accelerate import Accelerator
device = Accelerator().device

pipe = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=device)
segments = pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
segments[0]["label"]   # 'bird'

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
# {'text': ' He hoped there would be stew for dinner...'}
```

Inputs can be a local path, a URL, raw bytes, or a PIL image, depending on
the pipeline's modality.

## Batching over datasets (don't hand-roll a loop)

For iterating a full dataset, pass a `datasets.Dataset` (via `KeyDataset`) or
a generator directly to the pipeline — it handles batching internally and is
as fast as a custom loop on GPU:

```python
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)   # {"text": "..."}
```

Or a plain Python generator (note: `num_workers > 1` preprocessing doesn't
apply to generator inputs since they're inherently sequential):

```python
def data():
    while True:
        yield "This is a test"   # e.g. from a queue, DB, or HTTP request

for out in pipe(data()):
    print(out)
```

## Other notes

- **Chunk batching**: pipelines whose inputs can exceed a model's max length
  (e.g. long-document NER, ASR on long audio) automatically chunk the input
  and reassemble outputs — no special handling needed on your end.
- **FP16 inference**: pass `dtype=torch.float16` (or `"auto"`) to run in half
  precision on supported hardware.
- **Custom pipeline code**: a Hub repo can ship its own custom `Pipeline`
  subclass; loading it requires `trust_remote_code=True` just like custom
  models.
