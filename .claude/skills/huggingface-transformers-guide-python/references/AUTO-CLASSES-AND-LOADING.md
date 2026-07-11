# Auto Classes, Loading, and Custom Models

## Why Auto classes

The architecture can usually be inferred from a model's name/path.
`AutoConfig`/`AutoModel*`/`AutoTokenizer`/`AutoProcessor` resolve to the
correct concrete class automatically:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("google-bert/bert-base-cased", device_map="auto")
# -> an instance of BertModel
```

There's one `AutoModelFor*` class per **task** (not per architecture) — pick
the class matching what you want the model to output, not the model family.

## Auto class catalogue

**Config/tokenizer/processor:** `AutoConfig`, `AutoTokenizer`,
`AutoFeatureExtractor`, `AutoImageProcessor`, `AutoVideoProcessor`,
`AutoProcessor` (multimodal).

**Generic:** `AutoModel` (base, no task head), `AutoModelForPreTraining`.

**NLP:** `AutoModelForCausalLM`, `AutoModelForMaskedLM`,
`AutoModelForMaskGeneration`, `AutoModelForSeq2SeqLM`,
`AutoModelForSequenceClassification`, `AutoModelForMultipleChoice`,
`AutoModelForNextSentencePrediction`, `AutoModelForTokenClassification`,
`AutoModelForQuestionAnswering`, `AutoModelForTextEncoding`.

**Computer vision:** `AutoModelForDepthEstimation`,
`AutoModelForNormalEstimation`, `AutoModelForPointmapEstimation`,
`AutoModelForImageMatting`, `AutoModelForTextRecognition`,
`AutoModelForTableRecognition`, `AutoModelForImageClassification`,
`AutoModelForVideoClassification`, `AutoModelForPoseEstimation`,
`AutoModelForKeypointDetection`, `AutoModelForKeypointMatching`,
`AutoModelForMaskedImageModeling`, `AutoModelForObjectDetection`,
`AutoModelForImageSegmentation`, `AutoModelForImageToImage`,
`AutoModelForSemanticSegmentation`, `AutoModelForInstanceSegmentation`,
`AutoModelForUniversalSegmentation`, `AutoModelForZeroShotImageClassification`,
`AutoModelForZeroShotObjectDetection`.

**Audio:** `AutoModelForAudioClassification`,
`AutoModelForAudioFrameClassification`, `AutoModelForCTC`,
`AutoModelForTDT`, `AutoModelForRNNT`, `AutoModelForSpeechSeq2Seq`,
`AutoModelForAudioXVector`, `AutoModelForTextToSpectrogram`,
`AutoModelForTextToWaveform`, `AutoModelForAudioTokenization`.

**Multimodal:** `AutoModelForMultimodalLM`,
`AutoModelForTableQuestionAnswering`, `AutoModelForDocumentQuestionAnswering`,
`AutoModelForVisualQuestionAnswering`, `AutoModelForImageTextToText`.

**Time series:** `AutoModelForTimeSeriesPrediction`.

```python
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering

classifier_model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
```

## `from_pretrained()` — the load path

Works identically across `AutoConfig`, `AutoModel*`, `AutoTokenizer`,
`AutoProcessor`, and their model-specific concrete classes. First positional
arg is either a Hub repo id (`"org/model"`) or a local directory containing
files written by a previous `save_pretrained()` call.

### Key loading kwargs (models)

| Kwarg | Purpose |
|---|---|
| `dtype="auto"` \| `torch.float16` \| `"bfloat16"` | **Use `dtype`, not the older `torch_dtype`.** `"auto"` loads weights in whatever dtype they were saved in (reads `config.json`'s `dtype`/`torch_dtype` field, or the first floating-point weight's dtype) instead of upcasting to fp32 — avoids doubling memory for bf16/fp16 checkpoints. |
| `device_map="auto"` | Uses `accelerate` to place model shards across available devices (GPU first, then CPU/disk offload) automatically. Don't combine with a plain `device=` argument — they conflict. Can also be an explicit dict, an int (single GPU ordinal), or a device string. |
| `attn_implementation` | `"sdpa"` (default when available), `"eager"`, `"flash_attention_2"`, `"flash_attention_3"`, `"flash_attention_4"`. Also accepts an HF Hub kernel reference like `"org/model@revision:kernel_name"` to load a custom attention kernel. |
| `quantization_config` | A `QuantizationConfigMixin` (e.g. `BitsAndBytesConfig(load_in_4bit=True)`) or dict — reduces memory for large models. |
| `tp_plan="auto"` | Tensor-parallel plan; faster than `device_map` for multi-GPU but requires launching via `torchrun`. |
| `trust_remote_code=True` | Required to load models whose modeling/config/tokenizer code lives in the Hub repo itself rather than in the installed `transformers` package. **Only set this for repos you've read and trust** — it executes arbitrary code from the Hub locally. |
| `revision` | Branch/tag/commit id (git-based versioning); use `"refs/pr/<n>"` to test an open Hub PR. |
| `token` | Auth token for private/gated repos; `True` reuses the token from `hf auth login`. |
| `cache_dir`, `force_download`, `local_files_only`, `proxies` | Standard download/cache controls. |
| `subfolder` | If the model files live in a subfolder of the repo. |
| `use_safetensors` | Prefer `.safetensors` weight files over pickled `.bin` (default auto-detects). |
| `ignore_mismatched_sizes=True` | Needed when e.g. loading a checkpoint with a different number of classification labels than the config declares. |

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="sdpa",
)
```

Any `**kwargs` not consumed above are forwarded to the model's config (if no
explicit `config=` was given) or directly to the model's `__init__` (if a
`config=` was given) — e.g. `output_attentions=True`.

## Saving and sharing

```python
model.save_pretrained("./my_model_directory")       # writes config.json + weights
tokenizer.save_pretrained("./my_model_directory")    # writes tokenizer files alongside

model.push_to_hub("my-finetuned-bert")               # push to your namespace
model.push_to_hub("my-org/my-finetuned-bert")        # push to an organization
```

`push_to_hub()` accepts `commit_message`, `private`, `revision` (target
branch), `create_pr=True`, `max_shard_size` (default `"50GB"` per shard), and
`tags`. Config/model/tokenizer/processor classes all share this method via
`PushToHubMixin`.

## Custom models (`trust_remote_code`) and Auto-class registration

To ship a fully custom architecture (config + model code that doesn't exist
in the installed `transformers` package), subclass the base classes and give
the config a unique `model_type`:

```python
from transformers import PreTrainedConfig, PreTrainedModel

class ResnetConfig(PreTrainedConfig):
    model_type = "resnet"                    # must be unique; used for AutoClass dispatch

    def __init__(self, block_type="bottleneck", layers=[3, 4, 6, 3], num_classes=1000, **kwargs):
        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        super().__init__(**kwargs)           # REQUIRED: pass through unknown kwargs

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig              # REQUIRED: ties model to its config class

    def __init__(self, config):
        super().__init__(config)
        self.model = build_resnet_from(config)

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

Rules: the config's `__init__` must accept and forward arbitrary `**kwargs`
to `super().__init__()` (base `PreTrainedConfig` has more fields than your
subset); the model's `config_class` must point back at your config class.

Register with the Auto API so users get `AutoModel.from_pretrained(...)`
instead of importing your class directly:

```python
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

Push it like any other model (`resnet50d.push_to_hub("custom-resnet50d")`) —
the `modeling.py`/`configuration.py` files get uploaded alongside the
weights. Anyone loading it back needs `trust_remote_code=True`:

```python
model = AutoModel.from_pretrained("your-namespace/custom-resnet50d", trust_remote_code=True)
```

## `ModelOutput` — what `model(...)` returns

Every forward pass returns a `ModelOutput` subclass (e.g.
`SequenceClassifierOutput`, `CausalLMOutputWithPast`,
`Seq2SeqLMOutput`) — a dataclass usable as an object, a tuple, or a dict:

```python
outputs = model(**inputs, labels=labels)
outputs.loss, outputs.logits            # attribute access; None if not computed
outputs[:2]                              # tuple access — only non-None fields count
outputs["logits"]                        # dict-style access
```

Fields are only populated when requested (e.g. `hidden_states` requires
`output_hidden_states=True`, `attentions` requires `output_attentions=True`
on the forward call or config). Don't assume a field is present without
checking — unrequested fields are `None`, and tuple-style unpacking changes
length when a field is missing.
