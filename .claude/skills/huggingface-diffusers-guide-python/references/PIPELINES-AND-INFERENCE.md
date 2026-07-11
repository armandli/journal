# Pipelines and Inference

## `DiffusionPipeline` and the Auto classes

`DiffusionPipeline.from_pretrained(repo_id, **kwargs)` downloads and wires
up every component (model(s), scheduler, tokenizer(s), text encoder(s), VAE)
listed in the repo's `model_index.json`. Key `from_pretrained` kwargs:

| Kwarg | Purpose |
|---|---|
| `torch_dtype` | Weight dtype, e.g. `torch.float16`/`torch.bfloat16`. Halves memory vs the fp32 default. |
| `variant` | Load a named weight variant, e.g. `"fp16"` — fetches smaller pre-cast weight files instead of casting fp32 weights after download. |
| `use_safetensors` | Prefer `.safetensors` files over pickled `.bin` (safer, faster to load). |
| `safety_checker` | Set to `None` to disable the NSFW safety checker (SD 1.x/2.x only; know your deployment's content policy before disabling). |
| `custom_pipeline` | Load a community pipeline (see below). |

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
```

For most application code, use one of the **Auto** classes instead of a
named pipeline class — they inspect the checkpoint and resolve to the right
implementation, so the same call site works across SD 1.5, SDXL, SD3, and
Flux checkpoints:

```python
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

# Convert an already-loaded pipeline to a different task WITHOUT re-downloading/re-loading weights:
img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
```

## Task pipelines

### Text-to-image

```python
image = pipe(prompt="a serene mountain lake at sunrise", num_inference_steps=25, guidance_scale=7.5).images[0]
```

### Image-to-image

```python
from diffusers.utils import load_image

init_image = load_image("https://example.com/input.png")
image = img2img_pipe(prompt="turn it into a watercolor painting", image=init_image, strength=0.6).images[0]
```

`strength` (0-1) controls how much of the source image is overwritten —
1.0 ignores the source almost entirely, low values (0.2-0.4) stay close to
it. Effective `num_inference_steps` for img2img is roughly
`strength * num_inference_steps`.

### Inpainting

```python
image = pipe(
    prompt="a red sports car",
    image=init_image,
    mask_image=mask_image,     # white = regenerate, black = keep
    padding_mask_crop=32,      # crop+upscale around the mask for sharper detail, then paste back
).images[0]
```

### ControlNet

Adds a spatial conditioning signal (edges, depth, pose, ...) on top of a
base pipeline:

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

image = pipe(prompt="a photo of a room", image=canny_edge_image, controlnet_conditioning_scale=1.0).images[0]
```

`StableDiffusionXLControlNetPipeline` is the SDXL equivalent. Pass a list of
`ControlNetModel`s and a matching list of conditioning images/scales for
multi-ControlNet. `controlnet_conditioning_scale` trades adherence to the
control image against prompt adherence.

### IP-Adapter (image prompting)

```python
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.6)
image = pipe(prompt="best quality", ip_adapter_image=style_reference_image).images[0]
```

## Generation kwargs (apply across most task pipelines)

| Kwarg | Purpose |
|---|---|
| `negative_prompt` | Concepts to steer away from. |
| `num_inference_steps` | Denoising steps — quality/speed tradeoff, tune jointly with the scheduler (see [MODELS-AND-SCHEDULERS.md](MODELS-AND-SCHEDULERS.md)). |
| `guidance_scale` | Classifier-free guidance strength; `1.0` disables CFG (and roughly halves compute since the unconditional pass is skipped). Distilled models (SD-Turbo, some Flux variants) expect low/`1.0` guidance — check the model card. |
| `height` / `width` | Must be multiples of 8 (SD) or the model's VAE downsample factor; ignoring the checkpoint's native training resolution degrades quality. |
| `num_images_per_prompt` | Batches within one call; memory grows linearly — prefer this over a Python loop when VRAM allows. |
| `generator` | `torch.Generator(device).manual_seed(seed)` for reproducibility. A CPU generator and a CUDA generator produce **different** samples for the same seed — pick one and keep it consistent across runs you want to compare. |
| `latents` | Pass pre-built starting noise instead of letting the pipeline sample it — needed to reproduce a specific generation exactly, or to share the same starting point across prompts. |
| `output_type` | `"pil"` (default), `"np"`, or `"latent"` (skip VAE decode, useful when chaining pipelines on latents). |
| `callback_on_step_end` | Function called after each step with `(pipe, step, timestep, callback_kwargs)`; return a dict to mutate values (e.g. dynamically change `guidance_scale`) mid-generation. Replaces the deprecated `callback`/`callback_steps`. |

## Community / custom pipelines

```python
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", custom_pipeline="lpw_stable_diffusion"
)
```

Loads community-contributed pipeline code from the `diffusers` GitHub repo
or a Hub repo's own Python file — treat `custom_pipeline` like
`trust_remote_code` in `transformers`: only use it with sources you trust,
since it executes arbitrary Python.

## Saving and sharing

```python
pipe.save_pretrained("my-local-pipeline")
pipe.push_to_hub("my-username/my-pipeline")
```
