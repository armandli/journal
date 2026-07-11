---
name: huggingface-diffusers-guide-python
description: Write and debug Python code using the Hugging Face `diffusers` library — loading and running pretrained diffusion pipelines (text-to-image, image-to-image, inpainting, ControlNet, IP-Adapter), working with the underlying models (UNet, VAE, Transformer/DiT) and schedulers, training/using LoRA adapters (PEFT-based LoRA fine-tuning, DreamBooth, textual inversion), and memory/speed optimization (CPU offload, attention backends, torch.compile). Use when the user asks to "generate an image with diffusers", "use a diffusion pipeline", "load a stable diffusion / SD3 / Flux model", "train a LoRA for diffusion", "fine-tune with diffusers", or writes code importing `diffusers`. Do NOT use for the `transformers` model/tokenizer library alone, or for `datasets`/`peft`/`accelerate` used without `diffusers`.
argument-hint: "[task or description of what to implement]"
---

# Hugging Face Diffusers Python Guide

**New model families (SD3, Flux, and others) land in `diffusers` every few
months, often with new pipeline/scheduler/model classes.** Verify class
names against the installed version before trusting older tutorials or
training data:
```bash
python -c "import diffusers; print(diffusers.__version__)"
```
When the checkpoint's architecture isn't fixed in advance, prefer the
`AutoPipelineFor*` classes over hardcoding a specific pipeline class — they
resolve to the right implementation for the checkpoint automatically.

## Core concept: three building blocks

Every diffusion pipeline composes three pieces:

| Piece | Role |
|---|---|
| Model (`UNet2DModel` / `UNet2DConditionModel` / `*Transformer2DModel`) | Predicts the noise (or velocity/flow) to remove at each step. |
| Scheduler | Defines the noise schedule and the `step()` update rule (DDPM, DDIM, Euler, DPM-Solver, FlowMatch, ...). |
| (optional) VAE + text encoder(s) | Compress images to/from latent space; encode text conditioning. |

`DiffusionPipeline` (and its `AutoPipelineFor*` subclasses) wire these
together behind one callable.

## Loading & running a pipeline

```python
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

generator = torch.Generator("cuda").manual_seed(0)   # always seed for reproducibility
image = pipe(
    prompt="a photo of an astronaut riding a horse on mars",
    negative_prompt="blurry, low quality",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
).images[0]
image.save("astronaut.png")
```

`AutoPipelineForText2Image` / `ForImage2Image` / `ForInpainting` infer the
correct pipeline class from the checkpoint's `model_index.json` — prefer
these over a hardcoded `StableDiffusionPipeline` / `StableDiffusionXLPipeline`
/ etc. unless you need a capability the Auto classes don't cover
(ControlNet, IP-Adapter). `AutoPipelineFor*.from_pipe(existing_pipe)`
converts an already-loaded pipeline to a different task without re-loading
weights. Full pipeline catalogue (img2img `strength`, inpainting
`mask_image`, ControlNet, IP-Adapter), generation kwargs, callbacks, and
saving/hub push:
[references/PIPELINES-AND-INFERENCE.md](references/PIPELINES-AND-INFERENCE.md).

## Models & schedulers

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

Swapping the scheduler on an existing pipeline is the single
highest-leverage speed/quality knob available without retraining — most
SD-family checkpoints trained under DDPM sample fine (and 2-4x faster) with
DPM-Solver++ or UniPC at 20-25 steps instead of 50+. Always swap via
`.from_config(pipe.scheduler.config)` so checkpoint-specific settings
(`prediction_type`, `trained_betas`, ...) carry over. Core model classes
(`UNet2DModel`, `UNet2DConditionModel`, DiT-style `*Transformer2DModel` for
SD3/Flux, `AutoencoderKL`/`AutoencoderTiny`, `ControlNetModel`), the full
scheduler catalogue with speed/quality tradeoffs, and a manual denoising
loop (model + scheduler with no pipeline, for understanding what the
pipeline hides):
[references/MODELS-AND-SCHEDULERS.md](references/MODELS-AND-SCHEDULERS.md).

## LoRA: loading, fusing, training

```python
pipe.load_lora_weights(
    "path/or/hub-id", weight_name="pytorch_lora_weights.safetensors", adapter_name="pixel-art"
)
pipe.set_adapters(["pixel-art"], adapter_weights=[0.8])
image = pipe("a pixel art castle").images[0]
pipe.unload_lora_weights()
```

Training a new LoRA is most reliably done via the official `accelerate
launch examples/text_to_image/train_text_to_image_lora.py` scripts, or
manually by attaching a PEFT `LoraConfig` with `unet.add_adapter(...)` and
training only the LoRA params. Multi-adapter composition
(`set_adapters`/weights), permanent merging (`fuse_lora`/`unfuse_lora`),
DreamBooth (instance/class prompts, prior preservation), and textual
inversion:
[references/LORA-AND-FINETUNING.md](references/LORA-AND-FINETUNING.md).

## Memory & speed optimization

```python
pipe.enable_model_cpu_offload()      # keep weights on CPU, move submodules to GPU only when needed
pipe.enable_vae_slicing()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

Choosing between `enable_model_cpu_offload()` (fits large pipelines on
small GPUs, minor slowdown), `enable_sequential_cpu_offload()` (fits on
almost nothing, large slowdown), and plain `.to("cuda")` (fastest, needs
full VRAM); attention backends (SDPA is used automatically on torch>=2.0,
`enable_xformers_memory_efficient_attention()` for older stacks);
`torch.compile` pitfalls (recompiles on shape change); dtype/`variant`
selection; and quantizing large text encoders/transformers (T5, Flux) with
`diffusers.BitsAndBytesConfig`:
[references/OPTIMIZATION.md](references/OPTIMIZATION.md).

## Testing

After wiring pipeline or training code, actually run generation against a
small/fast checkpoint and inspect the output — a shape, a saved image, or a
histogram of pixel values. Silent failures here are usually black or NaN
images (dtype/device mismatch, e.g. fp16 on CPU), a resolution mismatch
(ignoring the checkpoint's native/trained resolution), or a LoRA that
doesn't visibly change output (wrong `adapter_name`, wrong `target_modules`
at train time, or `adapter_weights` left at 0). For fast smoke tests prefer
a tiny checkpoint (e.g. `hf-internal-testing/tiny-stable-diffusion-pipe`,
`segmind/small-sd`) with few inference steps
(`num_inference_steps=4-8` with a compatible fast scheduler) before pointing
at a full-size model.

## References

- [references/PIPELINES-AND-INFERENCE.md](references/PIPELINES-AND-INFERENCE.md) — `DiffusionPipeline`/`AutoPipelineFor*`, text2img/img2img/inpaint, ControlNet, IP-Adapter, generation kwargs, callbacks, saving/hub
- [references/MODELS-AND-SCHEDULERS.md](references/MODELS-AND-SCHEDULERS.md) — core model classes, scheduler catalogue and swapping, manual denoising loop
- [references/LORA-AND-FINETUNING.md](references/LORA-AND-FINETUNING.md) — LoRA loading/fusing/multi-adapter, LoRA/DreamBooth/textual-inversion training
- [references/OPTIMIZATION.md](references/OPTIMIZATION.md) — offloading, attention backends, `torch.compile`, dtype/variant, quantization

## External Docs

- Full docs: https://huggingface.co/docs/diffusers/index

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "huggingface-diffusers-guide-python"
```
