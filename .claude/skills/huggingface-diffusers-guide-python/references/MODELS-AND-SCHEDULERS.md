# Models and Schedulers

## Core model classes

Pipelines are thin orchestration around these building blocks — all
loadable standalone via `from_pretrained(repo_id, subfolder="unet")` etc.
when you need direct access (custom training loops, mixing components from
different checkpoints).

| Class | Role |
|---|---|
| `UNet2DModel` | Unconditional or class-conditioned UNet, e.g. basic DDPM/DDIM image generation. |
| `UNet2DConditionModel` | Text/embedding-conditioned UNet — the denoiser in SD 1.x/2.x/XL. Cross-attention layers attend to text encoder hidden states. |
| `Transformer2DModel` / `SD3Transformer2DModel` / `FluxTransformer2DModel` / `DiTTransformer2DModel` | DiT-style (Diffusion Transformer) denoisers used by SD3, Flux, and other newer model families — replace the UNet's convolutional backbone with transformer blocks operating on patchified latents. |
| `AutoencoderKL` | The VAE — encodes images to a compressed latent space and decodes latents back to pixels. Nearly every modern pipeline (SD, SDXL, SD3, Flux) operates in this VAE's latent space, not pixel space. |
| `AutoencoderTiny` (TAESD) | A much smaller, faster, lower-quality VAE — useful for fast local previews during iterative prompt tuning, not final output. |
| `ControlNetModel` | A trainable copy of (part of) a UNet's encoder that injects a spatial conditioning signal; paired with a base UNet via a ControlNet pipeline. |

## Manual denoising loop

Useful for understanding what a pipeline hides, or when you need control a
pipeline doesn't expose (custom guidance, hybrid samplers):

```python
import torch
from diffusers import UNet2DModel, DDPMScheduler

model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
scheduler.set_timesteps(50)   # fewer steps than the model was trained with (typically 1000) still works reasonably

sample = torch.randn(1, 3, 256, 256).to("cuda")
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(sample, t).sample
    sample = scheduler.step(noise_pred, t, sample).prev_sample

image = (sample / 2 + 0.5).clamp(0, 1)
```

The pattern generalizes: `scheduler.set_timesteps(n)` builds the step
schedule, `model(...)` predicts noise/velocity/flow for the current
timestep, `scheduler.step(prediction, t, sample)` applies the update rule
and returns `.prev_sample`.

## Scheduler catalogue

All schedulers implement the same `set_timesteps()` / `step()` interface,
so they're interchangeable on a given pipeline via
`pipe.scheduler = SomeScheduler.from_config(pipe.scheduler.config)` —
**always** swap through `.from_config` (not a fresh constructor call) so
checkpoint-specific settings (`prediction_type`, `beta_schedule`,
`trained_betas`, `timestep_spacing`) carry over correctly.

| Scheduler | Typical steps | Notes |
|---|---|---|
| `DDPMScheduler` | 1000 | Original formulation; slow but the reference/training-time schedule for most SD-family checkpoints. |
| `DDIMScheduler` | 50 | Deterministic (given a seed) subsampling of the DDPM chain; long-standing default for SD 1.x/2.x pipelines. |
| `PNDMScheduler` | 50 | Pseudo-numerical multistep method; was the original SD 1.x default. |
| `LMSDiscreteScheduler` | 20-50 | Linear multistep; solid general-purpose quality. |
| `EulerDiscreteScheduler` | 20-30 | Fast, simple, good default for many checkpoints. |
| `EulerAncestralDiscreteScheduler` | 20-30 | Euler + stochastic noise injection each step — more varied outputs, doesn't fully converge to a single image as steps increase. |
| `HeunDiscreteScheduler` | 20-30 | Higher-order (2nd order), more accurate per step at ~2x the model calls. |
| `KDPM2DiscreteScheduler` / `KDPM2AncestralDiscreteScheduler` | 20-30 | k-diffusion DPM-2 variants. |
| `DPMSolverMultistepScheduler` | 15-25 | Very common recommended default — good quality at low step counts. |
| `DPMSolverSinglestepScheduler` | 15-25 | Single-step variant of the same solver family. |
| `UniPCMultistepScheduler` | 15-20 | Often the fastest-converging option; strong choice when minimizing steps matters most. |
| `FlowMatchEulerDiscreteScheduler` | model-dependent (often 4-50) | Used by flow-matching model families (SD3, Flux) instead of the DDPM-style noise schedule above — not a drop-in swap for UNet-based checkpoints. |

For SD/SDXL-family (UNet-based, DDPM-trained) checkpoints,
`DPMSolverMultistepScheduler` or `UniPCMultistepScheduler` at 20-25 steps is
a strong default: comparable quality to 50-step DDIM at roughly half the
compute. Flow-matching models (SD3, Flux) ship with
`FlowMatchEulerDiscreteScheduler` already configured correctly — don't swap
it for a DDPM-family scheduler.
