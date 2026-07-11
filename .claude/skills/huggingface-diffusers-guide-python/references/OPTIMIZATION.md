# Memory and Speed Optimization

## Device placement / offloading

| Approach | VRAM needed | Speed | When |
|---|---|---|---|
| `pipe.to("cuda")` | Full pipeline resident | Fastest | Default when the pipeline fits in VRAM. |
| `pipe.enable_model_cpu_offload()` | One submodule's worth at a time | Small slowdown | Large pipelines (SDXL, SD3, Flux) on consumer GPUs. |
| `pipe.enable_sequential_cpu_offload()` | Minimal (near-zero) | Large slowdown | Barely-fits-at-all situations; last resort before giving up GPU inference. |

`enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` manage
device placement internally — **do not** also call `.to("cuda")` on a
pipeline with offloading enabled; let the offload hook move submodules as
needed.

## Attention

On `torch>=2.0`, `diffusers` uses PyTorch's built-in
`scaled_dot_product_attention` (SDPA) automatically — no action needed, and
it's typically as fast as xformers. For older stacks, or to force xformers
explicitly:

```python
pipe.enable_xformers_memory_efficient_attention()
```

`pipe.enable_attention_slicing("auto")` (or an int) trades a little speed
for lower peak memory by computing attention in chunks — useful on GPUs
with very limited VRAM (e.g. <=4GB) when offloading alone isn't enough.

## VAE memory

```python
pipe.enable_vae_slicing()   # decode one image at a time within a batch, instead of the whole batch at once
pipe.enable_vae_tiling()    # decode/encode large images in tiles — needed for high-res (>1024px) SDXL/img2img on limited VRAM
```

## `torch.compile`

```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# also worth compiling for full-pipeline speedup:
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
```

The first call after compiling pays a one-time (often 30s+) compilation
cost. Keep `height`/`width`/`batch_size` constant across calls — a shape
change triggers recompilation, which can make a naively-compiled pipeline
slower than an uncompiled one in workloads with varying input shapes.

## dtype and weight variants

```python
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,   # or torch.bfloat16 on hardware that supports it well (e.g. Ampere+/Apple Silicon)
    variant="fp16",              # fetch the repo's pre-cast fp16 weight files directly, instead of downloading fp32 and casting
    use_safetensors=True,
)
```

fp16 halves memory and roughly doubles throughput over fp32 on supported
GPUs with negligible quality loss for inference. `bfloat16` avoids fp16's
occasional overflow/NaN issues at a small memory cost on hardware with
native bf16 support.

## Quantization for large components

Newer, larger model families (Flux's transformer, SD3/Flux's T5 text
encoder) benefit from quantizing just the heaviest component rather than
the whole pipeline:

```python
from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", subfolder="transformer",
    quantization_config=quant_config, torch_dtype=torch.bfloat16,
)
```

Pass the quantized component into the pipeline constructor alongside the
other (non-quantized) parts. Combine with `enable_model_cpu_offload()` for
the biggest memory savings on consumer hardware.

## Batching

Prefer `num_images_per_prompt=N` over a Python loop calling the pipeline N
times — it batches the forward passes. Memory grows roughly linearly with
N, so on tight VRAM budgets a loop (or a small batch size with offloading)
may be the only option.
