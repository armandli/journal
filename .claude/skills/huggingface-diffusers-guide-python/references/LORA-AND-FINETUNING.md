# LoRA and Fine-Tuning

## Loading and using a trained LoRA

```python
pipe.load_lora_weights(
    "path/or/hub-id",
    weight_name="pytorch_lora_weights.safetensors",   # needed when a repo has more than one LoRA file
    adapter_name="pixel-art",
)
image = pipe("a pixel art castle").images[0]
```

### Multiple adapters

```python
pipe.load_lora_weights("hub-id-1", adapter_name="style")
pipe.load_lora_weights("hub-id-2", adapter_name="subject")
pipe.set_adapters(["style", "subject"], adapter_weights=[0.7, 1.0])   # compose both

pipe.disable_lora()   # temporarily fall back to the base model
pipe.enable_lora()
```

### Fusing (permanent merge) vs unloading

```python
pipe.fuse_lora(lora_scale=0.7)   # bakes LoRA weights into the base model weights — faster inference, but no longer swappable
pipe.unfuse_lora()               # reverse fuse_lora, if the unfused weights were kept

pipe.unload_lora_weights()       # drop LoRA weights entirely, restore original base model behavior
```

Fuse when shipping a fixed style/subject combination and inference speed
matters (no LoRA-layer overhead per forward pass). Keep unfused while still
experimenting with adapter weights or swapping adapters at runtime.

## Training a LoRA

### Official training scripts (recommended starting point)

`diffusers` ships maintained training scripts under `examples/` in its
GitHub repo, run via `accelerate launch`:

```bash
accelerate launch examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="path/to/images_and_captions" \
  --resolution=512 \
  --train_batch_size=1 \
  --rank=4 \
  --learning_rate=1e-4 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --validation_prompt="a photo of a <token> style painting" \
  --output_dir="lora-output"
```

Key args across these scripts: `--rank` (LoRA rank — higher captures more
detail, larger file, more overfitting risk on small datasets),
`--learning_rate` (LoRA tolerates much higher LR than full fine-tuning,
typically 1e-4 to 1e-5), `--validation_prompt` (periodic sample generation
during training so you can catch a collapsed/broken run early instead of
discovering it after `--max_train_steps`).
`train_text_to_image_lora_sdxl.py` is the SDXL variant (two text encoders).

### Manual LoRA attachment (when you need a custom training loop)

```python
from peft import LoraConfig

unet_lora_config = LoraConfig(
    r=4, lora_alpha=4, init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],   # the UNet's attention projection layers
)
unet.add_adapter(unet_lora_config)
unet.requires_grad_(False)
for name, param in unet.named_parameters():
    if "lora" in name:
        param.requires_grad_(True)

# ... standard diffusion training loop: sample noise, add to latents, predict, compute loss, backward ...

from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import StableDiffusionPipeline

lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
StableDiffusionPipeline.save_lora_weights(save_directory="output", unet_lora_layers=lora_state_dict)
```

This is what the official scripts do internally — reach for it when the
existing scripts don't cover your data pipeline or loss, not as the default
starting point.

## DreamBooth

Fine-tunes on a handful (3-20) of images of a specific subject, bound to a
rare token in the prompt:

```bash
accelerate launch examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="path/to/subject_images" \
  --instance_prompt="a photo of sks dog" \
  --class_data_dir="path/to/class_images" \
  --class_prompt="a photo of a dog" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=512 --train_batch_size=1 \
  --learning_rate=1e-4 --max_train_steps=800
```

`--with_prior_preservation` regularizes against catastrophic forgetting of
the general class ("dog") while learning the specific instance — omitting
it on small datasets tends to collapse the whole class concept into the
one trained subject. `train_dreambooth.py` (no `_lora` suffix) does full
UNet fine-tuning instead — much higher VRAM and disk cost, rarely needed
over the LoRA variant.

## Textual Inversion

Learns a new embedding for a special token without touching any model
weights — smallest possible artifact (a few KB), most limited expressivity:

```python
pipe.load_textual_inversion("sd-concepts-library/cat-toy", token="<cat-toy>")
image = pipe("a photo of <cat-toy> on a beach").images[0]
```

Train via `examples/textual_inversion/textual_inversion.py`, similarly run
under `accelerate launch`.
