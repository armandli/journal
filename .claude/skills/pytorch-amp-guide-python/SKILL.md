---
name: pytorch-amp-guide-python
description: Write, debug, and default to using PyTorch's torch.amp module (autocast + GradScaler) for mixed-precision model training. Covers the unified torch.amp API (device-generic autocast/GradScaler, replacing the deprecated torch.cuda.amp/torch.cpu.amp namespaces), float16 vs bfloat16 selection, CUDA/CPU op-eligibility tables, gradient clipping/accumulation/penalty under scaling, multiple models/losses/optimizers, DataParallel/DistributedDataParallel, and custom autograd Functions (custom_fwd/custom_bwd). Use automatically whenever code touches torch.amp, torch.autocast, torch.cuda.amp, torch.cpu.amp, GradScaler, or "mixed precision"/"fp16 training"/"bf16 training" — and by default whenever writing or editing any PyTorch training loop in this repo, since mixed precision is the default training mode here unless the user explicitly says otherwise. Do NOT use for MLX training (MLX has no autocast/GradScaler equivalent) or for non-training PyTorch code with no forward/backward pass.
argument-hint: "[task or description of what to implement]"
---

# PyTorch AMP (Automatic Mixed Precision) Guide

Reference for `torch.amp` — the unified, device-generic mixed-precision API.
`torch.cuda.amp.*` and `torch.cpu.amp.*` are **deprecated aliases**; always
import from `torch.amp` and pass an explicit `device_type`/`device`.

## Default behavior in this repo

Mixed precision is the default for every PyTorch training loop you write or
edit in this project — do not ask permission each time. Wrap the forward
pass + loss in `torch.autocast`, scale losses with `torch.amp.GradScaler`
when using `float16`, and only omit AMP when:
- the user explicitly asks for full-precision (fp32) training, or
- the target device has no autocast support in the installed torch version
  (verify with `torch.amp.autocast_mode.is_autocast_available(device_type)`
  rather than assuming — this matters most for `"mps"`, whose autocast/
  GradScaler support has evolved across torch releases).

## Imports

```python
import torch
from torch import autocast
from torch.amp import GradScaler
```

(`torch.autocast` and `torch.amp.autocast` are the same class; both
spellings appear in code and docs.)

## Canonical training loop

```python
device_type = "cuda"          # "cuda" | "cpu" | "xpu" | "mps" (see AUTOCAST.md)
amp_dtype = torch.float16     # float16 needs GradScaler; bfloat16 does not (see below)

model = Net().to(device_type)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler(device_type, enabled=(amp_dtype is torch.float16))

model.train()
for epoch in range(num_epochs):
    for x, y in train_loader:
        x, y = x.to(device_type), y.to(device_type)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, dtype=amp_dtype):
            output = model(x)
            loss = loss_fn(output, y)

        # Backward passes under autocast are not recommended — backward
        # ops run in whatever dtype the matching forward op used.
        scaler.scale(loss).backward()
        scaler.step(optimizer)   # unscales grads, skips step() on inf/NaN
        scaler.update()
```

## float16 vs bfloat16

| | Needs `GradScaler` | Exponent range | Mantissa | Typical use |
|---|---|---|---|---|
| `torch.float16` | **Yes** — narrow range under/overflows easily | ±65504 | 10-bit | CUDA (Volta+), best throughput |
| `torch.bfloat16` | **No** — same exponent range as float32 | ±3.4e38 | 7-bit | CPU default, Ampere+ CUDA, models pretrained in bf16 |

Casting a model that was *pretrained* in bfloat16 down to float16 can
overflow gradients instead of underflowing — prefer bfloat16 autocast for
such models (skip `GradScaler`, or construct it with `enabled=False`).

## Rules that are easy to get wrong

- `autocast` should wrap only the forward pass + loss computation —
  **never** wrap `.backward()`.
- `GradScaler` only matters for `float16`. For `bfloat16`, either don't
  create one, or pass `enabled=False` so `.scale()`/`.step()`/`.update()`
  become no-ops and you can keep one code path for both dtypes.
- `binary_cross_entropy` / `BCELoss` **raise an error** inside autocast —
  use `binary_cross_entropy_with_logits` / `BCEWithLogitsLoss` instead.
- In-place ops (`addmm_`), ops called with an explicit `out=` tensor, and
  ops called with an explicit `dtype=` argument never autocast.
- `scaler.unscale_(optimizer)` must be called **at most once** per
  optimizer per step, and only after every `.backward()` contributing to
  that optimizer's params has already run.

## References

- [references/AUTOCAST.md](references/AUTOCAST.md) — full `autocast` API, device support, CUDA/CPU op-eligibility tables, nested/local-disable patterns
- [references/GRADSCALER.md](references/GRADSCALER.md) — full `GradScaler` API, gradient clipping/accumulation/penalty, checkpointing scaler state
- [references/ADVANCED.md](references/ADVANCED.md) — multiple models/losses/optimizers, DataParallel/DDP, custom autograd `Function`s with `custom_fwd`/`custom_bwd`, and an AMP anti-pattern found across this repo's own diffusion/t2i_gen notebooks

## External Docs

- API reference: https://docs.pytorch.org/docs/2.13/amp.html
- Worked examples: https://docs.pytorch.org/docs/2.13/notes/amp_examples.html

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "pytorch-amp-guide-python"
```
