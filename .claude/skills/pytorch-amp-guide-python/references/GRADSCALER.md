# torch.amp.GradScaler Reference

## Why it exists

`float16` has a 10-bit mantissa and a narrow exponent range. Small gradient
magnitudes that are representable in `float32` can flush to zero
("underflow") in `float16`, silently dropping the update for the
corresponding parameters. `GradScaler` multiplies the loss by a large factor
before `.backward()` so gradients land in `float16`'s representable range,
then unscales them before the optimizer step. (`bfloat16` shares `float32`'s
exponent range and does not need this — see SKILL.md.)

## Signature

```python
class torch.amp.GradScaler(
    device="cuda",
    init_scale=2.0 ** 16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True,
)
```

- `device` — `"cuda"` or `"cpu"` (must match the device your model/optimizer run on — this is the parameter that replaces the deprecated `torch.cuda.amp.GradScaler()` / `torch.cpu.amp.GradScaler()` split).
- `init_scale` — starting scale factor.
- `growth_factor` — multiplier applied to the scale after `growth_interval` consecutive steps with no inf/NaN.
- `backoff_factor` — multiplier applied to the scale (i.e. scale shrinks) the moment an inf/NaN is seen; that step's optimizer update is skipped.
- `growth_interval` — number of consecutive good steps required before growing the scale again.
- `enabled` — set `False` to make every method below a no-op. Use this to keep one code path for `float16` (scaler active) and `bfloat16`/full-precision (scaler disabled) training.

## Core methods (per training step)

```python
scaler.scale(loss)          # -> Tensor: loss * current_scale. Call .backward() on the result.
scaler.step(optimizer)      # unscales optimizer's grads in-place, then calls optimizer.step()
                             # UNLESS any grad is inf/NaN, in which case the step is skipped.
                             # Returns optimizer.step()'s return value, or None if skipped.
scaler.update()             # adjusts the scale for the next iteration (grow/backoff);
                             # call exactly once per iteration, after all scaler.step() calls.
```

## unscale_ — for gradient inspection/clipping between backward and step

```python
scaler.unscale_(optimizer)   # unscales this optimizer's grads in-place, once
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # now operates on real-magnitude grads
scaler.step(optimizer)       # sees already-unscaled grads; still checks for inf/NaN before stepping
scaler.update()
```

Constraint: `unscale_(optimizer)` may be called **at most once** per
optimizer per step, and only after every `.backward()` contributing to that
optimizer's parameters has already run.

## Introspection / state methods

```python
scaler.get_scale() -> float          # current scale factor
scaler.get_growth_factor() -> float
scaler.set_growth_factor(new_factor)
scaler.get_backoff_factor() -> float
scaler.set_backoff_factor(new_factor)
scaler.get_growth_interval() -> int
scaler.set_growth_interval(new_interval)
scaler.is_enabled() -> bool
```

## Checkpointing scaler state

Persist the scaler alongside the model/optimizer so resumed training doesn't
restart the scale-growth schedule from `init_scale`:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
scaler.load_state_dict(ckpt["scaler"])
```

## Gradient accumulation

Scale the loss by `1/iters_to_accumulate` before backward so accumulated
(summed) scaled gradients match what a single large-batch step would have
produced; only step/update every `iters_to_accumulate` iterations:

```python
scaler = GradScaler("cuda")
iters_to_accumulate = 4

for i, (x, y) in enumerate(loader):
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(x), y) / iters_to_accumulate

    scaler.scale(loss).backward()   # accumulates scaled gradients

    if (i + 1) % iters_to_accumulate == 0:
        # scaler.unscale_(optimizer) here first if you want to clip
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

## Gradient penalty (double backward) under scaling

When a loss term itself depends on gradients (e.g. WGAN-GP style penalties),
scale via `torch.autograd.grad` directly and unscale by dividing, since
those gradients aren't owned by any optimizer (`unscale_` doesn't apply):

```python
scaler = GradScaler("cuda")

with autocast(device_type="cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)

# Scale via autograd.grad's backward pass to get scaled_grad_params
scaled_grad_params = torch.autograd.grad(
    outputs=scaler.scale(loss), inputs=model.parameters(), create_graph=True
)

# scaled_grad_params aren't optimizer-owned, so divide manually instead of unscale_
inv_scale = 1.0 / scaler.get_scale()
grad_params = [p * inv_scale for p in scaled_grad_params]

with autocast(device_type="cuda", dtype=torch.float16):
    grad_norm = sum(g.pow(2).sum() for g in grad_params).sqrt()
    loss = loss + grad_norm   # add the penalty to the primary loss

scaler.scale(loss).backward()   # accumulates correctly scaled leaf gradients
# scaler.unscale_(optimizer) here first if you want to clip
scaler.step(optimizer)
scaler.update()
```
