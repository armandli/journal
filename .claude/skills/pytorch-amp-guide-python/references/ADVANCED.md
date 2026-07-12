# Advanced torch.amp Patterns

## Multiple models, losses, and optimizers

One `GradScaler` is shared across every model/loss/optimizer in the step —
`scale()` before each relevant `.backward()`, `step()` once per optimizer,
`update()` once at the end. Call `unscale_()` only on the optimizers whose
gradients you actually need to inspect/clip:

```python
scaler = torch.amp.GradScaler("cuda")

for input, target in data:
    optimizer0.zero_grad()
    optimizer1.zero_grad()

    with autocast(device_type="cuda", dtype=torch.float16):
        output0 = model0(input)
        output1 = model1(input)
        loss0 = loss_fn(2 * output0 + 3 * output1, target)
        loss1 = loss_fn(3 * output0 - 5 * output1, target)

    # retain_graph is unrelated to AMP — it's needed here only because
    # loss0 and loss1 share part of the graph.
    scaler.scale(loss0).backward(retain_graph=True)
    scaler.scale(loss1).backward()

    scaler.unscale_(optimizer0)   # only unscale the optimizer(s) you need to inspect

    scaler.step(optimizer0)
    scaler.step(optimizer1)

    scaler.update()               # once, after all step() calls for this iteration
```

## DataParallel (single process, multiple GPUs)

Autocast is **thread-local**, and `DataParallel` replicates the forward pass
across worker threads — so autocast must be entered in the *main* thread
before calling the wrapped model; each replica thread inherits it:

```python
model = MyModel()
dp_model = nn.DataParallel(model)

with autocast(device_type="cuda", dtype=torch.float16):
    output = dp_model(input)     # dp_model's internal threads autocast correctly
    loss = loss_fn(output, target)
```

`DataParallel` is legacy — prefer `DistributedDataParallel` (below) for new
training code; it's covered here only because the thread-local-autocast
caveat is instructive.

## DistributedDataParallel (DDP)

- **One GPU per process (recommended, fastest):** enter `autocast` inside
  each process's training loop exactly as in the single-GPU canonical
  pattern in SKILL.md — no special handling needed, since each process has
  its own Python interpreter and thread-local autocast state.
- **Multiple GPUs per process:** `DistributedDataParallel`, like
  `DataParallel`, spawns per-device threads internally — autocast must be
  set *inside the model's `forward` method* (not just around the call site)
  so every internal replica thread sees it, mirroring the `DataParallel`
  caveat above.
- `GradScaler` state is per-process; DDP does not need special scaler
  synchronization since gradients are already all-reduced (averaged) before
  `optimizer.step()` runs — `scaler.step()`/`update()` operate the same as
  single-GPU.

## Custom autograd Functions

Wrap custom `torch.autograd.Function` forward/backward pairs with
`custom_fwd`/`custom_bwd` so casting behavior is *consistent* between the
forward and backward passes — otherwise the backward pass (which runs
outside of any autocast region by the time autograd invokes it) could see
different dtypes than the forward pass produced.

**Passthrough — let the surrounding autocast region decide the dtype:**

```python
class MyMM(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)

mymm = MyMM.apply
with autocast(device_type="cuda", dtype=torch.float16):
    output = mymm(input1, input2)
```

**Force a specific dtype regardless of the surrounding autocast state** (for
ops that are numerically fragile in reduced precision):

```python
class MyFloat32Func(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ...
        return fwd_output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        ...

func = MyFloat32Func.apply
with autocast(device_type="cuda", dtype=torch.float16):
    output = func(input)   # runs in float32 regardless of the outer autocast dtype
```

`torch.cuda.amp.custom_fwd`/`custom_bwd` (no `device_type` argument) are the
deprecated aliases — use the `torch.amp` versions with an explicit
`device_type` for new code.

## torch.compile interaction

`torch._functorch.config.backward_pass_autocast` defaults to
`"same_as_forward"`. If you follow the recommended pattern of keeping
`.backward()` outside the `autocast` block (as in every example in this
skill), set it to `"off"` for a `torch.compile`'d region to match that
convention exactly and avoid the compiler re-deriving it.

## Anti-pattern found in this repo — mismatched device between autocast and GradScaler

Several existing notebooks (`diffusion/ddpm.py`, `diffusion/ddim.py`,
`diffusion/vqvae1.py`, `diffusion/cfd_ddim.py`, `t2i_gen/image_caption.py`,
`t2i_gen/ldm_diffusion.py`, `t2i_gen/vit_transformer.py`) construct the
scaler on **CPU** while autocasting on **CUDA**:

```python
scaler = torch.cpu.amp.GradScaler()   # wrong device — and a deprecated alias
#scaler = torch.cuda.amp.GradScaler() # commented-out correct-device alternative, also deprecated
...
with torch.amp.autocast("cuda"):
    ...
```

Both lines use the deprecated per-device namespaces, and the active one
doesn't even match the autocast device. Write it as one device-derived line
instead, so the scaler always tracks whatever device the model actually
trains on:

```python
scaler = torch.amp.GradScaler(device_type, enabled=(amp_dtype is torch.float16))
...
with torch.autocast(device_type=device_type, dtype=amp_dtype):
    ...
```

This is the pattern used in this skill's canonical training loop (SKILL.md)
— apply it when writing new training code, and prefer it over copying the
older pattern from these existing notebooks.
