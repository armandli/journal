# torch.autocast Reference

## Signature

```python
class torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)
```

- `device_type` (`str`, required) — `"cuda"`, `"cpu"`, `"xpu"`, `"mps"`, `"mtia"`, `"maia"`, or `"hpu"`. Must match the device the tensors actually live on.
- `dtype` (`torch.dtype`, optional) — defaults to `torch.float16` on CUDA and `torch.bfloat16` on CPU. Always pass it explicitly rather than relying on the default, so the choice is visible in the code.
- `enabled` (`bool`, optional) — set `False` to make the block a no-op without deleting it (see "Locally disabling autocast" below).
- `cache_enabled` (`bool`, optional) — controls autocast's weight-cast cache (see "Weight cache" below); defaults to `True`.

Check support before relying on a device:

```python
torch.amp.autocast_mode.is_autocast_available(device_type)  # -> bool
```

## Usage modes

**Context manager** (most common):

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
```

**Decorator** on a `forward`/function:

```python
class Net(nn.Module):
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward(self, x):
        return self.layers(x)
```

**Locally disabling autocast** — nest a sub-region with `enabled=False` to
force full precision for a numerically sensitive piece (e.g. a custom loss
term), then resume autocasting outside it:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    out = model(x)
    with torch.autocast(device_type="cuda", enabled=False):
        # tensors entering here may still be float16 from the outer region —
        # cast explicitly if the op requires float32 inputs
        stable_term = sensitive_fn(out.float())
    loss = loss_fn(out, target) + stable_term
```

## Key constraint

> autocast should wrap only the forward pass(es) of your network, including
> the loss computation(s). Backward passes under autocast are not
> recommended — backward ops automatically run in the same dtype autocast
> chose for the corresponding forward op, so there's no need (and no
> benefit) to wrap `.backward()` itself.

## Dtype mismatches at the autocast boundary

Tensors produced inside an autocast region may be `float16`/`bfloat16`. If
code outside the region expects `float32`, cast explicitly:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    y = model(x)
y = y.float()   # no-op (and no overhead) if y is already float32
```

## Weight cache

Autocast maintains a per-step cache of casted weights (e.g. a `Linear`
layer's `float32` weight cast once to `float16`) so repeated calls to the
same layer within one autocast region/step don't repeat the cast. This is
transparent and rarely needs tuning — set `cache_enabled=False` only if
you're mutating parameters in-place *inside* an autocast region between
uses (rare; e.g. some meta-learning / hypernetwork patterns), since a stale
cached cast would otherwise be reused.

## Autocast is thread-local

Autocast state does not propagate automatically across threads. This
matters for `torch.nn.DataParallel` (spawns worker threads per replica) —
see `references/ADVANCED.md` for the required pattern.

## Ineligible ops (never autocast, regardless of dtype table below)

- In-place variants of ops (e.g. `addmm_` — the `_`-suffixed form)
- Ops called with an explicit `out=` tensor argument
- Ops called with an explicit `dtype=` argument
- `float64` and non-floating-point tensors

## Special case: BCE loss

`binary_cross_entropy` / `nn.BCELoss` **raise an error** inside an
autocast-enabled region (the naive sigmoid+log combination is numerically
unstable in reduced precision). Use `binary_cross_entropy_with_logits` /
`nn.BCEWithLogitsLoss` instead — they fuse the sigmoid and are autocast-safe.

## CUDA op-eligibility tables

**Autocast to `float16`** (the ops that benefit most from Tensor Cores):

- Matrix ops: `__matmul__`, `addbmm`, `addmm`, `addmv`, `addr`, `baddbmm`, `bmm`, `chain_matmul`, `linalg_multi_dot`, `matmul`, `mm`, `mv`
- Convolutions: `conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`
- RNN cells: `GRUCell`, `LSTMCell`, `RNNCell`
- Other: `linear`, `prelu`

**Kept in `float32`** (numerically sensitive — reductions, norms, losses, transcendentals):

- Reductions: `sum`, `prod`, `cumprod`, `cumsum`, `norm`
- Normalization / attention-adjacent: `softmax`, `log_softmax`, `layer_norm`, `group_norm`
- Losses: `cross_entropy`, `nll_loss`, `mse_loss`, `l1_loss`, `kl_div`, `binary_cross_entropy_with_logits`
- Transcendentals: `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`

**Promoted to the widest input dtype** (multi-input ops where mixed dtypes would be ambiguous):

- `addcdiv`, `addcmul`, `atan2`, `bilinear`, `cross`, `dot`, `grid_sample`, `index_put`, `scatter_add`, `tensordot`

## CPU op-eligibility tables

**Autocast to `bfloat16`:**

- `conv1d`, `conv2d`, `conv3d`, `linear`, `matmul`, `mm`, `bmm`, `baddbmm`, `addmm`, `addbmm`
- Attention: `scaled_dot_product_attention`, `_native_multi_head_attention`
- RNN: `mkldnn_rnn_layer`

**Kept in `float32`:**

- FFT family: `fft_fft`, `fft_ifft`, `fft_fft2`, `fft_ifft2`, etc.
- Linear algebra: `linalg_solve`, `linalg_cholesky`, `svd`, `qr`
- 3D pooling: `avg_pool3d`, `max_pool3d`, `adaptive_avg_pool3d`, `adaptive_max_pool3d`
- 40+ loss variants (cross-entropy family, embedding losses, distance metrics)
- Reflection/replication padding ops

**Promoted to the widest input dtype:** `cat`, `stack`, `index_copy`

## Other devices

- **XPU** — experimental; op coverage largely mirrors the CUDA table minus some advanced ops.
- **MPS (Apple Silicon)** — autocast support has evolved across torch releases and is not guaranteed in every version; call `torch.amp.autocast_mode.is_autocast_available("mps")` before relying on it rather than assuming parity with CUDA. `GradScaler` support on MPS is likewise less mature than on CUDA/CPU — prefer `bfloat16` autocast on MPS (no `GradScaler` needed) over `float16`.
- **MTIA / MAIA / HPU** — vendor-specific accelerators; supported but out of scope for this repo.
