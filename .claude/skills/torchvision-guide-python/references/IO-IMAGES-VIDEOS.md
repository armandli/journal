# torchvision.io — Decoding & Encoding Images and Video

## Image decoding

Torchvision decodes JPEG, PNG, WEBP, GIF, AVIF, and HEIC directly into
tensors — no PIL round-trip needed. JPEG decoding also works on CUDA GPUs.

```python
from torchvision.io import decode_image

img = decode_image("path_to_image", mode="RGB")   # torch.uint8 tensor, CHW
img.dtype   # torch.uint8

# Or decode from raw bytes already in memory
raw_encoded_bytes = ...
img = decode_image(raw_encoded_bytes, mode="RGB")
```

`decode_image()` is the recommended entry point — it auto-detects format and
dispatches to the right decoder (except AVIF/HEIC, which need their
dedicated functions below). Use the format-specific decoders directly when
you need extra control (e.g. CUDA JPEG decoding):

| Function | Signature | Notes |
|---|---|---|
| `decode_image` | `(input, mode=..., ...)` | Auto-detects format; path or raw bytes. |
| `decode_jpeg` | `(input, mode=..., device=..., ...)` | CPU or CUDA; pass `device="cuda"` for GPU decode. |
| `decode_png` | `(input, mode=..., apply_exif_orientation=...)` | |
| `decode_webp` | `(input, mode=...)` | |
| `decode_avif` | `(input, mode=...)` | |
| `decode_heic` | `(input, mode=...)` | |
| `decode_gif` | `(input)` | Returns a 3D or 4D (multi-frame) RGB tensor. |
| `ImageReadMode` | enum | Controls auto-conversion to RGB/RGBA/grayscale during decode. |

`read_image(path, mode=..., apply_exif_orientation=...)` still exists but is
**obsolete — use `decode_image()` instead.**

## Image encoding

JPEG (CPU and CUDA) and PNG are supported for encoding.

| Function | Signature |
|---|---|
| `encode_jpeg` | `(input, quality=...)` — returns raw encoded bytes tensor |
| `write_jpeg` | `(input, filename, quality=...)` — input in CHW layout |
| `encode_png` | `(input, compression_level=...)` — returns a buffer with PNG file contents |
| `write_png` | `(input, filename, compression_level=...)` — input CHW (or HW for grayscale) |

## Raw file I/O

| Function | Signature |
|---|---|
| `read_file` | `(path)` — file contents as a `uint8` 1D tensor |
| `write_file` | `(filename, data)` — write a `uint8` 1D tensor to a file |

## Video — DEPRECATED, migrate to TorchCodec

**All video decoding/encoding in torchvision (`read_video`,
`read_video_timestamps`, `write_video`, `VideoReader`) is deprecated as of
0.22 and scheduled for removal in 0.24.** Torchvision explicitly recommends
migrating to [TorchCodec](https://github.com/pytorch/torchcodec) for any new
video work, since future video decode/encode development is consolidating
there. Do not reach for these APIs in new code without flagging this to the
user — if the codebase is pinned to an older torchvision (<0.22), they still
work as documented below, but plan a TorchCodec migration for anything
long-lived.

```python
from torchvision.io import read_video   # [DEPRECATED]

vid, aud, info = read_video("video.mp4", start_pts=0, end_pts=None, pts_unit="sec")
```

| Function/class | Signature | Status |
|---|---|---|
| `read_video` | `(filename, start_pts=..., end_pts=..., pts_unit=..., output_format=...)` | Deprecated. Returns `(video_frames, audio_frames, info)`. |
| `read_video_timestamps` | `(filename, pts_unit=...)` | Deprecated. Lists frame timestamps without decoding. |
| `write_video` | `(filename, video_array, fps, ...)` | Deprecated. `video_array` is 4D `[T, H, W, C]`. |
| `VideoReader` | `(src, stream=..., num_threads=...)` | Deprecated. Fine-grained, TorchScript-friendly frame-by-frame reading. |

This is exactly the kind of API that has moved recently — if you already
know `read_video`/`VideoReader` from older torchvision experience, re-check
against the installed version before relying on it; a project on torchvision
≥0.24 will not have it at all.
