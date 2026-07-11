# TVTensors & transforms.v2

## What are TVTensors?

TVTensors are `torch.Tensor` subclasses — they behave exactly like regular
tensors (`.sum()`, any `torch.*` op works), but their *type* tells `v2`
transforms how to dispatch:

```python
from torchvision import tv_tensors

img_dp = tv_tensors.Image(torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8))
isinstance(img_dp, torch.Tensor)   # True
```

Available TVTensor classes (`torchvision.tv_tensors`):

| Class | Shape | Notes |
|---|---|---|
| `Image(data, *, dtype, device, requires_grad)` | `[..., C, H, W]` | |
| `Video(data, *, dtype, device, requires_grad)` | `[..., T, C, H, W]` | |
| `KeyPoints(data, *, canvas_size, dtype, ...)` | `[..., 2]` | points in an image |
| `BoundingBoxes(data, *, format, canvas_size)` | `[N, K]` | `format` is a `BoundingBoxFormat` (e.g. `"XYXY"`, `"XYWH"`, `"CXCYWH"`); `canvas_size=(H, W)` is required so boxes can be clamped/transformed correctly |
| `Mask(data, *, dtype, device, requires_grad)` | `[..., H, W]` | segmentation or detection masks |
| `TVTensor` | — | base class for all of the above |

Other `tv_tensors` utilities: `set_return_type(return_type)` (controls
whether torch ops on a TVTensor return a TVTensor or a plain `Tensor`), and
`wrap(wrappee, *, like, **kwargs)` (convert a plain `Tensor` into the same
TVTensor subclass as `like` — the standard way to construct a TVTensor that
shares another one's metadata, e.g. `canvas_size`).

## Dispatch rules for plain tensors

Transforms decide whether a bare `torch.Tensor` is "an image/video to
transform" or "pass-through data" (e.g. integer labels) like this:

- If the input contains an `Image`, `Video`, or `PIL.Image.Image`, every
  *other* plain tensor is passed through untouched.
- If it contains none of those, only the **first** plain tensor (in
  depth-first traversal order) is treated as an image/video; the rest pass
  through.

This is why `out_img, boxes_out = transforms(img, boxes)` works even when
`img` is a plain uint8 tensor rather than a `tv_tensors.Image` — but wrapping
the image explicitly with `tv_tensors.Image(...)` removes any ambiguity and
is safer once you have more than one plain tensor in the input.

## Pipelines operate on arbitrary structures

```python
transforms = v2.Compose([...])

out = transforms(img)                              # single image
out_img, out_boxes = transforms(img, boxes)         # tuple in, tuple out
out_dict = transforms({"image": img, "boxes": boxes, "path": "unchanged.jpg"})
```

Foreign objects (strings, ints, etc.) are passed through unchanged — handy for
carrying a file path alongside a sample through the pipeline for debugging.

## Building your own dataset's samples as TVTensors

For custom datasets (not using `wrap_dataset_for_transforms_v2`), construct
TVTensors either at the end of `__getitem__` or as the first step of your
transform pipeline:

```python
boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510]],
    format="XYXY", canvas_size=img.shape[-2:],
)
```

## V1 vs V2 — always prefer V2

`torchvision.transforms.v2` (introduced 0.15, March 2023) replaces
`torchvision.transforms` (v1). v2 is faster, handles boxes/masks/video/
keypoints (not just images), supports `CutMix`/`MixUp`, and accepts arbitrary
input structures (dicts/tuples/lists). Migration is normally just:

```python
# before
from torchvision import transforms
# after
from torchvision.transforms import v2 as transforms
```

v2 is fully backward compatible with v1 usage; expect only negligible output
differences from implementation details.

## Dtype & value-range convention

- Float dtype tensors: values expected in `[0, 1]`.
- Integer dtype tensors (typically `torch.uint8`): values expected in
  `[0, MAX_DTYPE]` (e.g. `[0, 255]` for `uint8`).
- Use `ToDtype(dtype, scale=True/False)` to convert dtype **and** rescale
  correctly — don't `.float()` an image and assume it's normalized.

## Performance guidelines

1. Use `torchvision.transforms.v2`, not v1.
2. Prefer tensors over PIL images as pipeline input.
3. Prefer `torch.uint8` for as long as possible, especially through resizing.
4. Resize with `interpolation=InterpolationMode.BILINEAR` or `BICUBIC`.

```python
transforms = v2.Compose([
    v2.ToImage(),                             # PIL/ndarray -> Image tensor (no value scaling)
    v2.ToDtype(torch.uint8, scale=True),      # normalize dtype, usually a no-op if already uint8
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),    # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

This ordering (resize/crop while still `uint8`, convert to float only right
before `Normalize`) gives close to the best throughput with a `DataLoader`
using `num_workers > 0`. Some transforms (e.g. `Resize`,
`RandomResizedCrop`) prefer channels-last input and don't benefit much from
`torch.compile()`; `Normalize` does benefit from `torch.compile()`.

## Classes, functionals, and kernels

- **Classes** (`v2.Resize`, `v2.RandomCrop`, ...) — stateful/randomized,
  typically what you put in a `Compose`.
- **Functionals** (`torchvision.transforms.v2.functional.resize`, `.crop`,
  ...) — stateless equivalents (like `torch.nn` vs `torch.nn.functional`).
  Random-class functionals (e.g. `crop()` for `RandomCrop`) take explicit
  parameters instead of sampling randomly — use the class's `get_params()`
  classmethod to sample parameters yourself if calling the functional
  directly.
- **Kernels** — low-level, type-specific functions inside `.functional`
  (e.g. `resize_bounding_boxes`), public but undocumented; needed mainly for
  TorchScript support of non-image types like boxes/masks.

## TorchScript support

```python
import torch.nn as nn
transforms = nn.Sequential(
    v2.CenterCrop(10),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
scripted = torch.jit.script(transforms)
```

Use `nn.Sequential` instead of `v2.Compose` when you need `torch.jit.script`.
**Caveat:** scripting a `v2` class transform actually scripts its v1
equivalent under the hood — for guaranteed v2 semantics under TorchScript,
script the functions in `v2.functional` directly. Functionals only support
scripting for plain tensors treated as images; box/mask TorchScript support
requires the low-level kernels. Any custom transform meant for
`torch.jit.script` should subclass `torch.nn.Module`.

## V2 API reference (recommended)

### Geometry — resizing
`Resize`, `ScaleJitter`, `RandomShortestSize`, `RandomResize`.
Functional: `resize`.

### Geometry — cropping
`RandomCrop`, `RandomResizedCrop`, `RandomIoUCrop`, `CenterCrop`, `FiveCrop`,
`TenCrop`.
Functional: `crop`, `resized_crop`, `ten_crop`, `center_crop`, `five_crop`.

### Geometry — other
`RandomHorizontalFlip`, `RandomVerticalFlip`, `Pad`, `RandomZoomOut`,
`RandomRotation`, `RandomAffine`, `RandomPerspective`, `ElasticTransform`.
Functional: `horizontal_flip`, `vertical_flip`, `pad`, `rotate`, `affine`,
`perspective`, `elastic`.

### Color
`ColorJitter`, `RandomChannelPermutation`, `RandomPhotometricDistort`,
`Grayscale`, `RGB`, `RandomGrayscale`, `GaussianBlur`, `GaussianNoise`,
`RandomInvert`, `RandomPosterize`, `RandomSolarize`, `RandomAdjustSharpness`,
`RandomAutocontrast`, `RandomEqualize`.
Functional: `permute_channels`, `rgb_to_grayscale`, `grayscale_to_rgb`,
`gaussian_blur`, `gaussian_noise`, `invert`, `posterize`, `solarize`,
`adjust_sharpness`, `autocontrast`, `adjust_contrast`, `equalize`,
`adjust_brightness`, `adjust_saturation`, `adjust_hue`, `adjust_gamma`.

### Composition
`Compose(transforms)`, `RandomApply(transforms, p)`,
`RandomChoice(transforms, p)`, `RandomOrder(transforms)`.

### Miscellaneous
`LinearTransformation`, `Normalize(mean, std, inplace=False)`,
`RandomErasing`, `Lambda(lambd, *types)`, `SanitizeBoundingBoxes` (drop
degenerate/invalid boxes + their labels/masks), `SanitizeKeyPoints`,
`ClampBoundingBoxes`, `ClampKeyPoints`, `UniformTemporalSubsample(num_samples)`
(subsample a video's temporal dim), `JPEG(quality)` (simulate JPEG
compression artifacts as augmentation).
Functional: `normalize`, `erase`, `sanitize_bounding_boxes`,
`sanitize_keypoints`, `clamp_bounding_boxes`, `clamp_keypoints`,
`uniform_temporal_subsample`, `jpeg`.

### Conversion
`ToImage()` (tensor/ndarray/PIL → `Image`, no value scaling), `ToPureTensor()`
(strip TVTensor metadata back to plain `Tensor`), `PILToTensor()` (no
scaling), `ToPILImage(mode=...)`, `ToDtype(dtype, scale=...)` (the
recommended way to change dtype **and** correctly rescale),
`ConvertBoundingBoxFormat(format)`.

**Deprecated:** `ToTensor()` → use
`v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])` instead;
`ConvertImageDtype(dtype)` → use `v2.ToDtype(dtype, scale=True)`.

### Auto-augmentation
`AutoAugment(policy=..., interpolation=..., fill=...)` (policies learned on
ImageNet/CIFAR10/SVHN), `RandAugment`, `TrivialAugmentWide`, `AugMix`.

### CutMix / MixUp
`CutMix(*, alpha=1.0, num_classes, labels_getter=...)` and
`MixUp(*, alpha=1.0, num_classes, labels_getter=...)` — applied to a
**batch** (after the `DataLoader` collates samples), not to individual
samples, since they combine pairs of images+labels. Typically used inside the
training loop or a custom collate function, not inside `Compose`.

### Developer tools
`v2.Transform()` (base class for writing your own v2 transform),
`v2.functional.register_kernel(functional, tv_tensor_cls)` (register a custom
kernel for a functional + TVTensor type), `v2.query_size(flat_inputs)`,
`v2.query_chw(flat_inputs)`, `v2.get_bounding_boxes(flat_inputs)`,
`v2.get_keypoints(flat_inputs)`.

## V1 API (legacy — prefer v2 above)

Same names without the `v2.` prefix live in `torchvision.transforms`:
`Resize`, `RandomCrop`, `RandomResizedCrop`, `CenterCrop`, `FiveCrop`,
`TenCrop`, `Pad`, `RandomRotation`, `RandomAffine`, `RandomPerspective`,
`ElasticTransform`, `RandomHorizontalFlip`, `RandomVerticalFlip`,
`ColorJitter`, `Grayscale`, `RandomGrayscale`, `GaussianBlur`,
`RandomInvert`, `RandomPosterize`, `RandomSolarize`,
`RandomAdjustSharpness`, `RandomAutocontrast`, `RandomEqualize`, `Compose`,
`RandomApply`, `RandomChoice`, `RandomOrder`, `LinearTransformation`,
`Normalize`, `RandomErasing`, `Lambda`, `AutoAugmentPolicy`, `AutoAugment`,
`RandAugment`, `TrivialAugmentWide`, `AugMix`. Plus the functional module
`torchvision.transforms.functional` with matching lower-case functions
(`resize`, `crop`, `adjust_brightness`, `to_tensor`, `to_pil_image`, etc.) —
same names as v2's functionals, but image-only (no TVTensor dispatch).
