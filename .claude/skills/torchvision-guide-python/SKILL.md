---
name: torchvision-guide-python
description: Write and debug Python computer-vision code using torchvision, PyTorch's companion library for vision. Covers built-in datasets, the transforms v2 / TVTensors API, pretrained models via the weights= API, image/video decoding-encoding (torchvision.io), and drawing/box/ROI utilities (torchvision.utils, torchvision.ops). Use when the user asks to "use torchvision", "load a torchvision dataset", "use a pretrained torchvision model", "torchvision transforms v2", "decode/encode an image or video with torchvision", or "draw bounding boxes/masks with torchvision". Do NOT use for plain PyTorch core (nn/autograd/training loops — use pytorch-guide-python) or for non-vision torch domains.
argument-hint: "[task or description of what to implement]"
---

# torchvision Python Guide

`torchvision` is PyTorch's vision companion library: datasets, pretrained
models, image/video transforms, and vision-specific ops.

## Imports

```python
import torch
import torchvision
from torchvision import datasets, models, tv_tensors
from torchvision.transforms import v2
from torchvision.io import decode_image
```

## The Big Picture

- **`torchvision.datasets`** — ~60 built-in datasets (`ImageFolder`/`ImageNet`/`CocoDetection`/...), all `torch.utils.data.Dataset` subclasses usable with `DataLoader`.
- **`torchvision.tv_tensors`** + **`torchvision.transforms.v2`** — typed tensor subclasses (`Image`, `BoundingBoxes`, `Mask`, `Video`, `KeyPoints`) that let one transform pipeline correctly handle images, boxes, masks, video, and keypoints together.
- **`torchvision.models`** — pretrained classification/detection/segmentation/video/optical-flow architectures via the modern `weights=` enum API, each weight bundling its own `.transforms()` preprocessing.
- **`torchvision.io`** — `decode_image`/`encode_jpeg`/`encode_png` etc. for tensor-native image codecs. **Video decode/encode (`read_video`, `write_video`, `VideoReader`) is deprecated since 0.22** — see [references/IO-IMAGES-VIDEOS.md](references/IO-IMAGES-VIDEOS.md).
- **`torchvision.ops`** / **`torchvision.utils`** — detection/segmentation primitives (`nms`, `roi_align`, `box_iou`) and visualization helpers (`draw_bounding_boxes`, `make_grid`).

## Quick pattern: pretrained classification inference

```python
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

img = decode_image("path/to/image.jpg")   # uint8 tensor, CHW — no PIL needed

weights = ResNet50_Weights.DEFAULT          # "best currently available" weights
model = resnet50(weights=weights).eval()
preprocess = weights.transforms()           # bundled resize/crop/normalize for THIS weight

batch = preprocess(img).unsqueeze(0)
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * prediction[class_id].item():.1f}%")
```

Always fetch preprocessing from `weights.transforms()` rather than hand-rolling
it — resize size/interpolation/normalization stats vary per model *and per
weight version*, and mismatching them silently degrades accuracy.

## Quick pattern: transforms v2 pipeline (works on images, boxes, masks, video)

```python
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),   # scale uint8 [0,255] -> float [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Plain image classification
img_out = transforms(img)

# Detection: pass an image AND its boxes through the SAME pipeline
boxes = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=img.shape[-2:])
img_out, boxes_out = transforms(img, boxes)

# Arbitrary structures (dicts, tuples, nested) also work — transforms dispatch
# by object type (Image/BoundingBoxes/Mask/Video/PIL), not position.
out_dict = transforms({"image": img, "boxes": boxes})
```

Use `torchvision.transforms.v2`, not the legacy `torchvision.transforms` (v1)
— v2 is faster, handles non-image types, and is where new features land.
Migrating is usually just changing the import.

## Loading a dataset

```python
from torchvision import datasets

train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
```

Detection/segmentation datasets like `CocoDetection` predate TVTensors and
return plain dicts/PIL by default — wrap them for v2 transforms:

```python
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

dataset = CocoDetection(root, annFile, transforms=my_v2_transforms)
dataset = wrap_dataset_for_transforms_v2(dataset)   # now yields TVTensors
```

Full dataset catalogue (classification, detection/segmentation, optical flow,
stereo, video, captioning): [references/DATASETS.md](references/DATASETS.md).

## Testing

After wiring a pipeline, actually run one batch through it end-to-end (dataset
→ transforms → model or visualization) and print shapes/dtypes — TVTensor
dispatch and the `weights=` preprocessing are easy to get subtly wrong (e.g.
missing `.unsqueeze(0)`, mismatched `ToDtype(scale=...)`), and errors often
surface as silently wrong accuracy rather than exceptions.

## References

- [references/DATASETS.md](references/DATASETS.md) — full `torchvision.datasets` catalogue by task, `DataLoader` usage, `wrap_dataset_for_transforms_v2`
- [references/TVTENSORS-AND-TRANSFORMS.md](references/TVTENSORS-AND-TRANSFORMS.md) — TVTensors concepts, full v2 transform-class/functional reference, v1-vs-v2, torchscript/performance notes
- [references/MODELS-AND-WEIGHTS.md](references/MODELS-AND-WEIGHTS.md) — `weights=` API, `list_models`/`get_model`/`get_weight`, model families per task (classification, quantized, segmentation, detection, video, optical flow)
- [references/IO-IMAGES-VIDEOS.md](references/IO-IMAGES-VIDEOS.md) — image decode/encode (JPEG/PNG/WEBP/GIF/AVIF/HEIC, incl. CUDA JPEG), deprecated video API + TorchCodec migration
- [references/OPS-AND-UTILS.md](references/OPS-AND-UTILS.md) — `torchvision.ops` (nms, box ops, roi_align, losses, layers), `torchvision.utils` (draw_bounding_boxes, make_grid, ...), feature extraction

## External Docs

- Full docs: https://docs.pytorch.org/vision/main/index.html

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "torchvision-guide-python"
```
