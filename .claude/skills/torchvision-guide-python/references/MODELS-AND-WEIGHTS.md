# Models and Pre-trained Weights

`torchvision.models` covers: image classification, semantic segmentation,
object detection, instance segmentation, person keypoint detection, video
classification, and optical flow. Weights download to a cache dir on first
use (override with the `TORCH_HOME` env var).

## The `weights=` API (current, since v0.13)

```python
from torchvision.models import resnet50, ResNet50_Weights

resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)   # older weights, 76.130% acc
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)   # newer weights, 80.858% acc
resnet50(weights=ResNet50_Weights.DEFAULT)         # "best available" (may change across releases)
resnet50(weights="IMAGENET1K_V2")                  # strings also accepted
resnet50(weights=None)                             # random init, no download
```

The old `pretrained=True/False` boolean argument is deprecated — don't use
it in new code. `weights=None` (or omitting `weights`) is the direct
replacement for `pretrained=False`.

## Always use the weight's bundled preprocessing

Every weight enum carries the exact preprocessing (resize, crop, normalize)
it was trained with, via `.transforms()`:

```python
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
img_transformed = preprocess(img)   # img: decoded uint8 tensor
```

Preprocessing differs across model families **and across weight versions of
the same model** — never assume a fixed `Resize(256)+CenterCrop(224)+ImageNet
mean/std` recipe; always pull it from the weight you're actually using.
Remember `model.eval()` before inference (batchnorm/dropout behave
differently in train vs eval mode).

## Listing/retrieving models and weights by name (since v0.14)

```python
from torchvision.models import list_models, get_model, get_weight, get_model_weights
import torchvision

all_models = list_models()
classification_models = list_models(module=torchvision.models)

m1 = get_model("mobilenet_v3_large", weights=None)
m2 = get_model("quantized_mobilenet_v3_large", weights="DEFAULT")

# Fetch a specific weight value by its full dotted name
weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")

# Fetch the weights *enum class* for a given model name (or builder function)
weights_enum = get_model_weights("quantized_mobilenet_v3_large")
assert weights_enum == MobileNet_V3_Large_QuantizedWeights
assert weights in weights_enum
```

| Function | Purpose |
|---|---|
| `get_model(name, **config)` | Instantiate a model by string name + kwargs (e.g. `weights=`). |
| `get_model_weights(name)` | Return the weights **enum class** for a model name or builder function. |
| `get_weight(name)` | Return a specific weight **enum value** given its full dotted name string. |
| `list_models(module=None, include=None, exclude=None)` | List registered model names, optionally filtered to a submodule. |

## Loading via `torch.hub` (no local torchvision clone needed)

```python
import torch
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet50")
print([w for w in weight_enum])
```

Exception: `torchvision.models.detection` models need torchvision actually
installed (they depend on custom C++ ops), so they can't be hub-loaded
standalone.

## Classification

Families: `AlexNet`, `ConvNeXt`, `DenseNet`, `EfficientNet`, `EfficientNetV2`,
`GoogLeNet`, `Inception V3`, `MaxVit`, `MNASNet`, `MobileNet V2`,
`MobileNet V3`, `RegNet`, `ResNet`, `ResNeXt`, `ShuffleNet V2`, `SqueezeNet`,
`SwinTransformer`, `VGG`, `VisionTransformer`, `Wide ResNet`.

```python
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

img = decode_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).eval()
preprocess = weights.transforms()

batch = preprocess(img).unsqueeze(0)
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * prediction[class_id].item():.1f}%")
```

Output class names for any classification/detection/segmentation weight are
always at `weights.meta["categories"]` (keypoint models instead expose
`weights.meta["keypoint_names"]`).

## Quantized (INT8) classification

Families: `Quantized GoogLeNet`, `Quantized InceptionV3`,
`Quantized MobileNet V2`, `Quantized MobileNet V3`, `Quantized ResNet`,
`Quantized ResNeXt`, `Quantized ShuffleNet V2`.

```python
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights

weights = ResNet50_QuantizedWeights.DEFAULT
model = resnet50(weights=weights, quantize=True).eval()
preprocess = weights.transforms()
```

## Semantic segmentation *(Beta — no backward-compat guarantee)*

Families: `DeepLabV3`, `FCN`, `LRASPP`.

```python
from torchvision.io.image import decode_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

img = decode_image("gallery/assets/dog1.jpg")
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
batch = weights.transforms()(img).unsqueeze(0)

prediction = model(batch)["out"]                       # note: dict output, key "out"
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {c: i for i, c in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["dog"]]
to_pil_image(mask).show()
```

## Object detection / instance segmentation / keypoint detection *(Beta)*

Detection models are built on top of classification backbones and expect a
**list** of `Tensor[C, H, W]` (not a batched 4D tensor).

- Object detection: `Faster R-CNN`, `FCOS`, `RetinaNet`, `SSD`, `SSDlite`.
- Instance segmentation: `Mask R-CNN`.
- Person keypoint detection: `Keypoint R-CNN`.

```python
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

img = decode_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9).eval()
preprocess = weights.transforms()

batch = [preprocess(img)]                 # list, not a stacked batch tensor
prediction = model(batch)[0]              # dict: "boxes", "labels", "scores" (+ "masks"/"keypoints")
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"], labels=labels, colors="red", width=4)
to_pil_image(box.detach()).show()
```

Mask R-CNN's prediction dict additionally has `"masks"`; Keypoint R-CNN's has
`"keypoints"`.

## Video classification *(Beta)*

Families: `Video MViT`, `Video ResNet` (`r3d_18`, `mc3_18`, `r2plus1d_18`),
`Video S3D`, `Video SwinTransformer` (`swin3d_*`).

```python
from torchvision.io.video import read_video   # deprecated since 0.22 — see IO-IMAGES-VIDEOS.md
from torchvision.models.video import r3d_18, R3D_18_Weights

vid, _, _ = read_video("test/assets/videos/v_SoccerJuggling_g23_c01.avi", output_format="TCHW")
vid = vid[:32]
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights).eval()
batch = weights.transforms()(vid).unsqueeze(0)
prediction = model(batch).squeeze(0).softmax(0)
category_name = weights.meta["categories"][prediction.argmax().item()]
```

## Optical flow

Family: `RAFT`.

## Licensing note

Pretrained weights may carry their own license/terms derived from the
training dataset — verify you're allowed to use a given weight for your use
case before shipping it.
