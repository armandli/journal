# torchvision.ops, torchvision.utils, and Feature Extraction

## torchvision.ops — detection/segmentation operators, losses, layers

All operators here have native TorchScript support.

### Detection & segmentation operators

| Function/class | Signature | Purpose |
|---|---|---|
| `nms` | `(boxes, scores, iou_threshold)` | Non-maximum suppression on one set of boxes. |
| `batched_nms` | `(boxes, scores, idxs, iou_threshold)` | NMS applied independently per group in `idxs` (e.g. per class). |
| `masks_to_boxes` | `(masks)` | Bounding boxes around each mask. |
| `roi_align` | `(input, boxes, output_size, ...)` | RoI Align w/ average pooling (Mask R-CNN). |
| `roi_pool` | `(input, boxes, output_size, ...)` | RoI Pool (Fast R-CNN). |
| `ps_roi_align` / `ps_roi_pool` | `(input, boxes, output_size, ...)` | Position-sensitive variants (Light-Head R-CNN / R-FCN). |
| `RoIAlign` / `RoIPool` / `PSRoIAlign` / `PSRoIPool` | `nn.Module` wrappers | Class form of the above functions. |
| `FeaturePyramidNetwork` | `(in_channels_list, out_channels, ...)` | Builds an FPN on top of a set of feature maps. |
| `MultiScaleRoIAlign` | `(featmap_names, ...)` | Multi-scale RoIAlign, used with or without an FPN. |

```python
from torchvision.ops import nms
keep_idx = nms(boxes, scores, iou_threshold=0.5)
boxes, scores = boxes[keep_idx], scores[keep_idx]
```

### Box operators

| Function | Signature | Purpose |
|---|---|---|
| `box_area` | `(boxes, fmt=...)` | Area of each box. |
| `box_convert` | `(boxes, in_fmt, out_fmt)` | Convert between `"xyxy"`, `"xywh"`, `"cxcywh"`. |
| `box_iou` | `(boxes1, boxes2, fmt=...)` | Pairwise IoU (Jaccard index). |
| `clip_boxes_to_image` | `(boxes, size)` | Clip boxes to lie inside an image of `size`. |
| `complete_box_iou` / `distance_box_iou` / `generalized_box_iou` | `(boxes1, boxes2, eps=...)` | CIoU / DIoU / GIoU variants. |
| `remove_small_boxes` | `(boxes, min_size)` | Drop boxes with any side `< min_size`. |

### Losses

| Function | Signature | Purpose |
|---|---|---|
| `complete_box_iou_loss` / `distance_box_iou_loss` / `generalized_box_iou_loss` | `(boxes1, boxes2, ...)` | Gradient-friendly IoU-family losses, penalize non-overlap. |
| `sigmoid_focal_loss` | `(inputs, targets, alpha=..., ...)` | Focal loss for dense detection (RetinaNet). |

### Layers (building blocks)

`Conv2dNormActivation`, `Conv3dNormActivation` (fused conv+norm+activation
blocks), `DeformConv2d` (+ functional `deform_conv2d` — Deformable ConvNets
v2), `DropBlock2d`/`DropBlock3d` (+ functional `drop_block2d`/`drop_block3d`),
`FrozenBatchNorm2d` (BatchNorm with fixed stats/affine params — common when
fine-tuning detection backbones), `MLP`, `Permute(dims)`,
`SqueezeExcitation`, `StochasticDepth` (+ functional `stochastic_depth`).

## torchvision.utils — visualization helpers

```python
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, draw_keypoints, make_grid, save_image

box_img = draw_bounding_boxes(image, boxes, labels=labels, colors="red", width=4, font_size=30)
mask_img = draw_segmentation_masks(image, masks, alpha=0.6, colors=["red", "blue"])
kp_img = draw_keypoints(image, keypoints, connectivity=skeleton_edges, colors="yellow")

grid = make_grid(batch_of_images, nrow=8, padding=2)
save_image(grid, "grid.png")
```

| Function | Signature | Purpose |
|---|---|---|
| `draw_bounding_boxes` | `(image, boxes, labels=..., colors=..., ...)` | Draws boxes on an RGB `uint8` image tensor. |
| `draw_segmentation_masks` | `(image, masks, alpha=..., colors=...)` | Overlays boolean segmentation/detection masks on an image. |
| `draw_keypoints` | `(image, keypoints, connectivity=..., colors=...)` | Draws keypoints (and optional skeleton edges) on an image. |
| `flow_to_image` | `(flow)` | Converts an optical-flow field to an RGB visualization. |
| `make_grid` | `(tensor, nrow=..., padding=..., ...)` | Arranges a batch of images into a single grid image. |
| `save_image` | `(tensor, fp, format=...)` | Saves a tensor (or `make_grid` output) directly to an image file. |

These all expect image tensors (not PIL), matching what `decode_image()`
already returns — no conversion needed if you're already tensor-native.

## Feature extraction for model inspection

`torchvision.models.feature_extraction.create_feature_extractor` taps
intermediate activations out of an existing model (via `torch.fx` symbolic
tracing) — useful for visualizing feature maps, computing embeddings, or
wiring a classification backbone into a detection/segmentation head (e.g. a
custom FPN backbone).

```python
import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

m = resnet50()
train_nodes, eval_nodes = get_graph_node_names(m)   # inspect available node names first

return_nodes = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
extractor = create_feature_extractor(m, return_nodes=return_nodes)

out = extractor(torch.randn(2, 3, 224, 224))
# out is a dict: {"layer1": Tensor, "layer2": Tensor, "layer3": Tensor, "layer4": Tensor}
```

Node names are dot-separated paths through the module hierarchy (e.g.
`"layer4.2.relu"`). You can pass a truncated name like `"layer4"` as a
shortcut — it resolves to that submodule's last executed node — but verify
against `get_graph_node_names` output when a layer has multiple internal
branches, since "last executed" isn't always the output you want. Repeated
ops within the same parent scope get a disambiguating `_N` suffix (e.g.
`"layer4.1.add"`, `"layer4.2.add"`).

Typical use: wrapping a `resnet50` backbone with a `FeaturePyramidNetwork` to
build a custom backbone for `MaskRCNN` — see the worked example in the
official docs (`feature_extraction.html`) if doing this for detection.

| Function | Signature | Purpose |
|---|---|---|
| `create_feature_extractor` | `(model, return_nodes=...)` | Build a module returning a dict of intermediate activations. |
| `get_graph_node_names` | `(model, tracer_kwargs=..., ...)` | List all traceable node names, in execution order. |
