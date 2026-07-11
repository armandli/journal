# torchvision.datasets — Full Catalogue

All datasets are `torch.utils.data.Dataset` subclasses (`__getitem__` +
`__len__`), so any of them can be dropped straight into a `DataLoader`:

```python
import torchvision
import torch

imagenet_data = torchvision.datasets.ImageNet("path/to/imagenet_root/")
data_loader = torch.utils.data.DataLoader(
    imagenet_data, batch_size=4, shuffle=True, num_workers=4
)
```

Nearly all datasets share two constructor arguments: `transform` (applied to
the input) and `target_transform` (applied to the label/target).

**Warning:** `download=True` is not multi-process safe. In a distributed
setting, instantiate a dummy dataset once (to trigger the download) *before*
initializing distributed mode, rather than letting every rank race to
download.

## Image classification

`Caltech101`, `Caltech256`, `CelebA`, `CIFAR10`, `CIFAR100`, `Country211`,
`DTD`, `EMNIST`, `EuroSAT`, `FakeData` (synthetic random images — useful for
pipeline smoke tests), `FashionMNIST`, `FER2013`, `FGVCAircraft`, `Flickr8k`,
`Flickr30k`, `Flowers102`, `Food101`, `GTSRB`, `INaturalist`, `ImageNet`,
`Imagenette`, `KMNIST`, `LFWPeople`, `LSUN`, `MNIST`, `Omniglot`,
`OxfordIIITPet`, `Places365`, `PCAM`, `QMNIST`, `RenderedSST2`, `SEMEION`,
`SBU`, `StanfordCars`, `STL10`, `SUN397`, `SVHN`, `USPS`.

```python
train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=my_transforms
)
```

## Image detection or segmentation

`CocoDetection`, `CelebA` (also has box/landmark targets), `Cityscapes`,
`Kitti`, `OxfordIIITPet` (also has segmentation trimaps), `SBDataset`
(Semantic Boundaries), `VOCSegmentation`, `VOCDetection`, `WIDERFace`.

## Optical flow

`FlyingChairs`, `FlyingThings3D`, `HD1K`, `KittiFlow`, `Sintel`.

## Stereo matching

`CarlaStereo`, `Kitti2012Stereo`, `Kitti2015Stereo`, `CREStereo`,
`FallingThingsStereo`, `SceneFlowStereo`, `SintelStereo`, `InStereo2k`,
`ETH3DStereo`, `Middlebury2014Stereo`.

## Image pairs

`LFWPairs`, `PhotoTour` (multi-view stereo correspondence).

## Image captioning

`CocoCaptions`.

## Video classification

`HMDB51`, `Kinetics`, `UCF101`.

## Video prediction

`MovingMNIST`.

## Base classes for building custom datasets

- `DatasetFolder(root, loader, extensions=..., transform=..., ...)` — generic
  loader that walks `root/<class>/<file>`, calling your `loader` function per
  file.
- `ImageFolder(root, transform=..., ...)` — `DatasetFolder` specialized for
  images (`root/<class>/<file>.jpg` layout); the common way to load a custom
  image-classification dataset with no special format.
- `VisionDataset(root=None, transforms=None, transform=None, target_transform=None)`
  — base class for writing torchvision-compatible custom datasets.

```python
# Custom image folder dataset, e.g. root/cat/1.jpg, root/dog/1.jpg, ...
dataset = torchvision.datasets.ImageFolder(root="path/to/root", transform=my_transforms)
```

## Transforms v2 interop

```python
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

dataset = CocoDetection(root, annFile, transforms=my_v2_transforms)
dataset = wrap_dataset_for_transforms_v2(dataset)
# dataset now yields tv_tensors.BoundingBoxes / tv_tensors.Mask instead of
# raw dicts/PIL, so it composes correctly with torchvision.transforms.v2.
```

Datasets that predate `transforms.v2` and `tv_tensors` (like `CocoDetection`,
`VOCDetection`, `VOCSegmentation`) don't return TVTensors by default —
`wrap_dataset_for_transforms_v2` is the one-line fix. Plain classification
datasets (`ImageFolder`, `CIFAR10`, `ImageNet`, ...) need no wrapping: just
pass a `v2.Compose(...)` as `transform=`.
