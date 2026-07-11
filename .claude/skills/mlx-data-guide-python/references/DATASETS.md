# Common Datasets & HuggingFace Integration

## `mlx.data.datasets` loaders

All loaders download to `~/.cache/mlx.data/<name>` on first use (override with
`root=`) and return a `Buffer` unless noted otherwise.

```python
from mlx.data.datasets import (
    load_mnist, load_fashion_mnist, load_cifar10, load_cifar100,
    load_imagenet, load_images_from_folder, load_librispeech,
    load_libritts_r, load_wikitext_lines, load_speechcommands,
)
```

| Function | Signature | Notes |
|---|---|---|
| `load_mnist` | `(root=None, train=True)` | Buffer, keys `{'image', 'label'}` |
| `load_fashion_mnist` | `(root=None, train=True)` | Buffer, same shape as MNIST |
| `load_cifar10` | `(root=None, train=True, quiet=False, validate_download=True)` | Buffer |
| `load_cifar100` | `(root=None, train=True, quiet=False, validate_download=True)` | Buffer |
| `load_imagenet` | `(root=None, split='train', quiet=False, validate_download=True, tar_index_threads=None)` | Buffer. **Must be downloaded manually** from image-net.org (data + devkit); cannot auto-download. `split` is `'train'` or `'val'`. `tar_index_threads` parallelizes indexing the nested training tar. |
| `load_images_from_folder` | `(image_folder)` | Buffer. Expects `image_folder/<class>/<file>` layout. Returns samples with keys `folder` (class name), `label` (0-based sorted class index), `file` (relative path), `image` (loaded array). |
| `load_librispeech` | `(root=None, split='dev-clean', quiet=False, validate_download=True)` | Buffer, loaded directly from the TAR archive. `split` in `dev-clean, dev-other, test-clean, test-other, train-clean-100, train-clean-360, train-other-500`. |
| `load_libritts_r` | `(root=None, split='dev-clean', quiet=False, validate_download=True)` | Same split options as LibriSpeech. |
| `load_wikitext_lines` | `(root=None, split='train', subset='wikitext-103-raw', quiet=False, validate_download=True)` | Returns a **Stream** of text lines (not a Buffer). `split` in `train, valid, test`. `subset` in `wikitext-103, wikitext-103-raw, wikitext-2, wikitext-2-raw`. |
| `load_speechcommands` | `(root=None, split='train', quiet=False, validate_download=True)` | Buffer, from TAR archive. `split` in `train, validation, test`. |

### Example: MNIST -> MLP-ready stream

```python
import mlx.data as dx
from mlx.data.datasets import load_mnist

mnist = load_mnist()
# Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ...
# Buffer(size=60000, keys={'label', 'image'})

mnist_iter = (
    mnist
    .shuffle()
    .to_stream()
    .key_transform("image", lambda x: (x.astype("float32") / 255).ravel())
    .batch(128)
    .prefetch(4, 2)
)
print(next(mnist_iter)["image"].shape)  # (128, 784)
```

### Example: WikiText -> tokenized sliding-window stream

```python
from mlx.data.datasets import load_wikitext_lines
from mlx.data.tokenizer_helpers import read_trie_from_vocab

wiki = load_wikitext_lines(split="train")   # Stream()

trie = read_trie_from_vocab("/path/to/vocab.txt")
wiki_iterator = (
    wiki
    .tokenize("line", trie, output_key="tokens")
    .filter_key("tokens")
    .prefetch(512, 8)
    .batch(128, dim=dict(tokens=0))     # gather everything into one big token array
    .sliding_window("tokens", 1025, 1025)
    .shape("tokens", "tokens_length", 0)
    .batch(32)                          # actual training batch size
    .prefetch(2, 1)
)
# Reported by MLX docs at ~2.5M tok/s on an M2 MacBook Air.
```

## HuggingFace `datasets` integration

`mlx.data` has no first-class HF loader. The supported pattern is: load with
🤗 `datasets`, convert each split to a plain list of numpy-friendly dicts, then
build a `Buffer` with `buffer_from_vector`.

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
print(ds["train"])
# Dataset({features: ['image', 'label'], num_rows: 60000})
```

HF datasets often store images as PIL objects — convert to numpy first:

```python
import numpy as np

def huggingface_to_array_of_dict(dataset):
    return [{"image": np.array(image).copy(), "label": label}
            for label, image in zip(dataset["label"], dataset["image"])]
```

**Requirement before `buffer_from_vector`:** you must have a `list[dict]`
where every value is numpy-array-castable (verify with
`type(dicts[0]["image"]) == np.ndarray`).

```python
import mlx.data as dx

dicts = huggingface_to_array_of_dict(ds["train"])
buffer = dx.buffer_from_vector(dicts)
```

Then build the stream, chaining `key_transform` for normalization/reshape:

```python
stream = (
    buffer
    .to_stream()
    .key_transform("image", lambda x: x.astype("float32") / 255)
    .batch(32)
    .prefetch(prefetch_size=8, num_threads=4)
)
```

### Full example (train/test split, one epoch)

```python
import numpy as np
import mlx.core as mx
import mlx.data as dx
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")

def huggingface_to_array_of_dict(dataset):
    return [{"image": np.array(image).copy(), "label": label}
            for label, image in zip(dataset["label"], dataset["image"])]

def hf_dataset_to_mlx_stream(dataset, shuffle=False):
    numpy_data = huggingface_to_array_of_dict(dataset)
    buffer = dx.buffer_from_vector(numpy_data)
    if shuffle:
        buffer = buffer.shuffle()
    return (
        buffer
        .to_stream()
        .key_transform("image", lambda x: x.astype("float32") / 255)
        .batch(32)
        .prefetch(prefetch_size=8, num_threads=4)
    )

train_stream = hf_dataset_to_mlx_stream(ds["train"], shuffle=True)
test_stream = hf_dataset_to_mlx_stream(ds["test"], shuffle=False)

train_stream.reset()   # streams are stateful iterators — reset before each epoch
for batch in train_stream:
    x, y = mx.array(batch["image"]), mx.array(batch["label"])
```

**Caveat:** always call `stream.reset()` before iterating a stream again
(e.g. at the start of every epoch) — streams don't auto-restart.
