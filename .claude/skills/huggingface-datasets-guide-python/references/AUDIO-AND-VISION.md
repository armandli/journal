# Audio and Vision Datasets

## Installation

```bash
pip install datasets[audio]     # torchcodec + ffmpeg-backed audio decoding
pip install datasets[vision]    # Pillow-backed image decoding
```

## Audio

### The `Audio` feature

**Audio decoding is torchcodec-based** (uses FFmpeg under the hood) — this
replaced the older array/dict-based decoding. Accessing an `audio` column
returns a torchcodec `AudioDecoder`-like object, not a raw numpy array
directly:

```python
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
audio = dataset[0]["audio"]
samples = audio.get_all_samples()
samples.data            # tensor of decoded audio
samples.sample_rate      # int
```

**Warning:** index row-first, then column (`dataset[0]["audio"]`), not
column-first — indexing the column across the whole dataset decodes every
file, which is slow for large datasets.

Get the raw path/bytes without decoding:
```python
dataset = dataset.cast_column("audio", Audio(decode=False))
dataset[0]["audio"]   # {'bytes': None, 'path': '/cache/.../file.wav'}
```

### Resampling

```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```
Resampling happens lazily on access — cast once, and every subsequent read of
that column returns audio at the new rate. Always match the sampling rate a
pretrained model/feature-extractor expects (check the model card) before
feeding audio into it.

### Loading local audio files

```python
from datasets import Dataset, Audio

audio_dataset = Dataset.from_dict({"audio": ["path/1.wav", "path/2.wav"]}).cast_column("audio", Audio())
```

### `AudioFolder` — folder of audio files, no custom code

```python
dataset = load_dataset("audiofolder", data_dir="/path/to/folder")
```

Directory structure mirrors `ImageFolder` (see below): split names inferred
from top-level directory names (`train`/`test`/`validation`), class labels
inferred from subdirectory names. Supports wav/mp3/mp4/etc. (anything ffmpeg
handles). Attach richer metadata (transcriptions, etc.) with a
`metadata.csv` (or `metadata.jsonl` for complex/nested values) next to the
audio files — needs a `file_name` column matching each audio filename:

```
folder/train/metadata.csv
folder/train/first_audio_file.mp3
folder/train/second_audio_file.mp3
```
```
file_name,transcription
first_audio_file.mp3,some transcription text
second_audio_file.mp3,another transcription
```

`drop_metadata=True` ignores the metadata file; `drop_labels=True` skips the
inferred label column (dataset then has just the `audio` column). The
`filters` argument (best combined with `streaming=True`, and fastest when
metadata is Parquet) lets you load only rows matching a condition:
```python
dataset = load_dataset("username/dataset_name", streaming=True, filters=[("label", "=", 0)])
```

### Feature-extractor / processor integration (for ASR/classification models)

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

def prepare_dataset(batch):
    audio = batch["audio"]
    samples = audio.get_all_samples()
    batch["input_values"] = processor(samples.data, sampling_rate=samples.sample_rate).input_values[0]
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

### Decoding performance

```python
import os
num_threads = min(32, (os.cpu_count() or 1) + 4)
dataset = dataset.decode(num_threads=num_threads)   # up to ~20x faster iteration, esp. for remote streaming
dataset = dataset.decode(False)                       # disable decoding entirely, get path/bytes
```
`num_threads` mainly helps remote/streamed data; for local files on a fast
disk, `num_threads=0` (default, sequential) can actually be faster. Note:
`.decode()` is currently only available on `IterableDataset` (streaming).

## Vision

### The `Image` feature

```python
from datasets import load_dataset

dataset = load_dataset("AI-Lab-Makerere/beans", split="train")
dataset[0]["image"]   # PIL.Image, decoded lazily
```

**Warning:** same row-first indexing caveat as audio —
`dataset[0]["image"]`, not `dataset["image"][0]`, to avoid decoding
everything.

```python
dataset = dataset.cast_column("image", Image(mode="RGB"))    # normalize color mode
dataset = dataset.cast_column("image", Image(decode=False))   # raw path/bytes instead of PIL
```

### Loading local image files

```python
from datasets import Dataset, Image

dataset = Dataset.from_dict({"image": ["path/1.png", "path/2.png"]}).cast_column("image", Image())
```

Images can also come from numpy arrays directly:
```python
import numpy as np
from datasets import Dataset, Features, Image

ds = Dataset.from_dict({"i": [np.zeros((16, 16, 3), dtype=np.uint8)]}, features=Features({"i": Image()}))
```
Multi-channel arrays (RGB/RGBA) must be `uint8` — higher precision gets
downcast with a warning. Grayscale accepts wider integer/float precision
(within what Pillow supports), also downcast (e.g. int64→int32, float64→
float32) with a warning if too wide.

### `ImageFolder` — folder of images, no custom code

```python
dataset = load_dataset("imagefolder", data_dir="/path/to/folder")
```

Expected layout — split from top-level dir name, label from subdirectory:
```
folder/train/dog/golden_retriever.png
folder/train/cat/maine_coon.png
folder/test/dog/chihuahua.png
```
Same `metadata.csv`/`metadata.jsonl`, `drop_metadata`, `drop_labels`, and
`filters` options as `AudioFolder` above.

### Data augmentation

Two ways to apply transforms, with different caching tradeoffs:

**`map()`** — runs once, result is cached to disk. Use for deterministic,
one-time preprocessing (resize, format conversion):
```python
def transforms(examples):
    examples["pixel_values"] = [image.convert("RGB").resize((100, 100)) for image in examples["image"]]
    return examples

dataset = dataset.map(transforms, remove_columns=["image"], batched=True)
```
Tune `batch_size`/`writer_batch_size` (both default 1000) down if `map()`
uses too much memory on image-heavy datasets.

**`set_transform()`** — applied on-the-fly at access time, every epoch, not
cached. Use for random augmentation you want re-rolled each epoch:
```python
from torchvision.transforms import Compose, ColorJitter, ToTensor
jitter = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])

def transforms(examples):
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset.set_transform(transforms)
```
Any augmentation library works (torchvision, Albumentations, Kornia,
imgaug) — `set_transform`/`with_transform` just needs a callable taking and
returning a batch dict.

### Decoding performance

Same `dataset.decode(num_threads=...)` / `dataset.decode(False)` pattern as
audio, currently streaming (`IterableDataset`) only.
