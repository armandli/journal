# Buffers, Streams, and the Shared Transform API

## Samples

A sample is `dict[str, array-like]`. Valid values:

```python
sample = {"hello": np.array(0)}       # numpy array
sample = {"scalar": 42}               # scalars auto-cast to scalar arrays
sample = {"key": "value"}             # strings work but become unicode
sample = {"key": b"path/to/my/file"}  # prefer bytes for filenames/text
```

## Buffer — indexable, known length

```python
import mlx.data as dx

numbers = dx.buffer_from_vector([{"x": i} for i in range(10)])
evens = numbers.key_transform("x", lambda x: 2 * x)

print(evens)        # Buffer(size=10, keys={'x'})
print(evens[3])      # {'x': array(6)}  -- random access works
print(len(evens))    # 10
```

Every op on a `Buffer` (see "Shared transform API" below) returns a new
`Buffer`, lazily evaluated on access — nothing runs until you index into it or
iterate.

### Buffer factory functions

```python
mlx.data.buffer_from_vector(data: list[dict]) -> Buffer
```
Main entry point. Build a buffer straight from a list of dicts:

```python
from pathlib import Path
import mlx.data as dx

def files_and_classes(root: Path):
    images = list(root.rglob("*.jpg"))
    categories = [p.relative_to(root).parent.name for p in images]
    category_map = {c: i for i, c in enumerate(sorted(set(categories)))}
    return [
        {"image": str(p.relative_to(root)).encode("ascii"), "category": c,
         "label": category_map[c]}
        for c, p in zip(categories, images)
    ]

dset = dx.buffer_from_vector(files_and_classes(Path("path/to/dataset")))
```

```python
mlx.data.files_from_tar(tarfile: str, nested: bool = False, num_threads: int = 1) -> Buffer
```
Returns the list of files contained in a tar archive (a `Buffer` of filenames).
`nested=True` recursively indexes archives-within-archives, parallelized with
`num_threads`.

### Buffer-only ops (need random access)

| Method | Signature | Behavior |
|---|---|---|
| `ordered_prefetch` | `(self, prefetch_size, num_thread) -> Stream` | Background-thread prefetch that **preserves ordering** (deterministic). Converts to a `Stream`. Use over `Stream.prefetch()` when reproducibility matters. |
| `partition` | `(self, num_partitions, partition) -> Buffer` | Equivalent to slicing with `step=num_partitions`, `offset=partition`. For sharding a dataset across distributed workers. |
| `perm` | `(self, perm: list[int]) -> Buffer` | Arbitrary reindex/reorder/filter via an explicit index list. |
| `shuffle` | `(self) -> Buffer` | Random permutation of the whole buffer. |
| `to_stream` | `(self) -> Stream` | Convert to a `Stream`. Do this right before you need `batch`/`prefetch`/training iteration. |

```python
# Deterministic prefetching (reproducible order across runs)
dset = (
    dset
    .shuffle()
    .batch(32)
    .ordered_prefetch(8, 4)   # 8 batches ahead, 4 threads, order preserved
)
sample = next(dset)
```

## Stream — iterator, possibly infinite, no random access

```python
import mlx.data as dx

numbers = dx.stream_python_iterable(lambda: ({"x": i} for i in range(10**10)))
evens = numbers.sample_transform(lambda s: s if s["x"] % 2 == 0 else dict())

print(next(numbers))  # {'x': array(0)}
print(next(numbers))  # {'x': array(1)}

# Streams share underlying state: advancing `numbers` advances `evens` too,
# since evens is built on top of numbers.
print(next(evens))    # {'x': array(2)}

# Streams are resettable (unless the underlying source truly can't rewind).
evens.reset()
```

Filtering on a `Stream` (or `Buffer`) is done via `sample_transform`/
`key_transform` functions that return an **empty dict** to drop a sample —
there is no separate "filter" primitive beyond `filter_key`/`filter_by_shape`.

### Stream factory functions

| Function | Signature | Notes |
|---|---|---|
| `stream_python_iterable` | `(iterable_factory: Callable[[], Iterable[dict]]) -> Stream` | The most useful one from Python. Takes a **factory function** (not the iterable itself) so the stream can be reset/restarted. |
| `stream_csv_reader` | `(file, sep=',', quote='"', *, local_prefix='', file_fetcher=None, file_fetcher_handle=None) -> Stream` | `file` can be a path or any object with `.read()`/`.seek()`. |
| `stream_csv_reader_from_string` | `(content: str, sep=',', quote='"') -> Stream` | Same as above but from an in-memory string. |
| `stream_line_reader` | `(file, key: str, unzip=False, *, local_prefix='', file_fetcher=None, file_fetcher_handle=None) -> Stream` | Streams file lines into `key`. `unzip=True` decompresses on the fly. Newlines are stripped. |

Also: from a `Buffer`, call `.to_stream()`.

### Stream-only ops (nesting / nondeterministic prefetch / windowing)

| Method | Signature | Behavior |
|---|---|---|
| `csv_reader_from_key` | `(self, key, sep=',', quote='"', from_memory=False, local_prefix='', file_fetcher=None) -> Stream` | For every sample, treat the array at `key` as a CSV filename (or content, if `from_memory=True`) and expand each row into its own sample. |
| `line_reader_from_key` | `(self, key, dst_key, from_memory=False, unzip=False, local_prefix='', file_fetcher=None) -> Stream` | Same idea, but for line-by-line files, writing lines to `dst_key`. |
| `dynamic_batch` | `(self, buffer_size, key, max_data_size=-1, pad={}, dim={}, shuffle=False, num_threads=1) -> Stream` | Batches by **total element count** at `key` rather than sample count — minimizes padding waste when sample sizes vary widely (e.g. variable-length token sequences). |
| `partition` | `(self, num_partitions, partition) -> Stream` | For every `num_partitions` consecutive samples, keep only the `partition`-th (0-based). For distributed sharding of a stream. |
| `buffered` | `(self, buffer_size, on_refill=None, num_threads=1) -> Stream` | Accumulate `buffer_size` samples into a `Buffer`, apply `on_refill` (e.g. `lambda b: b.shuffle()`, or a custom sort-by-length function), then iterate the result. Good for pseudo-shuffling or length-bucketing before padding. |
| `repeat` | `(self, num_time) -> Stream` | Reset the stream `num_time` times before declaring exhaustion (multi-epoch iteration without a manual loop). |
| `shuffle` | `(self, buffer_size) -> Stream` | **Shuffle buffer** semantics: fills a buffer of `buffer_size`, yields a random element from it, refills from upstream. Gives better mixing than `buffered(n, lambda b: b.shuffle())` because a sample can move arbitrarily far from its original position, not just within one bucket. |
| `sliding_window` | `(self, key, size, stride, dim=-1, index_key='') -> Stream` | Slides a window of `size` with `stride` over the array at `key` — common for chunking long sequences/documents. |
| `prefetch` | `(self, prefetch_size, num_threads) -> Stream` | Background-thread prefetch. **Not** order-preserving — use `Buffer.ordered_prefetch()` if determinism matters. |

```python
# dynamic_batch example: batches of ~constant total tokens instead of ~constant sample count
dset = dx.buffer_from_vector([random_sample() for _ in range(10_000)]).to_stream()
dset = dset.dynamic_batch(500, "tokens", max_data_size=16 * 1024)
```

```python
# sliding_window example
dset = dx.buffer_from_vector([{"x": np.arange(10)}]).to_stream()
for sample in dset.sliding_window("x", 3, 2):
    print(sample["x"])
# [0,1,2] [2,3,4] [4,5,6] [6,7,8] [8,9]
```

## Shared transform API (works on both `Buffer` and `Stream`)

All of these return a new object of the same kind (`Buffer` -> `Buffer`,
`Stream` -> `Stream`), applied lazily on access.

### General sample ops

| Method | Signature |
|---|---|
| `batch` | `(self, batch_size, pad={}, dim={}) -> Self` — stacks `batch_size` samples. If arrays differ in shape, pads to the smallest shape that fits all (fill value from `pad`). Pass `dim={key: d}` to **concatenate** along dim `d` instead of stacking (useful for variable-length sequences). |
| `filter_by_shape` | `(self, key, dim, low=-1, high=-1) -> Self` — keep samples whose array at `key` has size in `[low, high]` along `dim` (negative `high` = no upper bound). |
| `filter_key` | `(self, key, remove=False) -> Self` — keep only `key` (default), or drop it (`remove=True`). |
| `key_transform` | `(self, key, func, output_key='') -> Self` — apply `func(array) -> array` to the value at `key`; write to `output_key` if given, else overwrite. |
| `sample_transform` | `(self, func) -> Self` — apply `func(sample_dict) -> sample_dict` to whole samples. Return `{}` to drop the sample (this is *the* filtering primitive). GIL-bound — keep `func` cheap or numpy-vectorized. |
| `remove_value` | `(self, key, size_key, dim, value, pad=0) -> Self` — strip a specific `value` out of the array at `key` (shifting remaining elements left), updating `size_key`'s length tracking. |
| `rename_key` | `(self, key, output_key) -> Self` — rename a key (more efficient than a `sample_transform` doing the same). |

```python
dset = dset.batch(4, dim=dict(x=0))   # concat along dim 0 instead of stacking
```

### Image ops

| Method | Signature |
|---|---|
| `image_center_crop` | `(self, key, w, h, output_key='') -> Self` |
| `image_channel_reduction` | `(self, key, preset='default', output_key='') -> Self` — RGB→grayscale. `preset` in `default`/`rec601`, `rec709`, `rec2020`, `green`. |
| `image_random_area_crop` | `(self, key, area_range: (float,float), aspect_ratio_range: (float,float), num_trial=10, output_key='') -> Self` — rejection-sampled crop within area/aspect constraints; falls back to the original image if no valid crop is found in `num_trial` attempts. |
| `image_random_crop` | `(self, key, w, h, output_key='') -> Self` — fails if image smaller than `(w, h)`. |
| `image_random_h_flip` | `(self, key, prob, output_key='') -> Self` |
| `image_resize` | `(self, key, w, h, output_key='') -> Self` |
| `image_resize_smallest_side` | `(self, key, size, output_key='') -> Self` — resize so the smaller side equals `size`; commonly chained with a center/random-area crop. |
| `image_rotate` | `(self, key, angle, crop=False, output_key='') -> Self` — `angle` in degrees; `crop=True` crops back to original size. |

```python
# typical eval-time image pipeline
dset = (
    dset
    .load_image("file", output_key="image")
    .image_resize_smallest_side("image", 256)
    .image_center_crop("image", 224, 224)
)
```

### I/O ops (load from disk / memory / tar)

| Method | Signature |
|---|---|
| `load_file` | `(self, key, prefix=None, output_key='') -> Self` — raw bytes read of the file named at `key`. |
| `load_numpy` | `(self, key, prefix='', from_memory=False, output_key='') -> Self` — reads a `.npy` file (or in-memory bytes if `from_memory=True`). |
| `load_image` | `(self, key, prefix='', info=False, format='RGB', from_memory=False, output_key='') -> Self` — `info=True` loads width/height instead of pixel data. |
| `load_audio` | `(self, key, prefix='', info=False, from_memory=False, info_type=LoadAudioInfo.All, sample_rate=0, resampling_quality='sinc-fastest', info_key='', output_key='') -> Self` — set `sample_rate` to resample; `info_type` can restrict to `NumFrames`/`NumChannels`/`SampleRate`/`NumSeconds`. |
| `load_video` | `(self, key, prefix='', info=False, from_memory=False, output_key='') -> Self` — `info=True` loads width/height/frame count instead of video data. |
| `read_from_tar` | `(self, tarkey, ikey, okey, prefix='', tar_prefix='', from_key=False, file_fetcher=None, nested=False, num_threads=1) -> Self` — reads whole files out of one or many tar archives (commonly staged before `load_image`/`load_video`). Indexes the tar first, so it's efficient for reading many files per archive. |

```python
# Filter out audio clips under 10 seconds before decoding the full waveform
dset = (
    dset
    .load_audio("audio_file", info=True, info_type=LoadAudioInfo.NumSeconds, output_key="audio_info")
    .sample_transform(lambda s: s if s["audio_info"] >= 10 else dict())
)
```

### Padding ops

| Method | Signature |
|---|---|
| `pad` | `(self, key, dim, lpad, rpad, pad_value, output_key='') -> Self` — pad `lpad`/`rpad` positions on either side of `dim`. |
| `pad_to_multiple` | `(self, key, dim, pad_multiple, pad_value, output_key='') -> Self` — pad the end so size along `dim` is a multiple of `pad_multiple`. |
| `pad_to_size` | `(self, key, dim, size, pad_value, output_key='') -> Self` — pad the end so size along `dim` equals exactly `size`. |

```python
dset = dset.pad("text", 0, 1, 0, ord(" "))   # prepend one space character
```

### Shape ops

| Method | Signature |
|---|---|
| `shape` | `(self, key, output_key, dim=None) -> Self` — write the shape (or just size at `dim`) of the array at `key` into `output_key`. `output_key` is required (can't silently overwrite). |
| `shard` | `(self, key, num_shards, output_key='') -> Self` — numpy-style reshape: `x.reshape(num_shards, -1, *x.shape[1:])`. |
| `squeeze` | `(self, key, dim=None, output_key='') -> Self` — drop singleton dims (all of them if `dim` omitted). |

### Tokenization op

| Method | Signature |
|---|---|
| `tokenize` | `(self, key, trie, mode=TokenizeMode.Shortest, ignore_unk=False, trie_key_scores=[], output_key='') -> Self` — see [TOKENIZING.md](TOKENIZING.md) for the full `CharTrie`/`Tokenizer` picture. |

### Conditional variants — `*_if`

Every method above has a `_if(cond: bool, *args, **kwargs)` twin, so pipelines
built from CLI/config flags can read top-to-bottom without `if/else`
branching:

```python
dset = (
    dset
    .load_image("image_file", output_key="image")
    .image_random_crop_if(enable_random_crop, "image", 256, 256)
    .image_random_h_flip_if(flip_prob > 0, "image", flip_prob)
    .key_transform_if(
        brightness_range > 0, "image",
        lambda x: ((1 + brightness_range * np.random.rand(x.shape[:2])[..., None]) * x).astype(x.dtype),
    )
)
```
