# Feature Extraction & Remote File Fetching

## Audio features: `mlx.data.features.mfsc`

A numpy-based `key_transform` function factory for mel-frequency spectral
coefficient (MFSC) features — chosen for flexibility over raw throughput
(a C++ implementation would dodge the GIL entirely, but numpy is usually fast
enough).

```python
mlx.data.features.mfsc(
    n_filterbank, sampling_freq,
    frame_size_ms=25, frame_stride_ms=10,
    pre_emphasis_coeff=0.97,
    window_type=WindowType.Hamming,
    low_freq=0, high_freq=-1,
    mel_floor=1.0,
    freq_scale=FrequencyScale.MEL,
    post_process=None,
)
```

Pipeline: sliding window over audio → pre-emphasis filter → windowing →
power spectrum → triangular filterbank → log → optional `post_process`.

**Input must be mono, 1D (`(N,)` not `(N, 1)`)** — use `squeeze()` first if
needed.

```python
from mlx.data.datasets import load_librispeech
from mlx.data.features import mfsc

dset = (
    load_librispeech()
    .squeeze("audio")
    .key_transform("audio", mfsc(80, 16000))
    .to_stream()
    .prefetch(16, 8)
    .batch(16)
    .prefetch(2, 1)
)
```

Key params: `n_filterbank` (output feature dim), `sampling_freq` (Hz),
`high_freq=-1` means `sampling_freq // 2`.

### Enums

- `mlx.data.features.WindowType` — `Hamming`, `Hanning`.
- `mlx.data.features.FrequencyScale` — `MEL`, `LOG10`, `LINEAR`.

## Remote file fetching: `FileFetcher` / `AWSFileFetcher`

Several loaders (`load_file`, `read_from_tar`, `stream_csv_reader`, etc.)
accept a `file_fetcher=` argument so files can be pulled from remote storage
with local caching + background prefetch. The built-in implementation is
`mlx.data.core.AWSFileFetcher` (only available if mlx-data was built with AWS
S3 support — see the install docs).

```python
from pathlib import Path
from mlx.data.core import AWSFileFetcher

LOCAL_CACHE = Path("/path/to/local/cache")

ff = AWSFileFetcher(
    "my-cool-bucket",
    endpoint="https://my.endpoint.com/",
    local_prefix=LOCAL_CACHE,
    num_kept_files=100,   # 0 = keep everything; set a cap if data > local disk
)

ff.fetch("my/remote/path/foo.npy")   # blocks until cached
assert (LOCAL_CACHE / "my/remote/path/foo.npy").is_file()

# Background prefetch, overlapping download with processing
ff.prefetch(["foo_1.npy", "foo_2.npy"])
ff.fetch("foo_1.npy")   # returns fast if already prefetched/cached
```

### `AWSFileFetcher.__init__` — notable parameters

`bucket`, `endpoint=''`, `region=''`, `prefix=''` (remote path prefix),
`local_prefix=''` (local cache dir), `virtual_host=False`, `verify_ssl=True`,
`connect_timeout_ms=1000`, `num_retry_max=10` (exponential backoff),
`num_connection_max=25`, `buffer_size=100MB` (fetch chunk size),
`num_threads=4` (parallel chunks per file), `num_prefetch_max=1` /
`num_prefetch_threads=1` (how many/how parallel for the prefetch queue),
`num_kept_files=0` (LRU cache cap), plus explicit credential overrides
(`access_key_id`, `secret_access_key`, `session_token`, `expiration`) if you
don't want to rely on the default AWS credential chain.

### Methods

| Method | Signature | Behavior |
|---|---|---|
| `fetch` | `(self, filename) -> FileFetcherHandle` | Ensures `filename` is cached locally — fetches, waits on an in-progress prefetch, or returns immediately if already cached. |
| `prefetch` | `(self, filenames: list[str]) -> None` | Queues background downloads; `num_prefetch_max` files download at once with `num_prefetch_threads` parallelism each. As prefetched files get `fetch()`-ed, more of the queue starts downloading. |
