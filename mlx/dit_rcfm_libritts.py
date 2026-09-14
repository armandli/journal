import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import io
    import math
    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx import data as dx
    from mlx.data.datasets.libritts_r import load_libritts_r_tarfile

    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import resample_poly
    import soundfile as sf
    import librosa

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Text-to-Speech with a Diffusion Transformer trained via OT-Rectified Conditional Flow Matching on LibriTTS-R (MLX)

    ## Research Goal

    Train a **text-conditioned Diffusion Transformer (DiT)** to synthesize
    speech mel-spectrograms from character-level text prompts using
    **rectified conditional flow matching with minibatch optimal-transport
    (OT) coupling** on the **LibriTTS-R** corpus (loaded via `mlx.data`).

    **Scope note — no voice cloning / speaker conditioning.** The model
    conditions on text only. Speaker identity is treated as an untargeted
    latent (the model picks an averaged voice-like prior); we intentionally
    do **not** embed speaker id or a reference-audio embedding into the
    transformer.

    Griffin-Lim (via `librosa.feature.inverse.mel_to_audio`) is used as a
    classical, non-neural vocoder to turn generated log-mels back into
    audible waveforms — this bounds the achievable audio quality of the
    demonstration and is by design.

    ### Notebook Outline

    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation / preprocessing (how to prep a voice dataset for TTS)
    4. Model definition
    5. Training (OT-CFM with padding masks + text-prefix conditioning)
    6. Hyperparameter search (optional, checkbox-gated)
    7. Validation
    8. Results — loss curves, Euler ODE progression, step-count MSE
    9. Save trained model
    10. Text-to-Speech inference (interactive)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 2 — Data Exploration
    """)
    return


@app.function
def load_libritts_r_at_rate(
    root: str = "../data/libritts_r",
    split: str = "dev-clean",
    target_sample_rate: int = 22050,
    quiet: bool = True,
):
    """Mirror of ``mlx.data.datasets.load_libritts_r`` that additionally
    passes a target ``sample_rate`` to the built-in resampler inside
    ``load_audio``. This avoids hand-rolled resampling and yields all
    utterances at a single fixed sample rate.
    """
    target = load_libritts_r_tarfile(root=root, split=split, quiet=quiet)
    target = str(target)
    dset = (
        dx.files_from_tar(target)
        .to_stream()
        .sample_transform(
            lambda s: s if bytes(s["file"]).endswith(b".wav") else dict()
        )
        .sample_transform(attach_transcript_file)
        .read_from_tar(target, "transcript_file", "transcript")
        .read_from_tar(target, "file", "audio")
        .load_audio("audio", from_memory=True, sample_rate=target_sample_rate)
    )
    return dset


@app.function
def attach_transcript_file(sample: dict) -> dict:
    audio_file = Path(bytes(sample["file"]).decode("utf-8"))
    transcript_file = audio_file.with_suffix(".normalized.txt")
    sample["transcript_file"] = transcript_file.as_posix().encode("utf-8")
    return sample


@app.function
def parse_speaker_from_path(file_bytes: bytes) -> str:
    """LibriTTS paths look like <speaker>/<chapter>/<speaker>_<chapter>_<utt>.wav."""
    return Path(bytes(file_bytes).decode("utf-8")).parts[0]


@app.function
def collect_libritts_samples(
    root: str,
    split: str,
    target_sample_rate: int,
    max_samples: int,
) -> list:
    """Iterate the stream and materialize up to ``max_samples`` decoded
    utterances into memory as Python dicts. Each dict carries
    ``waveform`` (1-D float32 numpy array), ``transcript`` (str),
    ``sample_rate`` (int), ``speaker`` (str), and ``file`` (str).
    """
    stream = load_libritts_r_at_rate(
        root=root, split=split, target_sample_rate=target_sample_rate
    )
    samples: list = []
    for s in stream:
        wav = np.asarray(s["audio"]).astype(np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        transcript = bytes(s["transcript"]).decode("utf-8", errors="ignore").strip()
        speaker = parse_speaker_from_path(s["file"])
        samples.append(
            {
                "waveform": wav,
                "transcript": transcript,
                "sample_rate": target_sample_rate,
                "speaker": speaker,
                "file": bytes(s["file"]).decode("utf-8", errors="ignore"),
            }
        )
        if len(samples) >= max_samples:
            break
    return samples


@app.cell
def _(mo):
    split_ui = mo.ui.dropdown(
        options=[
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ],
        value="dev-clean",
        label="LibriTTS-R split",
    )
    max_samples_ui = mo.ui.number(
        value=200, label="Max utterances to load"
    )
    target_sr_ui = mo.ui.dropdown(
        options={"16000": 16000, "22050": 22050, "24000": 24000},
        value="22050",
        label="Target sample rate (Hz)",
    )
    load_data_btn = mo.ui.run_button(label="Download + Load LibriTTS-R samples")
    mo.vstack(
        [
            mo.md(
                "Loading `dev-clean` triggers a one-time ~1.4 GB download to "
                "`../data/libritts_r/`. `max_samples` caps how many "
                "utterances are decoded and mel-spectrogrammed for this run."
            ),
            mo.hstack([split_ui, max_samples_ui, target_sr_ui]),
            load_data_btn,
        ]
    )
    return load_data_btn, max_samples_ui, split_ui, target_sr_ui


@app.cell
def _(load_data_btn, max_samples_ui, mo, split_ui, target_sr_ui):
    raw_samples = []
    if not load_data_btn.value:
        mo.output.replace(
            mo.md(
                "Click **Download + Load LibriTTS-R samples** to fetch and decode "
                "the selected split (first time will download the tarball)."
            )
        )
    else:
        raw_samples = collect_libritts_samples(
            root="../data/libritts_r",
            split=split_ui.value,
            target_sample_rate=int(target_sr_ui.value),
            max_samples=int(max_samples_ui.value),
        )
        mo.output.replace(
            mo.md(
                f"Loaded **{len(raw_samples)}** utterances from split "
                f"`{split_ui.value}` at `{target_sr_ui.value} Hz`."
            )
        )
    return (raw_samples,)


@app.function
def encode_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform.astype(np.float32), sample_rate, format="WAV")
    return buf.getvalue()


@app.function
def plot_waveform(waveform: np.ndarray, sample_rate: int, title: str = "Waveform"):
    t = np.arange(waveform.shape[0]) / float(sample_rate)
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.plot(t, waveform, lw=0.5, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def compute_log_mel(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32),
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax if fmax is not None else sample_rate / 2.0,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


@app.function
def plot_log_mel(log_mel: np.ndarray, sample_rate: int, hop_length: int, title: str = "Log-mel"):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(0.0, log_mel.shape[1] * hop_length / float(sample_rate), 0, log_mel.shape[0]),
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


@app.function
def plot_duration_histogram(durations: list):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(durations, bins=30, color="seagreen", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Utterance count")
    ax.set_title("Utterance duration distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_transcript_length_histogram(lengths: list):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(lengths, bins=30, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Transcript length (characters)")
    ax.set_ylabel("Utterance count")
    ax.set_title("Transcript character-length distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def plot_speaker_histogram(speakers: list, top_k: int = 20):
    unique, counts = np.unique(np.array(speakers), return_counts=True)
    order = np.argsort(-counts)[:top_k]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(len(order)), counts[order], color="darkorange", edgecolor="black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(unique[order], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Utterance count")
    ax.set_title(f"Top {top_k} speakers by utterance count (illustration only — NOT used for conditioning)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(mo, raw_samples):
    if not raw_samples:
        _out = mo.md("_Load the dataset above to see samples._")
    else:
        _durations = [s["waveform"].shape[0] / float(s["sample_rate"]) for s in raw_samples]
        _out = mo.md(
            f"""
            ### Dataset overview

            | Metric | Value |
            |--------|-------|
            | Utterances loaded | **{len(raw_samples):,}** |
            | Sample rate | `{raw_samples[0]["sample_rate"]} Hz` |
            | Unique speakers in this pool | `{len(set(s["speaker"] for s in raw_samples))}` |
            | Duration min / mean / max (s) | `{min(_durations):.2f}` / `{float(np.mean(_durations)):.2f}` / `{max(_durations):.2f}` |
            | Total audio (s) | `{float(np.sum(_durations)):.1f}` |
            """
        )
    _out
    return


@app.cell
def _(mo, raw_samples):
    if not raw_samples:
        _out = mo.md("_Load the dataset above to inspect example utterances._")
    else:
        _items = []
        for _i in range(min(3, len(raw_samples))):
            _s = raw_samples[_i]
            _dur = _s["waveform"].shape[0] / float(_s["sample_rate"])
            _items.append(mo.md(f"**#{_i}** — speaker `{_s['speaker']}` — duration `{_dur:.2f}s`"))
            _items.append(mo.md(f"> {_s['transcript']}"))
            _items.append(mo.audio(src=encode_wav_bytes(_s["waveform"], _s["sample_rate"])))
            _items.append(plot_waveform(_s["waveform"], _s["sample_rate"], title=f"Waveform #{_i}"))
            _items.append(
                plot_log_mel(
                    compute_log_mel(_s["waveform"], _s["sample_rate"]),
                    sample_rate=_s["sample_rate"],
                    hop_length=256,
                    title=f"Log-mel #{_i}",
                )
            )
        _out = mo.vstack(_items)
    _out
    return


@app.cell
def _(mo, raw_samples):
    if not raw_samples:
        _out = mo.md("_Load the dataset above to see histograms._")
    else:
        _durations = [s["waveform"].shape[0] / float(s["sample_rate"]) for s in raw_samples]
        _out = plot_duration_histogram(_durations)
    _out
    return


@app.cell
def _(mo, raw_samples):
    if not raw_samples:
        _out = mo.md("_Load the dataset to see transcript-length distribution._")
    else:
        _lengths = [len(s["transcript"]) for s in raw_samples]
        _out = plot_transcript_length_histogram(_lengths)
    _out
    return


@app.cell
def _(mo, raw_samples):
    if not raw_samples:
        _out = mo.md("_Load the dataset to see the speaker histogram._")
    else:
        _speakers = [s["speaker"] for s in raw_samples]
        _out = plot_speaker_histogram(_speakers)
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 3 — Dataset Creation / Preprocessing (a general voice-dataset TTS pipeline)

    A typical text-to-speech preprocessing pipeline — not specific to
    LibriTTS-R — has these stages, all of which we implement here:

    1. **Resampling** to a single fixed sample rate (`22050 Hz` here).
       Model expects consistent time resolution; the source corpus may
       mix rates. We use `mlx.data`'s built-in `load_audio(sample_rate=...)`
       resampler.
    2. **Mel-spectrogram extraction (log-mel)**, not raw waveform, because
       (a) it is a compact perceptually-motivated representation
       (`n_mels=80` bins vs. thousands of samples per second), (b) it
       decouples the *content* model (this DiT) from the *waveform*
       generation problem (handled by a vocoder — Griffin-Lim here),
       (c) log-compressing the mel power (`power_to_db`) matches the
       roughly-Gaussian statistics that a diffusion/flow-matching model
       expects.
    3. **Text normalization & tokenization** — LibriTTS-R already ships
       *normalized* transcripts (digits spelled out, common
       abbreviations expanded, most punctuation removed). We tokenize
       character-by-character with a small self-contained vocabulary
       (a-z, space, apostrophe, a few punctuation marks, `PAD`, `EOS`).
    4. **Length handling — padding + masking**. Utterances have variable
       duration. We fix a `max_mel_frames` and `max_text_len` for
       batching; shorter samples are right-padded with zeros and a
       boolean valid-mask is carried alongside. The attention layers
       and the flow-matching loss both consume this mask so that padded
       positions are ignored.
    5. **Global mean/std normalization of log-mel** so the training
       target has roughly zero mean and unit variance, matching the
       Gaussian prior `x_0 ~ N(0, I)`. The inverse transform is applied
       before Griffin-Lim vocoding at inference.
    """)
    return


@app.function
def build_char_vocab(transcripts: list) -> dict:
    """PAD is 0 (also used as attention-padding sentinel), EOS is 1, then chars sorted."""
    chars = set()
    for t in transcripts:
        chars.update(t.lower())
    charset = sorted(chars)
    vocab = {"<pad>": 0, "<eos>": 1}
    for c in charset:
        vocab[c] = len(vocab)
    return vocab


@app.function
def text_to_ids(text: str, vocab: dict, max_len: int) -> tuple:
    lowered = text.lower()
    ids = [vocab.get(c, 0) for c in lowered]
    ids = ids[: max_len - 1] + [vocab["<eos>"]]
    length = len(ids)
    if length < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - length)
    return np.asarray(ids, dtype=np.int32), length


@app.function
def compute_mel_bank(
    samples: list,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> list:
    mels: list = []
    for s in samples:
        m = compute_log_mel(
            s["waveform"],
            sample_rate=s["sample_rate"],
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        mels.append(m)
    return mels


@app.function
def compute_mel_stats(mels: list) -> tuple:
    concat = np.concatenate([m.reshape(-1) for m in mels], axis=0)
    mean = float(concat.mean())
    std = float(concat.std() + 1e-6)
    return mean, std


@app.function
def round_up_to_multiple(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return value + (multiple - value % multiple)


@app.function
def build_tensor_dataset(
    samples: list,
    vocab: dict,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    win_length: int,
    max_mel_frames: int,
    max_text_len: int,
    patch_size: int,
) -> dict:
    """Return dict with keys: ``mel`` (B, T, M), ``mel_mask`` (B, T),
    ``text_ids`` (B, L), ``text_mask`` (B, L), plus normalization
    stats and helpers. ``max_mel_frames`` is rounded up to a multiple
    of ``patch_size`` so patchify is exact."""
    max_frames_pad = round_up_to_multiple(max_mel_frames, patch_size)
    raw_mels = compute_mel_bank(
        samples,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    mel_mean, mel_std = compute_mel_stats(raw_mels)
    n = len(samples)
    mel_arr = np.zeros((n, max_frames_pad, n_mels), dtype=np.float32)
    mel_mask = np.zeros((n, max_frames_pad), dtype=np.float32)
    text_ids = np.zeros((n, max_text_len), dtype=np.int32)
    text_mask = np.zeros((n, max_text_len), dtype=np.float32)
    kept = 0
    for i, s in enumerate(samples):
        m = raw_mels[i]
        num_frames = min(m.shape[1], max_frames_pad)
        m_use = ((m[:, :num_frames] - mel_mean) / mel_std).T
        mel_arr[i, :num_frames, :] = m_use
        mel_mask[i, :num_frames] = 1.0
        ids, length = text_to_ids(s["transcript"], vocab, max_text_len)
        text_ids[i] = ids
        text_mask[i, :length] = 1.0
        kept += 1
    return {
        "mel": mel_arr[:kept],
        "mel_mask": mel_mask[:kept],
        "text_ids": text_ids[:kept],
        "text_mask": text_mask[:kept],
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "max_mel_frames": max_frames_pad,
        "max_text_len": max_text_len,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "win_length": win_length,
    }


@app.function
def split_tensor_dataset(dataset: dict, val_fraction: float = 0.1, seed: int = 42) -> tuple:
    n = dataset["mel"].shape[0]
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    def _slice(idx: np.ndarray) -> dict:
        return {
            "mel": mx.array(dataset["mel"][idx]),
            "mel_mask": mx.array(dataset["mel_mask"][idx]),
            "text_ids": mx.array(dataset["text_ids"][idx]),
            "text_mask": mx.array(dataset["text_mask"][idx]),
        }

    return _slice(tr_idx), _slice(val_idx)


@app.function
def iter_batches(split: dict, batch_size: int = 8, shuffle: bool = True) -> list:
    n = split["mel"].shape[0]
    if shuffle:
        idx_np = np.random.permutation(n).astype(np.int32)
    else:
        idx_np = np.arange(n, dtype=np.int32)
    batches: list = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_idx = mx.array(idx_np[start:end])
        batches.append(
            {
                "mel": split["mel"][b_idx],
                "mel_mask": split["mel_mask"][b_idx],
                "text_ids": split["text_ids"][b_idx],
                "text_mask": split["text_mask"][b_idx],
            }
        )
    return batches


@app.cell
def _(mo):
    n_mels_ui = mo.ui.dropdown(
        options={"40": 40, "64": 64, "80": 80, "128": 128},
        value="80",
        label="n_mels",
    )
    hop_length_ui = mo.ui.dropdown(
        options={"128": 128, "256": 256, "512": 512},
        value="256",
        label="hop_length",
    )
    n_fft_ui = mo.ui.dropdown(
        options={"512": 512, "1024": 1024, "2048": 2048},
        value="1024",
        label="n_fft / win_length",
    )
    max_mel_frames_ui = mo.ui.slider(
        64, 1024, value=384, step=32, label="max_mel_frames"
    )
    max_text_len_ui = mo.ui.slider(
        32, 512, value=192, step=16, label="max_text_len"
    )
    patch_size_ui = mo.ui.dropdown(
        options={"1": 1, "2": 2, "4": 4, "8": 8},
        value="4",
        label="patch_size (mel frames per token)",
    )
    build_ds_btn = mo.ui.run_button(label="Build tensor dataset")
    mo.vstack(
        [
            mo.md("### Preprocessing hyperparameters"),
            mo.hstack([n_mels_ui, hop_length_ui, n_fft_ui]),
            mo.hstack([max_mel_frames_ui, max_text_len_ui, patch_size_ui]),
            build_ds_btn,
        ]
    )
    return (
        build_ds_btn,
        hop_length_ui,
        max_mel_frames_ui,
        max_text_len_ui,
        n_fft_ui,
        n_mels_ui,
        patch_size_ui,
    )


@app.cell
def _(
    build_ds_btn,
    hop_length_ui,
    max_mel_frames_ui,
    max_text_len_ui,
    mo,
    n_fft_ui,
    n_mels_ui,
    patch_size_ui,
    raw_samples,
):
    tensor_dataset = None
    char_vocab = None
    if not raw_samples:
        mo.output.replace(mo.md("_Load LibriTTS-R samples first (Section 2)._"))
    elif not build_ds_btn.value:
        mo.output.replace(
            mo.md("Click **Build tensor dataset** to run the preprocessing pipeline.")
        )
    else:
        char_vocab = build_char_vocab([s["transcript"] for s in raw_samples])
        tensor_dataset = build_tensor_dataset(
            samples=raw_samples,
            vocab=char_vocab,
            n_mels=int(n_mels_ui.value),
            hop_length=int(hop_length_ui.value),
            n_fft=int(n_fft_ui.value),
            win_length=int(n_fft_ui.value),
            max_mel_frames=int(max_mel_frames_ui.value),
            max_text_len=int(max_text_len_ui.value),
            patch_size=int(patch_size_ui.value),
        )
        mo.output.replace(
            mo.md(
                f"""
                Built tensor dataset:
                - `mel` shape: `{tensor_dataset["mel"].shape}` (dtype `float32`)
                - `mel_mask` shape: `{tensor_dataset["mel_mask"].shape}`
                - `text_ids` shape: `{tensor_dataset["text_ids"].shape}` (dtype `int32`)
                - `text_mask` shape: `{tensor_dataset["text_mask"].shape}`
                - Vocab size: `{len(char_vocab)}` (PAD=0, EOS=1)
                - Global log-mel mean/std: `{tensor_dataset["mel_mean"]:.3f}` / `{tensor_dataset["mel_std"]:.3f}`
                """
            )
        )
    return char_vocab, tensor_dataset


@app.cell
def _(mo, tensor_dataset):
    if tensor_dataset is None:
        _out = mo.md("_Build the tensor dataset above first to see a batch check._")
    else:
        _tr, _va = split_tensor_dataset(tensor_dataset, val_fraction=0.15)
        _sample_batches = iter_batches(_tr, batch_size=4, shuffle=True)
        _b = _sample_batches[0]
        _out = mo.md(
            f"""
            ### One-batch shape check (batch_size=4)

            | Tensor | Shape | dtype |
            |--------|-------|-------|
            | `mel` | `{tuple(_b["mel"].shape)}` | `{_b["mel"].dtype}` |
            | `mel_mask` | `{tuple(_b["mel_mask"].shape)}` | `{_b["mel_mask"].dtype}` |
            | `text_ids` | `{tuple(_b["text_ids"].shape)}` | `{_b["text_ids"].dtype}` |
            | `text_mask` | `{tuple(_b["text_mask"].shape)}` | `{_b["text_mask"].dtype}` |

            Total training utterances: `{_tr["mel"].shape[0]}`, validation: `{_va["mel"].shape[0]}`.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition

    A **text-conditioned Diffusion Transformer**. Text tokens (character
    embeddings + learned positional embedding) are prepended as a
    context prefix to the noised mel-frame patch tokens; the whole
    concatenated sequence is processed by a stack of `DiTBlockV1`s with
    AdaLN gating on the diffusion timestep. This "in-context
    conditioning" avoids adding a separate cross-attention module and
    lets the same self-attention layer learn text-audio alignment. The
    attention layer receives an additive key-padding mask so padded
    text and padded mel positions cannot leak into other positions.
    """)
    return


@app.class_definition
class SinusoidalTimestepEmbeddingV1(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # Leading underscore excludes this fixed buffer from
        # Module.valid_parameter_filter, so the optimizer never trains
        # or weight-decays the sinusoidal frequency bank.
        self._freqs = mx.exp(
            -math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / max(half, 1)
        )

    def __call__(self, t: mx.array) -> mx.array:
        return mx.concatenate(
            [
                mx.sin(t[:, None] * self._freqs[None, :]),
                mx.cos(t[:, None] * self._freqs[None, :]),
            ],
            axis=-1,
        )


@app.class_definition
class AdaptiveLayerNormV1(nn.Module):
    """adaLN-Zero (Peebles & Xie, DiT). Predicts scale/shift for the
    norm plus a residual-branch gate, with the projection zero-initialized
    so every block starts as an identity function. This is what lets a
    deep DiT stack train stably instead of the residual variance
    compounding layer over layer from random modulation at step zero."""

    def __init__(self, dim: int = 256, cond_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.proj = nn.Linear(cond_dim, 3 * dim)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.proj.bias = mx.zeros_like(self.proj.bias)

    def __call__(self, x: mx.array, cond: mx.array) -> tuple:
        scale, shift, gate = mx.split(self.proj(nn.silu(cond))[:, None, :], 3, axis=-1)
        return self.norm(x) * (1.0 + scale) + shift, gate


@app.class_definition
class MelPatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 4, n_mels: int = 80, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.n_mels = n_mels
        self.proj = nn.Linear(patch_size * n_mels, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, m = x.shape
        n_tokens = t // self.patch_size
        grouped = x.reshape(b, n_tokens, self.patch_size * m)
        return self.proj(grouped)


@app.class_definition
class MelUnpatchifyV1(nn.Module):
    def __init__(self, patch_size: int = 4, n_mels: int = 80, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.n_mels = n_mels
        self.proj = nn.Linear(embed_dim, patch_size * n_mels)

    def __call__(self, x: mx.array) -> mx.array:
        b, n_tokens, _ = x.shape
        raw = self.proj(x)
        return raw.reshape(b, n_tokens * self.patch_size, self.n_mels)


@app.class_definition
class DiTBlockV1(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_dim: int = 512, cond_dim: int = 256):
        super().__init__()
        self.attn_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.attn = nn.MultiHeadAttention(dim, num_heads)
        self.mlp_norm = AdaptiveLayerNormV1(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def __call__(self, x: mx.array, cond: mx.array, attn_mask: mx.array | None = None) -> mx.array:
        h, gate1 = self.attn_norm(x, cond)
        x = x + gate1 * self.attn(h, h, h, mask=attn_mask)
        h2, gate2 = self.mlp_norm(x, cond)
        return x + gate2 * self.mlp(h2)


@app.class_definition
class TextEncoderV1(nn.Module):
    def __init__(
        self,
        vocab_size: int = 40,
        embed_dim: int = 256,
        max_text_len: int = 192,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = mx.zeros((1, max_text_len, embed_dim))

    def __call__(self, text_ids: mx.array) -> mx.array:
        return self.token_embed(text_ids) + self.pos_embed[:, : text_ids.shape[1], :]


@app.class_definition
class TextToSpeechDiTV1(nn.Module):
    def __init__(
        self,
        vocab_size: int = 40,
        n_mels: int = 80,
        patch_size: int = 4,
        max_mel_frames: int = 384,
        max_text_len: int = 192,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        num_layers: int = 6,
    ):
        super().__init__()
        assert max_mel_frames % patch_size == 0, "max_mel_frames must be divisible by patch_size"
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.max_mel_frames = max_mel_frames
        self.max_text_len = max_text_len
        self.num_mel_tokens = max_mel_frames // patch_size
        self.embed_dim = embed_dim
        self.text_encoder = TextEncoderV1(vocab_size, embed_dim, max_text_len)
        self.patchify = MelPatchifyV1(patch_size, n_mels, embed_dim)
        self.mel_pos_embed = mx.zeros((1, self.num_mel_tokens, embed_dim))
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = [
            DiTBlockV1(embed_dim, num_heads, mlp_dim, embed_dim)
            for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatchify = MelUnpatchifyV1(patch_size, n_mels, embed_dim)

    def __call__(
        self,
        x_t: mx.array,
        t: mx.array,
        text_ids: mx.array,
        text_mask: mx.array,
        mel_mask: mx.array,
    ) -> mx.array:
        text_emb = self.text_encoder(text_ids)
        mel_tokens = self.patchify(x_t) + self.mel_pos_embed
        h = mx.concatenate([text_emb, mel_tokens], axis=1)
        cond = self.time_embed(t)
        attn_mask = build_attention_mask(text_mask, mel_mask, self.patch_size)
        for block in self.blocks:
            h = block(h, cond, attn_mask=attn_mask)
        h = self.final_norm(h)
        audio_h = h[:, text_emb.shape[1]:, :]
        return self.unpatchify(audio_h)


@app.function
def build_attention_mask(text_mask: mx.array, mel_mask: mx.array, patch_size: int) -> mx.array:
    """Build an additive attention mask (0 for keep, large-negative for
    ignore) with shape ``(B, 1, 1, L_total)`` where
    ``L_total = L_text + L_mel_tokens``. Broadcasts over heads and
    query positions.
    """
    b, t_frames = mel_mask.shape
    n_tokens = t_frames // patch_size
    mel_token_mask = mel_mask.reshape(b, n_tokens, patch_size).max(axis=-1)
    full = mx.concatenate([text_mask, mel_token_mask], axis=1)
    additive = (1.0 - full) * -1.0e9
    return additive[:, None, None, :]


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def build_tts_dit_model(
    vocab_size: int,
    n_mels: int,
    patch_size: int,
    max_mel_frames: int,
    max_text_len: int,
    embed_dim: int = 256,
    num_heads: int = 8,
    mlp_dim: int = 512,
    num_layers: int = 6,
) -> TextToSpeechDiTV1:
    model = TextToSpeechDiTV1(
        vocab_size=vocab_size,
        n_mels=n_mels,
        patch_size=patch_size,
        max_mel_frames=max_mel_frames,
        max_text_len=max_text_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
    )
    mx.eval(model.parameters())
    return model


@app.cell
def _(mo):
    mo.md("""
    ### Architecture Table

    | Component | Module | Shape / role |
    |-----------|--------|---------------|
    | Text token embedding | `TextEncoderV1` (`nn.Embedding` + learned pos) | `(B, L_text, D)` |
    | Mel patch embedding | `MelPatchifyV1` (`Linear(patch*M, D)`) + learned pos | `(B, T/patch, D)` |
    | Timestep embedding | `SinusoidalTimestepEmbeddingV1` + 2-layer MLP | `(B, D)` (used for AdaLN only) |
    | Backbone | `DiTBlockV1` × `num_layers` (AdaLN + masked self-attn + AdaLN + MLP) | `(B, L_text + T/patch, D)` |
    | Head | `LayerNorm` + `MelUnpatchifyV1` (`Linear(D, patch*M)`) | `(B, T, M)` |

    **Key differences vs. the CIFAR-10 DiT** (`mlx/dit_rcfm_cifar10.py`):

    - 1-D **audio** patchify (`MelPatchifyV1`) grouping `patch_size`
      consecutive mel frames into one token, instead of 2-D image patches.
    - **Text prefix conditioning** via `TextEncoderV1` concatenation +
      full self-attention (no cross-attention module).
    - AdaLN conditions on **timestep only** — the class-label conditioning
      from the CIFAR notebook is removed.
    - **Additive attention mask** built from combined text-validity and
      mel-frame-validity masks and applied inside every DiT block, so
      padded positions never affect valid positions.
    """)
    return


@app.cell
def _(char_vocab, mo, patch_size_ui, tensor_dataset):
    if tensor_dataset is None or char_vocab is None:
        _out = mo.md("_Build the tensor dataset above to instantiate the default model._")
    else:
        _m = build_tts_dit_model(
            vocab_size=len(char_vocab),
            n_mels=tensor_dataset["n_mels"],
            patch_size=int(patch_size_ui.value),
            max_mel_frames=tensor_dataset["mel"].shape[1],
            max_text_len=tensor_dataset["text_ids"].shape[1],
        )
        _out = mo.md(f"**Default model parameter count**: `{count_parameters(_m):,}`.")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training (OT-CFM with padding masks)
    """)
    return


@app.function
def sample_noise_like_mel(mel: mx.array) -> mx.array:
    return mx.random.normal(shape=mel.shape)


@app.function
def minibatch_ot_pairing(x0: mx.array, x1: mx.array, weight_mask: mx.array) -> mx.array:
    """Solve exact minibatch OT between ``x0`` and ``x1`` under a
    squared-L2 cost on the **masked** flattened representation, and
    return ``x0`` reordered to be optimally coupled with ``x1``. The
    ``weight_mask`` is broadcast (B, T, 1) so padded frames contribute
    zero to the cost.
    """
    x0_np = np.array(x0)
    x1_np = np.array(x1)
    w = np.array(weight_mask)[:, :, None]
    x0_flat = (x0_np * w).reshape(x0_np.shape[0], -1).astype(np.float64)
    x1_flat = (x1_np * w).reshape(x1_np.shape[0], -1).astype(np.float64)
    diffs = x0_flat[:, None, :] - x1_flat[None, :, :]
    cost = np.sum(diffs * diffs, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.argsort(col_ind).astype(np.int32)
    return x0[mx.array(perm)]


@app.function
def compute_flow_loss_masked(
    model: nn.Module,
    x1: mx.array,
    x0: mx.array,
    text_ids: mx.array,
    text_mask: mx.array,
    mel_mask: mx.array,
) -> mx.array:
    t = mx.random.uniform(shape=(x1.shape[0],))
    t_view = t.reshape(-1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0
    v_pred = model(x_t, t, text_ids, text_mask, mel_mask)
    mask3 = mel_mask[:, :, None]
    sq = ((v_pred - v_target) ** 2) * mask3
    denom = mx.maximum(mx.sum(mask3) * float(v_pred.shape[-1]), mx.array(1.0))
    return mx.sum(sq) / denom


@app.function
def run_train_epoch_ot_cfm_tts(model: nn.Module, optimizer, batches: list) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss_masked)
    epoch_loss = 0.0
    n = 0
    for b in batches:
        x0 = sample_noise_like_mel(b["mel"])
        x0_paired = minibatch_ot_pairing(x0, b["mel"], b["mel_mask"])
        loss, grads = loss_and_grad_fn(
            model,
            b["mel"],
            x0_paired,
            b["text_ids"],
            b["text_mask"],
            b["mel_mask"],
        )
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += float(loss.item())
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def run_train_epoch_cfm_tts(model: nn.Module, optimizer, batches: list) -> float:
    """No-OT variant (used only for hyperparameter search speed)."""
    loss_and_grad_fn = nn.value_and_grad(model, compute_flow_loss_masked)
    epoch_loss = 0.0
    n = 0
    for b in batches:
        x0 = sample_noise_like_mel(b["mel"])
        loss, grads = loss_and_grad_fn(
            model,
            b["mel"],
            x0,
            b["text_ids"],
            b["text_mask"],
            b["mel_mask"],
        )
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += float(loss.item())
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def evaluate_tts_model(model: nn.Module, batches: list) -> float:
    total = 0.0
    n = 0
    for b in batches:
        x0 = sample_noise_like_mel(b["mel"])
        loss = compute_flow_loss_masked(
            model, b["mel"], x0, b["text_ids"], b["text_mask"], b["mel_mask"]
        )
        mx.eval(loss)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@app.function
def euler_solve_tts(
    model: nn.Module,
    x0: mx.array,
    text_ids: mx.array,
    text_mask: mx.array,
    mel_mask: mx.array,
    num_steps: int = 40,
) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = model(x, t, text_ids, text_mask, mel_mask)
        x = x + dt * v
        mx.eval(x)
    return x


@app.function
def euler_solve_trajectory_tts(
    model: nn.Module,
    x0: mx.array,
    text_ids: mx.array,
    text_mask: mx.array,
    mel_mask: mx.array,
    num_steps: int = 40,
) -> list:
    dt = 1.0 / num_steps
    x = x0
    trajectory = [x]
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = model(x, t, text_ids, text_mask, mel_mask)
        x = x + dt * v
        mx.eval(x)
        trajectory.append(x)
    return trajectory


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3},
        value="3e-4",
        label="Learning rate",
    )
    bs_ui = mo.ui.dropdown(
        options={"4": 4, "8": 8, "16": 16, "32": 32},
        value="8",
        label="Batch size",
    )
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-4": 1e-4, "1e-3": 1e-3},
        value="1e-4",
        label="Weight decay",
    )
    epochs_ui = mo.ui.slider(1, 200, value=15, step=1, label="Epochs")
    embed_dim_ui = mo.ui.dropdown(
        options={"128": 128, "192": 192, "256": 256, "384": 384},
        value="192",
        label="embed_dim",
    )
    num_heads_ui = mo.ui.dropdown(
        options={"4": 4, "6": 6, "8": 8},
        value="6",
        label="num_heads",
    )
    mlp_dim_ui = mo.ui.dropdown(
        options={"256": 256, "384": 384, "512": 512, "768": 768},
        value="384",
        label="mlp_dim",
    )
    num_layers_ui = mo.ui.slider(1, 12, value=4, step=1, label="num_layers")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md("### Training hyperparameters"),
            mo.hstack([lr_ui, bs_ui, wd_ui, epochs_ui]),
            mo.hstack([embed_dim_ui, num_heads_ui, mlp_dim_ui, num_layers_ui]),
            train_btn,
        ]
    )
    return (
        bs_ui,
        embed_dim_ui,
        epochs_ui,
        lr_ui,
        mlp_dim_ui,
        num_heads_ui,
        num_layers_ui,
        train_btn,
        wd_ui,
    )


@app.function
def train_tts_model(
    train_split: dict,
    val_split: dict,
    vocab_size: int,
    n_mels: int,
    patch_size: int,
    max_mel_frames: int,
    max_text_len: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    num_layers: int,
    lr: float,
    wd: float,
    batch_size: int,
    epochs: int,
    epoch_fn=None,
    progress_cb=None,
) -> tuple:
    if epoch_fn is None:
        epoch_fn = run_train_epoch_ot_cfm_tts
    model = build_tts_dit_model(
        vocab_size=vocab_size,
        n_mels=n_mels,
        patch_size=patch_size,
        max_mel_frames=max_mel_frames,
        max_text_len=max_text_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
    )
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=wd)
    val_batches = iter_batches(val_split, batch_size=batch_size, shuffle=False)
    train_losses: list = []
    val_losses: list = []
    for epoch in range(epochs):
        train_batches = iter_batches(train_split, batch_size=batch_size, shuffle=True)
        tl = epoch_fn(model, optimizer, train_batches)
        vl = evaluate_tts_model(model, val_batches)
        train_losses.append(tl)
        val_losses.append(vl)
        if progress_cb is not None:
            progress_cb(epoch, epochs, tl, vl)
    return model, train_losses, val_losses


@app.cell
def _(
    bs_ui,
    char_vocab,
    embed_dim_ui,
    epochs_ui,
    lr_ui,
    mlp_dim_ui,
    mo,
    num_heads_ui,
    num_layers_ui,
    patch_size_ui,
    tensor_dataset,
    train_btn,
    wd_ui,
):
    train_losses = []
    val_losses = []
    trained_model = None
    train_split_gl = None
    val_split_gl = None
    if tensor_dataset is None or char_vocab is None:
        mo.output.replace(mo.md("_Build the tensor dataset first (Section 3)._"))
    elif not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to start OT-CFM training."))
    else:
        train_split_gl, val_split_gl = split_tensor_dataset(tensor_dataset, val_fraction=0.15)
        def _cb(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )
        trained_model, train_losses, val_losses = train_tts_model(
            train_split=train_split_gl,
            val_split=val_split_gl,
            vocab_size=len(char_vocab),
            n_mels=tensor_dataset["n_mels"],
            patch_size=int(patch_size_ui.value),
            max_mel_frames=tensor_dataset["mel"].shape[1],
            max_text_len=tensor_dataset["text_ids"].shape[1],
            embed_dim=int(embed_dim_ui.value),
            num_heads=int(num_heads_ui.value),
            mlp_dim=int(mlp_dim_ui.value),
            num_layers=int(num_layers_ui.value),
            lr=float(lr_ui.value),
            wd=float(wd_ui.value),
            batch_size=int(bs_ui.value),
            epochs=int(epochs_ui.value),
            epoch_fn=run_train_epoch_ot_cfm_tts,
            progress_cb=_cb,
        )
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train `{train_losses[-1]:.4f}` | "
                f"val `{val_losses[-1]:.4f}`. Params: `{count_parameters(trained_model):,}`."
            )
        )
    return (
        train_losses,
        train_split_gl,
        trained_model,
        val_losses,
        val_split_gl,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (optional)

    Small grid over learning rate and patch size, using **vanilla CFM
    without OT coupling** (faster per step). Only a few epochs each.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_search_cb
    return (hp_search_cb,)


@app.function
def run_hp_config_tts(
    train_split: dict,
    val_split: dict,
    vocab_size: int,
    n_mels: int,
    patch_size: int,
    max_mel_frames: int,
    max_text_len: int,
    lr: float,
    n_epochs: int,
    batch_size: int,
) -> float:
    model, _, val_losses = train_tts_model(
        train_split=train_split,
        val_split=val_split,
        vocab_size=vocab_size,
        n_mels=n_mels,
        patch_size=patch_size,
        max_mel_frames=max_mel_frames,
        max_text_len=max_text_len,
        embed_dim=192,
        num_heads=6,
        mlp_dim=384,
        num_layers=3,
        lr=lr,
        wd=1e-4,
        batch_size=batch_size,
        epochs=n_epochs,
        epoch_fn=run_train_epoch_cfm_tts,
        progress_cb=None,
    )
    _ = model
    return val_losses[-1] if val_losses else float("inf")


@app.cell
def _(char_vocab, hp_search_cb, mo, tensor_dataset):
    mo.stop(
        not hp_search_cb.value,
        mo.md("_Enable hyperparameter search above to run this section._"),
    )
    mo.stop(
        tensor_dataset is None or char_vocab is None,
        mo.md("_Build the tensor dataset first before searching._"),
    )
    _tr, _va = split_tensor_dataset(tensor_dataset, val_fraction=0.2)
    _search_space = {"lr": [1e-4, 3e-4], "patch_size": [2, 4]}
    _hp_epochs = 3
    _hp_bs = 8
    hp_results = []
    for _lr in _search_space["lr"]:
        for _ps in _search_space["patch_size"]:
            _T = round_up_to_multiple(tensor_dataset["mel"].shape[1], _ps)
            _vl = run_hp_config_tts(
                _tr,
                _va,
                vocab_size=len(char_vocab),
                n_mels=tensor_dataset["n_mels"],
                patch_size=_ps,
                max_mel_frames=_T,
                max_text_len=tensor_dataset["text_ids"].shape[1],
                lr=_lr,
                n_epochs=_hp_epochs,
                batch_size=_hp_bs,
            )
            hp_results.append({"lr": _lr, "patch_size": _ps, "val_loss": round(_vl, 4)})
            mo.output.replace(mo.md(f"lr={_lr}, patch_size={_ps} -> val={_vl:.4f}"))
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation

    We evaluate the trained model on the held-out validation split. **k-fold
    cross-validation is intentionally omitted** here: the entire loaded
    utterance pool is intentionally small (default ~200 utterances, capped
    by `max_samples`) to keep the demo tractable on Apple Silicon, and
    running 5 folds × N epochs of DiT training on audio would dominate the
    wall-clock budget of this notebook. Instead, we report the single
    held-out flow-matching loss as the primary generalization signal and
    complement it with the qualitative results in Section 8.
    """)
    return


@app.cell
def _(bs_ui, mo, trained_model, val_split_gl):
    if trained_model is None or val_split_gl is None:
        _out = mo.md("_Train the model first (Section 5)._")
    else:
        _val_batches = iter_batches(val_split_gl, batch_size=int(bs_ui.value), shuffle=False)
        val_flow_loss = evaluate_tts_model(trained_model, _val_batches)
        _out = mo.md(f"**Validation flow-matching loss (masked)**: `{val_flow_loss:.4f}`")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Results
    """)
    return


@app.function
def plot_loss_curves(train_losses: list, val_losses: list):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if train_losses:
        ax.plot(range(1, len(train_losses) + 1), train_losses, "b-o", lw=2, ms=4, label="Train")
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, "r-s", lw=2, ms=4, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked flow-matching MSE")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_losses, val_losses):
    if not train_losses:
        _out = mo.md("_Train the model first (Section 5)._")
    else:
        _out = plot_loss_curves(train_losses, val_losses)
    _out
    return


@app.function
def plot_euler_progression_tts(
    trajectory: list,
    valid_frames: int,
    frame_ts: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    title: str = "Euler ODE generation progression",
):
    num_steps = len(trajectory) - 1
    frame_steps = [int(round(t * num_steps)) for t in frame_ts]
    cols = len(frame_steps)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.6, 2.6))
    if cols == 1:
        axes = [axes]
    for c, step in enumerate(frame_steps):
        mel = np.array(trajectory[step][0])[:valid_frames, :].T
        axes[c].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        axes[c].set_title(f"t={step / num_steps:.2f}", fontsize=9)
        axes[c].set_xticks([])
        axes[c].set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, tensor_dataset, train_split_gl, trained_model):
    if trained_model is None or train_split_gl is None or tensor_dataset is None:
        _out = mo.md("_Train the model first to visualize its ODE process._")
    else:
        _b = iter_batches(train_split_gl, batch_size=1, shuffle=False)[0]
        mx.random.seed(11)
        _x0 = mx.random.normal(shape=_b["mel"].shape)
        _traj = euler_solve_trajectory_tts(
            trained_model,
            _x0,
            _b["text_ids"],
            _b["text_mask"],
            _b["mel_mask"],
            num_steps=40,
        )
        _valid = int(np.array(_b["mel_mask"][0]).sum())
        _out = plot_euler_progression_tts(_traj, valid_frames=_valid, title="Flow-matching ODE process (noise -> mel)")
    _out
    return


@app.function
def compute_step_count_mse_tts(
    model: nn.Module,
    text_ids: mx.array,
    text_mask: mx.array,
    mel_mask: mx.array,
    step_list: list,
    ref_steps: int = 100,
    seed: int = 7,
) -> list:
    mx.random.seed(seed)
    b, t_frames = mel_mask.shape
    # Infer n_mels from model
    n_mels = model.n_mels
    x0 = mx.random.normal(shape=(b, t_frames, n_mels))
    x_ref = euler_solve_tts(model, x0, text_ids, text_mask, mel_mask, ref_steps)
    mx.eval(x_ref)
    mses: list = []
    for ns in step_list:
        x_ns = euler_solve_tts(model, x0, text_ids, text_mask, mel_mask, ns)
        mx.eval(x_ns)
        m3 = mel_mask[:, :, None]
        diff = (x_ns - x_ref) * m3
        denom = mx.maximum(mx.sum(m3), mx.array(1.0))
        mse = mx.sum(diff * diff) / denom
        mses.append(float(mse.item()))
    return mses


@app.function
def plot_step_count_mse(step_list: list, mses: list):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(step_list, mses, "b-o", lw=2, ms=6, label="OT-CFM DiT")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Euler steps")
    ax.set_ylabel("MSE vs high-step reference (masked)")
    ax.set_title("Quality vs. number of ODE integration steps — flatter is straighter")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_split_gl, trained_model):
    if trained_model is None or train_split_gl is None:
        _out = mo.md("_Train the model first to run the step-count MSE experiment._")
    else:
        _b = iter_batches(train_split_gl, batch_size=2, shuffle=False)[0]
        _steps = [1, 2, 5, 10, 25, 50]
        _mses = compute_step_count_mse_tts(
            trained_model, _b["text_ids"], _b["text_mask"], _b["mel_mask"], _steps, ref_steps=100
        )
        _out = plot_step_count_mse(_steps, _mses)
    _out
    return


@app.function
def plot_reconstruction_comparison(
    mel_gt: np.ndarray,
    mel_pred: np.ndarray,
    valid_frames: int,
    title: str = "Ground truth vs. generated mel",
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].imshow(mel_gt[:valid_frames, :].T, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Ground truth (normalized log-mel)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(mel_pred[:valid_frames, :].T, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Generated (normalized log-mel)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, train_split_gl, trained_model):
    if trained_model is None or train_split_gl is None:
        _out = mo.md("_Train the model first to see a reconstruction comparison._")
    else:
        _b = iter_batches(train_split_gl, batch_size=1, shuffle=False)[0]
        mx.random.seed(42)
        _x0 = mx.random.normal(shape=_b["mel"].shape)
        _x_gen = euler_solve_tts(
            trained_model, _x0, _b["text_ids"], _b["text_mask"], _b["mel_mask"], num_steps=40
        )
        mx.eval(_x_gen)
        _valid = int(np.array(_b["mel_mask"][0]).sum())
        _out = plot_reconstruction_comparison(
            np.array(_b["mel"][0]),
            np.array(_x_gen[0]),
            valid_frames=_valid,
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ### Results summary

    - **Training curve**: shows the masked flow-matching MSE decreasing
      on train/val across epochs.
    - **ODE progression**: strip of intermediate mel-spectrograms as the
      Euler ODE integrates the learned velocity field from Gaussian
      noise (`t=0`) to a generated mel (`t=1`) conditioned on a real
      transcript.
    - **Step-count MSE**: how close the Euler solution at `k` steps
      gets to a 100-step reference for the same `(x_0, text)`. Because
      this DiT was trained with OT coupling, its trajectories should be
      relatively straight, i.e. small-`k` solutions should be close to
      the reference.
    - **Reconstruction comparison**: ground-truth mel next to a mel
      generated from the same transcript. At the tiny default training
      budget the generated mel captures coarse energy/time structure
      but not fine-grained voicing — expected for a demonstration-scale
      run on a small subset of the corpus.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 9 — Save Trained Model
    """)
    return


@app.cell
def _(mo):
    save_filename_ui = mo.ui.text(
        value="libritts_dit_rcfm_v1.safetensors",
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_model_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_model_btn])
    return save_filename_ui, save_model_btn


@app.cell
def _(mo, save_filename_ui, save_model_btn, trained_model):
    if trained_model is None:
        _out = mo.md("_Train the model first (Section 5) before saving._")
    elif not save_model_btn.value:
        _out = mo.md(
            "Enter a filename and click **Save Model** to write weights "
            "into `models/` at the repo root."
        )
    else:
        _models_dir = Path(__file__).resolve().parent.parent / "models"
        _models_dir.mkdir(parents=True, exist_ok=True)
        _save_path = _models_dir / save_filename_ui.value
        trained_model.save_weights(str(_save_path))
        _out = mo.md(f"**Saved!** Weights written to `{_save_path}`.")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 10 — Text-to-Speech Inference

    Type a transcript, click **Synthesize**, and the trained model will:

    1. Tokenize your input to character ids using the vocab from Section 3.
    2. Sample Gaussian noise of shape `(1, max_mel_frames, n_mels)`.
    3. Run `euler_solve_tts` for `num_steps` Euler steps, conditioned on
       the text prefix.
    4. De-normalize the generated log-mel using the dataset stats.
    5. Invert to a waveform via **Griffin-Lim** (`librosa.feature.inverse.mel_to_audio`) —
       this is a classical vocoder; expect audible artefacts.

    Given the small default training budget, generations mostly reveal
    coarse prosody/energy contour rather than intelligible speech — this
    is a scaffold that scales; enlarge the dataset and increase the
    epochs to improve quality.
    """)
    return


@app.function
def synthesize_speech(
    model: nn.Module,
    text: str,
    vocab: dict,
    mel_mean: float,
    mel_std: float,
    n_mels: int,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    win_length: int,
    max_mel_frames: int,
    max_text_len: int,
    num_ode_steps: int = 40,
    griffin_lim_iters: int = 32,
    seed: int = 0,
) -> tuple:
    ids, length = text_to_ids(text, vocab, max_text_len)
    text_ids = mx.array(ids[None, :])
    tmask = np.zeros((1, max_text_len), dtype=np.float32)
    tmask[0, :length] = 1.0
    text_mask = mx.array(tmask)
    mel_mask = mx.ones((1, max_mel_frames))
    mx.random.seed(seed)
    x0 = mx.random.normal(shape=(1, max_mel_frames, n_mels))
    mel_norm = euler_solve_tts(model, x0, text_ids, text_mask, mel_mask, num_ode_steps)
    mx.eval(mel_norm)
    log_mel_db = (np.array(mel_norm[0]).T * mel_std + mel_mean).astype(np.float32)
    power = librosa.db_to_power(log_mel_db)
    waveform = librosa.feature.inverse.mel_to_audio(
        power,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=griffin_lim_iters,
    )
    return waveform.astype(np.float32), log_mel_db


@app.cell
def _(mo):
    tts_text_ui = mo.ui.text_area(
        value="hello world this is a test of the text to speech model",
        label="Input text",
    )
    tts_num_steps_ui = mo.ui.slider(5, 200, value=40, step=5, label="Euler ODE steps")
    tts_gl_iters_ui = mo.ui.slider(4, 128, value=32, step=4, label="Griffin-Lim iterations")
    tts_seed_ui = mo.ui.number(value=0, label="Sampling seed")
    tts_synth_btn = mo.ui.run_button(label="Synthesize")
    mo.vstack(
        [
            tts_text_ui,
            mo.hstack([tts_num_steps_ui, tts_gl_iters_ui, tts_seed_ui]),
            tts_synth_btn,
        ]
    )
    return (
        tts_gl_iters_ui,
        tts_num_steps_ui,
        tts_seed_ui,
        tts_synth_btn,
        tts_text_ui,
    )


@app.cell
def _(
    char_vocab,
    mo,
    target_sr_ui,
    tensor_dataset,
    trained_model,
    tts_gl_iters_ui,
    tts_num_steps_ui,
    tts_seed_ui,
    tts_synth_btn,
    tts_text_ui,
):
    if trained_model is None or tensor_dataset is None or char_vocab is None:
        _out = mo.md("_Train the model first (Section 5) before synthesizing._")
    elif not tts_synth_btn.value:
        _out = mo.md("Type a transcript above and click **Synthesize** to generate speech.")
    else:
        _wav, _log_mel = synthesize_speech(
            model=trained_model,
            text=tts_text_ui.value,
            vocab=char_vocab,
            mel_mean=tensor_dataset["mel_mean"],
            mel_std=tensor_dataset["mel_std"],
            n_mels=tensor_dataset["n_mels"],
            sample_rate=int(target_sr_ui.value),
            hop_length=tensor_dataset["hop_length"],
            n_fft=tensor_dataset["n_fft"],
            win_length=tensor_dataset["win_length"],
            max_mel_frames=tensor_dataset["mel"].shape[1],
            max_text_len=tensor_dataset["text_ids"].shape[1],
            num_ode_steps=int(tts_num_steps_ui.value),
            griffin_lim_iters=int(tts_gl_iters_ui.value),
            seed=int(tts_seed_ui.value),
        )
        _items = [
            mo.md(f"**Synthesized {_wav.shape[0] / int(target_sr_ui.value):.2f}s of audio.**"),
            mo.audio(src=encode_wav_bytes(_wav, int(target_sr_ui.value))),
            plot_log_mel(
                _log_mel,
                sample_rate=int(target_sr_ui.value),
                hop_length=tensor_dataset["hop_length"],
                title="Generated log-mel spectrogram (de-normalized)",
            ),
            plot_waveform(_wav, int(target_sr_ui.value), title="Generated waveform (Griffin-Lim vocoded)"),
        ]
        _out = mo.vstack(_items)
    _out
    return


if __name__ == "__main__":
    app.run()
