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
    import soundfile as sf
    import librosa

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The flow decoder downsamples time twice (stride 2 each), so every
    # mel length it sees must be a multiple of 4.
    FRAME_MULTIPLE = 4


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Text-to-Speech on LibriTTS-R with a Matcha-style OT-CFM Acoustic Model (MLX)

    ## Research Goal

    Train a **text-conditioned acoustic model** that synthesizes speech
    mel-spectrograms from character-level transcripts on **LibriTTS-R**,
    using **optimal-transport conditional flow matching (OT-CFM)** as the
    generative objective — the same recipe as **Matcha-TTS** (Mehta et
    al., 2024), which is itself the flow-matching successor of Grad-TTS.

    ### Why the previous DiT design was replaced

    The first version of this notebook used an image-style Diffusion
    Transformer: mel frames grouped into 4-frame patches, text prepended
    as a prefix, and a naive *uniform* stretch of the text across the
    frame axis. That design has three structural defects that no amount
    of training fixes:

    1. **Patch-seam discontinuities.** Each 4-frame patch was produced
       by an independent linear "unpatchify" projection, so adjacent
       patches shared no output computation — the generated mels were
       visibly discontinuous at every patch boundary. The new decoder is
       a **frame-level 1-D convolutional U-Net** (with transformer
       blocks at the bottleneck); overlapping convolution kernels make
       the output continuous by construction.
    2. **Wrong alignment prior.** Uniformly stretching characters over
       frames assumes every character is spoken for the same duration.
       Real durations vary several-fold ("a" vs. a pause vs. "sh").
       The new model learns alignment during training with **Monotonic
       Alignment Search (MAS)** — the Viterbi-style dynamic program from
       Glow-TTS/Grad-TTS — and trains a **duration predictor** so that
       inference can size each character's segment correctly.
    3. **Unconditioned speaker averaging.** LibriTTS-R is multi-speaker;
       without a speaker signal the model must average over every voice,
       which blurs formants into noise. The new model conditions the
       encoder, duration predictor, and decoder on a **learned speaker
       embedding**, and inference lets you pick any training speaker.

    Griffin-Lim (`librosa.feature.inverse.mel_to_audio`) remains the
    vocoder — classical and artefact-prone, but sufficient to judge
    intelligibility once the mels themselves are sharp.

    ### Notebook Outline

    1. Title & research goal (this cell)
    2. Data exploration
    3. Dataset creation / preprocessing
    4. Model definition (encoder + MAS + duration predictor + flow decoder)
    5. Training (OT-CFM + prior loss + duration loss)
    6. Hyperparameter search (optional, checkbox-gated)
    7. Validation
    8. Results — losses, learned durations, ODE progression, step-count MSE
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
        value=4000, label="Max utterances to load"
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
                "`../data/libritts_r/`. Data volume is the binding constraint "
                "for intelligibility, so the default now loads **4000** "
                "utterances (~6 h of audio before length filtering). Decoding "
                "and mel extraction for that many clips takes a few minutes "
                "and ~3 GB of RAM."
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
    fmax: float = 8000.0,
) -> np.ndarray:
    """Magnitude (power=1) mel spectrogram compressed with a natural log.

    The previous version used ``power_to_db(ref=np.max)``, which
    normalizes every clip by its own peak — the same phoneme then had a
    different training target depending on the loudness of its clip, a
    hidden supervision inconsistency. ``log(mel)`` with a fixed floor is
    absolute, so targets are consistent across the whole corpus, and it
    inverts exactly with ``exp`` before Griffin-Lim. ``fmax=8000`` is the
    standard TTS band (Tacotron/Matcha) at 22.05 kHz.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32),
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        power=1.0,
    )
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32)


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
    fig.colorbar(im, ax=ax, label="ln magnitude")
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
    ax.set_title(f"Top {top_k} speakers by utterance count (speaker id IS a conditioning input)")
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
    ## Section 3 — Dataset Creation / Preprocessing

    The TTS preprocessing pipeline, stage by stage:

    1. **Resampling** to a single fixed sample rate (`22050 Hz`) via
       `mlx.data`'s built-in `load_audio(sample_rate=...)`.
    2. **Log-mel extraction** — magnitude (power=1) mel with a natural-log
       compression and a fixed `1e-5` floor. Absolute (no per-clip peak
       reference), so the same sound always maps to the same target, and
       exactly invertible with `exp` before Griffin-Lim vocoding.
       `fmax=8000 Hz` matches standard TTS practice.
    3. **Text tokenization** — LibriTTS-R ships normalized transcripts;
       we tokenize per character with a small vocabulary (`PAD`=0,
       `EOS`=1, then the observed characters). `EOS` conveniently absorbs
       trailing silence during alignment.
    4. **Speaker ids** — each utterance carries its LibriTTS speaker,
       mapped to an integer index for the learned speaker embedding.
    5. **Length handling** — utterances longer than `max_mel_frames` (or
       with transcripts longer than `max_text_len`) are **skipped, not
       truncated**. Truncation with a proportionally cropped transcript
       poisons the alignment search with mismatched (text, audio) pairs;
       dropping a minority of long clips is strictly safer supervision.
       Kept utterances are right-padded with zeros plus a validity mask
       consumed by attention, the losses, and MAS.
    6. **Global mean/std normalization** of the log-mel so the flow
       target is roughly zero-mean unit-variance, matching the Gaussian
       prior `x_0 ~ N(0, I)`. The corpus min/max are also stored so
       inference can clamp generated mels back into the trained domain.
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
    space_id = vocab.get(" ", 0)
    ids = [vocab.get(c, space_id) for c in lowered]
    ids = ids[: max_len - 1] + [vocab["<eos>"]]
    length = len(ids)
    if length < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - length)
    return np.asarray(ids, dtype=np.int32), length


@app.function
def ids_to_chars(ids: np.ndarray, vocab: dict) -> list:
    """Decode a row of token ids back to display characters (stops at PAD)."""
    inv = {v: k for k, v in vocab.items()}
    out: list = []
    for i in ids:
        ch = inv.get(int(i), "?")
        if ch == "<pad>":
            break
        if ch == "<eos>":
            out.append("$")
        elif ch == " ":
            out.append("␣")
        else:
            out.append(ch)
    return out


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
) -> dict:
    """Return dict with keys ``mel`` (B, T, M), ``mel_mask`` (B, T),
    ``text_ids`` (B, L), ``text_mask`` (B, L), ``spk_ids`` (B,), plus
    normalization stats and the speaker-name table. Utterances longer
    than the frame or text budget (or shorter than a sane minimum) are
    skipped rather than truncated."""
    max_frames_pad = round_up_to_multiple(max_mel_frames, FRAME_MULTIPLE)
    speakers = sorted(set(s["speaker"] for s in samples))
    spk_to_idx = {name: i for i, name in enumerate(speakers)}
    kept_mels: list = []
    kept_meta: list = []
    n_skipped = 0
    for s in samples:
        m = compute_log_mel(
            s["waveform"],
            sample_rate=s["sample_rate"],
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        num_frames = m.shape[1]
        text_len = len(s["transcript"]) + 1  # +1 for EOS
        if (
            num_frames > max_frames_pad
            or num_frames < 32
            or text_len > max_text_len
            or text_len > num_frames
        ):
            n_skipped += 1
            continue
        kept_mels.append(m)
        kept_meta.append(s)
    mel_mean, mel_std = compute_mel_stats(kept_mels)
    mel_min = float(min(m.min() for m in kept_mels))
    mel_max = float(max(m.max() for m in kept_mels))
    n = len(kept_mels)
    mel_arr = np.zeros((n, max_frames_pad, n_mels), dtype=np.float32)
    mel_mask = np.zeros((n, max_frames_pad), dtype=np.float32)
    text_ids = np.zeros((n, max_text_len), dtype=np.int32)
    text_mask = np.zeros((n, max_text_len), dtype=np.float32)
    spk_ids = np.zeros((n,), dtype=np.int32)
    for i, (m, s) in enumerate(zip(kept_mels, kept_meta)):
        num_frames = m.shape[1]
        mel_arr[i, :num_frames, :] = ((m - mel_mean) / mel_std).T
        mel_mask[i, :num_frames] = 1.0
        ids, length = text_to_ids(s["transcript"], vocab, max_text_len)
        text_ids[i] = ids
        text_mask[i, :length] = 1.0
        spk_ids[i] = spk_to_idx[s["speaker"]]
    return {
        "mel": mel_arr,
        "mel_mask": mel_mask,
        "text_ids": text_ids,
        "text_mask": text_mask,
        "spk_ids": spk_ids,
        "speakers": speakers,
        "n_skipped": n_skipped,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "mel_min": mel_min,
        "mel_max": mel_max,
        "max_mel_frames": max_frames_pad,
        "max_text_len": max_text_len,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "win_length": win_length,
        "fmin": 0.0,
        "fmax": 8000.0,
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
            "spk_ids": mx.array(dataset["spk_ids"][idx]),
        }

    return _slice(tr_idx), _slice(val_idx)


@app.function
def iter_batches(split: dict, batch_size: int = 16, shuffle: bool = True) -> list:
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
                "spk_ids": split["spk_ids"][b_idx],
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
        256, 1024, value=768, step=32, label="max_mel_frames"
    )
    max_text_len_ui = mo.ui.slider(
        64, 512, value=256, step=16, label="max_text_len"
    )
    build_ds_btn = mo.ui.run_button(label="Build tensor dataset")
    mo.vstack(
        [
            mo.md(
                "### Preprocessing hyperparameters\n"
                "`max_mel_frames=768` keeps utterances up to ~8.9 s at "
                "hop 256 / 22.05 kHz; longer clips are skipped."
            ),
            mo.hstack([n_mels_ui, hop_length_ui, n_fft_ui]),
            mo.hstack([max_mel_frames_ui, max_text_len_ui]),
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
        )
        mo.output.replace(
            mo.md(
                f"""
                Built tensor dataset:
                - `mel` shape: `{tensor_dataset["mel"].shape}` (dtype `float32`)
                - `text_ids` shape: `{tensor_dataset["text_ids"].shape}` (dtype `int32`)
                - Kept `{tensor_dataset["mel"].shape[0]}` utterances, skipped `{tensor_dataset["n_skipped"]}` (too long/short)
                - Speakers: `{len(tensor_dataset["speakers"])}` (conditioning input)
                - Vocab size: `{len(char_vocab)}` (PAD=0, EOS=1)
                - Log-mel mean/std: `{tensor_dataset["mel_mean"]:.3f}` / `{tensor_dataset["mel_std"]:.3f}`; range `[{tensor_dataset["mel_min"]:.2f}, {tensor_dataset["mel_max"]:.2f}]`
                - Mean valid frames: `{tensor_dataset["mel_mask"].sum(axis=1).mean():.0f}` of `{tensor_dataset["mel"].shape[1]}` budget
                """
            )
        )
    return char_vocab, tensor_dataset


@app.cell
def _(mo, tensor_dataset):
    if tensor_dataset is None:
        _out = mo.md("_Build the tensor dataset above first to see a batch check._")
    else:
        _tr, _va = split_tensor_dataset(tensor_dataset, val_fraction=0.1)
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
            | `spk_ids` | `{tuple(_b["spk_ids"].shape)}` | `{_b["spk_ids"].dtype}` |

            Total training utterances: `{_tr["mel"].shape[0]}`, validation: `{_va["mel"].shape[0]}`.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 4 — Model Definition

    The model follows the **Grad-TTS / Matcha-TTS** template, with three
    trainable components sharing one width `dim`:

    1. **Character encoder** — embedding + sinusoidal positions, a 3-layer
       convolutional prenet (local phonetic context), then transformer
       blocks whose AdaLN is conditioned on the **speaker embedding**.
       A linear head projects each character state to a mel-space mean
       vector `μ_j ∈ R^{n_mels}` — a rough per-character spectral target.
    2. **Duration predictor** — two masked convolutions + linear head
       predicting `log(duration)` per character. Its input is
       `stop_gradient`-ed so duration errors cannot corrupt the encoder
       (standard practice from FastSpeech/Glow-TTS).
    3. **Flow-matching decoder** — a frame-level **1-D convolutional
       U-Net**: two stride-2 downsampling stages of FiLM-conditioned
       residual conv blocks, transformer (DiT) blocks at the T/4
       bottleneck, then two upsampling stages with skip connections and
       a smoothing output conv. Input is `concat(x_t, μ_aligned)`;
       conditioning (timestep + speaker) enters every block. Because the
       receptive field overlaps everywhere and upsampling is
       nearest-neighbor + conv, the output has **no patch seams** — the
       defect that motivated this redesign. It is also fully
       **length-flexible**: no learned absolute positions, so inference
       can run at the exact predicted utterance length.

    **Alignment** is not a module but a training-time algorithm:
    **Monotonic Alignment Search** finds, for each (text, mel) pair, the
    monotonic frame→character assignment maximizing the likelihood of the
    mel under `N(μ_j, I)` — a Viterbi pass that is exact and fast. The
    resulting durations supervise the duration predictor, and the aligned
    `μ` sequence conditions the decoder. At inference the duration
    predictor replaces MAS.
    """)
    return


@app.function
def sinusoidal_positions(length: int, dim: int) -> mx.array:
    """Fixed sin/cos positional encodings, computed for any length so the
    model stays length-flexible at inference."""
    half = dim // 2
    freqs = mx.exp(
        -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / max(half, 1)
    )
    pos = mx.arange(length, dtype=mx.float32)[:, None] * freqs[None, :]
    return mx.concatenate([mx.sin(pos), mx.cos(pos)], axis=-1)


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
    so every block starts as an identity function."""

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
class CharacterEncoderV1(nn.Module):
    """Character embedding + conv prenet + speaker-conditioned transformer."""

    def __init__(
        self,
        vocab_size: int = 40,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        conv_kernel: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        pad = conv_kernel // 2
        self.prenet_convs = [
            nn.Conv1d(dim, dim, conv_kernel, padding=pad) for _ in range(3)
        ]
        self.prenet_norms = [nn.LayerNorm(dim) for _ in range(3)]
        self.blocks = [
            DiTBlockV1(dim, num_heads, 4 * dim, dim) for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(dim)

    def __call__(self, text_ids: mx.array, text_mask: mx.array, spk_emb: mx.array) -> mx.array:
        m3 = text_mask[:, :, None]
        x = self.token_embed(text_ids) + sinusoidal_positions(text_ids.shape[1], self.dim)[None]
        x = x * m3
        for conv, norm in zip(self.prenet_convs, self.prenet_norms):
            x = (x + nn.gelu(conv(norm(x) * m3))) * m3
        x = x + spk_emb[:, None, :]
        attn_mask = (1.0 - text_mask)[:, None, None, :] * -1.0e9
        for block in self.blocks:
            x = block(x, spk_emb, attn_mask=attn_mask)
        return self.final_norm(x) * m3


@app.class_definition
class DurationPredictorV1(nn.Module):
    """Two masked convolutions + linear head predicting log-duration per
    character. Callers pass a stop-gradient'ed encoder state so duration
    error never backpropagates into the encoder."""

    def __init__(self, dim: int = 256, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.norm2 = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 1)

    def __call__(self, h: mx.array, text_mask: mx.array) -> mx.array:
        m3 = text_mask[:, :, None]
        x = nn.gelu(self.norm1(self.conv1(h * m3))) * m3
        x = nn.gelu(self.norm2(self.conv2(x))) * m3
        return self.out(x)[:, :, 0]


@app.class_definition
class ResConvBlockV1(nn.Module):
    """FiLM-conditioned residual 1-D conv block (channels-last)."""

    def __init__(self, dim: int = 256, cond_dim: int = 256, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def __call__(self, x: mx.array, cond: mx.array, mask3: mx.array) -> mx.array:
        scale, shift = mx.split(self.film(nn.silu(cond))[:, None, :], 2, axis=-1)
        h = nn.gelu(self.conv1(self.norm1(x) * mask3))
        h = self.norm2(h) * (1.0 + scale) + shift
        h = nn.gelu(self.conv2(h * mask3))
        return (x + h) * mask3


@app.class_definition
class FlowDecoderV1(nn.Module):
    """Frame-level 1-D U-Net velocity field for OT-CFM.

    concat(x_t, mu_aligned) -> down x2 (stride-2) -> DiT blocks at T/4
    -> up x2 with skip connections -> smoothing conv head. FiLM/AdaLN
    conditioning on (timestep + speaker) in every block. No learned
    absolute positions, so any length divisible by FRAME_MULTIPLE works.
    """

    def __init__(
        self,
        n_mels: int = 80,
        dim: int = 256,
        num_heads: int = 4,
        mid_layers: int = 2,
        blocks_per_stage: int = 2,
        kernel: int = 5,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dim = dim
        self.in_proj = nn.Linear(2 * n_mels, dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbeddingV1(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.spk_proj = nn.Linear(dim, dim)
        self.down1_blocks = [ResConvBlockV1(dim, dim, kernel) for _ in range(blocks_per_stage)]
        self.down1_pool = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.down2_blocks = [ResConvBlockV1(dim, dim, kernel) for _ in range(blocks_per_stage)]
        self.down2_pool = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        self.mid_blocks = [
            DiTBlockV1(dim, num_heads, 4 * dim, dim) for _ in range(mid_layers)
        ]
        self.up2_conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.up2_fuse = nn.Linear(2 * dim, dim)
        self.up2_blocks = [ResConvBlockV1(dim, dim, kernel) for _ in range(blocks_per_stage)]
        self.up1_conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.up1_fuse = nn.Linear(2 * dim, dim)
        self.up1_blocks = [ResConvBlockV1(dim, dim, kernel) for _ in range(blocks_per_stage)]
        self.out_norm = nn.LayerNorm(dim)
        self.out_conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.out_proj = nn.Linear(dim, n_mels)

    def __call__(
        self,
        x_t: mx.array,
        t: mx.array,
        mu_frame: mx.array,
        spk_emb: mx.array,
        mel_mask: mx.array,
    ) -> mx.array:
        b, t_frames, _ = x_t.shape
        cond = self.time_embed(t) + self.spk_proj(spk_emb)
        mask1_tok = mel_mask.reshape(b, t_frames // 2, 2).max(axis=-1)
        mask2_tok = mask1_tok.reshape(b, t_frames // 4, 2).max(axis=-1)
        m0 = mel_mask[:, :, None]
        m1 = mask1_tok[:, :, None]
        m2 = mask2_tok[:, :, None]
        h = self.in_proj(mx.concatenate([x_t, mu_frame], axis=-1)) * m0
        for blk in self.down1_blocks:
            h = blk(h, cond, m0)
        skip1 = h
        h = self.down1_pool(h) * m1
        for blk in self.down2_blocks:
            h = blk(h, cond, m1)
        skip2 = h
        h = self.down2_pool(h) * m2
        h = h + sinusoidal_positions(h.shape[1], self.dim)[None]
        attn_mask = (1.0 - mask2_tok)[:, None, None, :] * -1.0e9
        for blk in self.mid_blocks:
            h = blk(h, cond, attn_mask=attn_mask)
        h = self.up2_conv(mx.repeat(h, 2, axis=1)) * m1
        h = self.up2_fuse(mx.concatenate([h, skip2], axis=-1))
        for blk in self.up2_blocks:
            h = blk(h, cond, m1)
        h = self.up1_conv(mx.repeat(h, 2, axis=1)) * m0
        h = self.up1_fuse(mx.concatenate([h, skip1], axis=-1))
        for blk in self.up1_blocks:
            h = blk(h, cond, m0)
        return self.out_proj(nn.gelu(self.out_conv(self.out_norm(h)))) * m0


@app.class_definition
class MatchaTTSV1(nn.Module):
    """Speaker-conditioned Matcha-style acoustic model: character encoder
    with mel-mean head, duration predictor, and OT-CFM flow decoder."""

    def __init__(
        self,
        vocab_size: int = 40,
        n_speakers: int = 40,
        n_mels: int = 80,
        dim: int = 256,
        enc_layers: int = 4,
        enc_heads: int = 4,
        dec_heads: int = 4,
        dec_mid_layers: int = 2,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dim = dim
        self.spk_embed = nn.Embedding(n_speakers, dim)
        self.encoder = CharacterEncoderV1(vocab_size, dim, enc_layers, enc_heads)
        self.mu_proj = nn.Linear(dim, n_mels)
        self.duration_predictor = DurationPredictorV1(dim)
        self.decoder = FlowDecoderV1(n_mels, dim, dec_heads, dec_mid_layers)

    def encode_text(
        self, text_ids: mx.array, text_mask: mx.array, spk_ids: mx.array
    ) -> tuple:
        spk_emb = self.spk_embed(spk_ids)
        h = self.encoder(text_ids, text_mask, spk_emb)
        mu_tok = self.mu_proj(h)
        log_dur = self.duration_predictor(mx.stop_gradient(h), text_mask)
        return h, mu_tok, log_dur, spk_emb


@app.function
def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


@app.function
def build_matcha_tts_model(
    vocab_size: int,
    n_speakers: int,
    n_mels: int,
    dim: int = 256,
    enc_layers: int = 4,
    enc_heads: int = 4,
    dec_heads: int = 4,
    dec_mid_layers: int = 2,
) -> MatchaTTSV1:
    model = MatchaTTSV1(
        vocab_size=vocab_size,
        n_speakers=n_speakers,
        n_mels=n_mels,
        dim=dim,
        enc_layers=enc_layers,
        enc_heads=enc_heads,
        dec_heads=dec_heads,
        dec_mid_layers=dec_mid_layers,
    )
    mx.eval(model.parameters())
    return model


@app.cell
def _(mo):
    mo.md("""
    ### Architecture Table

    | Component | Module | Shape / role |
    |-----------|--------|---------------|
    | Speaker embedding | `nn.Embedding(n_speakers, D)` | `(B, D)`; conditions everything |
    | Character encoder | `CharacterEncoderV1` (conv prenet + DiT blocks, AdaLN on speaker) | `(B, L, D)` |
    | Mel-mean head | `Linear(D, M)` | `μ_j` per character, drives MAS + prior loss |
    | Duration predictor | `DurationPredictorV1` (masked convs, stop-grad input) | `log d_j` per character |
    | Alignment | **MAS** (training) / duration predictor (inference) | frame→character map |
    | Flow decoder | `FlowDecoderV1` U-Net: `concat(x_t, μ_frame)` → down×2 → DiT×k → up×2 | velocity `(B, T, M)` |
    | Conditioning | timestep sinusoid MLP + speaker, via FiLM/AdaLN in every block | `(B, D)` |

    **Key properties vs. the previous patchified DiT:**

    - No unpatchify seams — overlapping convolutions at full frame rate.
    - Alignment is *learned* (MAS), not assumed uniform.
    - Speaker identity is explicit, so the model no longer averages voices.
    - Decoder is length-flexible: inference runs at the exact predicted
      utterance length instead of a fixed padded canvas.
    - Classifier-free guidance is dropped: the decoder is conditioned
      per-frame on `μ`, a far stronger signal than a global text prefix,
      so guidance is unnecessary (matching Matcha-TTS).
    """)
    return


@app.cell
def _(char_vocab, mo, tensor_dataset):
    if tensor_dataset is None or char_vocab is None:
        _out = mo.md("_Build the tensor dataset above to instantiate the default model._")
    else:
        _m = build_matcha_tts_model(
            vocab_size=len(char_vocab),
            n_speakers=len(tensor_dataset["speakers"]),
            n_mels=tensor_dataset["n_mels"],
        )
        _out = mo.md(f"**Default model parameter count**: `{count_parameters(_m):,}`.")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 5 — Training (OT-CFM + prior + duration losses)

    Each step:

    1. **Align (no gradient).** Run the encoder, compute the frame-vs-
       character log-likelihood matrix `logp[t, j] = -½‖mel_t − μ_j‖²`,
       and run **MAS** (a numpy Viterbi pass, vectorized over the batch)
       to get the best monotonic frame→character map and per-character
       durations.
    2. **Losses (with gradient), alignment held fixed:**
       - **CFM loss** — rectified-flow objective: `x_t = (1−t)x₀ + t·mel`
         with `x₀ ~ N(0, I)`, target velocity `mel − x₀`, masked MSE
         against the decoder's prediction. (The Hungarian minibatch-OT
         pairing from the CIFAR notebooks is dropped: with strong
         per-frame conditioning the coupling gain is negligible, and the
         conditional paths are already the straight OT-CFM paths.)
       - **Prior loss** — masked MSE between the aligned `μ` sequence and
         the target mel. This is what makes MAS meaningful: it pulls each
         character's `μ` toward the frames MAS assigned to it, and MAS in
         turn re-assigns frames to the closest `μ` — an EM-like loop.
       - **Duration loss** — masked MSE between predicted and MAS
         log-durations.

    Optimizer: AdamW with linear warmup → cosine decay, gradient-norm
    clipping at 1.0. **Runtime expectation:** at the defaults (~3–4 k
    utterances after filtering, batch 16, ~17 M parameters) one epoch is
    roughly 1–3 minutes on an M-series GPU, so **80 epochs ≈ 2–4 hours**.
    """)
    return


@app.function
def alignment_log_likelihood(mel: mx.array, mu_tok: mx.array, text_mask: mx.array) -> mx.array:
    """(B, T, L) isotropic-Gaussian log-likelihood (up to a constant) of
    each mel frame under each character's mu. Padded text positions get
    -1e9 so MAS never assigns frames to them."""
    mel2 = mx.sum(mel * mel, axis=-1)
    mu2 = mx.sum(mu_tok * mu_tok, axis=-1)
    cross = mel @ mu_tok.transpose(0, 2, 1)
    logp = -0.5 * (mel2[:, :, None] + mu2[:, None, :] - 2.0 * cross)
    return logp + (1.0 - text_mask)[:, None, :] * -1.0e9


@app.function
def monotonic_alignment_search(
    logp: np.ndarray, text_lens: np.ndarray, mel_lens: np.ndarray
) -> tuple:
    """Batched MAS (Glow-TTS): monotonic, no-skip Viterbi over the
    (frames x characters) log-likelihood grid. The forward pass is
    vectorized over batch and characters; backtracking is a cheap
    per-sample scalar loop. Returns ``align_idx`` (B, T) int32 mapping
    each frame to a character index, and ``durations`` (B, L) int32."""
    b, t_max, l_max = logp.shape
    neg = -1.0e9
    lp = logp.astype(np.float64)
    dp = np.full((b, t_max, l_max), neg, dtype=np.float64)
    back = np.zeros((b, t_max, l_max), dtype=np.int8)
    dp[:, 0, 0] = lp[:, 0, 0]
    for t in range(1, t_max):
        stay = dp[:, t - 1, :]
        move = np.concatenate([np.full((b, 1), neg), dp[:, t - 1, :-1]], axis=1)
        take = move > stay
        dp[:, t, :] = lp[:, t, :] + np.where(take, move, stay)
        back[:, t, :] = take
    align_idx = np.zeros((b, t_max), dtype=np.int32)
    durations = np.zeros((b, l_max), dtype=np.int32)
    for i in range(b):
        t_len = int(mel_lens[i])
        l_len = int(min(text_lens[i], t_len))
        j = l_len - 1
        for t in range(t_len - 1, -1, -1):
            align_idx[i, t] = j
            if t > 0 and back[i, t, j] and j > 0:
                j -= 1
        align_idx[i, t_len:] = max(l_len - 1, 0)
        counts = np.bincount(align_idx[i, :t_len], minlength=l_max)
        durations[i] = counts[:l_max]
    return align_idx, durations


@app.function
def compute_batch_alignment(model: nn.Module, batch: dict) -> tuple:
    """Encoder forward (no grad taken) + MAS. Returns mx arrays
    ``align_idx`` (B, T) and ``durations`` (B, L)."""
    _, mu_tok, _, _ = model.encode_text(
        batch["text_ids"], batch["text_mask"], batch["spk_ids"]
    )
    logp = alignment_log_likelihood(batch["mel"], mu_tok, batch["text_mask"])
    mx.eval(logp)
    text_lens = np.asarray(mx.sum(batch["text_mask"], axis=1)).astype(np.int64)
    mel_lens = np.asarray(mx.sum(batch["mel_mask"], axis=1)).astype(np.int64)
    align_np, dur_np = monotonic_alignment_search(np.asarray(logp), text_lens, mel_lens)
    return mx.array(align_np), mx.array(dur_np)


@app.function
def gather_mu_frames(mu_tok: mx.array, align_idx: mx.array, n_mels: int) -> mx.array:
    b, t_frames = align_idx.shape
    idx = mx.broadcast_to(align_idx[:, :, None], (b, t_frames, n_mels))
    return mx.take_along_axis(mu_tok, idx, axis=1)


@app.function
def compute_tts_losses(
    model: nn.Module,
    mel: mx.array,
    mel_mask: mx.array,
    text_ids: mx.array,
    text_mask: mx.array,
    spk_ids: mx.array,
    x0: mx.array,
    t: mx.array,
    align_idx: mx.array,
    durations: mx.array,
) -> tuple:
    _, mu_tok, log_dur_pred, spk_emb = model.encode_text(text_ids, text_mask, spk_ids)
    mu_frame = gather_mu_frames(mu_tok, align_idx, model.n_mels)
    m3 = mel_mask[:, :, None]
    feat_denom = mx.maximum(mx.sum(m3) * float(mel.shape[-1]), mx.array(1.0))
    prior_loss = mx.sum(((mu_frame - mel) ** 2) * m3) / feat_denom
    t3 = t.reshape(-1, 1, 1)
    x_t = (1.0 - t3) * x0 + t3 * mel
    v_target = mel - x0
    v_pred = model.decoder(x_t, t, mu_frame, spk_emb, mel_mask)
    cfm_loss = mx.sum(((v_pred - v_target) ** 2) * m3) / feat_denom
    dur_target = mx.log(mx.maximum(durations.astype(mx.float32), mx.array(1.0)))
    dur_denom = mx.maximum(mx.sum(text_mask), mx.array(1.0))
    dur_loss = mx.sum(((log_dur_pred - dur_target) ** 2) * text_mask) / dur_denom
    total = cfm_loss + prior_loss + dur_loss
    return total, cfm_loss, prior_loss, dur_loss


@app.function
def compute_tts_total_loss(model: nn.Module, *args) -> mx.array:
    return compute_tts_losses(model, *args)[0]


@app.function
def run_train_epoch_matcha(model: nn.Module, optimizer, batches: list) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, compute_tts_total_loss)
    epoch_loss = 0.0
    n = 0
    for b in batches:
        align_idx, durations = compute_batch_alignment(model, b)
        x0 = mx.random.normal(shape=b["mel"].shape)
        t = mx.random.uniform(shape=(b["mel"].shape[0],))
        loss, grads = loss_and_grad_fn(
            model,
            b["mel"],
            b["mel_mask"],
            b["text_ids"],
            b["text_mask"],
            b["spk_ids"],
            x0,
            t,
            align_idx,
            durations,
        )
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        epoch_loss += float(loss.item())
        n += 1
    return epoch_loss / max(n, 1)


@app.function
def evaluate_tts_model(model: nn.Module, batches: list) -> dict:
    """Component-wise validation losses with fixed noise/timestep keys so
    the numbers are comparable across epochs (the global RNG stream is
    left untouched)."""
    sums = {"total": 0.0, "cfm": 0.0, "prior": 0.0, "dur": 0.0}
    n = 0
    for i, b in enumerate(batches):
        align_idx, durations = compute_batch_alignment(model, b)
        x0 = mx.random.normal(shape=b["mel"].shape, key=mx.random.key(1000 + i))
        t = mx.random.uniform(shape=(b["mel"].shape[0],), key=mx.random.key(5000 + i))
        total, cfm, prior, dur = compute_tts_losses(
            model,
            b["mel"],
            b["mel_mask"],
            b["text_ids"],
            b["text_mask"],
            b["spk_ids"],
            x0,
            t,
            align_idx,
            durations,
        )
        mx.eval(total, cfm, prior, dur)
        sums["total"] += float(total.item())
        sums["cfm"] += float(cfm.item())
        sums["prior"] += float(prior.item())
        sums["dur"] += float(dur.item())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@app.function
def euler_solve_decoder(
    decoder: nn.Module,
    x0: mx.array,
    mu_frame: mx.array,
    spk_emb: mx.array,
    mel_mask: mx.array,
    num_steps: int = 40,
) -> mx.array:
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = decoder(x, t, mu_frame, spk_emb, mel_mask)
        x = x + dt * v
        mx.eval(x)
    return x


@app.function
def euler_solve_decoder_trajectory(
    decoder: nn.Module,
    x0: mx.array,
    mu_frame: mx.array,
    spk_emb: mx.array,
    mel_mask: mx.array,
    num_steps: int = 40,
) -> list:
    dt = 1.0 / num_steps
    x = x0
    trajectory = [x]
    for i in range(num_steps):
        t = mx.full((x.shape[0],), i * dt, dtype=mx.float32)
        v = decoder(x, t, mu_frame, spk_emb, mel_mask)
        x = x + dt * v
        mx.eval(x)
        trajectory.append(x)
    return trajectory


@app.function
def train_tts_model(
    train_split: dict,
    val_split: dict,
    vocab_size: int,
    n_speakers: int,
    n_mels: int,
    dim: int,
    enc_layers: int,
    enc_heads: int,
    dec_heads: int,
    dec_mid_layers: int,
    lr: float,
    wd: float,
    batch_size: int,
    epochs: int,
    seed: int = 0,
    progress_cb=None,
) -> tuple:
    mx.random.seed(seed)
    np.random.seed(seed)
    model = build_matcha_tts_model(
        vocab_size=vocab_size,
        n_speakers=n_speakers,
        n_mels=n_mels,
        dim=dim,
        enc_layers=enc_layers,
        enc_heads=enc_heads,
        dec_heads=dec_heads,
        dec_mid_layers=dec_mid_layers,
    )
    steps_per_epoch = max(math.ceil(train_split["mel"].shape[0] / batch_size), 1)
    total_steps = steps_per_epoch * epochs
    warmup = min(500, max(total_steps // 20, 1))
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, lr, warmup),
            optim.cosine_decay(lr, max(total_steps - warmup, 1)),
        ],
        [warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=wd)
    val_batches = iter_batches(val_split, batch_size=batch_size, shuffle=False)
    history = {
        "train_total": [],
        "val_total": [],
        "val_cfm": [],
        "val_prior": [],
        "val_dur": [],
    }
    for epoch in range(epochs):
        train_batches = iter_batches(train_split, batch_size=batch_size, shuffle=True)
        tl = run_train_epoch_matcha(model, optimizer, train_batches)
        vm = evaluate_tts_model(model, val_batches)
        history["train_total"].append(tl)
        history["val_total"].append(vm["total"])
        history["val_cfm"].append(vm["cfm"])
        history["val_prior"].append(vm["prior"])
        history["val_dur"].append(vm["dur"])
        if progress_cb is not None:
            progress_cb(epoch, epochs, tl, vm["total"])
    return model, history


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "2e-4": 2e-4, "3e-4": 3e-4},
        value="2e-4",
        label="Learning rate",
    )
    bs_ui = mo.ui.dropdown(
        options={"8": 8, "16": 16, "32": 32},
        value="16",
        label="Batch size",
    )
    wd_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "1e-6": 1e-6, "1e-4": 1e-4},
        value="0.0",
        label="Weight decay",
    )
    epochs_ui = mo.ui.slider(1, 200, value=80, step=1, label="Epochs")
    dim_ui = mo.ui.dropdown(
        options={"192": 192, "256": 256, "384": 384},
        value="256",
        label="model dim",
    )
    heads_ui = mo.ui.dropdown(
        options={"4": 4, "8": 8},
        value="4",
        label="attention heads",
    )
    enc_layers_ui = mo.ui.slider(2, 6, value=4, step=1, label="encoder layers")
    mid_layers_ui = mo.ui.slider(1, 4, value=2, step=1, label="decoder mid layers")
    train_btn = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md(
                "### Training hyperparameters\n"
                "Defaults follow the Matcha-TTS recipe scaled to this data "
                "budget: Adam-style optimizer without weight decay, lr "
                "`2e-4` with warmup + cosine decay, 80 epochs."
            ),
            mo.hstack([lr_ui, bs_ui, wd_ui, epochs_ui]),
            mo.hstack([dim_ui, heads_ui, enc_layers_ui, mid_layers_ui]),
            train_btn,
        ]
    )
    return (
        bs_ui,
        dim_ui,
        enc_layers_ui,
        epochs_ui,
        heads_ui,
        lr_ui,
        mid_layers_ui,
        train_btn,
        wd_ui,
    )


@app.cell
def _(
    bs_ui,
    char_vocab,
    dim_ui,
    enc_layers_ui,
    epochs_ui,
    heads_ui,
    lr_ui,
    mid_layers_ui,
    mo,
    tensor_dataset,
    train_btn,
    wd_ui,
):
    loss_history = None
    trained_model = None
    train_split_gl = None
    val_split_gl = None
    if tensor_dataset is None or char_vocab is None:
        mo.output.replace(mo.md("_Build the tensor dataset first (Section 3)._"))
    elif not train_btn.value:
        mo.output.replace(mo.md("Click **Train** to start MAS + OT-CFM training."))
    else:
        train_split_gl, val_split_gl = split_tensor_dataset(tensor_dataset, val_fraction=0.1)

        def _cb(epoch, n_epochs, tl, vl):
            mo.output.replace(
                mo.md(f"**Epoch {epoch + 1}/{n_epochs}** — train: {tl:.4f} | val: {vl:.4f}")
            )

        trained_model, loss_history = train_tts_model(
            train_split=train_split_gl,
            val_split=val_split_gl,
            vocab_size=len(char_vocab),
            n_speakers=len(tensor_dataset["speakers"]),
            n_mels=tensor_dataset["n_mels"],
            dim=int(dim_ui.value),
            enc_layers=int(enc_layers_ui.value),
            enc_heads=int(heads_ui.value),
            dec_heads=int(heads_ui.value),
            dec_mid_layers=int(mid_layers_ui.value),
            lr=float(lr_ui.value),
            wd=float(wd_ui.value),
            batch_size=int(bs_ui.value),
            epochs=int(epochs_ui.value),
            progress_cb=_cb,
        )
        mo.output.replace(
            mo.md(
                f"**Training complete!** Final train `{loss_history['train_total'][-1]:.4f}` | "
                f"val `{loss_history['val_total'][-1]:.4f}` "
                f"(cfm `{loss_history['val_cfm'][-1]:.4f}`, prior `{loss_history['val_prior'][-1]:.4f}`, "
                f"dur `{loss_history['val_dur'][-1]:.4f}`). "
                f"Params: `{count_parameters(trained_model):,}`."
            )
        )
    return loss_history, train_split_gl, trained_model, val_split_gl


@app.cell
def _(mo):
    mo.md("""
    ## Section 6 — Hyperparameter Search (optional)

    Small grid over learning rate and model width, a few epochs each.
    The final validation total loss ranks configurations. Left off by
    default: a full 80-epoch run should use the defaults above unless
    this table clearly disagrees.
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
    n_speakers: int,
    n_mels: int,
    dim: int,
    lr: float,
    n_epochs: int,
    batch_size: int,
) -> float:
    model, history = train_tts_model(
        train_split=train_split,
        val_split=val_split,
        vocab_size=vocab_size,
        n_speakers=n_speakers,
        n_mels=n_mels,
        dim=dim,
        enc_layers=3,
        enc_heads=4,
        dec_heads=4,
        dec_mid_layers=1,
        lr=lr,
        wd=0.0,
        batch_size=batch_size,
        epochs=n_epochs,
        progress_cb=None,
    )
    _ = model
    return history["val_total"][-1] if history["val_total"] else float("inf")


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
    _search_space = {"lr": [1e-4, 2e-4], "dim": [192, 256]}
    _hp_epochs = 3
    _hp_bs = 16
    hp_results = []
    for _lr in _search_space["lr"]:
        for _dim in _search_space["dim"]:
            _vl = run_hp_config_tts(
                _tr,
                _va,
                vocab_size=len(char_vocab),
                n_speakers=len(tensor_dataset["speakers"]),
                n_mels=tensor_dataset["n_mels"],
                dim=_dim,
                lr=_lr,
                n_epochs=_hp_epochs,
                batch_size=_hp_bs,
            )
            hp_results.append({"lr": _lr, "dim": _dim, "val_loss": round(_vl, 4)})
            mo.output.replace(mo.md(f"lr={_lr}, dim={_dim} -> val={_vl:.4f}"))
    hp_results.sort(key=lambda r: r["val_loss"])
    mo.output.replace(mo.ui.table(hp_results))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 7 — Validation

    Component-wise held-out losses. **k-fold cross-validation is
    intentionally omitted**: k full DiT+U-Net training runs on hours of
    audio would dominate the notebook's wall-clock budget, so the single
    held-out split is the generalization signal, complemented by the
    qualitative results in Section 8.

    Reading the components: `cfm` is the decoder's velocity-field error
    (the generative fidelity signal), `prior` measures how well the
    per-character μ vectors summarize their aligned frames (alignment
    quality), and `dur` is log-duration MSE (rhythm quality). A model
    that speaks intelligibly needs *all three* low — a low `cfm` with a
    high `prior` means the decoder paints fine texture on a wrong
    phonetic skeleton.
    """)
    return


@app.cell
def _(bs_ui, mo, trained_model, val_split_gl):
    if trained_model is None or val_split_gl is None:
        _out = mo.md("_Train the model first (Section 5)._")
    else:
        _val_batches = iter_batches(val_split_gl, batch_size=int(bs_ui.value), shuffle=False)
        _vm = evaluate_tts_model(trained_model, _val_batches)
        _out = mo.md(
            f"""
            | Validation loss | Value |
            |------|-------|
            | Total | `{_vm["total"]:.4f}` |
            | Flow matching (cfm) | `{_vm["cfm"]:.4f}` |
            | Prior (μ vs mel) | `{_vm["prior"]:.4f}` |
            | Duration (log-MSE) | `{_vm["dur"]:.4f}` |
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Section 8 — Results
    """)
    return


@app.function
def plot_loss_curves(history: dict):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = range(1, len(history["train_total"]) + 1)
    ax.plot(xs, history["train_total"], "b-o", lw=2, ms=3, label="Train total")
    ax.plot(xs, history["val_total"], "r-s", lw=2, ms=3, label="Val total")
    ax.plot(xs, history["val_cfm"], "r--", lw=1.2, alpha=0.7, label="Val cfm")
    ax.plot(xs, history["val_prior"], "g--", lw=1.2, alpha=0.7, label="Val prior")
    ax.plot(xs, history["val_dur"], "m--", lw=1.2, alpha=0.7, label="Val duration")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation losses (MAS + OT-CFM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(loss_history, mo):
    if not loss_history:
        _out = mo.md("_Train the model first (Section 5)._")
    else:
        _out = plot_loss_curves(loss_history)
    _out
    return


@app.function
def plot_char_durations(chars: list, durations: np.ndarray, title: str = "MAS character durations"):
    k = min(len(chars), 80)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(range(k), durations[:k], color="steelblue", edgecolor="black", lw=0.3)
    ax.set_xticks(range(k))
    ax.set_xticklabels(chars[:k], fontsize=7)
    ax.set_ylabel("Frames")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


@app.cell
def _(char_vocab, mo, train_split_gl, trained_model):
    if trained_model is None or train_split_gl is None or char_vocab is None:
        _out = mo.md("_Train the model first to inspect learned alignments._")
    else:
        _b = iter_batches(train_split_gl, batch_size=1, shuffle=False)[0]
        _, _durs = compute_batch_alignment(trained_model, _b)
        _ids = np.asarray(_b["text_ids"][0])
        _chars = ids_to_chars(_ids, char_vocab)
        _out = mo.vstack(
            [
                mo.md(
                    "**MAS-discovered durations** for one training utterance — "
                    "if alignment is healthy these vary per character "
                    "(vowels long, stops short, `␣` marks pauses) instead "
                    "of being flat."
                ),
                plot_char_durations(_chars, np.asarray(_durs[0])[: len(_chars)]),
            ]
        )
    _out
    return


@app.function
def teacher_forced_condition(model: nn.Module, batch: dict) -> tuple:
    """MAS-aligned mu sequence for a batch — the decoder condition with
    ground-truth durations, isolating decoder quality from duration
    prediction."""
    align_idx, durations = compute_batch_alignment(model, batch)
    _, mu_tok, _, spk_emb = model.encode_text(
        batch["text_ids"], batch["text_mask"], batch["spk_ids"]
    )
    mu_frame = gather_mu_frames(mu_tok, align_idx, model.n_mels)
    return mu_frame, spk_emb, durations


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
def _(mo, train_split_gl, trained_model):
    if trained_model is None or train_split_gl is None:
        _out = mo.md("_Train the model first to visualize its ODE process._")
    else:
        _b = iter_batches(train_split_gl, batch_size=1, shuffle=False)[0]
        _mu_frame, _spk, _ = teacher_forced_condition(trained_model, _b)
        _x0 = mx.random.normal(shape=_b["mel"].shape, key=mx.random.key(11)) * 0.667
        _traj = euler_solve_decoder_trajectory(
            trained_model.decoder, _x0, _mu_frame, _spk, _b["mel_mask"], num_steps=40
        )
        _valid = int(np.array(_b["mel_mask"][0]).sum())
        _out = plot_euler_progression_tts(
            _traj, valid_frames=_valid, title="Flow-matching ODE process (noise -> mel)"
        )
    _out
    return


@app.function
def compute_step_count_mse_tts(
    decoder: nn.Module,
    mu_frame: mx.array,
    spk_emb: mx.array,
    mel_mask: mx.array,
    n_mels: int,
    step_list: list,
    ref_steps: int = 100,
    seed: int = 7,
) -> list:
    b, t_frames = mel_mask.shape
    x0 = mx.random.normal(shape=(b, t_frames, n_mels), key=mx.random.key(seed))
    x_ref = euler_solve_decoder(decoder, x0, mu_frame, spk_emb, mel_mask, ref_steps)
    mx.eval(x_ref)
    mses: list = []
    for ns in step_list:
        x_ns = euler_solve_decoder(decoder, x0, mu_frame, spk_emb, mel_mask, ns)
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
    ax.plot(step_list, mses, "b-o", lw=2, ms=6, label="OT-CFM decoder")
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
        _mu_frame, _spk, _ = teacher_forced_condition(trained_model, _b)
        _steps = [1, 2, 5, 10, 25, 50]
        _mses = compute_step_count_mse_tts(
            trained_model.decoder,
            _mu_frame,
            _spk,
            _b["mel_mask"],
            trained_model.n_mels,
            _steps,
            ref_steps=100,
        )
        _out = plot_step_count_mse(_steps, _mses)
    _out
    return


@app.function
def plot_reconstruction_comparison(
    mel_gt: np.ndarray,
    mel_pred: np.ndarray,
    valid_frames: int,
    title: str = "Ground truth vs. generated mel (teacher-forced durations)",
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
        _mu_frame, _spk, _ = teacher_forced_condition(trained_model, _b)
        _x0 = mx.random.normal(shape=_b["mel"].shape, key=mx.random.key(42)) * 0.667
        _x_gen = euler_solve_decoder(
            trained_model.decoder, _x0, _mu_frame, _spk, _b["mel_mask"], num_steps=40
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

    - **Loss curves**: all three components should fall together. The
      `prior` curve is the alignment health signal — if it plateaus high,
      MAS never found a stable text↔audio correspondence and nothing
      downstream can be intelligible.
    - **MAS durations**: per-character durations should be visibly
      non-uniform. Flat bars would mean alignment collapsed to the old
      uniform-stretch behaviour.
    - **ODE progression**: intermediate states of the Euler integration
      from noise to mel. Note the absence of 4-frame blocking artefacts —
      the convolutional decoder output is continuous in time.
    - **Step-count MSE**: OT-CFM paths are near-straight, so few-step
      solutions should stay close to the 100-step reference.
    - **Reconstruction (teacher-forced)**: generated with MAS durations
      from the ground-truth pair, so both mels have identical timing and
      differences are purely spectral fidelity.

    ### What to scale next, in order of expected payoff

    1. **More audio.** Move `split` to `train-clean-100` and raise
       `max_samples`. Character-level TTS sharpens dramatically between
       ~5 h and ~25 h.
    2. **Phoneme input.** Replacing characters with phonemes (e.g. via
       `g2p-en`) removes English spelling irregularity — the largest
       remaining modeling burden at this scale.
    3. **A neural vocoder.** Griffin-Lim is now the perceptual floor;
       HiFi-GAN on top of these 80-bin mels is the standard next step.
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
        value="libritts_matcha_cfm_v1.safetensors",
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

    Type a transcript, pick a **speaker**, and click **Synthesize**:

    1. Characters are tokenized with the Section 3 vocabulary.
    2. The encoder produces per-character `μ` vectors and the duration
       predictor produces per-character frame counts (scaled by the
       **Speaking rate** slider), which are expanded into a
       frame-aligned `μ` sequence of the exact predicted length — no
       fixed padded canvas.
    3. Gaussian noise scaled by **temperature** (Matcha's default 0.667
       trades a little diversity for cleaner output) is integrated with
       `num_steps` Euler steps through the flow decoder.
    4. The mel is de-normalized, clamped to the trained log-mel range,
      exponentiated, and inverted with **Griffin-Lim** — a classical
      vocoder, so some metallic phasiness is expected even from good
      mels.
    """)
    return


@app.function
def infer_condition(
    model: nn.Module,
    text: str,
    vocab: dict,
    max_text_len: int,
    spk_id: int,
    speed: float = 1.0,
    max_total_frames: int = 2048,
) -> tuple:
    """Encoder + duration-predictor forward for one prompt. Returns the
    frame-aligned mu condition, speaker embedding, an all-valid mel mask
    at the exact predicted length (rounded to FRAME_MULTIPLE), and the
    per-character durations."""
    ids, length = text_to_ids(text, vocab, max_text_len)
    text_ids = mx.array(ids[None, :])
    tmask = np.zeros((1, max_text_len), dtype=np.float32)
    tmask[0, :length] = 1.0
    text_mask = mx.array(tmask)
    spk_ids = mx.array(np.asarray([spk_id], dtype=np.int32))
    _, mu_tok, log_dur, spk_emb = model.encode_text(text_ids, text_mask, spk_ids)
    mx.eval(mu_tok, log_dur)
    d = np.exp(np.asarray(log_dur[0, :length])) / max(speed, 1e-3)
    d = np.clip(np.round(d), 1, 60).astype(np.int64)
    frame_tok = np.repeat(np.arange(length), d)[:max_total_frames]
    pad = (-len(frame_tok)) % FRAME_MULTIPLE
    if pad:
        frame_tok = np.concatenate([frame_tok, np.full(pad, frame_tok[-1])])
    total = len(frame_tok)
    mu_frame = mx.take(mu_tok[0], mx.array(frame_tok.astype(np.int32)), axis=0)[None]
    mel_mask = mx.ones((1, total))
    return mu_frame, spk_emb, mel_mask, total, d


@app.function
def synthesize_speech(
    model: nn.Module,
    text: str,
    vocab: dict,
    spk_id: int,
    mel_mean: float,
    mel_std: float,
    mel_min: float,
    mel_max: float,
    n_mels: int,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    win_length: int,
    fmin: float,
    fmax: float,
    max_text_len: int,
    num_ode_steps: int = 40,
    griffin_lim_iters: int = 60,
    temperature: float = 0.667,
    speed: float = 1.0,
    seed: int = 0,
) -> tuple:
    mu_frame, spk_emb, mel_mask, total, _ = infer_condition(
        model, text, vocab, max_text_len, spk_id, speed
    )
    x0 = mx.random.normal(shape=(1, total, n_mels), key=mx.random.key(seed)) * temperature
    mel_norm = euler_solve_decoder(
        model.decoder, x0, mu_frame, spk_emb, mel_mask, num_ode_steps
    )
    mx.eval(mel_norm)
    # De-normalize into ln-magnitude mel and clamp to the trained domain
    # so Griffin-Lim never sees energies the corpus could not produce.
    log_mel = np.asarray(mel_norm[0]).T * mel_std + mel_mean
    log_mel = np.clip(log_mel, mel_min, mel_max).astype(np.float32)
    mel_mag = np.exp(log_mel)
    waveform = librosa.feature.inverse.mel_to_audio(
        mel_mag,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,
        n_iter=griffin_lim_iters,
        fmin=fmin,
        fmax=fmax,
    )
    peak = float(np.abs(waveform).max())
    if peak > 1e-6:
        waveform = waveform / peak * 0.95
    return waveform.astype(np.float32), log_mel


@app.cell
def _(mo, tensor_dataset):
    if tensor_dataset is None:
        tts_speaker_ui = None
        _out = mo.md("_Build the tensor dataset to populate the speaker list._")
    else:
        _names = tensor_dataset["speakers"]
        tts_speaker_ui = mo.ui.dropdown(
            options={name: idx for idx, name in enumerate(_names)},
            value=_names[0],
            label="Speaker (LibriTTS id)",
        )
        _out = tts_speaker_ui
    _out
    return (tts_speaker_ui,)


@app.cell
def _(mo):
    tts_text_ui = mo.ui.text_area(
        value="hello world this is a test of the text to speech model",
        label="Input text",
    )
    tts_num_steps_ui = mo.ui.slider(5, 200, value=40, step=5, label="Euler ODE steps")
    tts_gl_iters_ui = mo.ui.slider(8, 128, value=60, step=4, label="Griffin-Lim iterations")
    tts_seed_ui = mo.ui.number(value=0, label="Sampling seed")
    tts_temperature_ui = mo.ui.slider(
        0.1, 1.5, value=0.667, step=0.05, label="Noise temperature"
    )
    tts_speed_ui = mo.ui.slider(0.5, 2.0, value=1.0, step=0.05, label="Speaking rate")
    tts_synth_btn = mo.ui.run_button(label="Synthesize")
    mo.vstack(
        [
            tts_text_ui,
            mo.hstack([tts_num_steps_ui, tts_gl_iters_ui, tts_seed_ui]),
            mo.hstack([tts_temperature_ui, tts_speed_ui]),
            tts_synth_btn,
        ]
    )
    return (
        tts_gl_iters_ui,
        tts_num_steps_ui,
        tts_seed_ui,
        tts_speed_ui,
        tts_synth_btn,
        tts_temperature_ui,
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
    tts_speaker_ui,
    tts_speed_ui,
    tts_synth_btn,
    tts_temperature_ui,
    tts_text_ui,
):
    if trained_model is None or tensor_dataset is None or char_vocab is None or tts_speaker_ui is None:
        _out = mo.md("_Train the model first (Section 5) before synthesizing._")
    elif not tts_synth_btn.value:
        _out = mo.md("Type a transcript, pick a speaker, and click **Synthesize**.")
    else:
        _wav, _log_mel = synthesize_speech(
            model=trained_model,
            text=tts_text_ui.value,
            vocab=char_vocab,
            spk_id=int(tts_speaker_ui.value),
            mel_mean=tensor_dataset["mel_mean"],
            mel_std=tensor_dataset["mel_std"],
            mel_min=tensor_dataset["mel_min"],
            mel_max=tensor_dataset["mel_max"],
            n_mels=tensor_dataset["n_mels"],
            sample_rate=int(target_sr_ui.value),
            hop_length=tensor_dataset["hop_length"],
            n_fft=tensor_dataset["n_fft"],
            win_length=tensor_dataset["win_length"],
            fmin=tensor_dataset["fmin"],
            fmax=tensor_dataset["fmax"],
            max_text_len=tensor_dataset["text_ids"].shape[1],
            num_ode_steps=int(tts_num_steps_ui.value),
            griffin_lim_iters=int(tts_gl_iters_ui.value),
            temperature=float(tts_temperature_ui.value),
            speed=float(tts_speed_ui.value),
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
