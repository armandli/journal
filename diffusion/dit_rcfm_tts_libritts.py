import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import hashlib
    import json
    import math
    import random
    import time
    import wave
    import warnings
    from dataclasses import dataclass, asdict
    from pathlib import Path
    from typing import Callable, Dict, List, Optional, Tuple

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.io.wavfile as scipy_wavfile
    import torch
    import torch.nn.functional as F
    import torchaudio
    import torchaudio.functional as AF
    import torchaudio.transforms as AT
    from torch import nn


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diffusion Transformer + 2D RoPE + Rectified Flow Matching for LibriTTS TTS

    ## Research Goal

    Train a **Diffusion Transformer** (adaLN-Zero DiT with **2D axial rotary
    position embeddings**) with **rectified conditional flow matching** on the
    LibriTTS corpus for text-to-speech synthesis. The model takes character
    tokens and a speaker id, and generates a normalized log-mel spectrogram
    which is then vocoded to audio with Griffin-Lim (an intentionally simple
    vocoder that lets us focus on the acoustic modelling and reflow behaviour).
    After the base rectified flow ("gen1") is trained, the notebook performs a
    **reverse-ODE reflow**: original real mel spectrograms ($x_1$) are inverted
    through gen1's ODE back to $x_0$, and a second flow model is trained on
    those $(x_0, x_1)$ couples (2-rectified flow).

    ## Flow Convention

    Let $x_0 \sim \mathcal{N}(0, I)$ be the source (noise) and $x_1$ be a
    normalized log-mel spectrogram. We use straight, linear conditional paths:

    $$x_t \;=\; (1 - t)\, x_0 \;+\; t\, x_1, \qquad t \in [0, 1],$$

    with velocity target

    $$v^{*}(x_t, t) \;=\; \frac{d x_t}{d t} \;=\; x_1 - x_0.$$

    The conditional flow-matching loss (masked to valid mel frames only) is

    $$\mathcal{L}_{\text{CFM}}(\theta) \;=\; \mathbb{E}_{t,\, x_0,\, (x_1, c, s)}
        \Big\lVert\, \big(v_{\theta}(x_t,\, t,\, c,\, s) \,-\, (x_1 - x_0)\big) \odot m \,\Big\rVert^{2} / \lVert m \rVert_1.$$

    Generation solves $dx/dt = v_\theta(x, t, c, s)$ forward from $t = 0$ to
    $t = 1$; **inversion (used for reflow)** solves the same ODE backward from
    $t = 1$ to $t = 0$.

    ## Classifier-Free Guidance

    During training we jointly drop the text and speaker conditioning to a
    dedicated null token / null speaker with probability `p_uncond`. At sampling
    the guided velocity is

    $$\tilde{v} \;=\; v_{\text{uncond}} \,+\, w \cdot \big(v_{\text{cond}} - v_{\text{uncond}}\big).$$

    ## Reflow (2-Rectified Flow, Reverse Coupling)

    For every training utterance $(x_1, c, s)$, integrate the trained gen1
    velocity field **backward** from $t = 1$ to $t = 0$ to obtain
    $x_0 = \Phi^{-1}(x_1, c, s)$. Store $(x_0, x_1, c, s)$ pairs on CPU in fp16
    and train the reflow model on that fixed dataset with the same masked flow
    matching objective. In this reverse coupling $x_1$ stays exactly on the
    data distribution and labels are the real ones; the cost is that $x_0$
    only approximately matches $\mathcal{N}(0, I)$, and enough ODE steps are
    needed for that gap to close.

    ## Outline (12 sections)

    1. **Title & Research Goal** (this cell).
    2. **Data Exploration** — download / list LibriTTS via `_walker`, duration
       and text histograms, per-speaker counts, one sample waveform + log-mel,
       Griffin-Lim vocoder ceiling audio.
    3. **Dataset Creation** — mel extraction + on-disk cache, normalization
       stats, char tokenizer, speaker map, splits, dataset class + collate,
       length-bucketed batch sampler.
    4. **Model Definition** — `DiffusionTransformerTTSV1` composed of
       versioned building blocks; adaLN-Zero self-attn 2D RoPE + cross-attn
       with soft diagonal alignment prior.
    5. **Training — 1-Rectified Flow (gen1)** — AdamW + warmup + AMP + EMA.
    6. **Hyperparameter Search** (optional).
    7. **Validation & Cross-Validation** — masked flow-matching loss, per-timestep-bin
       loss, straightness, k-fold CV, optional ASR-based CER (with vocoder ceiling).
    8. **Reflow (2-Rectified Flow, reverse-ODE couplings)**.
    9. **Results** — loss curves, ground-truth vs gen1 vs reflow mel grid with audio
       players, NFE-sweep deviation, denoising trajectory, cross-attention alignment.
    10. **Sampling from the Trained Model** — text + speaker + duration UI.
    11. **Save Trained Model** — state dict + JSON sidecar under `models/`.
    12. **Load Saved Model for Sampling** — reload + sample from any saved checkpoint.
    """)
    return


@app.function
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.function
def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_cuda:
        warnings.warn(
            "CUDA is not available; falling back to CPU. Training this diffusion "
            "transformer on CPU will be very slow.",
            RuntimeWarning,
            stacklevel=2,
        )
    return torch.device("cpu")


@app.function
def select_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


@app.function
def make_grad_scaler(device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> "torch.amp.GradScaler":
    return torch.amp.GradScaler(device.type, enabled=(use_amp and amp_dtype == torch.float16))


@app.function
def amp_backward_supported(device: torch.device, amp_dtype: torch.dtype) -> Tuple[bool, str]:
    probe = nn.ModuleDict(
        {
            "patch": nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(2, 2)),
            "norm": nn.LayerNorm(8, elementwise_affine=False),
            "qkv": nn.Linear(8, 24),
            "out": nn.Linear(8, 8),
        }
    ).to(device)
    x = torch.randn(2, 1, 4, 4, device=device)
    try:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            h = probe["patch"](x).flatten(2).transpose(1, 2)
            q, k, v = probe["qkv"](probe["norm"](h)).view(2, 4, 3, 2, 4).permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(2, 4, 8)
            out = probe["out"](F.gelu(attn, approximate="tanh") * F.silu(h))
        out.float().pow(2).mean().backward()
    except RuntimeError as err:
        return False, str(err).splitlines()[0]
    return True, ""


@app.function
def format_amp_status(
    device: torch.device,
    amp_dtype: torch.dtype,
    requested: bool,
    supported: bool,
    probe_error: str,
) -> str:
    dtype_name = str(amp_dtype).replace("torch.", "")
    if not requested:
        return f"`disabled` (unchecked) — training runs in float32 on `{device.type}`"
    if supported:
        return f"`enabled`, dtype = `{dtype_name}`"
    return (
        f"`disabled automatically` — a {dtype_name} autocast backward probe failed on "
        f"`{device.type}` (`{probe_error}`); training runs in float32. This is a "
        "backend/hardware limitation (e.g. oneDNN has no bf16 convolution backward "
        "on AVX-VNNI-2 CPUs, which the mel patch embedding needs), not a model error."
    )


@app.function
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@app.function
def cpu_warning_callout(device: torch.device, mo) -> object:
    if device.type == "cpu":
        return mo.callout(
            mo.md(
                "**CPU-only device detected.** Training this diffusion transformer "
                "on CPU will be very slow; a laptop-class GPU is strongly recommended."
            ),
            kind="warn",
        )
    return mo.md("")


@app.function
def device_summary_markdown(
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_requested: bool,
    amp_supported: bool,
    amp_probe_error: str,
    seed: int,
    mo,
) -> object:
    return mo.md(
        f"""**Active device**: `{device}`

**AMP**: {format_amp_status(device, amp_dtype, amp_requested, amp_supported, amp_probe_error)}

**Seed**: `{int(seed)}`
"""
    )


@app.cell
def _(mo):
    seed_ui = mo.ui.number(value=1337, label="Seed", start=0, stop=2**31 - 1)
    amp_ui = mo.ui.checkbox(value=True, label="Use Mixed Precision (AMP)")
    mo.hstack([seed_ui, amp_ui])
    return amp_ui, seed_ui


@app.cell
def _(amp_ui, mo, seed_ui):
    device = select_device(prefer_cuda=True)
    amp_dtype = select_amp_dtype(device)
    amp_supported, amp_probe_error = amp_backward_supported(device, amp_dtype)
    use_amp = bool(amp_ui.value) and amp_supported
    set_seed(int(seed_ui.value))
    mo.vstack(
        [
            cpu_warning_callout(device, mo),
            device_summary_markdown(
                device, amp_dtype, bool(amp_ui.value), amp_supported, amp_probe_error, int(seed_ui.value), mo
            ),
        ]
    )
    return amp_dtype, device, use_amp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data Exploration

    LibriTTS is downloaded (if missing) into `~/data/LibriTTS/<subset>` via
    `torchaudio.datasets.LIBRITTS(..., download=True)`. This notebook only
    uses that class for **downloading and file discovery** — the current
    torchaudio 2.10 requires TorchCodec (plus system FFmpeg libraries) to
    decode audio through `torchaudio.load`, and neither is available here.
    We instead:

    - list utterances via the dataset's `_walker` (sorted id strings
      `<speaker>_<chapter>_<segment>_<utterance>`) and its `_path`
      (`~/data/LibriTTS/<subset>`),
    - read raw waveforms with `scipy.io.wavfile.read` (verified 24 kHz,
      int16 mono; cast to float32 and divided by 32768 to reach $[-1, 1]$),
    - read normalized text from the sibling `<id>.normalized.txt` file.

    Everything downstream (mel spectrogram, inverse mel, Griffin-Lim, ASR,
    resampling) still uses `torchaudio.transforms` / `torchaudio.functional`
    / `torchaudio.pipelines`.

    **Subset sizes** — dev-clean / dev-other / test-clean / test-other are
    each ~1.2 GB archives (~8-10 h of audio). train-clean-100 is ~7.7 GB
    (~54 h). The larger train-clean-360 (~27 GB) and train-other-500 (~44 GB)
    are excluded from this dropdown because the whole mel cache is held in
    RAM here and 15 GB of system memory would not accommodate them.
    """)
    return


@app.cell
def _(mo):
    subset_ui = mo.ui.dropdown(
        options=[
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
        ],
        value="dev-clean",
        label="LibriTTS Subset",
    )
    min_dur_ui = mo.ui.dropdown(
        options={"0.5": 0.5, "1.0": 1.0, "1.5": 1.5, "2.0": 2.0},
        value="1.0",
        label="Min Duration (s)",
    )
    max_dur_ui = mo.ui.dropdown(
        options={"4.0": 4.0, "6.0": 6.0, "8.0": 8.0, "10.0": 10.0, "15.0": 15.0},
        value="8.0",
        label="Max Duration (s)",
    )
    mo.hstack([subset_ui, min_dur_ui, max_dur_ui])
    return max_dur_ui, min_dur_ui, subset_ui


@app.function
def data_root() -> Path:
    return Path.home() / "data"


@app.function
def load_libritts_walker(subset: str, root: Optional[Path] = None) -> Tuple[Path, List[str]]:
    r = root if root is not None else data_root()
    r.mkdir(parents=True, exist_ok=True)
    subset_dir = r / "LibriTTS" / subset
    should_download = not subset_dir.exists() or not any(subset_dir.iterdir())
    ds = torchaudio.datasets.LIBRITTS(
        root=str(r),
        url=subset,
        folder_in_archive="LibriTTS",
        download=should_download,
    )
    return Path(ds._path), list(ds._walker)


@app.function
def read_wav_scipy(wav_path: Path) -> Tuple[int, np.ndarray]:
    sr, x = scipy_wavfile.read(str(wav_path))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return int(sr), x


@app.function
def utterance_paths(subset_path: Path, utt_id: str) -> Tuple[Path, Path]:
    parts = utt_id.split("_")
    speaker = parts[0]
    chapter = parts[1]
    return (
        subset_path / speaker / chapter / f"{utt_id}.wav",
        subset_path / speaker / chapter / f"{utt_id}.normalized.txt",
    )


@app.function
def read_normalized_text(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8").strip()


@app.function
def wav_header_info(wav_path: Path) -> Tuple[int, int]:
    with wave.open(str(wav_path), "rb") as wf:
        return int(wf.getframerate()), int(wf.getnframes())


@app.function
def survey_utterances(
    subset_path: Path,
    walker: List[str],
    max_ids: Optional[int] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ids = walker if max_ids is None else walker[:max_ids]
    for utt_id in ids:
        wav_path, txt_path = utterance_paths(subset_path, utt_id)
        if not wav_path.exists() or not txt_path.exists():
            continue
        try:
            sr, n_samples = wav_header_info(wav_path)
            text = read_normalized_text(txt_path)
        except Exception:
            continue
        rows.append(
            {
                "utt_id": utt_id,
                "speaker": utt_id.split("_")[0],
                "wav_path": wav_path,
                "txt_path": txt_path,
                "num_samples": int(n_samples),
                "sample_rate": int(sr),
                "duration": float(n_samples / sr),
                "text": text,
                "num_chars": len(text),
            }
        )
    return rows


@app.cell
def _(subset_ui):
    subset_name = str(subset_ui.value)
    subset_path, walker = load_libritts_walker(subset_name)
    return subset_name, subset_path, walker


@app.cell
def _(subset_path, walker):
    all_utterances = survey_utterances(subset_path, walker)
    dataset_sample_rate = int(all_utterances[0]["sample_rate"]) if all_utterances else 24000
    return all_utterances, dataset_sample_rate


@app.function
def filter_by_duration(
    utterances: List[Dict[str, object]],
    min_seconds: float,
    max_seconds: float,
) -> List[Dict[str, object]]:
    return [u for u in utterances if min_seconds <= float(u["duration"]) <= max_seconds]


@app.cell
def _(all_utterances, max_dur_ui, min_dur_ui):
    filtered_utterances = filter_by_duration(
        all_utterances, float(min_dur_ui.value), float(max_dur_ui.value)
    )
    return (filtered_utterances,)


@app.function
def dataset_summary_markdown(
    subset_name: str,
    all_utterances: List[Dict[str, object]],
    filtered_utterances: List[Dict[str, object]],
    mo,
) -> object:
    n_all = len(all_utterances)
    hours_all = sum(float(u["duration"]) for u in all_utterances) / 3600.0
    n_kept = len(filtered_utterances)
    hours_kept = sum(float(u["duration"]) for u in filtered_utterances) / 3600.0
    n_speakers = len({str(u["speaker"]) for u in filtered_utterances})
    return mo.md(
        f"""
    **Subset**: `{subset_name}` — {n_all:,} utterances, total {hours_all:.2f} h.

    **After duration filter**: {n_kept:,} utterances kept (total {hours_kept:.2f} h), {n_speakers} unique speakers.
    """
    )


@app.cell
def _(all_utterances, filtered_utterances, mo, subset_name):
    dataset_summary_markdown(subset_name, all_utterances, filtered_utterances, mo)
    return


@app.function
def plot_duration_histogram(utterances: List[Dict[str, object]], title: str = "Utterance duration"):
    durs = [float(u["duration"]) for u in utterances]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(durs, bins=60, color="steelblue", alpha=0.85)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(all_utterances):
    plot_duration_histogram(all_utterances, title="LibriTTS utterance duration (all)")
    return


@app.function
def plot_text_length_histogram(utterances: List[Dict[str, object]], title: str = "Normalized text length"):
    lens = [int(u["num_chars"]) for u in utterances]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(lens, bins=60, color="darkgreen", alpha=0.85)
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(all_utterances):
    plot_text_length_histogram(all_utterances, title="LibriTTS normalized text length (all)")
    return


@app.function
def plot_utterances_per_speaker(utterances: List[Dict[str, object]], title: str = "Utterances per speaker"):
    counts: Dict[str, int] = {}
    for u in utterances:
        counts[str(u["speaker"])] = counts.get(str(u["speaker"]), 0) + 1
    sorted_counts = sorted(counts.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(sorted_counts)), sorted_counts, color="darkorange", alpha=0.85)
    ax.set_xlabel("Speaker (rank)")
    ax.set_ylabel("Utterances")
    ax.set_title(f"{title} ({len(sorted_counts)} speakers)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(all_utterances):
    plot_utterances_per_speaker(all_utterances, title="LibriTTS utterances per speaker")
    return


@app.function
def plot_frames_vs_characters(
    utterances: List[Dict[str, object]],
    hop_length: int,
    title: str = "Frames vs characters",
):
    chars = np.array([int(u["num_chars"]) for u in utterances], dtype=np.float64)
    frames = np.array(
        [int(round(float(u["num_samples"]) / hop_length)) for u in utterances],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(chars, frames, s=4, alpha=0.4)
    if chars.size > 0 and (chars > 0).any():
        slope = float(np.median(frames[chars > 0] / chars[chars > 0]))
        xs = np.linspace(0, float(chars.max()) + 1, 2)
        ax.plot(xs, slope * xs, "r-", lw=1.2, label=f"median frames/char = {slope:.2f}")
        ax.legend()
    ax.set_xlabel("Characters")
    ax.set_ylabel(f"Mel frames (hop={hop_length})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(all_utterances):
    plot_frames_vs_characters(all_utterances, hop_length=256, title="Mel frames vs char count (hop=256)")
    return


@app.function
def waveform_and_mel_figure(
    utterance: Dict[str, object],
    mel_config: Dict[str, object],
    title_prefix: str = "Sample",
):
    sr, x = read_wav_scipy(Path(str(utterance["wav_path"])))
    mel = AT.MelSpectrogram(
        sample_rate=sr,
        n_fft=int(mel_config["n_fft"]),
        win_length=int(mel_config["win_length"]),
        hop_length=int(mel_config["hop_length"]),
        n_mels=int(mel_config["n_mels"]),
        f_min=float(mel_config["f_min"]),
        f_max=float(mel_config["f_max"]),
        power=float(mel_config["power"]),
        center=True,
    )
    m = mel(torch.from_numpy(x))
    log_m = torch.log(torch.clamp(m, min=1e-5))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    axes[0].plot(np.arange(x.shape[0]) / sr, x, color="steelblue", lw=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{title_prefix} — waveform ({utterance['utt_id']})")
    axes[0].grid(True, alpha=0.3)
    axes[1].imshow(
        log_m.numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.shape[0] / sr, 0, int(mel_config["n_mels"])],
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel bin")
    axes[1].set_title(
        f"{title_prefix} — log-mel (n_mels={int(mel_config['n_mels'])}, hop={int(mel_config['hop_length'])})"
    )
    fig.tight_layout()
    return fig


@app.function
def default_mel_config(sample_rate: int = 24000) -> Dict[str, object]:
    return {
        "sample_rate": int(sample_rate),
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 100,
        "f_min": 0.0,
        "f_max": 12000.0,
        "power": 1.0,
        "log_eps": 1e-5,
        "center": True,
    }


@app.cell
def _(dataset_sample_rate):
    mel_config = default_mel_config(dataset_sample_rate)
    return (mel_config,)


@app.function
def sample_waveform_mel_view(
    filtered_utterances: List[Dict[str, object]],
    mel_config: Dict[str, object],
    mo,
) -> object:
    if not filtered_utterances:
        return mo.md("_No utterances match the duration filter — widen the range._")
    return waveform_and_mel_figure(filtered_utterances[0], mel_config, title_prefix="Sample #0")


@app.cell
def _(filtered_utterances, mel_config, mo):
    sample_waveform_mel_view(filtered_utterances, mel_config, mo)
    return


@app.function
def griffin_lim_from_log_mel(
    log_mel: torch.Tensor,
    mel_config: Dict[str, object],
    n_iter: int = 32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    target_device = device if device is not None else log_mel.device
    mel = log_mel.to(target_device)
    mag = torch.exp(mel)
    n_stft = int(mel_config["n_fft"]) // 2 + 1
    inv = AT.InverseMelScale(
        n_stft=n_stft,
        n_mels=int(mel_config["n_mels"]),
        sample_rate=int(mel_config["sample_rate"]),
        f_min=float(mel_config["f_min"]),
        f_max=float(mel_config["f_max"]),
        mel_scale="htk",
        norm=None,
        driver="gels",
    ).to(target_device)
    gl = AT.GriffinLim(
        n_fft=int(mel_config["n_fft"]),
        win_length=int(mel_config["win_length"]),
        hop_length=int(mel_config["hop_length"]),
        power=float(mel_config["power"]),
        n_iter=int(n_iter),
    ).to(target_device)
    spec = inv(mag)
    return gl(spec).detach().cpu()


@app.function
def vocoder_ceiling_view(
    filtered_utterances: List[Dict[str, object]],
    mel_config: Dict[str, object],
    device: torch.device,
    mo,
    n_iter: int = 32,
) -> object:
    if not filtered_utterances:
        return mo.md("_No utterances match the duration filter._")
    utt = filtered_utterances[0]
    sr, x = read_wav_scipy(Path(str(utt["wav_path"])))
    mel = AT.MelSpectrogram(
        sample_rate=sr,
        n_fft=int(mel_config["n_fft"]),
        win_length=int(mel_config["win_length"]),
        hop_length=int(mel_config["hop_length"]),
        n_mels=int(mel_config["n_mels"]),
        f_min=float(mel_config["f_min"]),
        f_max=float(mel_config["f_max"]),
        power=float(mel_config["power"]),
        center=True,
    ).to(device)
    log_m = torch.log(torch.clamp(mel(torch.from_numpy(x).to(device)), min=1e-5))
    recon = griffin_lim_from_log_mel(log_m, mel_config, n_iter=n_iter, device=device).numpy()
    return mo.vstack(
        [
            mo.md(
                f"**Vocoder ceiling** — ground-truth mel through Griffin-Lim "
                f"({int(mel_config['n_mels'])} mels, hop={int(mel_config['hop_length'])}, "
                f"{n_iter} iterations) for `{utt['utt_id']}`. Text: _{str(utt['text'])[:120]}_"
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md("**Original**"), mo.audio(src=x, rate=sr)]),
                    mo.vstack([mo.md("**Griffin-Lim resynth**"), mo.audio(src=recon, rate=sr)]),
                ]
            ),
        ]
    )


@app.cell
def _(device, filtered_utterances, mel_config, mo):
    vocoder_ceiling_view(filtered_utterances, mel_config, device, mo, n_iter=32)
    return


@app.function
def deterministic_split(
    items: List[object],
    fractions: Tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 1234,
) -> Tuple[List[object], List[object], List[object]]:
    n = len(items)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return (
        [items[i] for i in train_idx],
        [items[i] for i in val_idx],
        [items[i] for i in test_idx],
    )


@app.cell
def _(filtered_utterances):
    train_utts, val_utts, test_utts = deterministic_split(
        filtered_utterances, fractions=(0.9, 0.05, 0.05), seed=1234
    )
    return test_utts, train_utts, val_utts


@app.function
def splits_summary_markdown(
    train_utts: List[Dict[str, object]],
    val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    mo,
) -> object:
    def _row(name, items):
        spk = len({str(u["speaker"]) for u in items})
        h = sum(float(u["duration"]) for u in items) / 3600.0
        return f"| {name} | {len(items):,} | {spk} | {h:.2f} |"

    return mo.md(
        "| Split | Utterances | Unique speakers | Total duration (h) |\n"
        "|---|---|---|---|\n"
        + _row("Train", train_utts)
        + "\n"
        + _row("Val", val_utts)
        + "\n"
        + _row("Test", test_utts)
    )


@app.cell
def _(mo, test_utts, train_utts, val_utts):
    splits_summary_markdown(train_utts, val_utts, test_utts, mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Dataset Creation

    - **Mel extraction** — a batched, GPU-if-available extractor computes each
      utterance's log-mel and caches the whole subset once to
      `~/data/LibriTTS/mel_cache/<subset>_<hash>.pt` in fp16 together with the
      metadata (ids, texts, speakers, frame counts). Reruns hit the cache.
    - **Normalization** — per mel-bin mean and std computed on the **training
      split only**, so the normalized $x_1$ has roughly unit variance and
      matches the $\mathcal{N}(0, I)$ source used at $t = 0$.
    - **Tokenizer** — character tokenizer built from the training split's
      lowercased normalized text with `<pad>` = 0, `<null>` = 1 (the
      classifier-free-guidance unconditional token) and `<unk>` = 2.
    - **Speaker map** — index 0 is the null speaker used for CFG dropout;
      LibriTTS speakers seen in the split are mapped to 1..S.
    - **Collate** — pads each batch to the longest utterance in it,
      trims each utterance's frame count to a multiple of `patch_time` so no
      2D patch straddles the padding boundary, and returns boolean masks
      (`True = valid`) over mel-time and text-token axes.
    - **Bucketed batch sampler** — shuffle → chunk → sort each chunk by
      length → batch → shuffle batches, so batches don't waste compute on
      padding. Determinism: the sampler seeds `default_rng(seed + epoch)`;
      call `set_epoch(epoch)` to advance.

    The `loaders` cell below is a **display-only preview** built with a
    default `patch_time=4`. Training / evaluation / reflow cells always
    rebuild loaders through `make_datasets` with the trained model's actual
    `patch_time`, so a `patch_time=2` or `patch_time=8` model does not
    accidentally use these preview loaders.
    """)
    return


@app.function
def hash_mel_config(mel_config: Dict[str, object]) -> str:
    canonical = json.dumps(
        {k: mel_config[k] for k in sorted(mel_config.keys())}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()[:12]


@app.function
def mel_cache_path(
    subset: str,
    mel_config: Dict[str, object],
    cache_root: Optional[Path] = None,
) -> Path:
    root = cache_root if cache_root is not None else data_root() / "LibriTTS" / "mel_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{subset}_{hash_mel_config(mel_config)}.pt"


@app.function
def build_mel_extractor(mel_config: Dict[str, object], device: torch.device) -> AT.MelSpectrogram:
    return AT.MelSpectrogram(
        sample_rate=int(mel_config["sample_rate"]),
        n_fft=int(mel_config["n_fft"]),
        win_length=int(mel_config["win_length"]),
        hop_length=int(mel_config["hop_length"]),
        n_mels=int(mel_config["n_mels"]),
        f_min=float(mel_config["f_min"]),
        f_max=float(mel_config["f_max"]),
        power=float(mel_config["power"]),
        center=bool(mel_config["center"]),
    ).to(device)


@app.function
def compute_log_mel(
    waveform: torch.Tensor,
    extractor: AT.MelSpectrogram,
    log_eps: float,
) -> torch.Tensor:
    m = extractor(waveform)
    return torch.log(torch.clamp(m, min=float(log_eps)))


@app.function
def build_or_load_mel_cache(
    subset: str,
    utterances: List[Dict[str, object]],
    mel_config: Dict[str, object],
    device: torch.device,
    cache_root: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    path = mel_cache_path(subset, mel_config, cache_root=cache_root)
    if path.exists():
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        ids_in_cache = set(payload["utt_ids"])
        ids_needed = {str(u["utt_id"]) for u in utterances}
        if ids_needed.issubset(ids_in_cache):
            return payload
    extractor = build_mel_extractor(mel_config, device)
    logs: List[torch.Tensor] = []
    frames: List[int] = []
    ids: List[str] = []
    texts: List[str] = []
    speakers: List[str] = []
    total = len(utterances)
    for i, u in enumerate(utterances):
        _sr, x = read_wav_scipy(Path(str(u["wav_path"])))
        wav = torch.from_numpy(x).to(device)
        with torch.no_grad():
            lm = compute_log_mel(wav, extractor, float(mel_config["log_eps"]))
        logs.append(lm.to(torch.float16).cpu())
        frames.append(int(lm.shape[-1]))
        ids.append(str(u["utt_id"]))
        texts.append(str(u["text"]))
        speakers.append(str(u["speaker"]))
        if progress_cb is not None and (i + 1) % 32 == 0:
            progress_cb(i + 1, total)
    if progress_cb is not None:
        progress_cb(total, total)
    payload = {
        "utt_ids": ids,
        "texts": texts,
        "speakers": speakers,
        "frames": frames,
        "log_mels": logs,
        "mel_config": mel_config,
        "subset": subset,
    }
    torch.save(payload, str(path))
    return payload


@app.function
def mel_cache_building_view(
    subset_name: str,
    filtered_utterances: List[Dict[str, object]],
    mel_config: Dict[str, object],
    device: torch.device,
    mo,
) -> Optional[Dict[str, object]]:
    if not filtered_utterances:
        mo.output.replace(mo.md("_No utterances match the duration filter — widen the range._"))
        return None
    mo.output.replace(mo.md(f"Building mel cache for `{subset_name}` — this may take a while on first run..."))
    cache = build_or_load_mel_cache(
        subset=subset_name,
        utterances=filtered_utterances,
        mel_config=mel_config,
        device=device,
        progress_cb=lambda done, total: mo.output.replace(mo.md(f"Mel extraction: {done}/{total}")),
    )
    mo.output.replace(
        mo.md(
            f"**Mel cache ready** at `{mel_cache_path(subset_name, mel_config)}` — "
            f"{len(cache['utt_ids']):,} utterances."
        )
    )
    return cache


@app.cell
def _(device, filtered_utterances, mel_config, mo, subset_name):
    mel_cache = mel_cache_building_view(subset_name, filtered_utterances, mel_config, device, mo)
    return (mel_cache,)


@app.function
def compute_mel_stats(
    cache: Dict[str, object],
    utt_id_subset: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    wanted = set(utt_id_subset) if utt_id_subset is not None else None
    total_sum: Optional[torch.Tensor] = None
    total_sqsum: Optional[torch.Tensor] = None
    total_frames = 0
    for utt_id, lm in zip(cache["utt_ids"], cache["log_mels"]):
        if wanted is not None and utt_id not in wanted:
            continue
        lm_f = lm.to(torch.float32)
        if total_sum is None:
            total_sum = torch.zeros(lm_f.shape[0], dtype=torch.float64)
            total_sqsum = torch.zeros(lm_f.shape[0], dtype=torch.float64)
        total_sum += lm_f.sum(dim=1).to(torch.float64)
        total_sqsum += (lm_f * lm_f).sum(dim=1).to(torch.float64)
        total_frames += int(lm_f.shape[1])
    if total_sum is None or total_frames == 0:
        raise RuntimeError("Empty subset for mel stats.")
    mean = (total_sum / total_frames).to(torch.float32)
    var = (total_sqsum / total_frames - (total_sum / total_frames) ** 2).clamp_min(1e-6).to(torch.float32)
    return mean, var.sqrt()


@app.function
def mel_stats_view(
    mel_cache: Optional[Dict[str, object]],
    train_utts: List[Dict[str, object]],
    mo,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mel_cache is None:
        mo.output.replace(mo.md("_No cache yet._"))
        return torch.zeros(1), torch.ones(1)
    train_ids = [str(u["utt_id"]) for u in train_utts]
    mean, std = compute_mel_stats(mel_cache, utt_id_subset=train_ids)
    mo.output.replace(
        mo.md(
            f"**Training-set mel normalization** — mean range "
            f"[{mean.min().item():.3f}, {mean.max().item():.3f}], "
            f"std range [{std.min().item():.3f}, {std.max().item():.3f}] "
            f"over {mean.shape[0]} bins."
        )
    )
    return mean, std


@app.cell
def _(mel_cache, mo, train_utts):
    mel_norm_mean, mel_norm_std = mel_stats_view(mel_cache, train_utts, mo)
    return mel_norm_mean, mel_norm_std


@app.class_definition
class CharTokenizerV1:
    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        pad_id: int = 0,
        null_id: int = 1,
        unk_id: int = 2,
        lowercase: bool = True,
    ):
        self.pad_id = int(pad_id)
        self.null_id = int(null_id)
        self.unk_id = int(unk_id)
        self.lowercase = bool(lowercase)
        if vocab is None:
            vocab = ["<pad>", "<null>", "<unk>"]
        self.vocab = list(vocab)
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}

    @classmethod
    def build_from_texts(cls, texts: List[str], lowercase: bool = True) -> "CharTokenizerV1":
        chars: Dict[str, int] = {}
        for t in texts:
            for c in (t.lower() if lowercase else t):
                chars[c] = chars.get(c, 0) + 1
        specials = ["<pad>", "<null>", "<unk>"]
        ordered = sorted(chars.keys())
        vocab = specials + [c for c in ordered if c not in specials]
        return cls(vocab=vocab, lowercase=lowercase)

    def encode(self, text: str) -> List[int]:
        t = text.lower() if self.lowercase else text
        return [self.char_to_id.get(c, self.unk_id) for c in t]

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def null_sequence(self) -> List[int]:
        return [self.null_id]

    def __len__(self) -> int:
        return len(self.vocab)


@app.cell
def _(train_utts):
    text_tokenizer = CharTokenizerV1.build_from_texts([str(u["text"]) for u in train_utts])
    return (text_tokenizer,)


@app.function
def build_speaker_map(train_utts: List[Dict[str, object]]) -> Dict[str, int]:
    ordered = sorted({str(u["speaker"]) for u in train_utts})
    return {s: i + 1 for i, s in enumerate(ordered)}


@app.function
def tokenizer_and_speaker_summary(
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mo,
) -> object:
    return mo.md(
        f"**Tokenizer**: {len(tokenizer)} chars in the vocabulary "
        f"(pad={tokenizer.pad_id}, null={tokenizer.null_id}, "
        f"unk={tokenizer.unk_id}).\n\n"
        f"**Speakers**: {len(speaker_to_id)} (index 0 is the null speaker for CFG dropout)."
    )


@app.cell
def _(mo, text_tokenizer, train_utts):
    speaker_to_id = build_speaker_map(train_utts)
    tokenizer_and_speaker_summary(text_tokenizer, speaker_to_id, mo)
    return (speaker_to_id,)


@app.function
def frames_per_char_stat(
    train_utts: List[Dict[str, object]],
    hop_length: int,
) -> float:
    ratios = []
    for u in train_utts:
        n_chars = max(int(u["num_chars"]), 1)
        n_frames = max(int(round(float(u["num_samples"]) / hop_length)), 1)
        ratios.append(n_frames / n_chars)
    return float(np.median(ratios)) if ratios else 5.5


@app.cell
def _(mel_config, train_utts):
    frames_per_char = frames_per_char_stat(train_utts, hop_length=int(mel_config["hop_length"]))
    return (frames_per_char,)


@app.class_definition
class LibriTTSMelDatasetV1(torch.utils.data.Dataset):
    def __init__(
        self,
        cache: Dict[str, object],
        utt_id_subset: List[str],
        tokenizer: CharTokenizerV1,
        speaker_to_id: Dict[str, int],
        mel_mean: torch.Tensor,
        mel_std: torch.Tensor,
    ):
        id_to_idx = {uid: i for i, uid in enumerate(cache["utt_ids"])}
        self.indices = [id_to_idx[u] for u in utt_id_subset if u in id_to_idx]
        self.cache = cache
        self.tokenizer = tokenizer
        self.speaker_to_id = speaker_to_id
        self.mel_mean = mel_mean.to(torch.float32)
        self.mel_std = mel_std.to(torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def frame_length(self, i: int) -> int:
        return int(self.cache["frames"][self.indices[i]])

    def __getitem__(self, i: int) -> Dict[str, object]:
        idx = self.indices[i]
        lm = self.cache["log_mels"][idx].to(torch.float32)
        lm_norm = (lm - self.mel_mean[:, None]) / self.mel_std[:, None]
        text = self.cache["texts"][idx]
        tok = torch.tensor(self.tokenizer.encode(text), dtype=torch.int64)
        speaker = str(self.cache["speakers"][idx])
        spk_id = int(self.speaker_to_id.get(speaker, 0))
        return {
            "mel": lm_norm,
            "frames": int(lm.shape[-1]),
            "tokens": tok,
            "num_tokens": int(tok.shape[0]),
            "speaker": spk_id,
            "utt_id": str(self.cache["utt_ids"][idx]),
            "text": text,
        }


@app.function
def collate_mel_batch(
    samples: List[Dict[str, object]],
    pad_id: int,
    patch_time: int = 4,
) -> Dict[str, torch.Tensor]:
    n_mels = samples[0]["mel"].shape[0]
    trim_frames = [int(s["frames"]) - (int(s["frames"]) % max(int(patch_time), 1)) for s in samples]
    trim_frames = [max(f, int(patch_time)) for f in trim_frames]
    max_frames = max(trim_frames)
    max_tokens = max(int(s["num_tokens"]) for s in samples)
    max_tokens = max(max_tokens, 1)
    B = len(samples)
    mel = torch.zeros(B, n_mels, max_frames, dtype=torch.float32)
    mel_mask = torch.zeros(B, max_frames, dtype=torch.bool)
    tokens = torch.full((B, max_tokens), int(pad_id), dtype=torch.int64)
    token_mask = torch.zeros(B, max_tokens, dtype=torch.bool)
    speakers = torch.zeros(B, dtype=torch.int64)
    frames_out: List[int] = []
    tokens_out: List[int] = []
    ids: List[str] = []
    texts: List[str] = []
    for b, s in enumerate(samples):
        f = trim_frames[b]
        mel[b, :, :f] = s["mel"][:, :f]
        mel_mask[b, :f] = True
        n_t = int(s["num_tokens"])
        tokens[b, :n_t] = s["tokens"][:n_t]
        token_mask[b, :n_t] = True
        speakers[b] = int(s["speaker"])
        frames_out.append(int(f))
        tokens_out.append(int(n_t))
        ids.append(str(s["utt_id"]))
        texts.append(str(s["text"]))
    return {
        "mel": mel,
        "mel_mask": mel_mask,
        "tokens": tokens,
        "token_mask": token_mask,
        "speakers": speakers,
        "frames": torch.tensor(frames_out, dtype=torch.int64),
        "num_tokens": torch.tensor(tokens_out, dtype=torch.int64),
        "utt_ids": ids,
        "texts": texts,
    }


@app.class_definition
class BucketedBatchSamplerV1(torch.utils.data.Sampler):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int = 16,
        bucket_multiplier: int = 8,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_multiplier = int(bucket_multiplier)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / max(self.batch_size, 1))

    def __iter__(self):
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + int(self.epoch))
        idxs = np.arange(n)
        if self.shuffle:
            rng.shuffle(idxs)
        chunk = self.batch_size * max(self.bucket_multiplier, 1)
        batches: List[List[int]] = []
        for start in range(0, n, chunk):
            group = idxs[start : start + chunk].tolist()
            group.sort(key=lambda i: self.lengths[i])
            for bs in range(0, len(group), self.batch_size):
                batches.append(group[bs : bs + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        for b in batches:
            yield b


@app.function
def make_datasets(
    mel_cache: Dict[str, object],
    train_utts: List[Dict[str, object]],
    val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    batch_size: int = 8,
    patch_time: int = 4,
    seed: int = 0,
    bucket_multiplier: int = 8,
    num_workers: int = 0,
) -> Dict[str, object]:
    train_ds = LibriTTSMelDatasetV1(
        cache=mel_cache,
        utt_id_subset=[str(u["utt_id"]) for u in train_utts],
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        mel_mean=mel_mean,
        mel_std=mel_std,
    )
    val_ds = LibriTTSMelDatasetV1(
        cache=mel_cache,
        utt_id_subset=[str(u["utt_id"]) for u in val_utts],
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        mel_mean=mel_mean,
        mel_std=mel_std,
    )
    test_ds = LibriTTSMelDatasetV1(
        cache=mel_cache,
        utt_id_subset=[str(u["utt_id"]) for u in test_utts],
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        mel_mean=mel_mean,
        mel_std=mel_std,
    )
    collate = lambda batch: collate_mel_batch(batch, pad_id=tokenizer.pad_id, patch_time=int(patch_time))
    train_sampler = BucketedBatchSamplerV1(
        lengths=[train_ds.frame_length(i) for i in range(len(train_ds))],
        batch_size=batch_size,
        bucket_multiplier=bucket_multiplier,
        shuffle=True,
        seed=seed,
    )
    val_sampler = BucketedBatchSamplerV1(
        lengths=[val_ds.frame_length(i) for i in range(len(val_ds))],
        batch_size=batch_size,
        bucket_multiplier=bucket_multiplier,
        shuffle=False,
        seed=seed + 1,
    )
    test_sampler = BucketedBatchSamplerV1(
        lengths=[test_ds.frame_length(i) for i in range(len(test_ds))],
        batch_size=batch_size,
        bucket_multiplier=bucket_multiplier,
        shuffle=False,
        seed=seed + 2,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=collate, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_sampler=test_sampler, collate_fn=collate, num_workers=num_workers
    )
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
        "test_sampler": test_sampler,
    }


@app.function
def format_batch_summary(batch: Dict[str, object]) -> str:
    return (
        "**Loaders ready** — one train batch:\n\n"
        f"- mel: `{tuple(batch['mel'].shape)}` `{batch['mel'].dtype}`\n"
        f"- mel_mask: `{tuple(batch['mel_mask'].shape)}` `{batch['mel_mask'].dtype}`\n"
        f"- tokens: `{tuple(batch['tokens'].shape)}` `{batch['tokens'].dtype}`\n"
        f"- token_mask: `{tuple(batch['token_mask'].shape)}` `{batch['token_mask'].dtype}`\n"
        f"- speakers: `{tuple(batch['speakers'].shape)}` `{batch['speakers'].dtype}`\n"
        f"- frames per utt: `{batch['frames'].tolist()}`\n"
        f"- tokens per utt: `{batch['num_tokens'].tolist()}`"
    )


@app.function
def preview_loaders_view(
    mel_cache: Optional[Dict[str, object]],
    train_utts: List[Dict[str, object]],
    val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mo,
) -> Optional[Dict[str, object]]:
    if mel_cache is None:
        mo.output.replace(mo.md("_Mel cache not ready yet._"))
        return None
    ldrs = make_datasets(
        mel_cache=mel_cache,
        train_utts=train_utts,
        val_utts=val_utts,
        test_utts=test_utts,
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        mel_mean=mel_mean,
        mel_std=mel_std,
        batch_size=4,
        patch_time=4,
        seed=0,
    )
    batch = next(iter(ldrs["train_loader"]))
    mo.output.replace(mo.md(format_batch_summary(batch)))
    return ldrs


@app.cell
def _(
    mel_cache,
    mel_norm_mean,
    mel_norm_std,
    mo,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    train_utts,
    val_utts,
):
    preview_loaders = preview_loaders_view(
        mel_cache, train_utts, val_utts, test_utts, text_tokenizer,
        speaker_to_id, mel_norm_mean, mel_norm_std, mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Model Definition — DiT with 2D RoPE, adaLN-Zero, cross-attention alignment prior

    | Component | Class | Notes |
    |---|---|---|
    | 2D RoPE cache | `RotaryEmbedding2DV1` | half of `head_dim` rotates with freq row, half with time col |
    | 1D RoPE (text) | `RotaryEmbedding1DV1` | same code path, single axis |
    | Text embedding + transformer | `TextEncoderV1` + `TextEncoderBlockV1` | pre-norm, 1D RoPE self-attn, GELU MLP, key padding mask |
    | Mel patch embed | `PatchEmbed2DV1` | `Conv2d(kernel=stride=(patch_freq, patch_time))` |
    | Time embedding | `TimestepEmbedV1` | sinusoidal @ `t * 1000` + 2-layer MLP |
    | Speaker embedding | `SpeakerEmbedV1` | table with null-speaker index 0 + CFG dropout |
    | Self-attention (mel) | `MultiHeadSelfAttentionRoPE2DV1` | 2D RoPE on q, k; time-axis key padding mask |
    | Cross-attention | `MultiHeadCrossAttentionAlignmentV1` | soft diagonal Gaussian alignment prior + text mask |
    | Feed-forward | `FeedForwardV1` | GELU MLP |
    | DiT block | `DiTBlockTTSV1` | adaLN-Zero self-attn + cross-attn + MLP |
    | Final layer | `FinalLayerV1` | adaLN + zero-init linear + unpatchify |
    | Top-level | `DiffusionTransformerTTSV1` | composes all of the above, predicts velocity |

    The **soft diagonal alignment prior** adds a log-Gaussian additive bias
    $-(u - s)^2 / (2 \sigma^2)$ to the cross-attention logits, with $u$ the
    normalized time-column position and $s$ the normalized text-token position.
    That gives the model a monotonic-alignment prior, which is critical when
    training on only a few hours of TTS data. Set `alignment_sigma <= 0` to
    disable it (ablation). For diagnostic plots, pass
    `use_align_prior=False` to `DiffusionTransformerTTSV1.forward` to see the
    content-only cross-attention alongside the prior-augmented one.
    """)
    return


@app.class_definition
@dataclass
class DiTTTSConfigV1:
    n_mels: int = 100
    patch_freq: int = 20
    patch_time: int = 4
    hidden_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    text_layers: int = 4
    text_heads: int = 6
    mlp_ratio: float = 4.0
    vocab_size: int = 64
    num_speakers: int = 128
    rope_base: float = 10000.0
    time_max_period: float = 10000.0
    time_scale: float = 1000.0
    alignment_sigma: float = 0.10
    use_freq_row_embed: bool = True
    pad_id: int = 0
    null_id: int = 1

    def __post_init__(self) -> None:
        if self.n_mels % self.patch_freq != 0:
            raise ValueError(f"n_mels ({self.n_mels}) must be divisible by patch_freq ({self.patch_freq}).")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")
        head = self.hidden_dim // self.num_heads
        if head % 4 != 0:
            raise ValueError(f"head_dim ({head}) must be divisible by 4 for 2D axial RoPE.")
        if self.hidden_dim % self.text_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by text_heads ({self.text_heads}).")
        th = self.hidden_dim // self.text_heads
        if th % 2 != 0:
            raise ValueError(f"text head_dim ({th}) must be divisible by 2 for 1D RoPE.")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def text_head_dim(self) -> int:
        return self.hidden_dim // self.text_heads

    @property
    def freq_rows(self) -> int:
        return self.n_mels // self.patch_freq


@app.function
def dit_tts_presets() -> Dict[str, Dict[str, object]]:
    return {
        "tiny": {
            "hidden_dim": 192, "depth": 4, "num_heads": 3,
            "text_layers": 2, "text_heads": 3,
            "patch_freq": 20, "patch_time": 4, "mlp_ratio": 4.0,
        },
        "small": {
            "hidden_dim": 384, "depth": 8, "num_heads": 6,
            "text_layers": 4, "text_heads": 6,
            "patch_freq": 20, "patch_time": 4, "mlp_ratio": 4.0,
        },
        "base": {
            "hidden_dim": 512, "depth": 12, "num_heads": 8,
            "text_layers": 6, "text_heads": 8,
            "patch_freq": 20, "patch_time": 4, "mlp_ratio": 4.0,
        },
    }


@app.function
def make_dit_tts_config(**fields) -> Tuple[Optional[DiTTTSConfigV1], str]:
    try:
        return DiTTTSConfigV1(**fields), ""
    except (TypeError, ValueError) as err:
        return None, str(err)


@app.function
def build_1d_rope_cache(
    max_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 2 for 1D RoPE.")
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


@app.function
def build_2d_rope_cache_dynamic(
    freq_rows: int,
    time_cols: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for 2D axial RoPE.")
    d_axis = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, d_axis, 2, dtype=torch.float32, device=device) / d_axis))
    row_positions = (
        torch.arange(freq_rows, dtype=torch.float32, device=device)
        .view(-1, 1).expand(-1, time_cols).reshape(-1)
    )
    col_positions = (
        torch.arange(time_cols, dtype=torch.float32, device=device)
        .view(1, -1).expand(freq_rows, -1).reshape(-1)
    )
    angles_row = torch.outer(row_positions, freqs)
    angles_col = torch.outer(col_positions, freqs)
    cos_row = torch.cat([angles_row.cos(), angles_row.cos()], dim=-1)
    sin_row = torch.cat([angles_row.sin(), angles_row.sin()], dim=-1)
    cos_col = torch.cat([angles_col.cos(), angles_col.cos()], dim=-1)
    sin_col = torch.cat([angles_col.sin(), angles_col.sin()], dim=-1)
    return cos_row, sin_row, cos_col, sin_col


@app.function
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


@app.function
def apply_1d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig = x.dtype
    x_f = x.to(torch.float32)
    L = x_f.shape[-2]
    c = cos[:L].to(x_f.dtype)
    s = sin[:L].to(x_f.dtype)
    while c.ndim < x_f.ndim:
        c = c.unsqueeze(0)
        s = s.unsqueeze(0)
    return (x_f * c + rotate_half(x_f) * s).to(orig)


@app.function
def apply_2d_rope(
    x: torch.Tensor,
    cos_row: torch.Tensor,
    sin_row: torch.Tensor,
    cos_col: torch.Tensor,
    sin_col: torch.Tensor,
) -> torch.Tensor:
    orig = x.dtype
    x_f = x.to(torch.float32)
    d = x_f.shape[-1]
    d_axis = d // 2
    x_row = x_f[..., :d_axis]
    x_col = x_f[..., d_axis:]
    L = x_row.shape[-2]
    cr = cos_row[:L].to(x_row.dtype)
    sr = sin_row[:L].to(x_row.dtype)
    cc = cos_col[:L].to(x_col.dtype)
    sc = sin_col[:L].to(x_col.dtype)
    while cr.ndim < x_row.ndim:
        cr = cr.unsqueeze(0)
        sr = sr.unsqueeze(0)
        cc = cc.unsqueeze(0)
        sc = sc.unsqueeze(0)
    x_row = x_row * cr + rotate_half(x_row) * sr
    x_col = x_col * cc + rotate_half(x_col) * sc
    return torch.cat([x_row, x_col], dim=-1).to(orig)


@app.class_definition
class RotaryEmbedding1DV1(nn.Module):
    def __init__(self, head_dim: int = 64, base: float = 10000.0, max_len: int = 2048):
        super().__init__()
        cos, sin = build_1d_rope_cache(max_len, head_dim, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.max_len = int(max_len)

    def get(self, length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if length > self.cos.shape[0]:
            cos, sin = build_1d_rope_cache(length, self.head_dim, self.base, device=device)
            return cos, sin
        return self.cos.to(device), self.sin.to(device)


@app.class_definition
class RotaryEmbedding2DV1(nn.Module):
    def __init__(self, head_dim: int = 64, base: float = 10000.0, max_rows: int = 16, max_cols: int = 512):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)

    def get(
        self, rows: int, cols: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return build_2d_rope_cache_dynamic(rows, cols, self.head_dim, self.base, device=device)


@app.class_definition
class PatchEmbed2DV1(nn.Module):
    def __init__(
        self, in_channels: int = 1, patch_freq: int = 20, patch_time: int = 4, hidden_dim: int = 384
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=(patch_freq, patch_time), stride=(patch_freq, patch_time)
        )
        self.patch_freq = int(patch_freq)
        self.patch_time = int(patch_time)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h = self.proj(x)
        b, d, fr, tc = h.shape
        return h.flatten(2).transpose(1, 2), int(fr), int(tc)


@app.class_definition
class TimestepEmbedV1(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        frequency_dim: int = 384,
        max_period: float = 10000.0,
        scale: float = 1000.0,
    ):
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.max_period = float(max_period)
        self.scale = float(scale)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t.to(torch.float32) * self.scale
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = t_scaled[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.frequency_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


@app.class_definition
class SpeakerEmbedV1(nn.Module):
    def __init__(self, num_speakers: int = 128, hidden_dim: int = 384):
        super().__init__()
        self.num_speakers = int(num_speakers)
        self.embedding = nn.Embedding(self.num_speakers + 1, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, speaker: torch.Tensor) -> torch.Tensor:
        return self.embedding(speaker)


@app.class_definition
class FeedForwardV1(nn.Module):
    def __init__(self, hidden_dim: int = 384, mlp_dim: int = 1536):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@app.class_definition
class MultiHeadSelfAttentionRoPE1DV1(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // self.num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, h = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_1d_rope(q, cos, sin)
        k = apply_1d_rope(k, cos, sin)
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(b, 1, 1, l).expand(b, self.num_heads, l, l).contiguous()
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(b, l, h)
        return self.proj(out)


@app.class_definition
class MultiHeadSelfAttentionRoPE2DV1(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // self.num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos_row: torch.Tensor,
        sin_row: torch.Tensor,
        cos_col: torch.Tensor,
        sin_col: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, h = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_2d_rope(q, cos_row, sin_row, cos_col, sin_col)
        k = apply_2d_rope(k, cos_row, sin_row, cos_col, sin_col)
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(b, 1, 1, l).expand(b, self.num_heads, l, l).contiguous()
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(b, l, h)
        return self.proj(out)


@app.function
def diagonal_alignment_bias(
    freq_rows: int,
    time_cols: List[int],
    text_lens: List[int],
    max_time: int,
    max_text: int,
    sigma: float,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    B = len(time_cols)
    bias = torch.zeros(B, num_heads, freq_rows * max_time, max_text, device=device, dtype=torch.float32)
    if sigma <= 0.0:
        return bias
    for b in range(B):
        Cb = int(time_cols[b])
        Nb = int(text_lens[b])
        if Cb <= 0 or Nb <= 0:
            continue
        u = (torch.arange(Cb, device=device, dtype=torch.float32) + 0.5) / max(Cb, 1)
        s = (torch.arange(Nb, device=device, dtype=torch.float32) + 0.5) / max(Nb, 1)
        gauss = -((u[:, None] - s[None, :]) ** 2) / (2.0 * (sigma ** 2))
        for row in range(freq_rows):
            start = row * max_time
            bias[b, :, start : start + Cb, :Nb] = gauss.unsqueeze(0).expand(num_heads, -1, -1)
    return bias


@app.class_definition
class MultiHeadCrossAttentionAlignmentV1(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // self.num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        align_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, l, h = x.shape
        _, t_len, _ = text.shape
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(text).reshape(b, t_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn_mask = None
        if text_mask is not None:
            base = text_mask.view(b, 1, 1, t_len).expand(b, self.num_heads, l, t_len).contiguous()
            attn_mask = torch.zeros_like(base, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(~base, float("-inf"))
        if align_bias is not None:
            if attn_mask is None:
                attn_mask = align_bias.to(torch.float32)
            else:
                attn_mask = attn_mask + align_bias.to(torch.float32)
        if return_attn:
            scale = 1.0 / math.sqrt(self.head_dim)
            logits = torch.matmul(q, k.transpose(-1, -2)) * scale
            if attn_mask is not None:
                logits = logits + attn_mask
            attn = torch.softmax(logits, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(b, l, h)
            return self.out_proj(out), attn
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, l, h)
        return self.out_proj(out), None


@app.class_definition
class TextEncoderBlockV1(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttentionRoPE1DV1(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = FeedForwardV1(hidden_dim, int(hidden_dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


@app.class_definition
class TextEncoderV1(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        pad_id: int = 0,
        max_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()
        self.blocks = nn.ModuleList(
            [TextEncoderBlockV1(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.rope = RotaryEmbedding1DV1(head_dim=hidden_dim // num_heads, base=rope_base, max_len=max_len)

    def forward(self, tokens: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.embed(tokens)
        cos, sin = self.rope.get(h.shape[1], h.device)
        for block in self.blocks:
            h = block(h, cos, sin, key_padding_mask=token_mask)
        return self.norm(h)


@app.class_definition
class DiTBlockTTSV1(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadSelfAttentionRoPE2DV1(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadCrossAttentionAlignmentV1(hidden_dim, num_heads)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForwardV1(hidden_dim, int(hidden_dim * mlp_ratio))
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos_row: torch.Tensor,
        sin_row: torch.Tensor,
        cos_col: torch.Tensor,
        sin_col: torch.Tensor,
        mel_key_padding_mask: Optional[torch.Tensor],
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        align_bias: Optional[torch.Tensor],
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mods = self.ada_mod(c)
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
        h1 = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            h1, cos_row, sin_row, cos_col, sin_col, key_padding_mask=mel_key_padding_mask
        )
        h2 = self.norm2(x)
        ca_out, attn = self.cross_attn(
            h2, text, text_mask=text_mask, align_bias=align_bias, return_attn=return_attn
        )
        x = x + ca_out
        h3 = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h3)
        return x, attn


@app.class_definition
class FinalLayerV1(nn.Module):
    def __init__(
        self, hidden_dim: int = 384, patch_freq: int = 20, patch_time: int = 4, out_channels: int = 1
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_freq * patch_time * out_channels, bias=True)
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mods = self.ada_mod(c)
        shift, scale = mods.chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(h)


@app.class_definition
class DiffusionTransformerTTSV1(nn.Module):
    def __init__(self, config: DiTTTSConfigV1):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed2DV1(
            in_channels=1,
            patch_freq=config.patch_freq,
            patch_time=config.patch_time,
            hidden_dim=config.hidden_dim,
        )
        self.time_embed = TimestepEmbedV1(
            hidden_dim=config.hidden_dim,
            frequency_dim=config.hidden_dim,
            max_period=config.time_max_period,
            scale=config.time_scale,
        )
        self.speaker_embed = SpeakerEmbedV1(
            num_speakers=config.num_speakers,
            hidden_dim=config.hidden_dim,
        )
        self.text_encoder = TextEncoderV1(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.text_layers,
            num_heads=config.text_heads,
            mlp_ratio=config.mlp_ratio,
            rope_base=config.rope_base,
            pad_id=config.pad_id,
            max_len=1024,
        )
        self.rope_2d = RotaryEmbedding2DV1(head_dim=config.head_dim, base=config.rope_base)
        self.blocks = nn.ModuleList(
            [
                DiTBlockTTSV1(config.hidden_dim, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final = FinalLayerV1(
            hidden_dim=config.hidden_dim,
            patch_freq=config.patch_freq,
            patch_time=config.patch_time,
            out_channels=1,
        )
        if config.use_freq_row_embed:
            self.freq_row_embed = nn.Parameter(torch.zeros(config.freq_rows, config.hidden_dim))
            nn.init.normal_(self.freq_row_embed, mean=0.0, std=0.02)
        else:
            self.register_parameter("freq_row_embed", None)

    def unpatchify(self, h: torch.Tensor, freq_rows: int, time_cols: int) -> torch.Tensor:
        b, _, _d = h.shape
        pf = self.config.patch_freq
        pt = self.config.patch_time
        h = h.reshape(b, freq_rows, time_cols, pf, pt, 1)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
        return h.reshape(b, 1, freq_rows * pf, time_cols * pt)

    def encode_text_and_speaker(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        speaker: torch.Tensor,
        joint_dropout_prob: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        cfg = self.config
        b = tokens.shape[0]
        if joint_dropout_prob > 0.0:
            drop = (torch.rand(b, device=tokens.device) < float(joint_dropout_prob))
            null_row = torch.full(
                (tokens.shape[1],), fill_value=int(cfg.pad_id), dtype=tokens.dtype, device=tokens.device
            )
            null_row[0] = int(cfg.null_id)
            tokens = torch.where(drop.unsqueeze(1), null_row.unsqueeze(0), tokens)
            null_mask_row = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
            null_mask_row[0] = True
            if token_mask is not None:
                token_mask = torch.where(drop.unsqueeze(1), null_mask_row.unsqueeze(0), token_mask)
            else:
                token_mask = torch.where(
                    drop.unsqueeze(1),
                    null_mask_row.unsqueeze(0),
                    torch.ones(b, tokens.shape[1], dtype=torch.bool, device=tokens.device),
                )
            speaker = torch.where(drop, torch.zeros_like(speaker), speaker)
        text = self.text_encoder(tokens, token_mask=token_mask)
        spk_emb = self.speaker_embed(speaker)
        return text, token_mask, spk_emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        speaker: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
        joint_dropout_prob: float = 0.0,
        return_attn_layer: Optional[int] = None,
        use_align_prior: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config
        text, text_mask, spk_emb = self.encode_text_and_speaker(
            tokens=tokens,
            token_mask=token_mask,
            speaker=speaker,
            joint_dropout_prob=float(joint_dropout_prob) if self.training else 0.0,
        )
        x_in = x.unsqueeze(1) if x.ndim == 3 else x
        h, freq_rows, time_cols = self.patch_embed(x_in)
        if self.freq_row_embed is not None:
            row_bias = self.freq_row_embed.view(1, freq_rows, 1, cfg.hidden_dim).expand(
                h.shape[0], freq_rows, time_cols, cfg.hidden_dim
            ).reshape(h.shape[0], freq_rows * time_cols, cfg.hidden_dim)
            h = h + row_bias.to(h.dtype)
        cos_row, sin_row, cos_col, sin_col = self.rope_2d.get(freq_rows, time_cols, h.device)
        t_emb = self.time_embed(t)
        cond = t_emb + spk_emb
        if mel_mask is not None:
            valid_time_cols = torch.clamp(
                (mel_mask.sum(dim=1).to(torch.int64) // max(cfg.patch_time, 1)),
                min=0, max=time_cols,
            ).tolist()
            token_lens = (
                text_mask.sum(dim=1).to(torch.int64).tolist()
                if text_mask is not None else [tokens.shape[1]] * tokens.shape[0]
            )
            mel_key_padding_mask = torch.zeros(
                h.shape[0], freq_rows * time_cols, dtype=torch.bool, device=h.device
            )
            for b in range(h.shape[0]):
                Cb = int(valid_time_cols[b])
                for r in range(freq_rows):
                    start = r * time_cols
                    mel_key_padding_mask[b, start : start + Cb] = True
        else:
            valid_time_cols = [time_cols] * h.shape[0]
            token_lens = (
                text_mask.sum(dim=1).to(torch.int64).tolist()
                if text_mask is not None else [tokens.shape[1]] * tokens.shape[0]
            )
            mel_key_padding_mask = None
        sigma_used = cfg.alignment_sigma if use_align_prior else -1.0
        align_bias = diagonal_alignment_bias(
            freq_rows=freq_rows,
            time_cols=valid_time_cols,
            text_lens=token_lens,
            max_time=time_cols,
            max_text=tokens.shape[1],
            sigma=sigma_used,
            num_heads=cfg.num_heads,
            device=h.device,
        )
        attn_returned = None
        for i, block in enumerate(self.blocks):
            want_attn = return_attn_layer is not None and i == int(return_attn_layer)
            h, attn = block(
                h, cond, cos_row, sin_row, cos_col, sin_col,
                mel_key_padding_mask, text, text_mask, align_bias, return_attn=want_attn,
            )
            if want_attn:
                attn_returned = attn
        h = self.final(h, cond)
        v = self.unpatchify(h, freq_rows, time_cols)
        v = v.squeeze(1)
        if mel_mask is not None:
            v = v * mel_mask.unsqueeze(1).to(v.dtype)
        return v, attn_returned


@app.function
def build_model_from_config(
    config: DiTTTSConfigV1,
    device: Optional[torch.device] = None,
) -> DiffusionTransformerTTSV1:
    model = DiffusionTransformerTTSV1(copy.deepcopy(config))
    if device is not None:
        model = model.to(device)
    return model


@app.cell
def _(mo):
    preset_ui = mo.ui.dropdown(
        options=list(dit_tts_presets().keys()),
        value="small",
        label="Model Preset",
    )
    mo.vstack([mo.md("### Model Config"), preset_ui])
    return (preset_ui,)


@app.cell
def _(mo, preset_ui):
    pv = dit_tts_presets()[preset_ui.value]
    hidden_dim_ui = mo.ui.dropdown(
        options=[128, 192, 256, 384, 512, 768], value=int(pv["hidden_dim"]), label="Hidden Dim"
    )
    depth_ui = mo.ui.slider(2, 16, value=int(pv["depth"]), step=1, label="Depth")
    num_heads_ui = mo.ui.dropdown(options=[2, 3, 4, 6, 8, 12], value=int(pv["num_heads"]), label="Num Heads")
    text_layers_ui = mo.ui.slider(1, 8, value=int(pv["text_layers"]), step=1, label="Text Layers")
    text_heads_ui = mo.ui.dropdown(options=[2, 3, 4, 6, 8, 12], value=int(pv["text_heads"]), label="Text Heads")
    patch_freq_ui = mo.ui.dropdown(options=[10, 20, 25, 50], value=int(pv["patch_freq"]), label="Patch Freq")
    patch_time_ui = mo.ui.dropdown(options=[2, 4, 8], value=int(pv["patch_time"]), label="Patch Time")
    mlp_ratio_ui = mo.ui.dropdown(
        options={"2.0": 2.0, "3.0": 3.0, "4.0": 4.0}, value=f"{float(pv['mlp_ratio']):.1f}", label="MLP Ratio"
    )
    alignment_sigma_ui = mo.ui.dropdown(
        options={"off": -1.0, "0.05": 0.05, "0.10": 0.10, "0.20": 0.20, "0.30": 0.30},
        value="0.10",
        label="Alignment sigma",
    )
    freq_row_embed_ui = mo.ui.checkbox(value=True, label="Freq-Row Absolute Embed")
    mo.vstack(
        [
            mo.md(f"_Preset `{preset_ui.value}` — edit any field to override._"),
            mo.hstack([hidden_dim_ui, depth_ui, num_heads_ui, mlp_ratio_ui]),
            mo.hstack([text_layers_ui, text_heads_ui, patch_freq_ui, patch_time_ui]),
            mo.hstack([alignment_sigma_ui, freq_row_embed_ui]),
        ]
    )
    return (
        alignment_sigma_ui,
        depth_ui,
        freq_row_embed_ui,
        hidden_dim_ui,
        mlp_ratio_ui,
        num_heads_ui,
        patch_freq_ui,
        patch_time_ui,
        text_heads_ui,
        text_layers_ui,
    )


@app.cell
def _(
    alignment_sigma_ui,
    depth_ui,
    freq_row_embed_ui,
    hidden_dim_ui,
    mel_config,
    mlp_ratio_ui,
    mo,
    num_heads_ui,
    patch_freq_ui,
    patch_time_ui,
    speaker_to_id,
    text_heads_ui,
    text_layers_ui,
    text_tokenizer,
):
    model_cfg, model_cfg_error = make_dit_tts_config(
        n_mels=int(mel_config["n_mels"]),
        patch_freq=int(patch_freq_ui.value),
        patch_time=int(patch_time_ui.value),
        hidden_dim=int(hidden_dim_ui.value),
        depth=int(depth_ui.value),
        num_heads=int(num_heads_ui.value),
        text_layers=int(text_layers_ui.value),
        text_heads=int(text_heads_ui.value),
        mlp_ratio=float(mlp_ratio_ui.value),
        vocab_size=len(text_tokenizer),
        num_speakers=max(len(speaker_to_id), 1),
        alignment_sigma=float(alignment_sigma_ui.value),
        use_freq_row_embed=bool(freq_row_embed_ui.value),
        pad_id=text_tokenizer.pad_id,
        null_id=text_tokenizer.null_id,
    )
    mo.stop(model_cfg is None, mo.md(f"**Invalid model configuration** — {model_cfg_error}"))
    return (model_cfg,)


@app.function
def model_summary_markdown(
    model_cfg: DiTTTSConfigV1,
    mel_config: Dict[str, object],
    mo,
) -> object:
    model = build_model_from_config(model_cfg)
    params = count_parameters(model)
    fps = int(mel_config["sample_rate"]) / int(mel_config["hop_length"])
    tokens_per_sec = fps * (model_cfg.n_mels / model_cfg.patch_freq) / model_cfg.patch_time
    return mo.md(
        f"""
    ### Instantiated `DiffusionTransformerTTSV1`

    | Field | Value |
    |---|---|
    | n_mels | {model_cfg.n_mels} |
    | patch_freq x patch_time | {model_cfg.patch_freq} x {model_cfg.patch_time} |
    | freq_rows | {model_cfg.freq_rows} |
    | hidden_dim | {model_cfg.hidden_dim} |
    | depth | {model_cfg.depth} |
    | num_heads (head_dim) | {model_cfg.num_heads} ({model_cfg.head_dim}) |
    | text_layers / text_heads | {model_cfg.text_layers} / {model_cfg.text_heads} |
    | mlp_ratio | {model_cfg.mlp_ratio} |
    | vocab_size | {model_cfg.vocab_size} |
    | num_speakers (+null) | {model_cfg.num_speakers} + 1 |
    | alignment_sigma | {model_cfg.alignment_sigma:.3f} |
    | freq_row_embed | {model_cfg.use_freq_row_embed} |

    **Total parameters**: `{params:,}`

    **Tokens per second of audio**: `{tokens_per_sec:.1f}` (hop {int(mel_config['hop_length'])} @ {int(mel_config['sample_rate'])} Hz).
    """
    )


@app.cell
def _(mel_config, mo, model_cfg):
    model_summary_markdown(model_cfg, mel_config, mo)
    return


@app.function
def sample_timesteps(
    batch_size: int,
    scheme: str = "uniform",
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if scheme == "uniform":
        if generator is not None:
            t = torch.rand(batch_size, generator=generator)
            return t.to(device) if device is not None else t
        return torch.rand(batch_size, device=device)
    if scheme == "logit_normal":
        if generator is not None:
            u = torch.randn(batch_size, generator=generator)
            u = u.to(device) if device is not None else u
        else:
            u = torch.randn(batch_size, device=device)
        return torch.sigmoid(u)
    raise ValueError(f"Unknown timestep scheme: {scheme!r}")


@app.function
def interpolate_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_view = t.view(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - t_view) * x0 + t_view * x1


@app.function
def masked_flow_matching_loss(
    model: nn.Module,
    x1: torch.Tensor,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    speaker: torch.Tensor,
    mel_mask: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    t_scheme: str = "uniform",
    joint_dropout_prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if x0 is None:
        x0 = torch.randn_like(x1)
    t = sample_timesteps(x1.shape[0], t_scheme, device=x1.device, generator=generator)
    xt = interpolate_path(x0, x1, t)
    target_v = x1 - x0
    pred_v, _ = model(
        xt, t, tokens, token_mask, speaker,
        mel_mask=mel_mask,
        joint_dropout_prob=float(joint_dropout_prob),
    )
    err2 = ((pred_v.to(torch.float32) - target_v.to(torch.float32)) ** 2).mean(dim=1)
    mask_f = mel_mask.to(torch.float32)
    num = (err2 * mask_f).sum()
    den = mask_f.sum().clamp_min(1.0)
    return num / den


@app.function
def classifier_free_velocity(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    speaker: torch.Tensor,
    mel_mask: torch.Tensor,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        pred_v, _ = model(
            x_t, t, tokens, token_mask, speaker, mel_mask=mel_mask, joint_dropout_prob=0.0
        )
        return pred_v
    null_tokens = torch.full(
        (tokens.shape[0], 1), fill_value=int(model.config.null_id), dtype=tokens.dtype, device=tokens.device
    )
    null_mask = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
    v_cond, _ = model(
        x_t, t, tokens, token_mask, speaker, mel_mask=mel_mask, joint_dropout_prob=0.0
    )
    null_speaker = torch.zeros_like(speaker)
    v_uncond, _ = model(
        x_t, t, null_tokens, null_mask, null_speaker, mel_mask=mel_mask, joint_dropout_prob=0.0
    )
    return v_uncond + guidance_scale * (v_cond - v_uncond)


@app.function
def ode_integrate(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_start: torch.Tensor,
    t_start: float,
    t_end: float,
    num_steps: int,
    method: str = "euler",
) -> Tuple[torch.Tensor, int]:
    x = x_start.to(torch.float32)
    ts = torch.linspace(float(t_start), float(t_end), int(num_steps) + 1, device=x.device, dtype=torch.float32)
    nfe = 0
    for i in range(int(num_steps)):
        t_i = ts[i]
        dt = ts[i + 1] - ts[i]
        t_batch = t_i.expand(x.shape[0])
        if method == "euler":
            v = velocity_fn(x, t_batch)
            nfe += 1
            x = x + v.to(torch.float32) * dt
        elif method == "midpoint":
            v1 = velocity_fn(x, t_batch)
            nfe += 1
            x_mid = x + v1.to(torch.float32) * (dt * 0.5)
            t_mid = (t_i + dt * 0.5).expand(x.shape[0])
            v2 = velocity_fn(x_mid, t_mid)
            nfe += 1
            x = x + v2.to(torch.float32) * dt
        elif method == "heun":
            v1 = velocity_fn(x, t_batch)
            nfe += 1
            x_pred = x + v1.to(torch.float32) * dt
            t_next = (t_i + dt).expand(x.shape[0])
            v2 = velocity_fn(x_pred, t_next)
            nfe += 1
            x = x + (dt * 0.5) * (v1.to(torch.float32) + v2.to(torch.float32))
        elif method == "rk4":
            k1 = velocity_fn(x, t_batch)
            nfe += 1
            k2 = velocity_fn(x + k1.to(torch.float32) * (dt * 0.5), (t_i + dt * 0.5).expand(x.shape[0]))
            nfe += 1
            k3 = velocity_fn(x + k2.to(torch.float32) * (dt * 0.5), (t_i + dt * 0.5).expand(x.shape[0]))
            nfe += 1
            k4 = velocity_fn(x + k3.to(torch.float32) * dt, (t_i + dt).expand(x.shape[0]))
            nfe += 1
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4).to(torch.float32)
        else:
            raise ValueError(f"Unknown ODE method: {method!r}")
    return x, nfe


@app.function
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float, num_updates: Optional[int] = None) -> None:
    if num_updates is not None:
        decay = min(decay, (1.0 + num_updates) / (10.0 + num_updates))
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, param in model.named_parameters():
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        for name, buf in model.named_buffers():
            ema_buffers[name].data.copy_(buf.data)


@app.function
def make_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int):
    if warmup_steps <= 0:
        return None

    def _lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


@app.function
def batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@app.function
def run_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    joint_dropout_prob: float = 0.0,
    grad_clip: float = 1.0,
    t_scheme: str = "uniform",
    warmup_scheduler=None,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.0,
    max_steps: Optional[int] = None,
    start_step: int = 0,
) -> Tuple[float, int]:
    model.train()
    losses: List[float] = []
    step = int(start_step)
    for i, batch in enumerate(loader):
        if max_steps is not None and i >= int(max_steps):
            break
        b = batch_to_device(batch, device)
        x1 = b["mel"].to(torch.float32)
        mel_mask = b["mel_mask"]
        tokens = b["tokens"]
        token_mask = b["token_mask"]
        speaker = b["speakers"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = masked_flow_matching_loss(
                model, x1, tokens, token_mask, speaker, mel_mask,
                x0=None, t_scheme=t_scheme, joint_dropout_prob=float(joint_dropout_prob),
            )
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema(ema_model, model, ema_decay, num_updates=step)
        losses.append(float(loss.detach().item()))
        step += 1
    return sum(losses) / max(len(losses), 1), step - int(start_step)


@app.function
def run_evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    t_scheme: str = "uniform",
    seed: int = 12345,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    losses: List[float] = []
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            b = batch_to_device(batch, device)
            x1 = b["mel"].to(torch.float32)
            mel_mask = b["mel_mask"]
            tokens = b["tokens"]
            token_mask = b["token_mask"]
            speaker = b["speakers"]
            x0 = torch.randn(x1.shape, generator=cpu_gen).to(device)
            t = sample_timesteps(x1.shape[0], t_scheme, device=None, generator=cpu_gen).to(device)
            xt = interpolate_path(x0, x1, t)
            target_v = x1 - x0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_v, _ = model(
                    xt, t, tokens, token_mask, speaker, mel_mask=mel_mask, joint_dropout_prob=0.0
                )
            err2 = ((pred_v.to(torch.float32) - target_v.to(torch.float32)) ** 2).mean(dim=1)
            mask_f = mel_mask.to(torch.float32)
            losses.append(float(((err2 * mask_f).sum() / mask_f.sum().clamp_min(1.0)).item()))
    return sum(losses) / max(len(losses), 1)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Training — 1-Rectified Flow (gen1)

    The DiT is trained to regress the linear-path velocity $v^{*} = x_1 - x_0$
    (masked to valid mel frames) with fresh $x_0 \sim \mathcal{N}(0, I)$ per
    training example. AdamW + linear warmup + gradient clip + optional EMA +
    mixed precision (bf16 preferred on CUDA). Text and speaker conditioning
    is jointly dropped with probability `p_uncond` for classifier-free
    guidance at sampling.

    The EMA schedule uses the shared **global step counter** (accumulated
    across epochs, not restarted per epoch), so the warmup
    $\beta_n = \min(\beta, (1+n)/(10+n))$ progresses continuously and the
    averaged weights track the whole training run.
    """)
    return


@app.class_definition
@dataclass
class TrainConfigV1:
    lr: float = 3e-4
    batch_size: int = 16
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_steps: int = 200
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    t_scheme: str = "uniform"
    p_uncond: float = 0.1
    max_steps_per_epoch: int = 0
    seed: int = 1337


@app.cell
def _(mo):
    lr_ui = mo.ui.dropdown(
        options={"1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3, "3e-3": 3e-3},
        value="3e-4",
        label="Learning Rate",
    )
    bs_ui = mo.ui.dropdown(options=[4, 8, 16, 32], value=16, label="Batch Size")
    wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-2": 1e-2, "5e-2": 5e-2},
        value="1e-2",
        label="Weight Decay",
    )
    epochs_ui = mo.ui.slider(1, 200, value=30, step=1, label="Epochs")
    warmup_ui = mo.ui.dropdown(options=[0, 100, 200, 500, 1000], value=200, label="Warmup Steps")
    grad_clip_ui = mo.ui.dropdown(
        options={"none": 0.0, "0.5": 0.5, "1.0": 1.0, "2.0": 2.0},
        value="1.0",
        label="Grad Clip",
    )
    ema_ui = mo.ui.dropdown(
        options={"off": 0.0, "0.999": 0.999, "0.9995": 0.9995, "0.9999": 0.9999},
        value="0.999",
        label="EMA Decay",
    )
    t_scheme_ui = mo.ui.dropdown(
        options=["uniform", "logit_normal"], value="uniform", label="Timestep Scheme"
    )
    p_uncond_ui = mo.ui.dropdown(
        options={"0.0": 0.0, "0.05": 0.05, "0.1": 0.1, "0.2": 0.2}, value="0.1", label="p_uncond"
    )
    max_steps_ui = mo.ui.number(value=0, label="Max Steps / Epoch (0 = full)", start=0, stop=100000)
    train_btn = mo.ui.run_button(label="Train (1-Rectified Flow)")
    mo.vstack(
        [
            mo.md("### Training Hyperparameters"),
            mo.hstack([lr_ui, bs_ui, wd_ui, epochs_ui]),
            mo.hstack([warmup_ui, grad_clip_ui, ema_ui, t_scheme_ui]),
            mo.hstack([p_uncond_ui, max_steps_ui, train_btn]),
        ]
    )
    return (
        bs_ui,
        ema_ui,
        epochs_ui,
        grad_clip_ui,
        lr_ui,
        max_steps_ui,
        p_uncond_ui,
        t_scheme_ui,
        train_btn,
        warmup_ui,
        wd_ui,
    )


@app.function
def fit_flow_model(
    model: nn.Module,
    train_cfg: TrainConfigV1,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, object]:
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.99),
    )
    scaler = make_grad_scaler(device, use_amp, amp_dtype)
    warmup = make_warmup_scheduler(optimizer, train_cfg.warmup_steps)
    ema_model: Optional[nn.Module] = None
    if train_cfg.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.requires_grad_(False)
        ema_model.eval()
    train_losses: List[float] = []
    val_losses: List[float] = []
    start_time = time.perf_counter()
    max_steps = train_cfg.max_steps_per_epoch if train_cfg.max_steps_per_epoch > 0 else None
    global_step = 0
    for epoch in range(train_cfg.epochs):
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_loss, steps_taken = run_train_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            joint_dropout_prob=train_cfg.p_uncond,
            grad_clip=train_cfg.grad_clip,
            t_scheme=train_cfg.t_scheme,
            warmup_scheduler=warmup,
            ema_model=ema_model,
            ema_decay=train_cfg.ema_decay,
            max_steps=max_steps,
            start_step=global_step,
        )
        global_step += int(steps_taken)
        val_loss = run_evaluate(
            model=ema_model if ema_model is not None else model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            t_scheme=train_cfg.t_scheme,
            seed=train_cfg.seed + 1000,
            max_batches=max_steps,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if progress_cb is not None:
            progress_cb(epoch + 1, train_cfg.epochs, train_loss, val_loss)
    final_model = ema_model if ema_model is not None else model
    final_model.eval()
    return {
        "model": final_model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "wall_time": time.perf_counter() - start_time,
        "ema_used": ema_model is not None,
        "global_step": global_step,
    }


@app.function
def run_gen1_training(
    model_cfg: DiTTTSConfigV1,
    train_cfg: TrainConfigV1,
    mel_cache: Dict[str, object],
    train_utts: List[Dict[str, object]],
    val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, object]:
    set_seed(train_cfg.seed)
    loaders = make_datasets(
        mel_cache=mel_cache,
        train_utts=train_utts,
        val_utts=val_utts,
        test_utts=test_utts,
        tokenizer=tokenizer,
        speaker_to_id=speaker_to_id,
        mel_mean=mel_mean,
        mel_std=mel_std,
        batch_size=train_cfg.batch_size,
        patch_time=model_cfg.patch_time,
        seed=train_cfg.seed,
    )
    result = fit_flow_model(
        model=build_model_from_config(model_cfg, device=device),
        train_cfg=train_cfg,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        progress_cb=progress_cb,
    )
    result["loaders"] = loaders
    result["train_config"] = train_cfg
    return result


@app.cell
def _(
    amp_dtype,
    bs_ui,
    device,
    ema_ui,
    epochs_ui,
    grad_clip_ui,
    lr_ui,
    max_steps_ui,
    mel_cache,
    mel_norm_mean,
    mel_norm_std,
    mo,
    model_cfg,
    p_uncond_ui,
    seed_ui,
    speaker_to_id,
    t_scheme_ui,
    test_utts,
    text_tokenizer,
    train_btn,
    train_utts,
    use_amp,
    val_utts,
    warmup_ui,
    wd_ui,
):
    gen1_run: Optional[Dict[str, object]] = None
    if mel_cache is None:
        mo.output.replace(mo.md("_Data pipeline not ready — check Section 2 filters._"))
    elif not train_btn.value:
        mo.output.replace(mo.md("Click **Train (1-Rectified Flow)** to begin training."))
    else:
        gen1_run = run_gen1_training(
            model_cfg=model_cfg,
            train_cfg=TrainConfigV1(
                lr=float(lr_ui.value),
                batch_size=int(bs_ui.value),
                weight_decay=float(wd_ui.value),
                epochs=int(epochs_ui.value),
                warmup_steps=int(warmup_ui.value),
                grad_clip=float(grad_clip_ui.value),
                ema_decay=float(ema_ui.value),
                t_scheme=str(t_scheme_ui.value),
                p_uncond=float(p_uncond_ui.value),
                max_steps_per_epoch=int(max_steps_ui.value),
                seed=int(seed_ui.value),
            ),
            mel_cache=mel_cache,
            train_utts=train_utts,
            val_utts=val_utts,
            test_utts=test_utts,
            tokenizer=text_tokenizer,
            speaker_to_id=speaker_to_id,
            mel_mean=mel_norm_mean,
            mel_std=mel_norm_std,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            progress_cb=lambda epoch, total, tl, vl: mo.output.replace(
                mo.md(f"**Epoch {epoch}/{total}** — train loss: {tl:.4f} | val loss: {vl:.4f}")
            ),
        )
        mo.output.replace(
            mo.md(
                f"**Training complete** in {gen1_run['wall_time']:.1f}s — final train "
                f"{gen1_run['train_losses'][-1]:.4f} | final val {gen1_run['val_losses'][-1]:.4f} "
                f"| using EMA weights: {gen1_run['ema_used']} | global steps: {gen1_run['global_step']}"
            )
        )
    train_losses: List[float] = gen1_run["train_losses"] if gen1_run else []
    val_losses: List[float] = gen1_run["val_losses"] if gen1_run else []
    trained_model: Optional[nn.Module] = gen1_run["model"] if gen1_run else None
    train_wall_time: float = gen1_run["wall_time"] if gen1_run else 0.0
    trained_ema_used: bool = bool(gen1_run["ema_used"]) if gen1_run else False
    gen1_train_cfg_used: Optional[TrainConfigV1] = gen1_run["train_config"] if gen1_run else None
    gen1_loaders: Optional[Dict[str, object]] = gen1_run["loaders"] if gen1_run else None
    return (
        gen1_loaders,
        gen1_train_cfg_used,
        train_losses,
        train_wall_time,
        trained_ema_used,
        trained_model,
        val_losses,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Hyperparameter Search (optional)

    Small grid over learning rate, model width and `p_uncond`. Uses the
    **tiny preset** and a small training subset (a few hundred utterances
    at most) plus 3-5 epochs to stay tractable, then reports per-config
    validation loss in a sortable table.
    """)
    return


@app.cell
def _(mo):
    hp_search_cb = mo.ui.checkbox(label="Enable Hyperparameter Search", value=False)
    hp_epochs_ui = mo.ui.slider(1, 10, value=3, step=1, label="HP-Search Epochs")
    hp_subset_ui = mo.ui.dropdown(options=[64, 128, 256, 512], value=128, label="Train Subset")
    hp_run_btn = mo.ui.run_button(label="Run HP Search")
    mo.vstack(
        [
            hp_search_cb,
            mo.hstack([hp_epochs_ui, hp_subset_ui, hp_run_btn]),
        ]
    )
    return hp_epochs_ui, hp_run_btn, hp_search_cb, hp_subset_ui


@app.function
def hp_search_grid() -> List[Dict[str, object]]:
    grid: List[Dict[str, object]] = []
    for lr in [1e-4, 3e-4, 1e-3]:
        for width in [128, 192]:
            for pu in [0.0, 0.1]:
                grid.append({"lr": lr, "hidden_dim": width, "p_uncond": pu})
    return grid


@app.function
def run_hp_search(
    grid: List[Dict[str, object]],
    train_utts: List[Dict[str, object]],
    val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    mel_cache: Dict[str, object],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    hp_epochs: int,
    hp_subset: int,
    progress_cb: Optional[Callable[[int, int, Dict[str, object]], None]] = None,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    tr = train_utts[:hp_subset]
    va = val_utts[: max(hp_subset // 4, 4)]
    for i, cfg_dict in enumerate(grid):
        mcfg, err = make_dit_tts_config(
            n_mels=100, patch_freq=20, patch_time=4,
            hidden_dim=int(cfg_dict["hidden_dim"]),
            depth=3, num_heads=4, text_layers=2, text_heads=4,
            mlp_ratio=4.0, vocab_size=len(tokenizer),
            num_speakers=max(len(speaker_to_id), 1),
            pad_id=tokenizer.pad_id, null_id=tokenizer.null_id,
        )
        if mcfg is None:
            continue
        run = run_gen1_training(
            model_cfg=mcfg,
            train_cfg=TrainConfigV1(
                lr=float(cfg_dict["lr"]),
                batch_size=4, epochs=int(hp_epochs), warmup_steps=50,
                grad_clip=1.0, ema_decay=0.0, t_scheme="uniform",
                p_uncond=float(cfg_dict["p_uncond"]),
                seed=999 + i,
            ),
            mel_cache=mel_cache,
            train_utts=tr, val_utts=va, test_utts=test_utts,
            tokenizer=tokenizer, speaker_to_id=speaker_to_id,
            mel_mean=mel_mean, mel_std=mel_std,
            device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        )
        row = {
            "lr": cfg_dict["lr"],
            "hidden_dim": int(cfg_dict["hidden_dim"]),
            "p_uncond": float(cfg_dict["p_uncond"]),
            "params": count_parameters(run["model"]),
            "val_loss": round(run["val_losses"][-1], 4),
        }
        results.append(row)
        if progress_cb is not None:
            progress_cb(i + 1, len(grid), row)
    results.sort(key=lambda r: r["val_loss"])
    return results


@app.cell
def _(
    amp_dtype,
    device,
    hp_epochs_ui,
    hp_run_btn,
    hp_search_cb,
    hp_subset_ui,
    mel_cache,
    mel_norm_mean,
    mel_norm_std,
    mo,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    train_utts,
    use_amp,
    val_utts,
):
    mo.stop(not hp_search_cb.value, mo.md("_Enable **Hyperparameter Search** above to run this section._"))
    mo.stop(not hp_run_btn.value, mo.md("Toggle enabled — now click **Run HP Search** to launch."))
    hp_results = run_hp_search(
        grid=hp_search_grid(),
        train_utts=train_utts, val_utts=val_utts, test_utts=test_utts,
        mel_cache=mel_cache, tokenizer=text_tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_norm_mean, mel_std=mel_norm_std,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        hp_epochs=int(hp_epochs_ui.value), hp_subset=int(hp_subset_ui.value),
        progress_cb=lambda i, total, row: mo.output.replace(
            mo.md(f"[{i}/{total}] lr={row['lr']}, dim={row['hidden_dim']}, p_uncond={row['p_uncond']} -> val {row['val_loss']:.4f}")
        ),
    )
    mo.ui.table(hp_results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Validation & Cross-Validation

    Test-set metrics for the 1st-generation model:

    - **Masked flow-matching loss** on the test loader.
    - **Per-timestep-bin loss** (10 bins of $t \in [0, 1]$).
    - **Straightness** — deviation of the trained velocity from the
      chord-average $x_1 - x_0$ over the Euler trajectory.

    Optional k-fold CV on a small training subset with the tiny architecture
    (uses `make_tiny_cv_config` — CV trains `k` models, so keep it small).

    Optional ASR-based CER on Griffin-Lim resynthesis vs the vocoder ceiling
    (ground-truth mel → Griffin-Lim → ASR).
    """)
    return


@app.function
def straightness_metric(
    model: nn.Module,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    speaker: torch.Tensor,
    mel_mask: torch.Tensor,
    shape: Tuple[int, int, int],
    num_steps: int,
    device: torch.device,
    seed: int = 0,
    guidance_scale: float = 1.0,
) -> float:
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x0 = torch.randn(shape, generator=gen).to(device)

    def _velocity(x, t):
        return classifier_free_velocity(
            model, x, t, tokens, token_mask, speaker, mel_mask,
            guidance_scale=guidance_scale,
        )

    with torch.no_grad():
        x_final, _ = ode_integrate(_velocity, x0, 0.0, 1.0, num_steps, method="euler")
        target = x_final - x0
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        x = x0.clone()
        total = 0.0
        n = 0
        for i in range(num_steps + 1):
            t_batch = ts[i].expand(x.shape[0])
            v_pred = _velocity(x, t_batch).to(torch.float32)
            total += float(((target - v_pred) ** 2).mean().item())
            n += 1
            if i < num_steps:
                dt = ts[i + 1] - ts[i]
                x = x + v_pred * dt
    if was_training:
        model.train()
    return total / max(n, 1)


@app.function
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    num_bins: int = 10,
    straightness_steps: int = 16,
    seed: int = 2026,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()
    sum_loss = 0.0
    count = 0
    bin_sums = [0.0] * num_bins
    bin_counts = [0] * num_bins
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    last_batch: Optional[Dict[str, object]] = None
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            b = batch_to_device(batch, device)
            x1 = b["mel"].to(torch.float32)
            mel_mask = b["mel_mask"]
            tokens = b["tokens"]
            token_mask = b["token_mask"]
            speaker = b["speakers"]
            x0 = torch.randn(x1.shape, generator=cpu_gen).to(device)
            t = torch.rand(x1.shape[0], generator=cpu_gen).to(device)
            xt = interpolate_path(x0, x1, t)
            target_v = x1 - x0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_v, _ = model(
                    xt, t, tokens, token_mask, speaker, mel_mask=mel_mask, joint_dropout_prob=0.0
                )
            per_frame = ((pred_v.to(torch.float32) - target_v.to(torch.float32)) ** 2).mean(dim=1)
            mask_f = mel_mask.to(torch.float32)
            per_sample = (per_frame * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            sum_loss += float(per_sample.sum().item())
            count += int(per_sample.shape[0])
            bins = (t * num_bins).long().clamp(0, num_bins - 1)
            for bidx in range(num_bins):
                sel = bins == bidx
                if sel.any():
                    bin_sums[bidx] += float(per_sample[sel].sum().item())
                    bin_counts[bidx] += int(sel.sum().item())
            last_batch = b
    straight = 0.0
    if last_batch is not None:
        b = last_batch
        n_sub = min(4, b["mel"].shape[0])
        shape = (n_sub, b["mel"].shape[1], b["mel"].shape[2])
        straight = straightness_metric(
            model=model,
            tokens=b["tokens"][:n_sub],
            token_mask=b["token_mask"][:n_sub],
            speaker=b["speakers"][:n_sub],
            mel_mask=b["mel_mask"][:n_sub],
            shape=shape,
            num_steps=int(straightness_steps),
            device=device,
            seed=seed,
        )
    loss_per_bin = [bin_sums[b] / max(bin_counts[b], 1) for b in range(num_bins)]
    return {
        "mean_loss": sum_loss / max(count, 1),
        "loss_per_bin": loss_per_bin,
        "bin_counts": bin_counts,
        "straightness": straight,
    }


@app.function
def format_loss_bin_table(loss_per_bin: List[float], bin_counts: List[int]) -> str:
    num_bins = len(loss_per_bin)
    rows = [
        f"| bin {b} (t in [{b / num_bins:.2f}, {(b + 1) / num_bins:.2f}]) | {loss_per_bin[b]:.4f} | {bin_counts[b]} |"
        for b in range(num_bins)
    ]
    return "\n".join(["| Timestep bin | Mean loss | Count |", "|---|---|---|", *rows])


@app.function
def gen1_test_metrics_view(
    trained_model: Optional[nn.Module],
    gen1_loaders: Optional[Dict[str, object]],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    mo,
) -> Tuple[Optional[Dict[str, object]], object]:
    if trained_model is None or gen1_loaders is None:
        return None, mo.md("_Train the 1st-generation model first (Section 5) to see test metrics._")
    metrics = evaluate_model(
        model=trained_model,
        loader=gen1_loaders["test_loader"],
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        num_bins=10, straightness_steps=16, seed=2026,
    )
    md = mo.md(
        f"""
    **Test mean loss**: {metrics['mean_loss']:.4f}

    **Straightness (16-step)**: {metrics['straightness']:.4f}

    {format_loss_bin_table(metrics['loss_per_bin'], metrics['bin_counts'])}
        """
    )
    return metrics, md


@app.cell
def _(
    amp_dtype,
    device,
    gen1_loaders: Optional[Dict[str, object]],
    mo,
    trained_model: Optional[nn.Module],
    use_amp,
):
    gen1_test_metrics, _out = gen1_test_metrics_view(
        trained_model, gen1_loaders, device, use_amp, amp_dtype, mo
    )
    _out
    return (gen1_test_metrics,)


@app.function
def plot_per_timestep_bins(loss_per_bin: List[float], title: str = "Loss per timestep bin"):
    xs = [(b + 0.5) / len(loss_per_bin) for b in range(len(loss_per_bin))]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(xs, loss_per_bin, width=1.0 / len(loss_per_bin) * 0.9, color="darkorange", alpha=0.8)
    ax.set_xlabel("t (mid of bin)")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(gen1_test_metrics, mo):
    if gen1_test_metrics is None:
        _out = mo.md("_Train first to see the per-timestep loss plot._")
    else:
        _out = plot_per_timestep_bins(gen1_test_metrics["loss_per_bin"], title="1st-gen test loss per t bin")
    _out
    return


@app.cell
def _(mo):
    cv_cb = mo.ui.checkbox(label="Enable k-Fold CV", value=False)
    cv_epochs_ui = mo.ui.slider(1, 10, value=2, step=1, label="Epochs per Fold")
    cv_subset_ui = mo.ui.dropdown(options=[64, 128, 256, 512], value=128, label="CV Subset")
    cv_folds_ui = mo.ui.slider(2, 5, value=5, step=1, label="Folds (k)")
    cv_run_btn = mo.ui.run_button(label="Run k-Fold CV")
    mo.vstack(
        [
            cv_cb,
            mo.hstack([cv_epochs_ui, cv_subset_ui, cv_folds_ui, cv_run_btn]),
        ]
    )
    return cv_cb, cv_epochs_ui, cv_folds_ui, cv_run_btn, cv_subset_ui


@app.function
def make_tiny_cv_config(tokenizer: CharTokenizerV1, speaker_to_id: Dict[str, int]) -> DiTTTSConfigV1:
    cfg, err = make_dit_tts_config(
        n_mels=100, patch_freq=20, patch_time=4,
        hidden_dim=128, depth=3, num_heads=4,
        text_layers=2, text_heads=4, mlp_ratio=4.0,
        vocab_size=len(tokenizer),
        num_speakers=max(len(speaker_to_id), 1),
        pad_id=tokenizer.pad_id, null_id=tokenizer.null_id,
    )
    if cfg is None:
        raise RuntimeError(f"tiny CV config invalid: {err}")
    return cfg


@app.function
def run_fold(
    fold_train_utts: List[Dict[str, object]],
    fold_val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    model_cfg: DiTTTSConfigV1,
    mel_cache: Dict[str, object],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    lr: float,
    n_epochs: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    seed: int,
) -> float:
    run = run_gen1_training(
        model_cfg=model_cfg,
        train_cfg=TrainConfigV1(
            lr=lr, batch_size=batch_size, epochs=n_epochs,
            warmup_steps=50, grad_clip=1.0, ema_decay=0.0, seed=seed,
        ),
        mel_cache=mel_cache,
        train_utts=fold_train_utts, val_utts=fold_val_utts, test_utts=test_utts,
        tokenizer=tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_mean, mel_std=mel_std,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
    )
    return float(run["val_losses"][-1])


@app.function
def run_kfold_cv(
    train_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    mel_cache: Dict[str, object],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    k: int,
    subset: int,
    epochs_per_fold: int,
    lr: float,
    batch_size: int = 4,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, object]:
    subset = min(int(subset), len(train_utts))
    items = train_utts[:subset]
    perm = np.random.default_rng(42).permutation(subset)
    fold_size = subset // int(k)
    losses: List[float] = []
    tiny_cfg = make_tiny_cv_config(tokenizer, speaker_to_id)
    for f in range(int(k)):
        val_idx = set(perm[f * fold_size : (f + 1) * fold_size].tolist())
        tr_utts = [items[i] for i in range(subset) if i not in val_idx]
        va_utts = [items[i] for i in range(subset) if i in val_idx]
        vl = run_fold(
            fold_train_utts=tr_utts, fold_val_utts=va_utts, test_utts=test_utts,
            model_cfg=tiny_cfg, mel_cache=mel_cache, tokenizer=tokenizer,
            speaker_to_id=speaker_to_id, mel_mean=mel_mean, mel_std=mel_std,
            lr=lr, n_epochs=epochs_per_fold, batch_size=batch_size,
            device=device, use_amp=use_amp, amp_dtype=amp_dtype,
            seed=1000 + f,
        )
        losses.append(vl)
        if progress_cb is not None:
            progress_cb(f + 1, int(k), vl)
    mean = float(np.mean(losses))
    std = float(np.std(losses))
    return {"fold_losses": losses, "mean": mean, "std": std}


@app.function
def format_cv_summary(cv: Dict[str, object]) -> str:
    losses = cv["fold_losses"]
    return (
        f"**{len(losses)}-Fold CV mean loss**: {cv['mean']:.4f} ± {cv['std']:.4f}\n\n"
        "| Fold | Val loss |\n|---|---|\n"
        + "\n".join(f"| {i + 1} | {v:.4f} |" for i, v in enumerate(losses))
    )


@app.cell
def _(
    amp_dtype,
    cv_cb,
    cv_epochs_ui,
    cv_folds_ui,
    cv_run_btn,
    cv_subset_ui,
    device,
    lr_ui,
    mel_cache,
    mel_norm_mean,
    mel_norm_std,
    mo,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    train_utts,
    use_amp,
):
    mo.stop(not cv_cb.value, mo.md("_Enable **k-Fold CV** above to run this section._"))
    mo.stop(not cv_run_btn.value, mo.md("Toggle enabled — now click **Run k-Fold CV** to launch."))
    cv_results = run_kfold_cv(
        train_utts=train_utts, test_utts=test_utts, mel_cache=mel_cache,
        tokenizer=text_tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_norm_mean, mel_std=mel_norm_std,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        k=int(cv_folds_ui.value),
        subset=int(cv_subset_ui.value),
        epochs_per_fold=int(cv_epochs_ui.value),
        lr=float(lr_ui.value),
        progress_cb=lambda i, total, vl: mo.output.replace(mo.md(f"Fold {i}/{total} — val loss: {vl:.4f}")),
    )
    mo.output.replace(mo.md(format_cv_summary(cv_results)))
    return


@app.cell
def _(mo):
    asr_cer_cb = mo.ui.checkbox(label="Enable ASR-based CER (optional)", value=False)
    asr_num_ui = mo.ui.slider(1, 32, value=4, step=1, label="Utterances to Score")
    asr_run_btn = mo.ui.run_button(label="Run ASR CER")
    mo.vstack([asr_cer_cb, mo.hstack([asr_num_ui, asr_run_btn])])
    return asr_cer_cb, asr_num_ui, asr_run_btn


@app.function
def synthesize_speech(
    model: nn.Module,
    text: str,
    speaker_id: int,
    tokenizer: CharTokenizerV1,
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    num_steps: int = 32,
    guidance_scale: float = 1.0,
    method: str = "euler",
    manual_duration_s: Optional[float] = None,
    speed: float = 1.0,
    n_iter_gl: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    cfg = model.config
    patch_time = int(cfg.patch_time)
    n_mels = int(cfg.n_mels)
    tok_ids = tokenizer.encode(text)
    tok = torch.tensor([tok_ids], dtype=torch.int64, device=device)
    tok_mask = torch.ones((1, tok.shape[1]), dtype=torch.bool, device=device)
    speaker = torch.tensor([int(speaker_id)], dtype=torch.int64, device=device)
    if manual_duration_s is not None:
        n_frames = int(round(float(manual_duration_s) * int(mel_config["sample_rate"]) / int(mel_config["hop_length"])))
    else:
        n_frames = int(round(len(text) * float(frames_per_char) / max(float(speed), 1e-3)))
    n_frames = max(patch_time, n_frames)
    if n_frames % patch_time != 0:
        n_frames = n_frames + (patch_time - n_frames % patch_time)
    mel_mask = torch.ones((1, n_frames), dtype=torch.bool, device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x0 = torch.randn((1, n_mels, n_frames), generator=gen).to(device)

    def _vel(x, t):
        return classifier_free_velocity(
            model, x, t, tok, tok_mask, speaker, mel_mask, guidance_scale=guidance_scale
        )

    t0 = time.perf_counter()
    with torch.no_grad():
        x_final, nfe = ode_integrate(_vel, x0, 0.0, 1.0, int(num_steps), method=method)
    wall_time = time.perf_counter() - t0
    mel_norm = x_final[0]
    mel_mean_dev = mel_mean.to(device)
    mel_std_dev = mel_std.to(device)
    mel_log = mel_norm * mel_std_dev[:, None] + mel_mean_dev[:, None]
    waveform = griffin_lim_from_log_mel(mel_log, mel_config, n_iter=n_iter_gl, device=device).numpy()
    if was_training:
        model.train()
    return {
        "mel_log": mel_log.detach().cpu(),
        "mel_norm": mel_norm.detach().cpu(),
        "waveform": waveform,
        "nfe": int(nfe),
        "wall_time": float(wall_time),
    }


@app.function
def normalize_for_asr(text: str) -> str:
    up = text.upper()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ '|")
    replaced = "".join(c if c in allowed else " " for c in up)
    return " ".join(replaced.split())


@app.function
def load_asr_model(device: torch.device) -> Tuple[nn.Module, List[str], int]:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model_asr = bundle.get_model().to(device)
    labels = list(bundle.get_labels())
    return model_asr, labels, int(bundle.sample_rate)


@app.function
def cer_via_wav2vec(
    waveform: np.ndarray,
    reference_text: str,
    device: torch.device,
    asr_model: nn.Module,
    asr_labels: List[str],
    asr_sample_rate: int,
    src_sr: int = 24000,
) -> Tuple[float, str]:
    wav_t = torch.from_numpy(np.asarray(waveform)).to(device).float()
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0)
    if int(asr_sample_rate) != int(src_sr):
        wav_t = AF.resample(wav_t, int(src_sr), int(asr_sample_rate))
    with torch.no_grad():
        emissions, _ = asr_model(wav_t)
    ids = emissions[0].argmax(dim=-1).cpu().tolist()
    tokens: List[str] = []
    prev = -1
    for tid in ids:
        if tid == prev:
            continue
        prev = tid
        if tid == 0:
            continue
        tokens.append(asr_labels[tid])
    hyp_text = "".join(tokens).replace("|", " ").strip()
    ref = normalize_for_asr(reference_text)
    hyp = normalize_for_asr(hyp_text)
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0, hyp
    dist = int(AF.edit_distance(ref_chars, hyp_chars))
    return dist / len(ref_chars), hyp


@app.function
def run_asr_cer(
    trained_model: nn.Module,
    test_utts: List[Dict[str, object]],
    mel_cache: Dict[str, object],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    dataset_sample_rate: int,
    num_to_score: int,
    num_steps: int = 32,
) -> List[Dict[str, object]]:
    asr_model, asr_labels, asr_sr = load_asr_model(device)
    rows: List[Dict[str, object]] = []
    for i in range(min(int(num_to_score), len(test_utts))):
        utt = test_utts[i]
        spk = int(speaker_to_id.get(str(utt["speaker"]), 0))
        syn = synthesize_speech(
            model=trained_model, text=str(utt["text"]), speaker_id=spk,
            tokenizer=tokenizer, mel_mean=mel_mean, mel_std=mel_std,
            mel_config=mel_config, frames_per_char=frames_per_char, device=device,
            num_steps=int(num_steps), guidance_scale=1.0, method="euler",
            n_iter_gl=32, seed=i,
        )
        cer_syn, hyp_syn = cer_via_wav2vec(
            syn["waveform"], str(utt["text"]), device,
            asr_model, asr_labels, asr_sr, src_sr=int(dataset_sample_rate),
        )
        idx = mel_cache["utt_ids"].index(str(utt["utt_id"])) if str(utt["utt_id"]) in mel_cache["utt_ids"] else -1
        if idx >= 0:
            lm = mel_cache["log_mels"][idx].to(torch.float32)
            gt_wav = griffin_lim_from_log_mel(lm.to(device), mel_config, n_iter=32, device=device).numpy()
            cer_gt, _ = cer_via_wav2vec(
                gt_wav, str(utt["text"]), device,
                asr_model, asr_labels, asr_sr, src_sr=int(dataset_sample_rate),
            )
        else:
            cer_gt = float("nan")
        rows.append(
            {
                "utt_id": str(utt["utt_id"]),
                "text_len": int(utt["num_chars"]),
                "cer_synth": round(cer_syn, 3),
                "cer_baseline": round(cer_gt, 3),
                "hyp_synth": hyp_syn[:64],
            }
        )
    return rows


@app.function
def asr_cer_view(
    trained_model: Optional[nn.Module],
    test_utts: List[Dict[str, object]],
    mel_cache: Optional[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    dataset_sample_rate: int,
    num_to_score: int,
    mo,
) -> object:
    if trained_model is None or mel_cache is None:
        return mo.md("_Train the 1st-generation model (Section 5) first._")
    rows = run_asr_cer(
        trained_model=trained_model, test_utts=test_utts, mel_cache=mel_cache,
        tokenizer=tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_mean, mel_std=mel_std, mel_config=mel_config,
        frames_per_char=frames_per_char, device=device,
        dataset_sample_rate=int(dataset_sample_rate),
        num_to_score=int(num_to_score),
    )
    return mo.vstack(
        [
            mo.md("**ASR CER** — `cer_baseline` is the vocoder ceiling (ground-truth mel -> GL -> ASR)."),
            mo.ui.table(rows),
        ]
    )


@app.cell
def _(
    asr_cer_cb,
    asr_num_ui,
    asr_run_btn,
    dataset_sample_rate,
    device,
    frames_per_char,
    mel_cache,
    mel_config,
    mel_norm_mean,
    mel_norm_std,
    mo,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    trained_model: Optional[nn.Module],
):
    mo.stop(not asr_cer_cb.value, mo.md("_Optional. Enable **ASR-based CER** above._"))
    mo.stop(not asr_run_btn.value, mo.md("Click **Run ASR CER**."))
    asr_cer_view(
        trained_model=trained_model, test_utts=test_utts, mel_cache=mel_cache,
        tokenizer=text_tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_norm_mean, mel_std=mel_norm_std, mel_config=mel_config,
        frames_per_char=frames_per_char, device=device,
        dataset_sample_rate=dataset_sample_rate,
        num_to_score=int(asr_num_ui.value), mo=mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Reflow — 2-Rectified Flow (reverse-ODE couplings)

    For every training utterance $(x_1, c, s)$, integrate gen1's velocity
    field **backward** from $t = 1$ to $t = 0$ to obtain
    $x_0 = \Phi^{-1}(x_1, c, s)$. Store $(x_0, x_1, c, s)$ pairs on CPU
    in fp16 and train the reflow model on that fixed dataset with the same
    masked flow-matching objective and the same CFG dropout.

    The inversion guidance scale is configurable (default 1.0 = no guidance,
    which is the recommended setting for reflow pair generation). Inversion
    uses its own **non-shuffled** loader with a configurable inversion batch
    size so pairs are collected deterministically.

    We report `reflow_prior_stats` (mean / std of derived $x_0$ over valid
    entries, targets 0 and 1) and a round-trip check ($x_1 \to x_0 \to \hat{x}_1$
    mean squared error). If std $< 1$ significantly, raise the ODE step count.

    Reflow validation loss is computed on a **held-out subset of the pairs
    themselves** (the same coupled objective the reflow model is trained on),
    not on independent Gaussian x0.
    """)
    return


@app.cell
def _(mo, train_utts):
    n_train = max(len(train_utts), 1)
    reflow_steps_ui = mo.ui.slider(10, 200, value=100, step=10, label="Inversion Steps")
    reflow_method_ui = mo.ui.dropdown(options=["euler", "midpoint", "heun", "rk4"], value="euler", label="Inversion Method")
    reflow_guidance_ui = mo.ui.dropdown(
        options={"1.0": 1.0, "1.5": 1.5, "2.0": 2.0}, value="1.0", label="Inversion Guidance"
    )
    reflow_pairs_num_ui = mo.ui.number(value=int(n_train), label="Pairs to Build", start=8, stop=int(n_train))
    reflow_invert_bs_ui = mo.ui.dropdown(options=[4, 8, 16, 32], value=16, label="Inversion Batch Size")
    reflow_val_frac_ui = mo.ui.dropdown(
        options={"2%": 0.02, "5%": 0.05, "10%": 0.10}, value="5%", label="Reflow Val Fraction"
    )
    reflow_pairs_btn = mo.ui.run_button(label="Build Reflow Pairs")
    mo.vstack(
        [
            mo.md("### Reflow pair-generation controls"),
            mo.hstack([reflow_steps_ui, reflow_method_ui, reflow_guidance_ui]),
            mo.hstack([reflow_pairs_num_ui, reflow_invert_bs_ui, reflow_val_frac_ui, reflow_pairs_btn]),
        ]
    )
    return (
        reflow_guidance_ui,
        reflow_invert_bs_ui,
        reflow_method_ui,
        reflow_pairs_btn,
        reflow_pairs_num_ui,
        reflow_steps_ui,
        reflow_val_frac_ui,
    )


@app.function
def invert_batch(
    model: nn.Module,
    x1: torch.Tensor,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    speaker: torch.Tensor,
    mel_mask: torch.Tensor,
    num_steps: int,
    method: str,
    guidance_scale: float,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    def _vel(x, t):
        return classifier_free_velocity(
            model, x, t, tokens, token_mask, speaker, mel_mask, guidance_scale=guidance_scale
        )

    with torch.no_grad():
        x0, _ = ode_integrate(_vel, x1, 1.0, 0.0, int(num_steps), method=method)
    if was_training:
        model.train()
    return x0


@app.function
def build_reflow_pairs(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: int,
    method: str,
    guidance_scale: float,
    max_pairs: int,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, object]]:
    stored: List[Dict[str, object]] = []
    total = 0
    for batch in loader:
        if total >= int(max_pairs):
            break
        b = batch_to_device(batch, device)
        x1 = b["mel"].to(torch.float32)
        x0 = invert_batch(
            model, x1, b["tokens"], b["token_mask"], b["speakers"], b["mel_mask"],
            num_steps=int(num_steps), method=method, guidance_scale=float(guidance_scale),
        )
        for i in range(x1.shape[0]):
            if total >= int(max_pairs):
                break
            valid_frames = int(b["mel_mask"][i].sum().item())
            stored.append(
                {
                    "x0": x0[i].detach().cpu().to(torch.float16),
                    "x1": x1[i].detach().cpu().to(torch.float16),
                    "tokens": b["tokens"][i].detach().cpu().clone(),
                    "num_tokens": int(b["num_tokens"][i].item()),
                    "speaker": int(b["speakers"][i].item()),
                    "mel_mask": b["mel_mask"][i].detach().cpu().clone(),
                    "frames": valid_frames,
                    "utt_id": str(b["utt_ids"][i]),
                    "text": str(b["texts"][i]),
                }
            )
            total += 1
        if progress_cb is not None:
            progress_cb(total, int(max_pairs))
    return stored


@app.function
def reflow_prior_stats(pairs: List[Dict[str, object]]) -> Dict[str, float]:
    if not pairs:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    total_sum = 0.0
    total_sqsum = 0.0
    total_n = 0
    for p in pairs:
        n_frames = int(p["frames"])
        if n_frames == 0:
            continue
        vals = p["x0"][:, :n_frames].to(torch.float32)
        total_sum += float(vals.sum().item())
        total_sqsum += float((vals ** 2).sum().item())
        total_n += int(vals.numel())
    if total_n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = total_sum / total_n
    var = max(total_sqsum / total_n - mean * mean, 1e-8)
    return {"mean": mean, "std": math.sqrt(var), "n": total_n}


@app.function
def roundtrip_check(
    model: nn.Module,
    pairs: List[Dict[str, object]],
    device: torch.device,
    num_steps: int,
    method: str,
    guidance_scale: float,
    max_check: int = 4,
) -> Dict[str, float]:
    if not pairs:
        return {"roundtrip_mse": 0.0}
    model.eval()
    errs: List[float] = []
    for _i, p in enumerate(pairs[:max_check]):
        x0 = p["x0"].to(device).to(torch.float32).unsqueeze(0)
        x1 = p["x1"].to(device).to(torch.float32).unsqueeze(0)
        tokens = p["tokens"].to(device).unsqueeze(0)
        n_t = int(p["num_tokens"])
        tok_mask = torch.zeros((1, tokens.shape[1]), dtype=torch.bool, device=device)
        tok_mask[0, :n_t] = True
        speaker = torch.tensor([int(p["speaker"])], dtype=torch.int64, device=device)
        mel_mask = p["mel_mask"].to(device).unsqueeze(0)

        def _vel(x, t):
            return classifier_free_velocity(
                model, x, t, tokens, tok_mask, speaker, mel_mask, guidance_scale=guidance_scale
            )

        with torch.no_grad():
            x1_hat, _ = ode_integrate(_vel, x0, 0.0, 1.0, int(num_steps), method=method)
        f = int(p["frames"])
        errs.append(float(((x1_hat[0, :, :f] - x1[0, :, :f]) ** 2).mean().item()))
    return {"roundtrip_mse": float(np.mean(errs))}


@app.function
def build_reflow_pairs_view(
    trained_model: Optional[nn.Module],
    train_utts: List[Dict[str, object]],
    val_utts: List[Dict[str, object]],
    test_utts: List[Dict[str, object]],
    mel_cache: Optional[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    device: torch.device,
    inversion_bs: int,
    num_steps: int,
    method: str,
    guidance_scale: float,
    max_pairs: int,
    is_clicked: bool,
    mo,
) -> Tuple[Optional[List[Dict[str, object]]], float, Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if trained_model is None or mel_cache is None:
        mo.output.replace(mo.md("_Train the 1st-generation model first (Section 5)._"))
        return None, 0.0, None, None
    if not is_clicked:
        mo.output.replace(mo.md("Click **Build Reflow Pairs**."))
        return None, 0.0, None, None
    ldrs = make_datasets(
        mel_cache=mel_cache,
        train_utts=train_utts, val_utts=val_utts, test_utts=test_utts,
        tokenizer=tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_mean, mel_std=mel_std,
        batch_size=int(inversion_bs),
        patch_time=int(trained_model.config.patch_time),
        seed=0,
    )
    ldrs["train_sampler"].shuffle = False
    t0 = time.perf_counter()
    pairs = build_reflow_pairs(
        model=trained_model, loader=ldrs["train_loader"], device=device,
        num_steps=int(num_steps), method=str(method),
        guidance_scale=float(guidance_scale), max_pairs=int(max_pairs),
        progress_cb=lambda done, total: mo.output.replace(mo.md(f"Reflow pairs built: {done}/{total}")),
    )
    gen_time = time.perf_counter() - t0
    stats = reflow_prior_stats(pairs)
    rt = roundtrip_check(
        model=trained_model, pairs=pairs, device=device,
        num_steps=int(num_steps), method=str(method), guidance_scale=float(guidance_scale),
    )
    mo.output.replace(
        mo.md(
            f"**Reflow pairs built** in {gen_time:.1f}s — {len(pairs)} pairs.\n\n"
            f"- Derived `x0` mean: {stats['mean']:.4f} (target 0)\n"
            f"- Derived `x0` std: {stats['std']:.4f} (target 1)\n"
            f"- Round-trip MSE ($x_1 \\to x_0 \\to \\hat{{x}}_1$): {rt['roundtrip_mse']:.4f}"
        )
    )
    return pairs, gen_time, stats, rt


@app.cell
def _(
    device,
    mel_cache,
    mel_norm_mean,
    mel_norm_std,
    mo,
    reflow_guidance_ui,
    reflow_invert_bs_ui,
    reflow_method_ui,
    reflow_pairs_btn,
    reflow_pairs_num_ui,
    reflow_steps_ui,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    train_utts,
    trained_model: Optional[nn.Module],
    val_utts,
):
    reflow_pairs, reflow_gen_time, reflow_prior, reflow_roundtrip = build_reflow_pairs_view(
        trained_model=trained_model,
        train_utts=train_utts, val_utts=val_utts, test_utts=test_utts,
        mel_cache=mel_cache, tokenizer=text_tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_norm_mean, mel_std=mel_norm_std,
        device=device,
        inversion_bs=int(reflow_invert_bs_ui.value),
        num_steps=int(reflow_steps_ui.value),
        method=str(reflow_method_ui.value),
        guidance_scale=float(reflow_guidance_ui.value),
        max_pairs=int(reflow_pairs_num_ui.value),
        is_clicked=bool(reflow_pairs_btn.value),
        mo=mo,
    )
    return reflow_pairs, reflow_prior, reflow_roundtrip


@app.function
def plot_reflow_x0_hist(pairs: List[Dict[str, object]], title: str = "Derived x0 histogram"):
    vals: List[np.ndarray] = []
    for p in pairs[: min(len(pairs), 64)]:
        f = int(p["frames"])
        vals.append(p["x0"][:, :f].to(torch.float32).flatten().numpy())
    if not vals:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title("No reflow pairs")
        return fig
    v = np.concatenate(vals)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(v, bins=80, density=True, color="steelblue", alpha=0.75, label="Derived x0")
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi), "r--", lw=1.2, label="N(0, 1)")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell
def _(mo, reflow_pairs):
    if reflow_pairs is None:
        _out = mo.md("_Build reflow pairs first._")
    else:
        _out = plot_reflow_x0_hist(reflow_pairs, title="Derived x0 vs N(0, 1)")
    _out
    return


@app.class_definition
class ReflowPairDatasetV1(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Dict[str, object]]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Dict[str, object]:
        p = self.pairs[i]
        return {
            "mel": p["x1"].to(torch.float32),
            "x0": p["x0"].to(torch.float32),
            "frames": int(p["frames"]),
            "tokens": p["tokens"],
            "num_tokens": int(p["num_tokens"]),
            "speaker": int(p["speaker"]),
            "mel_mask_1d": p["mel_mask"],
            "utt_id": str(p["utt_id"]),
            "text": str(p["text"]),
        }


@app.function
def collate_reflow(samples: List[Dict[str, object]], pad_id: int, patch_time: int) -> Dict[str, torch.Tensor]:
    n_mels = samples[0]["mel"].shape[0]
    trim = [int(s["frames"]) - (int(s["frames"]) % max(int(patch_time), 1)) for s in samples]
    trim = [max(t, int(patch_time)) for t in trim]
    max_frames = max(trim)
    max_tokens = max(int(s["num_tokens"]) for s in samples)
    max_tokens = max(max_tokens, 1)
    B = len(samples)
    mel = torch.zeros(B, n_mels, max_frames, dtype=torch.float32)
    x0 = torch.randn(B, n_mels, max_frames, dtype=torch.float32)
    mel_mask = torch.zeros(B, max_frames, dtype=torch.bool)
    tokens = torch.full((B, max_tokens), int(pad_id), dtype=torch.int64)
    token_mask = torch.zeros(B, max_tokens, dtype=torch.bool)
    speakers = torch.zeros(B, dtype=torch.int64)
    frames_out: List[int] = []
    tokens_out: List[int] = []
    ids: List[str] = []
    texts: List[str] = []
    for b, s in enumerate(samples):
        f = trim[b]
        mel[b, :, :f] = s["mel"][:, :f]
        x0[b, :, :f] = s["x0"][:, :f]
        mel_mask[b, :f] = True
        n_t = int(s["num_tokens"])
        src = s["tokens"][:n_t] if s["tokens"].numel() >= n_t else s["tokens"]
        tokens[b, :n_t] = src
        token_mask[b, :n_t] = True
        speakers[b] = int(s["speaker"])
        frames_out.append(int(f))
        tokens_out.append(int(n_t))
        ids.append(str(s["utt_id"]))
        texts.append(str(s["text"]))
    return {
        "mel": mel,
        "x0": x0,
        "mel_mask": mel_mask,
        "tokens": tokens,
        "token_mask": token_mask,
        "speakers": speakers,
        "frames": torch.tensor(frames_out, dtype=torch.int64),
        "num_tokens": torch.tensor(tokens_out, dtype=torch.int64),
        "utt_ids": ids,
        "texts": texts,
    }


@app.function
def make_reflow_loader(
    pairs: List[Dict[str, object]],
    pad_id: int,
    patch_time: int,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    ds = ReflowPairDatasetV1(pairs)
    lengths = [int(p["frames"]) for p in pairs]
    sampler = BucketedBatchSamplerV1(
        lengths=lengths, batch_size=batch_size, bucket_multiplier=4, shuffle=shuffle, seed=seed,
    )
    collate = lambda batch: collate_reflow(batch, pad_id=pad_id, patch_time=patch_time)
    return torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate)


@app.function
def split_reflow_pairs(
    pairs: List[Dict[str, object]],
    val_fraction: float,
    seed: int = 4242,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not pairs:
        return [], []
    n = len(pairs)
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n_val, max(n - 1, 1))
    perm = np.random.default_rng(seed).permutation(n)
    val_idx = set(perm[:n_val].tolist())
    train = [pairs[i] for i in range(n) if i not in val_idx]
    val = [pairs[i] for i in range(n) if i in val_idx]
    return train, val


@app.function
def run_reflow_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    joint_dropout_prob: float = 0.0,
    grad_clip: float = 1.0,
    t_scheme: str = "uniform",
    warmup_scheduler=None,
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.0,
    start_step: int = 0,
) -> Tuple[float, int]:
    model.train()
    losses: List[float] = []
    step = int(start_step)
    for batch in loader:
        b = batch_to_device(batch, device)
        x1 = b["mel"].to(torch.float32)
        x0 = b["x0"].to(torch.float32)
        mel_mask = b["mel_mask"]
        tokens = b["tokens"]
        token_mask = b["token_mask"]
        speaker = b["speakers"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = masked_flow_matching_loss(
                model, x1, tokens, token_mask, speaker, mel_mask,
                x0=x0, t_scheme=t_scheme, joint_dropout_prob=float(joint_dropout_prob),
            )
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        if ema_model is not None and ema_decay > 0.0:
            update_ema(ema_model, model, ema_decay, num_updates=step)
        losses.append(float(loss.detach().item()))
        step += 1
    return sum(losses) / max(len(losses), 1), step - int(start_step)


@app.function
def run_reflow_evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    t_scheme: str = "uniform",
    seed: int = 12345,
) -> float:
    model.eval()
    losses: List[float] = []
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for batch in loader:
            b = batch_to_device(batch, device)
            x1 = b["mel"].to(torch.float32)
            x0 = b["x0"].to(torch.float32)
            mel_mask = b["mel_mask"]
            tokens = b["tokens"]
            token_mask = b["token_mask"]
            speaker = b["speakers"]
            t = sample_timesteps(x1.shape[0], t_scheme, device=None, generator=cpu_gen).to(device)
            xt = interpolate_path(x0, x1, t)
            target_v = x1 - x0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_v, _ = model(
                    xt, t, tokens, token_mask, speaker, mel_mask=mel_mask, joint_dropout_prob=0.0
                )
            err2 = ((pred_v.to(torch.float32) - target_v.to(torch.float32)) ** 2).mean(dim=1)
            mask_f = mel_mask.to(torch.float32)
            losses.append(float(((err2 * mask_f).sum() / mask_f.sum().clamp_min(1.0)).item()))
    return sum(losses) / max(len(losses), 1)


@app.function
def fit_reflow_model(
    model: nn.Module,
    train_cfg: TrainConfigV1,
    reflow_train_loader: torch.utils.data.DataLoader,
    reflow_val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, object]:
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.99)
    )
    scaler = make_grad_scaler(device, use_amp, amp_dtype)
    warmup = make_warmup_scheduler(optimizer, train_cfg.warmup_steps)
    ema_model: Optional[nn.Module] = None
    if train_cfg.ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.requires_grad_(False)
        ema_model.eval()
    train_losses: List[float] = []
    val_losses: List[float] = []
    start_time = time.perf_counter()
    global_step = 0
    for epoch in range(train_cfg.epochs):
        if hasattr(reflow_train_loader, "batch_sampler") and hasattr(reflow_train_loader.batch_sampler, "set_epoch"):
            reflow_train_loader.batch_sampler.set_epoch(epoch)
        tl, steps_taken = run_reflow_train_epoch(
            model=model, optimizer=optimizer, scaler=scaler,
            loader=reflow_train_loader, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
            joint_dropout_prob=train_cfg.p_uncond,
            grad_clip=train_cfg.grad_clip, t_scheme=train_cfg.t_scheme,
            warmup_scheduler=warmup, ema_model=ema_model, ema_decay=train_cfg.ema_decay,
            start_step=global_step,
        )
        global_step += int(steps_taken)
        vl = run_reflow_evaluate(
            model=ema_model if ema_model is not None else model,
            loader=reflow_val_loader, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
            t_scheme=train_cfg.t_scheme, seed=train_cfg.seed + 2000,
        )
        train_losses.append(tl)
        val_losses.append(vl)
        if progress_cb is not None:
            progress_cb(epoch + 1, train_cfg.epochs, tl, vl)
    final = ema_model if ema_model is not None else model
    final.eval()
    return {
        "model": final,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "wall_time": time.perf_counter() - start_time,
        "ema_used": ema_model is not None,
        "global_step": global_step,
        "train_config": train_cfg,
    }


@app.cell
def _(mo):
    reflow_lr_ui = mo.ui.dropdown(
        options={"3e-5": 3e-5, "1e-4": 1e-4, "3e-4": 3e-4}, value="1e-4", label="Reflow LR"
    )
    reflow_bs_ui = mo.ui.dropdown(options=[4, 8, 16, 32], value=16, label="Reflow Batch Size")
    reflow_wd_ui = mo.ui.dropdown(
        options={"0": 0.0, "1e-4": 1e-4, "1e-2": 1e-2}, value="1e-2", label="Reflow Weight Decay"
    )
    reflow_epochs_ui = mo.ui.slider(1, 100, value=10, step=1, label="Reflow Epochs")
    reflow_warmup_ui = mo.ui.dropdown(options=[0, 50, 100, 200], value=100, label="Reflow Warmup Steps")
    reflow_grad_clip_ui = mo.ui.dropdown(
        options={"none": 0.0, "0.5": 0.5, "1.0": 1.0, "2.0": 2.0}, value="1.0", label="Reflow Grad Clip"
    )
    reflow_ema_ui = mo.ui.dropdown(
        options={"off": 0.0, "0.999": 0.999, "0.9999": 0.9999}, value="0.999", label="Reflow EMA"
    )
    reflow_t_scheme_ui = mo.ui.dropdown(
        options=["uniform", "logit_normal"], value="uniform", label="Reflow Timestep Scheme"
    )
    reflow_init_ui = mo.ui.dropdown(
        options=["from-previous", "from-scratch"], value="from-previous", label="Reflow Init"
    )
    reflow_train_btn = mo.ui.run_button(label="Train Reflow Model")
    mo.vstack(
        [
            mo.md("### Reflow training hyperparameters"),
            mo.hstack([reflow_lr_ui, reflow_bs_ui, reflow_wd_ui, reflow_epochs_ui]),
            mo.hstack([reflow_warmup_ui, reflow_grad_clip_ui, reflow_ema_ui, reflow_t_scheme_ui]),
            mo.hstack([reflow_init_ui, reflow_train_btn]),
        ]
    )
    return (
        reflow_bs_ui,
        reflow_ema_ui,
        reflow_epochs_ui,
        reflow_grad_clip_ui,
        reflow_init_ui,
        reflow_lr_ui,
        reflow_t_scheme_ui,
        reflow_train_btn,
        reflow_warmup_ui,
        reflow_wd_ui,
    )


@app.function
def init_next_generation_model(
    previous_model: nn.Module, mode: str, device: torch.device
) -> nn.Module:
    if mode == "from-previous":
        return copy.deepcopy(previous_model).to(device)
    if mode == "from-scratch":
        return build_model_from_config(previous_model.config, device=device)
    raise ValueError(f"Unknown reflow init mode: {mode!r}")


@app.function
def run_reflow_training_view(
    trained_model: Optional[nn.Module],
    reflow_pairs: Optional[List[Dict[str, object]]],
    reflow_train_cfg: TrainConfigV1,
    reflow_val_fraction: float,
    init_mode: str,
    tokenizer: CharTokenizerV1,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    is_clicked: bool,
    mo,
) -> Optional[Dict[str, object]]:
    if trained_model is None or reflow_pairs is None:
        mo.output.replace(mo.md("_Generate reflow pairs first (Section 8 button)._"))
        return None
    if not is_clicked:
        mo.output.replace(mo.md("Click **Train Reflow Model**."))
        return None
    set_seed(reflow_train_cfg.seed)
    train_pairs, val_pairs = split_reflow_pairs(
        reflow_pairs, val_fraction=float(reflow_val_fraction), seed=reflow_train_cfg.seed + 5000,
    )
    train_loader = make_reflow_loader(
        train_pairs, pad_id=tokenizer.pad_id,
        patch_time=int(trained_model.config.patch_time),
        batch_size=int(reflow_train_cfg.batch_size), shuffle=True, seed=reflow_train_cfg.seed,
    )
    val_loader = make_reflow_loader(
        val_pairs, pad_id=tokenizer.pad_id,
        patch_time=int(trained_model.config.patch_time),
        batch_size=int(reflow_train_cfg.batch_size), shuffle=False, seed=reflow_train_cfg.seed + 1,
    )
    run = fit_reflow_model(
        model=init_next_generation_model(trained_model, str(init_mode), device),
        train_cfg=reflow_train_cfg,
        reflow_train_loader=train_loader,
        reflow_val_loader=val_loader,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        progress_cb=lambda epoch, total, tl, vl: mo.output.replace(
            mo.md(f"**Reflow epoch {epoch}/{total}** — train {tl:.4f} | val (held-out pairs) {vl:.4f}")
        ),
    )
    mo.output.replace(
        mo.md(
            f"**Reflow training complete** in {run['wall_time']:.1f}s — "
            f"final train {run['train_losses'][-1]:.4f} | "
            f"final val (held-out pairs) {run['val_losses'][-1]:.4f}"
        )
    )
    return run


@app.cell
def _(
    amp_dtype,
    device,
    gen1_train_cfg_used: Optional[TrainConfigV1],
    mo,
    reflow_bs_ui,
    reflow_ema_ui,
    reflow_epochs_ui,
    reflow_grad_clip_ui,
    reflow_init_ui,
    reflow_lr_ui,
    reflow_pairs,
    reflow_t_scheme_ui,
    reflow_train_btn,
    reflow_val_frac_ui,
    reflow_warmup_ui,
    reflow_wd_ui,
    seed_ui,
    text_tokenizer,
    trained_model: Optional[nn.Module],
    use_amp,
):
    reflow_run = run_reflow_training_view(
        trained_model=trained_model,
        reflow_pairs=reflow_pairs,
        reflow_train_cfg=TrainConfigV1(
            lr=float(reflow_lr_ui.value),
            batch_size=int(reflow_bs_ui.value),
            weight_decay=float(reflow_wd_ui.value),
            epochs=int(reflow_epochs_ui.value),
            warmup_steps=int(reflow_warmup_ui.value),
            grad_clip=float(reflow_grad_clip_ui.value),
            ema_decay=float(reflow_ema_ui.value),
            t_scheme=str(reflow_t_scheme_ui.value),
            p_uncond=float(gen1_train_cfg_used.p_uncond) if gen1_train_cfg_used is not None else 0.1,
            seed=int(seed_ui.value) + 7,
        ),
        reflow_val_fraction=float(reflow_val_frac_ui.value),
        init_mode=str(reflow_init_ui.value),
        tokenizer=text_tokenizer,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        is_clicked=bool(reflow_train_btn.value),
        mo=mo,
    )
    reflow_train_losses: List[float] = reflow_run["train_losses"] if reflow_run else []
    reflow_val_losses: List[float] = reflow_run["val_losses"] if reflow_run else []
    reflow_model: Optional[nn.Module] = reflow_run["model"] if reflow_run else None
    reflow_wall_time: float = reflow_run["wall_time"] if reflow_run else 0.0
    reflow_ema_used: bool = bool(reflow_run["ema_used"]) if reflow_run else False
    reflow_train_cfg_used: Optional[TrainConfigV1] = reflow_run["train_config"] if reflow_run else None
    return (
        reflow_ema_used,
        reflow_model,
        reflow_train_cfg_used,
        reflow_train_losses,
        reflow_val_losses,
        reflow_wall_time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Results

    Loss curves for the two generations, a ground-truth vs gen1 vs reflow
    mel grid with audio players for the same test texts, an NFE-sweep of
    few-step deviation vs a many-step reference, a denoising trajectory
    snapshot, and cross-attention alignment maps (with the diagonal prior on
    and off, so you can tell whether the model has learned monotonic
    alignment from content rather than only from the prior).
    """)
    return


@app.function
def plot_loss_curves(
    gen1_train: List[float],
    gen1_val: List[float],
    reflow_train: List[float],
    reflow_val: List[float],
):
    fig, ax = plt.subplots(figsize=(9, 4))
    if gen1_train:
        ax.plot(range(1, len(gen1_train) + 1), gen1_train, "b-o", ms=4, label="1st gen train")
    if gen1_val:
        ax.plot(range(1, len(gen1_val) + 1), gen1_val, "b--s", ms=4, label="1st gen val")
    if reflow_train:
        ax.plot(range(1, len(reflow_train) + 1), reflow_train, "r-o", ms=4, label="reflow train")
    if reflow_val:
        ax.plot(range(1, len(reflow_val) + 1), reflow_val, "r--s", ms=4, label="reflow val (held-out pairs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training / validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


@app.cell
def _(
    mo,
    reflow_train_losses: List[float],
    reflow_val_losses: List[float],
    train_losses: List[float],
    val_losses: List[float],
):
    if not train_losses and not reflow_train_losses:
        _out = mo.md("_Train something to see the loss curves._")
    else:
        _out = plot_loss_curves(train_losses, val_losses, reflow_train_losses, reflow_val_losses)
    _out
    return


@app.function
def plot_mel_grid(entries: List[Dict[str, object]], title: str = "Mel comparison"):
    n = len(entries)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title("No entries")
        return fig
    n_cols = max(len(entries[0]["mels"]), 1)
    fig, axes = plt.subplots(n, n_cols, figsize=(3.5 * n_cols, 2.4 * n), squeeze=False)
    for i, entry in enumerate(entries):
        for j, (name, mel) in enumerate(entry["mels"]):
            ax = axes[i, j]
            m = mel.detach().cpu().numpy()
            ax.imshow(m, origin="lower", aspect="auto", cmap="magma")
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(str(entry["text"])[:32], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


@app.function
def build_comparison_entries(
    utts: List[Dict[str, object]],
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    mel_cache: Dict[str, object],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    num_utts: int = 2,
    gen1_steps: int = 32,
    reflow_steps: int = 8,
    guidance_scale: float = 1.0,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for i, u in enumerate(utts[:num_utts]):
        spk_id = int(speaker_to_id.get(str(u["speaker"]), 0))
        mels: List[Tuple[str, torch.Tensor]] = []
        idx = mel_cache["utt_ids"].index(str(u["utt_id"])) if str(u["utt_id"]) in mel_cache["utt_ids"] else -1
        if idx >= 0:
            gt = mel_cache["log_mels"][idx].to(torch.float32)
            gt_norm = (gt - mel_mean[:, None]) / mel_std[:, None]
            mels.append(("ground truth", gt_norm))
        if trained_model is not None:
            gen1_syn = synthesize_speech(
                model=trained_model, text=str(u["text"]), speaker_id=spk_id,
                tokenizer=tokenizer, mel_mean=mel_mean, mel_std=mel_std,
                mel_config=mel_config, frames_per_char=frames_per_char, device=device,
                num_steps=int(gen1_steps), guidance_scale=guidance_scale,
                method="euler", n_iter_gl=16, seed=i,
            )
            mels.append((f"gen1 ({int(gen1_steps)} steps)", gen1_syn["mel_norm"]))
        if reflow_model is not None:
            rf_syn = synthesize_speech(
                model=reflow_model, text=str(u["text"]), speaker_id=spk_id,
                tokenizer=tokenizer, mel_mean=mel_mean, mel_std=mel_std,
                mel_config=mel_config, frames_per_char=frames_per_char, device=device,
                num_steps=int(reflow_steps), guidance_scale=guidance_scale,
                method="euler", n_iter_gl=16, seed=i,
            )
            mels.append((f"reflow ({int(reflow_steps)} steps)", rf_syn["mel_norm"]))
        entries.append({"utt_id": str(u["utt_id"]), "text": str(u["text"]), "mels": mels})
    return entries


@app.function
def build_mel_comparison_view(
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    test_utts: List[Dict[str, object]],
    mel_cache: Optional[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    dataset_sample_rate: int,
    mo,
) -> object:
    if trained_model is None and reflow_model is None:
        return mo.md("_Train first (Section 5) to see mel comparison._")
    if not test_utts or mel_cache is None:
        return mo.md("_No test utterances._")
    entries = build_comparison_entries(
        utts=test_utts, trained_model=trained_model, reflow_model=reflow_model,
        mel_cache=mel_cache, tokenizer=tokenizer, speaker_to_id=speaker_to_id,
        mel_mean=mel_mean, mel_std=mel_std, mel_config=mel_config,
        frames_per_char=frames_per_char, device=device,
        num_utts=2, gen1_steps=32, reflow_steps=8, guidance_scale=1.0,
    )
    fig = plot_mel_grid(entries, title="Ground truth vs gen1 vs reflow (test utterances)")
    audio_blocks = []
    for e in entries:
        elts = [mo.md(f"**{e['text'][:80]}**")]
        for name, mel in e["mels"]:
            log_m = mel * mel_std[:, None] + mel_mean[:, None]
            w = griffin_lim_from_log_mel(log_m.to(device), mel_config, n_iter=16, device=device).numpy()
            elts.append(mo.hstack([mo.md(f"_{name}_"), mo.audio(src=w, rate=dataset_sample_rate)]))
        audio_blocks.append(mo.vstack(elts))
    return mo.vstack([fig, *audio_blocks])


@app.cell
def _(
    dataset_sample_rate,
    device,
    frames_per_char,
    mel_cache,
    mel_config,
    mel_norm_mean,
    mel_norm_std,
    mo,
    reflow_model: Optional[nn.Module],
    speaker_to_id,
    test_utts,
    text_tokenizer,
    trained_model: Optional[nn.Module],
):
    build_mel_comparison_view(
        trained_model, reflow_model, test_utts, mel_cache, text_tokenizer,
        speaker_to_id, mel_norm_mean, mel_norm_std, mel_config,
        frames_per_char, device, dataset_sample_rate, mo,
    )
    return


@app.function
def few_step_deviation(
    model: nn.Module,
    utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    step_counts: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    reference_steps: int = 64,
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> Dict[int, float]:
    ref_syn = synthesize_speech(
        model=model, text=str(utts[0]["text"]),
        speaker_id=int(speaker_to_id.get(str(utts[0]["speaker"]), 0)),
        tokenizer=tokenizer, mel_mean=mel_mean, mel_std=mel_std,
        mel_config=mel_config, frames_per_char=frames_per_char, device=device,
        num_steps=int(reference_steps), guidance_scale=guidance_scale,
        method="euler", n_iter_gl=1, seed=seed,
    )
    ref = ref_syn["mel_norm"].to(device)
    out: Dict[int, float] = {}
    for s in step_counts:
        syn = synthesize_speech(
            model=model, text=str(utts[0]["text"]),
            speaker_id=int(speaker_to_id.get(str(utts[0]["speaker"]), 0)),
            tokenizer=tokenizer, mel_mean=mel_mean, mel_std=mel_std,
            mel_config=mel_config, frames_per_char=frames_per_char, device=device,
            num_steps=int(s), guidance_scale=guidance_scale,
            method="euler", n_iter_gl=1, seed=seed,
        )
        cur = syn["mel_norm"].to(device)
        T_common = min(cur.shape[-1], ref.shape[-1])
        out[int(s)] = float(((cur[:, :T_common] - ref[:, :T_common]) ** 2).mean().item())
    return out


@app.function
def plot_nfe_sweep(dev_by_model: Dict[str, Dict[int, float]], title: str = "Few-step deviation vs many-step reference"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, dev in dev_by_model.items():
        xs = sorted(dev.keys())
        ys = [dev[x] for x in xs]
        ax.plot(xs, ys, "-o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("NFE (Euler steps)")
    ax.set_ylabel("MSE vs reference")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return fig


@app.function
def nfe_sweep_view(
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    mo,
) -> object:
    if trained_model is None and reflow_model is None:
        return mo.md("_Train first for the NFE sweep._")
    if not test_utts:
        return mo.md("_No test utterances._")
    devs: Dict[str, Dict[int, float]] = {}
    if trained_model is not None:
        devs["gen1"] = few_step_deviation(
            model=trained_model, utts=test_utts, tokenizer=tokenizer,
            speaker_to_id=speaker_to_id, mel_mean=mel_mean, mel_std=mel_std,
            mel_config=mel_config, frames_per_char=frames_per_char,
            device=device, step_counts=(1, 2, 4, 8, 16), reference_steps=32,
        )
    if reflow_model is not None:
        devs["reflow"] = few_step_deviation(
            model=reflow_model, utts=test_utts, tokenizer=tokenizer,
            speaker_to_id=speaker_to_id, mel_mean=mel_mean, mel_std=mel_std,
            mel_config=mel_config, frames_per_char=frames_per_char,
            device=device, step_counts=(1, 2, 4, 8, 16), reference_steps=32,
        )
    return plot_nfe_sweep(devs, title="Few-step deviation (log-log)")


@app.cell
def _(
    device,
    frames_per_char,
    mel_config,
    mel_norm_mean,
    mel_norm_std,
    mo,
    reflow_model: Optional[nn.Module],
    speaker_to_id,
    test_utts,
    text_tokenizer,
    trained_model: Optional[nn.Module],
):
    nfe_sweep_view(
        trained_model, reflow_model, test_utts, text_tokenizer, speaker_to_id,
        mel_norm_mean, mel_norm_std, mel_config, frames_per_char, device, mo,
    )
    return


@app.function
def plot_denoise_trajectory(
    model: nn.Module,
    text: str,
    speaker_id: int,
    tokenizer: CharTokenizerV1,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    num_steps: int = 16,
    num_snapshots: int = 6,
    seed: int = 0,
):
    was_training = model.training
    model.eval()
    tok = torch.tensor([tokenizer.encode(text)], dtype=torch.int64, device=device)
    tok_mask = torch.ones((1, tok.shape[1]), dtype=torch.bool, device=device)
    speaker = torch.tensor([int(speaker_id)], dtype=torch.int64, device=device)
    n_frames = max(model.config.patch_time, int(round(len(text) * float(frames_per_char))))
    if n_frames % model.config.patch_time != 0:
        n_frames = n_frames + (model.config.patch_time - n_frames % model.config.patch_time)
    mel_mask = torch.ones((1, n_frames), dtype=torch.bool, device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn((1, model.config.n_mels, n_frames), generator=gen).to(device)
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    snap_idxs = sorted({int(round(v)) for v in np.linspace(0, num_steps, num_snapshots)})
    snaps: Dict[int, torch.Tensor] = {}
    if 0 in snap_idxs:
        snaps[0] = x.detach().clone()
    with torch.no_grad():
        for i in range(num_steps):
            t_batch = ts[i].expand(1)
            v = classifier_free_velocity(model, x, t_batch, tok, tok_mask, speaker, mel_mask, guidance_scale=1.0)
            x = x + v.to(torch.float32) * (ts[i + 1] - ts[i])
            if (i + 1) in snap_idxs:
                snaps[i + 1] = x.detach().clone()
    if was_training:
        model.train()
    fig, axes = plt.subplots(1, len(snap_idxs), figsize=(2.2 * len(snap_idxs), 3))
    axes = np.atleast_1d(axes)
    for j, idx in enumerate(snap_idxs):
        m = snaps[idx][0].detach().cpu().numpy()
        axes[j].imshow(m, origin="lower", aspect="auto", cmap="magma")
        axes[j].set_title(f"t={float(ts[idx].item()):.2f}", fontsize=9)
        axes[j].axis("off")
    fig.suptitle("Denoising trajectory (mel_norm)", fontsize=11)
    fig.tight_layout()
    return fig


@app.function
def denoise_trajectory_view(
    trained_model: Optional[nn.Module],
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    mo,
) -> object:
    if trained_model is None or not test_utts:
        return mo.md("_Train first to see denoising trajectory._")
    u = test_utts[0]
    return plot_denoise_trajectory(
        model=trained_model, text=str(u["text"]),
        speaker_id=int(speaker_to_id.get(str(u["speaker"]), 0)),
        tokenizer=tokenizer, mel_config=mel_config,
        frames_per_char=frames_per_char, device=device,
        num_steps=16, num_snapshots=6,
    )


@app.cell
def _(
    device,
    frames_per_char,
    mel_config,
    mo,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    trained_model: Optional[nn.Module],
):
    denoise_trajectory_view(
        trained_model, test_utts, text_tokenizer, speaker_to_id,
        mel_config, frames_per_char, device, mo,
    )
    return


@app.function
def compute_cross_attention_map(
    model: nn.Module,
    text: str,
    speaker_id: int,
    tokenizer: CharTokenizerV1,
    frames_per_char: float,
    device: torch.device,
    layer: int,
    t_val: float,
    use_align_prior: bool,
    seed: int,
) -> Tuple[Optional[np.ndarray], int]:
    was_training = model.training
    model.eval()
    cfg = model.config
    tok = torch.tensor([tokenizer.encode(text)], dtype=torch.int64, device=device)
    tok_mask = torch.ones((1, tok.shape[1]), dtype=torch.bool, device=device)
    speaker = torch.tensor([int(speaker_id)], dtype=torch.int64, device=device)
    n_frames = max(cfg.patch_time, int(round(len(text) * float(frames_per_char))))
    if n_frames % cfg.patch_time != 0:
        n_frames = n_frames + (cfg.patch_time - n_frames % cfg.patch_time)
    mel_mask = torch.ones((1, n_frames), dtype=torch.bool, device=device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x0 = torch.randn((1, cfg.n_mels, n_frames), generator=gen).to(device)
    ts = torch.linspace(0.0, 1.0, 17, device=device)
    x = x0.clone()
    with torch.no_grad():
        for i in range(16):
            if float(ts[i].item()) >= float(t_val):
                break
            t_batch = ts[i].expand(1)
            v = classifier_free_velocity(model, x, t_batch, tok, tok_mask, speaker, mel_mask, guidance_scale=1.0)
            x = x + v.to(torch.float32) * (ts[i + 1] - ts[i])
        t_batch = torch.tensor([float(t_val)], device=device)
        _, attn = model(
            x, t_batch, tok, tok_mask, speaker,
            mel_mask=mel_mask, joint_dropout_prob=0.0,
            return_attn_layer=int(layer), use_align_prior=bool(use_align_prior),
        )
    if was_training:
        model.train()
    if attn is None:
        return None, cfg.freq_rows
    a = attn.mean(dim=1)[0].detach().cpu().numpy()
    return a, cfg.freq_rows


@app.function
def plot_cross_attention_alignment(
    model: nn.Module,
    text: str,
    speaker_id: int,
    tokenizer: CharTokenizerV1,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    layer: int = 2,
    t_val: float = 0.5,
    seed: int = 0,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j, (label, use_prior) in enumerate([("with prior (what the model uses)", True), ("content-only (prior removed)", False)]):
        a, freq_rows = compute_cross_attention_map(
            model=model, text=text, speaker_id=speaker_id, tokenizer=tokenizer,
            frames_per_char=frames_per_char, device=device,
            layer=layer, t_val=t_val, use_align_prior=use_prior, seed=seed,
        )
        if a is None:
            axes[j].text(0.5, 0.5, "no attention returned", ha="center", va="center")
            continue
        time_cols = a.shape[0] // freq_rows
        a2d = a.reshape(freq_rows, time_cols, -1).mean(axis=0)
        axes[j].imshow(a2d, origin="lower", aspect="auto", cmap="viridis")
        axes[j].set_xlabel("Text tokens")
        axes[j].set_ylabel("Mel-time patches")
        axes[j].set_title(f"{label} (layer {layer}, t={t_val})", fontsize=9)
    fig.suptitle(
        "Cross-attention: with vs without the diagonal alignment prior. "
        "The 'content-only' panel reveals what the model has learned from text/mel alone; "
        "the 'with prior' panel is what actually drives the forward pass at training and sampling.",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


@app.function
def alignment_view(
    trained_model: Optional[nn.Module],
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    mo,
) -> object:
    if trained_model is None or not test_utts:
        return mo.md("_Train first to see alignment map._")
    u = test_utts[0]
    layer = min(len(trained_model.blocks) // 2, len(trained_model.blocks) - 1)
    return plot_cross_attention_alignment(
        model=trained_model, text=str(u["text"]),
        speaker_id=int(speaker_to_id.get(str(u["speaker"]), 0)),
        tokenizer=tokenizer, mel_config=mel_config,
        frames_per_char=frames_per_char, device=device,
        layer=layer, t_val=0.5,
    )


@app.cell
def _(
    device,
    frames_per_char,
    mel_config,
    mo,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    trained_model: Optional[nn.Module],
):
    alignment_view(
        trained_model, test_utts, text_tokenizer, speaker_to_id,
        mel_config, frames_per_char, device, mo,
    )
    return


@app.function
def summarize_generation(
    name: str,
    model: nn.Module,
    val_losses: List[float],
    wall_time: float,
    device: torch.device,
    utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    seed: int = 808,
) -> Dict[str, object]:
    dev = few_step_deviation(
        model=model, utts=utts, tokenizer=tokenizer,
        speaker_to_id=speaker_to_id, mel_mean=mel_mean, mel_std=mel_std,
        mel_config=mel_config, frames_per_char=frames_per_char,
        device=device, step_counts=(1, 2, 4, 8), reference_steps=32, seed=seed,
    )
    return {
        "model": name,
        "params": count_parameters(model),
        **{f"dev@{k}": round(v, 4) for k, v in dev.items()},
        "final_val_loss": round(val_losses[-1], 4) if val_losses else None,
        "train_time_s": round(wall_time, 1),
    }


@app.function
def comparison_table_view(
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    val_losses: List[float],
    reflow_val_losses: List[float],
    train_wall_time: float,
    reflow_wall_time: float,
    test_utts: List[Dict[str, object]],
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    mo,
) -> object:
    if trained_model is None and reflow_model is None:
        return mo.md("_Train something to see the comparison table._")
    if not test_utts:
        return mo.md("_No test utterances for comparison._")
    rows: List[Dict[str, object]] = []
    if trained_model is not None:
        rows.append(
            summarize_generation(
                "1st gen", trained_model, val_losses, train_wall_time, device,
                test_utts, tokenizer, speaker_to_id, mel_mean, mel_std,
                mel_config, frames_per_char,
            )
        )
    if reflow_model is not None:
        rows.append(
            summarize_generation(
                "reflow", reflow_model, reflow_val_losses, reflow_wall_time, device,
                test_utts, tokenizer, speaker_to_id, mel_mean, mel_std,
                mel_config, frames_per_char,
            )
        )
    return mo.vstack(
        [
            mo.md(
                "**Model comparison** — `dev@k` is the MSE of a k-step sample "
                "against a 32-step reference (lower = straighter)."
            ),
            mo.ui.table(rows),
        ]
    )


@app.cell
def _(
    device,
    frames_per_char,
    mel_config,
    mel_norm_mean,
    mel_norm_std,
    mo,
    reflow_model: Optional[nn.Module],
    reflow_val_losses: List[float],
    reflow_wall_time: float,
    speaker_to_id,
    test_utts,
    text_tokenizer,
    train_wall_time: float,
    trained_model: Optional[nn.Module],
    val_losses: List[float],
):
    comparison_table_view(
        trained_model, reflow_model, val_losses, reflow_val_losses,
        train_wall_time, reflow_wall_time,
        test_utts, text_tokenizer, speaker_to_id, mel_norm_mean, mel_norm_std,
        mel_config, frames_per_char, device, mo,
    )
    return


@app.function
def summary_markdown(
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    mo,
) -> object:
    bits: List[str] = []
    if trained_model is not None:
        bits.append("- **1st-generation model** trained with rectified flow matching on log-mel spectrograms.")
    if reflow_model is not None:
        bits.append(
            "- **Reflow model** trained on reverse-ODE couplings — real mel kept as x1, "
            "x0 derived by inverting gen1."
        )
    bits.append(
        "- **Limitations**: LibriTTS dev-clean is only ~9 h; a base preset needs "
        "hundreds of hours to reach production quality. Griffin-Lim caps audio "
        "quality (measure the ceiling in Section 7's ASR block). Cross-attention "
        "alignment relies on the soft diagonal prior when data is small — see the "
        "content-only panel in Section 9. Scale next by (i) train-clean-100 (~54 h), "
        "(ii) neural vocoder, (iii) larger depth and hidden dim, "
        "(iv) longer reflow inversion (200+ steps)."
    )
    if not bits:
        bits = ["_No trained models yet — run Sections 5 and/or 8._"]
    return mo.md("### Summary\n\n" + "\n\n".join(bits))


@app.cell
def _(
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    summary_markdown(trained_model, reflow_model, mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Sampling from the Trained Model

    Interactive TTS from the trained model. Choose gen1 or reflow (EMA
    weights are the default), type text, pick a speaker, choose duration
    (automatic from text length × frames_per_char, adjustable by a speed
    factor, or override with a manual duration), pick the ODE solver, step
    count, guidance scale, Griffin-Lim iterations, and click **Generate**.
    """)
    return


@app.function
def model_choice_options(
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    if trained_model is not None:
        opts["gen1"] = "gen1"
    if reflow_model is not None:
        opts["reflow"] = "reflow"
    if not opts:
        opts = {"(train first)": "none"}
    return opts


@app.function
def speaker_dropdown_options(speaker_to_id: Dict[str, int]) -> Dict[str, int]:
    opts: Dict[str, int] = {"(null / cfg unconditional)": 0}
    for name, sid in sorted(speaker_to_id.items(), key=lambda kv: kv[1]):
        opts[f"speaker {name}"] = int(sid)
    return opts


@app.function
def default_speaker_label(opts: Dict[str, int]) -> str:
    keys = list(opts.keys())
    return keys[1] if len(keys) > 1 else keys[0]


@app.cell
def _(
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    sample_which_ui = mo.ui.dropdown(
        options=model_choice_options(trained_model, reflow_model),
        value=list(model_choice_options(trained_model, reflow_model).keys())[0],
        label="Model",
    )
    sample_which_ui
    return (sample_which_ui,)


@app.function
def default_sample_text(test_utts: List[Dict[str, object]]) -> str:
    return str(test_utts[0]["text"]) if test_utts else "Hello world."


@app.cell
def _(mo, speaker_to_id, test_utts):
    sample_text_ui = mo.ui.text_area(value=default_sample_text(test_utts), label="Text")
    sample_speaker_ui = mo.ui.dropdown(
        options=speaker_dropdown_options(speaker_to_id),
        value=default_speaker_label(speaker_dropdown_options(speaker_to_id)),
        label="Speaker",
    )
    sample_speed_ui = mo.ui.dropdown(
        options={"0.7": 0.7, "0.85": 0.85, "1.0": 1.0, "1.2": 1.2, "1.5": 1.5},
        value="1.0", label="Speed",
    )
    sample_manual_dur_ui = mo.ui.number(value=0.0, label="Manual Duration (s, 0=auto)", start=0.0, stop=30.0, step=0.1)
    sample_steps_ui = mo.ui.slider(1, 128, value=32, step=1, label="ODE Steps")
    sample_method_ui = mo.ui.dropdown(options=["euler", "midpoint", "heun", "rk4"], value="euler", label="Solver")
    sample_cfg_ui = mo.ui.dropdown(
        options={"1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "3.0": 3.0, "5.0": 5.0}, value="1.5", label="CFG"
    )
    sample_gl_ui = mo.ui.dropdown(options=[8, 16, 32, 64], value=32, label="Griffin-Lim iters")
    sample_seed_ui = mo.ui.number(value=0, label="Seed", start=0, stop=2**31 - 1)
    sample_btn = mo.ui.run_button(label="Generate")
    mo.vstack(
        [
            mo.md("### Sampling controls"),
            sample_text_ui,
            mo.hstack([sample_speaker_ui, sample_speed_ui, sample_manual_dur_ui]),
            mo.hstack([sample_steps_ui, sample_method_ui, sample_cfg_ui, sample_gl_ui]),
            mo.hstack([sample_seed_ui, sample_btn]),
        ]
    )
    return (
        sample_btn,
        sample_cfg_ui,
        sample_gl_ui,
        sample_manual_dur_ui,
        sample_method_ui,
        sample_seed_ui,
        sample_speaker_ui,
        sample_speed_ui,
        sample_steps_ui,
        sample_text_ui,
    )


@app.function
def render_synthesis_output(
    synth: Dict[str, object],
    dataset_sample_rate: int,
    mo,
) -> object:
    mel = synth["mel_norm"].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"Sampled mel (nfe={synth['nfe']}, wall={synth['wall_time']:.2f}s)")
    ax.axis("off")
    fig.tight_layout()
    return mo.vstack([fig, mo.audio(src=synth["waveform"], rate=int(dataset_sample_rate))])


@app.function
def sampling_view(
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    which: str,
    text: str,
    speaker_id: int,
    tokenizer: CharTokenizerV1,
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    mel_config: Dict[str, object],
    frames_per_char: float,
    device: torch.device,
    dataset_sample_rate: int,
    num_steps: int,
    guidance_scale: float,
    method: str,
    manual_duration_s: Optional[float],
    speed: float,
    n_iter_gl: int,
    seed: int,
    is_clicked: bool,
    mo,
) -> object:
    if which == "none":
        return mo.md("_Train first — no model available._")
    if not is_clicked:
        return mo.md("Type text and click **Generate**.")
    model = trained_model if which == "gen1" else reflow_model
    if model is None:
        return mo.md("_Requested model is not trained yet._")
    syn = synthesize_speech(
        model=model, text=text, speaker_id=int(speaker_id),
        tokenizer=tokenizer, mel_mean=mel_mean, mel_std=mel_std,
        mel_config=mel_config, frames_per_char=frames_per_char, device=device,
        num_steps=int(num_steps), guidance_scale=float(guidance_scale),
        method=str(method), manual_duration_s=manual_duration_s, speed=float(speed),
        n_iter_gl=int(n_iter_gl), seed=int(seed),
    )
    return render_synthesis_output(syn, int(dataset_sample_rate), mo)


@app.cell
def _(
    dataset_sample_rate,
    device,
    frames_per_char,
    mel_config,
    mel_norm_mean,
    mel_norm_std,
    mo,
    reflow_model: Optional[nn.Module],
    sample_btn,
    sample_cfg_ui,
    sample_gl_ui,
    sample_manual_dur_ui,
    sample_method_ui,
    sample_seed_ui,
    sample_speaker_ui,
    sample_speed_ui,
    sample_steps_ui,
    sample_text_ui,
    sample_which_ui,
    text_tokenizer,
    trained_model: Optional[nn.Module],
):
    sampling_view(
        trained_model=trained_model, reflow_model=reflow_model,
        which=str(sample_which_ui.value),
        text=str(sample_text_ui.value),
        speaker_id=int(sample_speaker_ui.value),
        tokenizer=text_tokenizer,
        mel_mean=mel_norm_mean, mel_std=mel_norm_std,
        mel_config=mel_config, frames_per_char=frames_per_char,
        device=device, dataset_sample_rate=dataset_sample_rate,
        num_steps=int(sample_steps_ui.value),
        guidance_scale=float(sample_cfg_ui.value),
        method=str(sample_method_ui.value),
        manual_duration_s=(float(sample_manual_dur_ui.value) if float(sample_manual_dur_ui.value) > 0.0 else None),
        speed=float(sample_speed_ui.value),
        n_iter_gl=int(sample_gl_ui.value),
        seed=int(sample_seed_ui.value),
        is_clicked=bool(sample_btn.value),
        mo=mo,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Save Trained Model

    The chosen generation's state dict is written to the repo-root `models/`
    directory (created if missing). A JSON sidecar carries everything needed
    to sample later without touching the dataset — model + mel config, mel
    normalization stats, vocabulary, speaker ids, `frames_per_char`, and the
    training-config snapshot actually used. Reflow checkpoints also carry
    the pair provenance (inversion steps / method / guidance / pair count /
    prior stats / round-trip MSE).
    """)
    return


@app.cell
def _(
    mo,
    reflow_model: Optional[nn.Module],
    trained_model: Optional[nn.Module],
):
    save_which_ui = mo.ui.dropdown(
        options=model_choice_options(trained_model, reflow_model),
        value=list(model_choice_options(trained_model, reflow_model).keys())[0],
        label="Which Model",
    )
    save_which_ui
    return (save_which_ui,)


@app.function
def default_checkpoint_name(generation: str) -> str:
    return {
        "gen1": "libritts_dit_rcfm_tts_v1.pt",
        "reflow": "libritts_dit_rcfm_tts_reflow_v1.pt",
    }.get(generation, "libritts_dit_rcfm_tts_v1.pt")


@app.cell
def _(mo, save_which_ui):
    save_filename_ui = mo.ui.text(
        value=default_checkpoint_name(str(save_which_ui.value)),
        label="Filename (saved into models/)",
        full_width=True,
    )
    save_btn = mo.ui.run_button(label="Save Model")
    mo.vstack([save_filename_ui, save_btn])
    return save_btn, save_filename_ui


@app.function
def format_tag() -> str:
    return "dit_rcfm_tts_libritts_v1"


@app.function
def save_model_bundle(
    model: nn.Module,
    save_dir: Path,
    filename: str,
    generation: str,
    mel_config: Dict[str, object],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    frames_per_char: float,
    train_cfg_snapshot: Dict[str, object],
    ema_used: bool,
    reflow_provenance: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / filename
    sidecar_path = weights_path.with_suffix(".json")
    torch.save(model.state_dict(), weights_path)
    ordered_speakers = [""] * (max(speaker_to_id.values(), default=0) + 1)
    for name, idx in speaker_to_id.items():
        if 0 <= int(idx) < len(ordered_speakers):
            ordered_speakers[int(idx)] = str(name)
    payload = {
        "format": format_tag(),
        "generation": generation,
        "model_config": asdict(model.config),
        "mel_config": {k: (v if not isinstance(v, torch.Tensor) else v.tolist()) for k, v in mel_config.items()},
        "mel_mean": mel_mean.to(torch.float32).tolist(),
        "mel_std": mel_std.to(torch.float32).tolist(),
        "vocab": list(tokenizer.vocab),
        "speaker_ids": ordered_speakers,
        "frames_per_char": float(frames_per_char),
        "train_config": train_cfg_snapshot,
        "ema_used": bool(ema_used),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if reflow_provenance is not None:
        payload["reflow_provenance"] = reflow_provenance
    with open(sidecar_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return weights_path, sidecar_path


@app.function
def save_checkpoint(
    which: str,
    filename: str,
    trained_model: Optional[nn.Module],
    reflow_model: Optional[nn.Module],
    gen1_train_cfg_used: Optional[TrainConfigV1],
    reflow_train_cfg_used: Optional[TrainConfigV1],
    trained_ema_used: bool,
    reflow_ema_used: bool,
    reflow_prior: Optional[Dict[str, float]],
    reflow_roundtrip: Optional[Dict[str, float]],
    reflow_pairs: Optional[List[Dict[str, object]]],
    reflow_invert_steps: int,
    reflow_invert_method: str,
    reflow_invert_guidance: float,
    mel_config: Dict[str, object],
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
    tokenizer: CharTokenizerV1,
    speaker_to_id: Dict[str, int],
    frames_per_char: float,
    save_dir: Path,
    mo,
) -> object:
    target = trained_model if which == "gen1" else reflow_model
    if target is None:
        return mo.md("_Requested model is not trained yet._")
    fname = str(filename).strip() or default_checkpoint_name(str(which))
    if which == "gen1":
        cfg_used = gen1_train_cfg_used
        ema_used = bool(trained_ema_used)
        provenance = None
    else:
        cfg_used = reflow_train_cfg_used
        ema_used = bool(reflow_ema_used)
        provenance = {
            "inversion_steps": int(reflow_invert_steps),
            "inversion_method": str(reflow_invert_method),
            "inversion_guidance_scale": float(reflow_invert_guidance),
            "pair_count": len(reflow_pairs) if reflow_pairs is not None else 0,
            "prior_mean": float(reflow_prior["mean"]) if reflow_prior is not None else None,
            "prior_std": float(reflow_prior["std"]) if reflow_prior is not None else None,
            "roundtrip_mse": float(reflow_roundtrip["roundtrip_mse"]) if reflow_roundtrip is not None else None,
        }
    train_cfg_snapshot = asdict(cfg_used) if cfg_used is not None else {}
    weights_path, sidecar_path = save_model_bundle(
        model=target, save_dir=save_dir, filename=fname, generation=str(which),
        mel_config=mel_config, mel_mean=mel_mean, mel_std=mel_std,
        tokenizer=tokenizer, speaker_to_id=speaker_to_id,
        frames_per_char=frames_per_char,
        train_cfg_snapshot=train_cfg_snapshot, ema_used=ema_used,
        reflow_provenance=provenance,
    )
    return mo.md(
        f"**Saved.**\n\n- weights: `{weights_path}`\n- sidecar: `{sidecar_path}`\n\n"
        "Saved weights are the EMA weights when EMA was enabled during training "
        "(sample from them at inference)."
    )


@app.cell
def _(
    frames_per_char,
    gen1_train_cfg_used: Optional[TrainConfigV1],
    mel_config,
    mel_norm_mean,
    mel_norm_std,
    mo,
    reflow_ema_used: bool,
    reflow_guidance_ui,
    reflow_method_ui,
    reflow_model: Optional[nn.Module],
    reflow_pairs,
    reflow_prior,
    reflow_roundtrip,
    reflow_steps_ui,
    reflow_train_cfg_used: Optional[TrainConfigV1],
    save_btn,
    save_filename_ui,
    save_which_ui,
    speaker_to_id,
    text_tokenizer,
    trained_ema_used: bool,
    trained_model: Optional[nn.Module],
):
    if str(save_which_ui.value) == "none":
        _out = mo.md("_Train first (Section 5 and/or Section 8)._")
    elif not save_btn.value:
        _out = mo.md("Choose which model and click **Save Model**.")
    else:
        _out = save_checkpoint(
            which=str(save_which_ui.value),
            filename=str(save_filename_ui.value),
            trained_model=trained_model, reflow_model=reflow_model,
            gen1_train_cfg_used=gen1_train_cfg_used,
            reflow_train_cfg_used=reflow_train_cfg_used,
            trained_ema_used=bool(trained_ema_used),
            reflow_ema_used=bool(reflow_ema_used),
            reflow_prior=reflow_prior, reflow_roundtrip=reflow_roundtrip,
            reflow_pairs=reflow_pairs,
            reflow_invert_steps=int(reflow_steps_ui.value),
            reflow_invert_method=str(reflow_method_ui.value),
            reflow_invert_guidance=float(reflow_guidance_ui.value),
            mel_config=mel_config, mel_mean=mel_norm_mean, mel_std=mel_norm_std,
            tokenizer=text_tokenizer, speaker_to_id=speaker_to_id,
            frames_per_char=frames_per_char,
            save_dir=Path(__file__).resolve().parent.parent / "models",
            mo=mo,
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Load a Saved Model for Sampling

    Pick any `.pt` file in `models/` whose sidecar `.json` has the format
    tag `dit_rcfm_tts_libritts_v1`. The checkpoint's config drives model
    rebuilding, so this section is independent of Sections 3–5: it needs
    only the checkpoint files.
    """)
    return


@app.function
def list_saved_checkpoints(models_dir: Path, tag: str) -> List[Path]:
    if not models_dir.exists():
        return []
    out: List[Path] = []
    for p in sorted(models_dir.glob("*.pt")):
        side = p.with_suffix(".json")
        if not side.exists():
            continue
        try:
            payload = json.loads(side.read_text())
            if str(payload.get("format", "")) == str(tag):
                out.append(p)
        except (json.JSONDecodeError, OSError):
            continue
    return out


@app.cell
def _(mo):
    ckpt_refresh_ui = mo.ui.run_button(label="Refresh List")
    load_btn = mo.ui.run_button(label="Load Model")
    mo.hstack([ckpt_refresh_ui, load_btn])
    return ckpt_refresh_ui, load_btn


@app.function
def checkpoint_dropdown_options(models_dir: Path, tag: str) -> Dict[str, str]:
    paths = list_saved_checkpoints(models_dir, tag)
    if not paths:
        return {"(no checkpoints found)": ""}
    return {p.name: str(p) for p in paths}


@app.cell
def _(ckpt_refresh_ui, mo):
    ckpt_refresh_ui.value
    ckpt_pick_ui = mo.ui.dropdown(
        options=checkpoint_dropdown_options(
            Path(__file__).resolve().parent.parent / "models", format_tag()
        ),
        value=list(
            checkpoint_dropdown_options(
                Path(__file__).resolve().parent.parent / "models", format_tag()
            ).keys()
        )[0],
        label="Checkpoint",
    )
    ckpt_pick_ui
    return (ckpt_pick_ui,)


@app.function
def load_saved_bundle(
    checkpoint_path: Path,
    device: torch.device,
    tag: str,
) -> Tuple[Optional[nn.Module], Optional[Dict[str, object]], str]:
    try:
        sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text())
    except (OSError, json.JSONDecodeError) as err:
        return None, None, f"Failed to read sidecar: {err}"
    if str(sidecar.get("format", "")) != str(tag):
        return None, None, f"Incompatible sidecar format: {sidecar.get('format')!r}"
    cfg_dict = dict(sidecar["model_config"])
    cfg, err = make_dit_tts_config(**cfg_dict)
    if cfg is None:
        return None, None, f"Invalid model config: {err}"
    model = build_model_from_config(cfg, device=device)
    try:
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except (RuntimeError, FileNotFoundError) as err:
        return None, None, f"Failed to load weights: {err}"
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as err:
        return None, None, f"State dict mismatch: {err}"
    model.eval()
    return model, sidecar, ""


@app.function
def load_checkpoint_view(
    checkpoint_path: str,
    device: torch.device,
    is_clicked: bool,
    mo,
) -> Tuple[Optional[nn.Module], Optional[Dict[str, object]], object]:
    if not is_clicked:
        return None, None, mo.md("Click **Load Model** to load the selected checkpoint.")
    if not checkpoint_path:
        return None, None, mo.md("_No checkpoint selected._")
    model, sidecar, err = load_saved_bundle(Path(checkpoint_path), device, format_tag())
    if model is None:
        return None, None, mo.md(f"**Load failed** — {err}")
    return model, sidecar, mo.md(
        f"**Loaded** `{Path(checkpoint_path).name}` "
        f"(generation={sidecar['generation']}, params={count_parameters(model):,})"
    )


@app.cell
def _(ckpt_pick_ui, device, load_btn, mo):
    loaded_model, loaded_sidecar, _out = load_checkpoint_view(
        str(ckpt_pick_ui.value) if ckpt_pick_ui.value else "",
        device, bool(load_btn.value), mo,
    )
    _out
    return loaded_model, loaded_sidecar


@app.function
def rebuild_tokenizer_from_sidecar(sidecar: Dict[str, object]) -> CharTokenizerV1:
    vocab = list(sidecar["vocab"])
    return CharTokenizerV1(vocab=vocab)


@app.function
def rebuild_speaker_map_from_sidecar(sidecar: Dict[str, object]) -> Dict[str, int]:
    ids = list(sidecar["speaker_ids"])
    out: Dict[str, int] = {}
    for i, name in enumerate(ids):
        if i > 0 and str(name):
            out[str(name)] = i
    return out


@app.function
def loaded_speaker_options(loaded_sidecar: Optional[Dict[str, object]]) -> Dict[str, int]:
    if loaded_sidecar is None:
        return {"(null / cfg unconditional)": 0}
    spk_map = rebuild_speaker_map_from_sidecar(loaded_sidecar)
    return speaker_dropdown_options(spk_map)


@app.cell
def _(loaded_sidecar, mo):
    load_text_ui = mo.ui.text_area(value="Hello world.", label="Text (loaded)")
    load_speaker_ui = mo.ui.dropdown(
        options=loaded_speaker_options(loaded_sidecar),
        value=default_speaker_label(loaded_speaker_options(loaded_sidecar)),
        label="Speaker (loaded)",
    )
    load_steps_ui = mo.ui.slider(1, 128, value=32, step=1, label="ODE Steps (loaded)")
    load_method_ui = mo.ui.dropdown(options=["euler", "midpoint", "heun", "rk4"], value="euler", label="Solver (loaded)")
    load_cfg_ui = mo.ui.dropdown(
        options={"1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "3.0": 3.0}, value="1.5", label="CFG (loaded)"
    )
    load_gl_ui = mo.ui.dropdown(options=[8, 16, 32, 64], value=32, label="Griffin-Lim iters (loaded)")
    load_speed_ui = mo.ui.dropdown(
        options={"0.7": 0.7, "0.85": 0.85, "1.0": 1.0, "1.2": 1.2}, value="1.0", label="Speed (loaded)"
    )
    load_manual_dur_ui = mo.ui.number(
        value=0.0, label="Manual Duration (s, 0=auto) (loaded)", start=0.0, stop=30.0, step=0.1
    )
    load_seed_ui = mo.ui.number(value=0, label="Seed (loaded)", start=0, stop=2**31 - 1)
    load_sample_btn = mo.ui.run_button(label="Generate (Loaded)")
    mo.vstack(
        [
            mo.md("### Sampling controls (loaded checkpoint)"),
            load_text_ui,
            mo.hstack([load_speaker_ui, load_speed_ui, load_manual_dur_ui]),
            mo.hstack([load_steps_ui, load_method_ui, load_cfg_ui, load_gl_ui]),
            mo.hstack([load_seed_ui, load_sample_btn]),
        ]
    )
    return (
        load_cfg_ui,
        load_gl_ui,
        load_manual_dur_ui,
        load_method_ui,
        load_sample_btn,
        load_seed_ui,
        load_speaker_ui,
        load_speed_ui,
        load_steps_ui,
        load_text_ui,
    )


@app.function
def loaded_sampling_view(
    loaded_model: Optional[nn.Module],
    loaded_sidecar: Optional[Dict[str, object]],
    text: str,
    speaker_id: int,
    num_steps: int,
    guidance_scale: float,
    method: str,
    manual_duration_s: Optional[float],
    speed: float,
    n_iter_gl: int,
    seed: int,
    device: torch.device,
    is_clicked: bool,
    mo,
) -> object:
    if loaded_model is None or loaded_sidecar is None:
        return mo.md("_Load a checkpoint first._")
    if not is_clicked:
        return mo.md("Click **Generate (Loaded)**.")
    tok = rebuild_tokenizer_from_sidecar(loaded_sidecar)
    mel_mean = torch.tensor(list(loaded_sidecar["mel_mean"]), dtype=torch.float32)
    mel_std = torch.tensor(list(loaded_sidecar["mel_std"]), dtype=torch.float32)
    mel_cfg = {k: v for k, v in loaded_sidecar["mel_config"].items()}
    syn = synthesize_speech(
        model=loaded_model, text=text, speaker_id=int(speaker_id),
        tokenizer=tok, mel_mean=mel_mean, mel_std=mel_std,
        mel_config=mel_cfg, frames_per_char=float(loaded_sidecar["frames_per_char"]),
        device=device, num_steps=int(num_steps), guidance_scale=float(guidance_scale),
        method=str(method), manual_duration_s=manual_duration_s, speed=float(speed),
        n_iter_gl=int(n_iter_gl), seed=int(seed),
    )
    return render_synthesis_output(syn, int(mel_cfg["sample_rate"]), mo)


@app.cell
def _(
    device,
    load_cfg_ui,
    load_gl_ui,
    load_manual_dur_ui,
    load_method_ui,
    load_sample_btn,
    load_seed_ui,
    load_speaker_ui,
    load_speed_ui,
    load_steps_ui,
    load_text_ui,
    loaded_model,
    loaded_sidecar,
    mo,
):
    loaded_sampling_view(
        loaded_model=loaded_model, loaded_sidecar=loaded_sidecar,
        text=str(load_text_ui.value),
        speaker_id=int(load_speaker_ui.value),
        num_steps=int(load_steps_ui.value),
        guidance_scale=float(load_cfg_ui.value),
        method=str(load_method_ui.value),
        manual_duration_s=(float(load_manual_dur_ui.value) if float(load_manual_dur_ui.value) > 0.0 else None),
        speed=float(load_speed_ui.value),
        n_iter_gl=int(load_gl_ui.value),
        seed=int(load_seed_ui.value),
        device=device,
        is_clicked=bool(load_sample_btn.value),
        mo=mo,
    )
    return


if __name__ == "__main__":
    app.run()
