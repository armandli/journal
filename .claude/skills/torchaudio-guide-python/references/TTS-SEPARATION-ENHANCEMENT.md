# Text-to-Speech, Source Separation, and Speech Enhancement

## Text-to-Speech with Tacotron2

A TTS pipeline has three stages: text → tokens (`TextProcessor`), tokens →
spectrogram (Tacotron2), spectrogram → waveform (`Vocoder`).

```python
import torch
import torchaudio

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

text = "Hello world! Text to speech!"
with torch.inference_mode():
    processed, lengths = processor(text)          # tokenize
    processed, lengths = processed.to(device), lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)   # spectrogram
    waveforms, out_lengths = vocoder(spec, spec_lengths)          # vocode -> waveform

torchaudio.save("output.wav", waveforms[0:1].cpu(), vocoder.sample_rate)
```

Swap `get_vocoder()`'s underlying bundle to change vocoder quality/speed
tradeoff:
- `TACOTRON2_WAVERNN_CHAR_LJSPEECH` / `TACOTRON2_WAVERNN_PHONE_LJSPEECH` —
  WaveRNN vocoder (learned, higher quality).
- `TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH` / `TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH`
  — GriffinLim vocoder (no learned weights, faster to set up, lower quality).

Phoneme-based (`_PHONE_`) bundles require the `DeepPhonemizer` package for
grapheme-to-phoneme conversion; character-based (`_CHAR_`) bundles don't.

`processor.tokens` lets you inspect the intermediate token sequence (useful
for debugging why a particular input produced odd output — unsupported
characters are silently dropped, not errored on).

## Music/Speech Source Separation with Hybrid Demucs

`SourceSeparationBundle`s take single-channel (or stereo) mixed audio and
return one waveform per source (e.g. drums/bass/other/vocals for music).

```python
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().to(device)
sample_rate = bundle.sample_rate
```

Hybrid Demucs is memory-hungry, so full songs are processed in overlapping
chunks with cross-fades to avoid edge artifacts, then stitched back together:

```python
def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
    """Apply model to `mix` in overlapping chunks of `segment` seconds."""
    device = device or mix.device
    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start, end = 0, chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

waveform, sr = torchaudio.load("song.wav")
waveform = waveform.to(device)

# Normalize before separating, denormalize after — Hybrid Demucs expects
# roughly zero-mean, unit-variance input.
ref = waveform.mean(0)
normalized = (waveform - ref.mean()) / ref.std()
sources = separate_sources(model, normalized[None], device=device)[0]
sources = sources * ref.std() + ref.mean()

sources_dict = dict(zip(model.sources, list(sources)))   # {"drums": ..., "bass": ..., "other": ..., "vocals": ...}
```

For 2-speaker speech separation instead of music stems, use
`CONVTASNET_BASE_LIBRI2MIX` — same `bundle.get_model()` /
`bundle.sample_rate` interface, simpler model (no chunking usually needed for
short utterances).

## Speech Enhancement with MVDR Beamforming (multi-channel)

MVDR beamforming enhances speech using **multiple microphone channels** plus
a time-frequency mask (or estimate) of where speech vs. noise dominates.
Pipeline: STFT → per-source PSD matrices → MVDR weights → enhanced STFT →
inverse STFT.

```python
import torchaudio.functional as F
import torchaudio.transforms as T

REFERENCE_CHANNEL = 0
stft = T.Spectrogram(n_fft=1024, hop_length=256, power=None)   # power=None -> complex STFT
istft = T.InverseSpectrogram(n_fft=1024, hop_length=256)

stft_mix = stft(waveform_mix)          # multi-channel complex STFT: [channel, freq, time]

# Ideal ratio masks (if you have separate clean/noise references, e.g. for eval);
# in production you'd estimate these masks with a separation/VAD model instead.
def get_irms(stft_clean, stft_noise):
    mag_clean, mag_noise = stft_clean.abs() ** 2, stft_noise.abs() ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise)
    irm_noise = mag_noise / (mag_clean + mag_noise)
    return irm_speech[REFERENCE_CHANNEL], irm_noise[REFERENCE_CHANNEL]

irm_speech, irm_noise = get_irms(stft_clean, stft_noise)

psd_transform = T.PSD()
psd_speech = psd_transform(stft_mix, irm_speech)
psd_noise = psd_transform(stft_mix, irm_noise)

# Option A: SoudenMVDR — directly from PSD matrices
mvdr_transform = T.SoudenMVDR()
stft_enhanced = mvdr_transform(stft_mix, psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL)
waveform_enhanced = istft(stft_enhanced, length=waveform_mix.shape[-1])

# Option B: RTFMVDR — via a relative transfer function (RTF) estimate
rtf = F.rtf_evd(psd_speech)                                   # eigenvalue-decomposition estimate
# or: rtf = F.rtf_power(psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL)
mvdr_transform = T.RTFMVDR()
stft_enhanced = mvdr_transform(stft_mix, rtf, psd_noise, reference_channel=REFERENCE_CHANNEL)
waveform_enhanced = istft(stft_enhanced, length=waveform_mix.shape[-1])
```

`rtf_evd` (eigenvalue decomposition) is simpler; `rtf_power` (power
iteration method, `n_iter=` controls iterations) can be more robust — try
both if beamforming quality is unsatisfactory.

## Speech Quality Assessment with SQUIM (no clean reference needed)

Useful for estimating intelligibility/quality metrics **without** access to
clean ground-truth audio (e.g. scoring real-world recordings where no clean
reference exists).

```python
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

# Objective: predicts STOI, PESQ, SI-SDR directly from the degraded waveform
objective_model = SQUIM_OBJECTIVE.get_model()
stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(waveform_distorted[0:1, :])

# Subjective: predicts Mean Opinion Score (MOS), given the degraded waveform
# plus ANY non-matching clean reference clip (doesn't need to be the same utterance)
subjective_model = SQUIM_SUBJECTIVE.get_model()
mos = subjective_model(waveform_distorted[0:1, :], waveform_non_matching_reference)
```

This is the natural tool for evaluating output quality after source
separation or MVDR enhancement, when you don't have a clean reference for
the specific recording.
