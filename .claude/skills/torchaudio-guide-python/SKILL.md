---
name: torchaudio-guide-python
description: Write and debug Python audio-processing code using torchaudio, PyTorch's companion library for audio. Covers built-in datasets, waveform I/O, transforms/functional (resampling, spectrograms, MFCC), pretrained pipelines (torchaudio.pipelines) for wav2vec2/HuBERT/WavLM ASR, forced alignment, Tacotron2 text-to-speech, Hybrid Demucs source separation, and MVDR/SQUIM speech enhancement. Use when the user asks to "use torchaudio", "load an audio dataset with torchaudio", "use wav2vec2", "do source separation", "text to speech with torchaudio", or "speech enhancement with torchaudio". Do NOT use for plain PyTorch core (use pytorch-guide-python) or non-audio torch domains.
argument-hint: "[task or description of what to implement]"
---

# torchaudio Python Guide

**Read this first — torchaudio is in maintenance phase.** As of version 2.9,
torchaudio's decoding/encoding has been consolidated into
[TorchCodec](https://github.com/pytorch/torchcodec), APIs deprecated in 2.8
were removed in 2.9, and no major new development is expected. Concretely:
- `torchaudio.io` (`StreamReader`/`StreamWriter`) no longer exists in current docs.
- `torchaudio.load()` / `torchaudio.save()` still work but are now thin aliases
  for `load_with_torchcodec()` / `save_with_torchcodec()` — new code should
  prefer calling TorchCodec directly.
- `torchaudio.functional.forced_align` and `rnnt_loss` are deprecated (slated
  for removal), though `torchaudio.pipelines` forced-alignment bundles still
  use them internally as of this writing.
- `torchaudio.models.decoder.cuda_ctc_decoder` is deprecated.

Everything else below (`datasets`, `transforms`, `functional`,
`pipelines`, `models`) is current and actively documented — just don't be
surprised if a project pinned to an older torchaudio has APIs (like
`torchaudio.info`/`AudioMetaData`, `StreamReader`) that current docs no
longer mention. Check the installed version before assuming an old snippet
still applies.

## Imports

```python
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
```

## Loading and saving audio

```python
waveform, sample_rate = torchaudio.load("path/to/audio.wav")   # -> Tensor[channels, time], float32 in [-1, 1]
torchaudio.save("out.wav", waveform, sample_rate)
```

`load`/`save` call TorchCodec's `AudioDecoder`/`AudioEncoder` under the hood.

## Building a processing pipeline (transforms are `nn.Module`s)

```python
class MyPipeline(torch.nn.Module):
    def __init__(self, input_freq=16000, resample_freq=8000, n_fft=1024, n_mel=256):
        super().__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = T.Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        spec = self.spec(resampled)
        return self.mel_scale(spec)

pipeline = MyPipeline().to(device="cuda", dtype=torch.float32)
features = pipeline(waveform)
```

Chain `transforms` classes with `torch.nn.Sequential` (e.g. `FrequencyMasking`
+ `TimeMasking` for SpecAugment), or use the stateless equivalents in
`torchaudio.functional` directly. Full transform/functional catalogue:
[references/IO-TRANSFORMS-FUNCTIONAL.md](references/IO-TRANSFORMS-FUNCTIONAL.md).

## The Bundle pattern — pretrained pipelines

Every pretrained capability (ASR, SSL features, TTS, source separation,
speech-quality assessment) is exposed through `torchaudio.pipelines` as a
**Bundle**: an object that packages a pretrained model with the exact
feature-extraction/post-processing it needs.

```python
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)          # instantiate + download weights
sample_rate = bundle.sample_rate                # the rate the model expects
```

Different bundle *types* expose different accessor methods
(`get_model()`, `get_text_processor()`, `get_vocoder()`, `get_tokenizer()`,
`get_aligner()`, ...) — see
[references/PIPELINES.md](references/PIPELINES.md) for the full bundle
catalogue (RNN-T ASR, wav2vec2/HuBERT/WavLM SSL, wav2vec2 ASR, wav2vec2
forced alignment, Tacotron2 TTS, source separation, SQUIM).

## Task quick-reference

| Task | Bundle / API | Details |
|---|---|---|
| ASR / feature extraction with wav2vec2, HuBERT, WavLM | `torchaudio.pipelines.WAV2VEC2_*`, `HUBERT_*`, `WAVLM_*` | [references/WAV2VEC2-AND-ASR.md](references/WAV2VEC2-AND-ASR.md) |
| Forced alignment (incl. multilingual) | `torchaudio.pipelines.MMS_FA` (`Wav2Vec2FABundle`) | [references/WAV2VEC2-AND-ASR.md](references/WAV2VEC2-AND-ASR.md) |
| Text-to-speech | `torchaudio.pipelines.TACOTRON2_*` | [references/TTS-SEPARATION-ENHANCEMENT.md](references/TTS-SEPARATION-ENHANCEMENT.md) |
| Music/speech source separation | `torchaudio.pipelines.HDEMUCS_*`, `CONVTASNET_BASE_LIBRI2MIX` | [references/TTS-SEPARATION-ENHANCEMENT.md](references/TTS-SEPARATION-ENHANCEMENT.md) |
| Speech enhancement (beamforming) | `torchaudio.transforms.{PSD,SoudenMVDR,RTFMVDR}` | [references/TTS-SEPARATION-ENHANCEMENT.md](references/TTS-SEPARATION-ENHANCEMENT.md) |
| Speech quality assessment (no reference needed) | `torchaudio.pipelines.SQUIM_OBJECTIVE` / `SQUIM_SUBJECTIVE` | [references/TTS-SEPARATION-ENHANCEMENT.md](references/TTS-SEPARATION-ENHANCEMENT.md) |
| Built-in datasets | `torchaudio.datasets.*` | [references/DATASETS.md](references/DATASETS.md) |

## Testing

After wiring a pipeline, run it end-to-end on one real audio file and inspect
the actual output (waveform shape/sample rate, transcript text, or a saved
`.wav`) rather than trusting shapes alone — sample-rate mismatches between a
bundle's expected `sample_rate` and your input audio are the most common
silent failure (resample explicitly with `torchaudio.functional.resample` or
`T.Resample` if they don't match).

## References

- [references/DATASETS.md](references/DATASETS.md) — full `torchaudio.datasets` catalogue, `DataLoader` usage
- [references/IO-TRANSFORMS-FUNCTIONAL.md](references/IO-TRANSFORMS-FUNCTIONAL.md) — load/save, full `transforms`/`functional` reference, `compliance.kaldi`
- [references/PIPELINES.md](references/PIPELINES.md) — the Bundle abstraction and every pretrained bundle category/model name
- [references/WAV2VEC2-AND-ASR.md](references/WAV2VEC2-AND-ASR.md) — wav2vec2/HuBERT/WavLM feature extraction, ASR decoding, CTC forced alignment (incl. multilingual)
- [references/TTS-SEPARATION-ENHANCEMENT.md](references/TTS-SEPARATION-ENHANCEMENT.md) — Tacotron2 TTS, Hybrid Demucs source separation, MVDR beamforming, SQUIM speech-quality assessment

## External Docs

- Full docs: https://docs.pytorch.org/audio/stable/index.html
- Maintenance-phase / TorchCodec migration notice: https://github.com/pytorch/audio/issues/3902

---

### Final Step — Record Usage

```bash
python3 ${PWD}/.claude/skills/skill-stat/scripts/record-stat.py "torchaudio-guide-python"
```
