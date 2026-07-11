# torchaudio.pipelines — the Bundle Abstraction

A pretrained model alone isn't enough — you also need the exact feature
extraction and post-processing used during its training (sample rate, FFT
size, tokenizer, vocoder, ...). `torchaudio.pipelines` packages a pretrained
model with its matching pipeline components into a **Bundle**.

Different Bundle *types* share a common shape for a given task, but wrap
different underlying architectures. For instance `SourceSeparationBundle`
instances `CONVTASNET_BASE_LIBRI2MIX` (a ConvTasNet model) and
`HDEMUCS_HIGH_MUSDB` (an HDemucs model) expose the *same* interface despite
different implementations underneath.

```python
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)   # downloads + caches weights on first use
print(bundle.sample_rate)                # the rate this bundle expects, e.g. 16000
```

## RNN-T Streaming/Non-Streaming ASR

`RNNTBundle` — feature extraction, inference, and de-tokenization for RNN-T
ASR models. Accessors: `.FeatureExtractor`, `.TokenProcessor`.

- **`EMFORMER_RNNT_BASE_LIBRISPEECH`** — Emformer-RNNT trained on LibriSpeech;
  supports both streaming and non-streaming inference.

## wav2vec 2.0 / HuBERT / WavLM — self-supervised (SSL) feature extraction

`Wav2Vec2Bundle` — instantiates models producing acoustic features for
downstream inference/fine-tuning (not directly a transcript).

Pretrained models (all via `bundle.get_model()`):
- **wav2vec 2.0:** `WAV2VEC2_BASE`, `WAV2VEC2_LARGE`, `WAV2VEC2_LARGE_LV60K`,
  `WAV2VEC2_XLSR53` (multilingual), `WAV2VEC2_XLSR_300M` / `_1B` / `_2B`
  (XLS-R, 128 languages, increasing size).
- **HuBERT:** `HUBERT_BASE`, `HUBERT_LARGE`, `HUBERT_XLARGE`.
- **WavLM:** `WAVLM_BASE`, `WAVLM_BASE_PLUS`, `WAVLM_LARGE`.

None of these are fine-tuned for a task — use them for feature extraction or
as a base for your own fine-tuning. See
[WAV2VEC2-AND-ASR.md](WAV2VEC2-AND-ASR.md) for `model.extract_features(...)`.

## wav2vec 2.0 / HuBERT — fine-tuned ASR

`Wav2Vec2ASRBundle` — instantiates models producing a probability
distribution over output labels (a transcript, via CTC decoding).

Pretrained models: `WAV2VEC2_ASR_BASE_10M` / `_100H` / `_960H` and
`WAV2VEC2_ASR_LARGE_10M` / `_100H` / `_960H` and
`WAV2VEC2_ASR_LARGE_LV60K_10M` / `_100H` / `_960H` (suffix = amount of
labeled fine-tuning data); `VOXPOPULI_ASR_BASE_10K_{DE,EN,ES,FR,IT}`
(per-language VoxPopuli fine-tunes); `HUBERT_ASR_LARGE`, `HUBERT_ASR_XLARGE`.

```python
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
print(bundle.get_labels())   # output label/character set, e.g. ('-', '|', 'E', 'T', ...)
```

See [WAV2VEC2-AND-ASR.md](WAV2VEC2-AND-ASR.md) for the full decode-to-text
workflow.

## wav2vec 2.0 / HuBERT — forced alignment

`Wav2Vec2FABundle` — bundles a pretrained acoustic model with a matching
tokenizer, and supports appending a `<star>` token dimension for
robustness to insertions/unknown segments.

- **`MMS_FA`** — trained on 31K hours across 1,100+ languages (from
  *Scaling Speech Technology to 1,000+ Languages*). The go-to bundle for
  multilingual forced alignment.

```python
from torchaudio.pipelines import MMS_FA as bundle
model = bundle.get_model()          # pass with_star=False to disable <star> token dim
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
```

Full workflow: [WAV2VEC2-AND-ASR.md](WAV2VEC2-AND-ASR.md).

## Tacotron2 Text-to-Speech

`Tacotron2TTSBundle` — three stages: tokenization (`TextProcessor`),
spectrogram generation (Tacotron2), and vocoding (`Vocoder`: GriffinLim,
WaveRNN, or Waveglow).

Pretrained models (all trained on LJSpeech):
- `TACOTRON2_WAVERNN_PHONE_LJSPEECH` / `TACOTRON2_WAVERNN_CHAR_LJSPEECH` —
  phoneme- or character-based tokenization, WaveRNN vocoder.
- `TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH` / `TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH`
  — same, with GriffinLim (no learned vocoder weights) instead.

```python
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)
```

Full workflow: [TTS-SEPARATION-ENHANCEMENT.md](TTS-SEPARATION-ENHANCEMENT.md).

## Source Separation

`SourceSeparationBundle` — takes single-channel audio, returns multi-channel
(per-source) audio.

- **`CONVTASNET_BASE_LIBRI2MIX`** — ConvTasNet trained on Libri2Mix (speech
  separation, 2 speakers).
- **`HDEMUCS_HIGH_MUSDB_PLUS`** — Hybrid Demucs trained on MUSDB-HQ + extra
  internal Meta data (music: drums/bass/other/vocals).
- **`HDEMUCS_HIGH_MUSDB`** — Hybrid Demucs trained on MUSDB-HQ training set only.

Full workflow: [TTS-SEPARATION-ENHANCEMENT.md](TTS-SEPARATION-ENHANCEMENT.md).

## SQUIM — Speech Quality/Intelligibility Assessment (no reference audio needed for objective metrics)

- **`SquimObjectiveBundle` → `SQUIM_OBJECTIVE`** — predicts objective metrics
  (STOI, PESQ, SI-SDR) directly from a degraded waveform, no clean reference
  needed.
- **`SquimSubjectiveBundle` → `SQUIM_SUBJECTIVE`** — predicts a subjective
  Mean Opinion Score (MOS) given the degraded waveform plus a non-matching
  reference (any clean speech clip, not necessarily the same utterance).

Full workflow: [TTS-SEPARATION-ENHANCEMENT.md](TTS-SEPARATION-ENHANCEMENT.md).

## Note on internal dependencies

Bundle implementations may pull in components from `torchaudio.models`,
`torchaudio.transforms`, or third-party libraries (SentencePiece,
DeepPhonemizer) — this is abstracted away; you don't need those dependencies
installed unless a specific bundle's docs say so (e.g. phoneme-based
Tacotron2 bundles need DeepPhonemizer).
