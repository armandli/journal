# I/O, Transforms, and Functional

## I/O — `torchaudio.load` / `torchaudio.save`

**Maintenance-phase note:** since 2.9, decoding/encoding is consolidated into
TorchCodec. `torchaudio.load()`/`torchaudio.save()` are now aliases for
`load_with_torchcodec()`/`save_with_torchcodec()`. The old `torchaudio.io`
module (`StreamReader`/`StreamWriter` for streaming decode) and
`torchaudio.info()`/`AudioMetaData` no longer appear in current docs — if you
need streaming decode or format metadata without loading the full file,
reach for TorchCodec's native API directly rather than assuming torchaudio
still wraps it.

```python
waveform, sample_rate = torchaudio.load("audio.wav")   # Tensor[channels, time], float32
torchaudio.save("out.wav", waveform, sample_rate)

# Equivalent, explicit form:
waveform, sample_rate = torchaudio.load_with_torchcodec("audio.wav")
torchaudio.save_with_torchcodec("out.wav", waveform, sample_rate)
```

## `torchaudio.transforms` — `nn.Module`-based processing

Transforms are `torch.nn.Module`s: instantiate once, move to device/dtype,
call repeatedly. Chain them with a custom `nn.Module` or `nn.Sequential`.

```python
class MyPipeline(torch.nn.Module):
    def __init__(self, input_freq=16000, resample_freq=8000, n_fft=1024,
                 n_mel=256, stretch_factor=0.8):
        super().__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spec = T.Spectrogram(n_fft=n_fft, power=2)
        self.spec_aug = torch.nn.Sequential(
            T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=80),
            T.TimeMasking(time_mask_param=80),
        )
        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform):
        resampled = self.resample(waveform)
        spec = self.spec(resampled)
        spec = self.spec_aug(spec)
        return self.mel_scale(spec)

pipeline = MyPipeline().to(device=torch.device("cuda"), dtype=torch.float32)
features = pipeline(waveform)
```

### Utility

`AmplitudeToDB`, `MuLawEncoding`, `MuLawDecoding`, `Resample`, `Fade`, `Vol`,
`Loudness` (ITU-R BS.1770-4), `AddNoise` (scale+add noise at a target SNR),
`Convolve` / `FFTConvolve` (direct vs. FFT-based convolution), `Speed`,
`SpeedPerturbation`, `Deemphasis`, `Preemphasis`.

```python
resampler = T.Resample(orig_freq=44100, new_freq=16000, dtype=waveform.dtype)
resampled = resampler(waveform)
```

### Feature extraction

`Spectrogram`, `InverseSpectrogram`, `MelScale`, `InverseMelScale`,
`MelSpectrogram`, `GriffinLim` (magnitude spectrogram → waveform),
`MFCC`, `LFCC`, `ComputeDeltas`, `PitchShift`, `SlidingWindowCmn`,
`SpectralCentroid`, `Vad` (voice activity detector).

```python
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate, n_fft=1024, win_length=None, hop_length=512,
    center=True, pad_mode="reflect", power=2.0, norm="slaney",
    n_mels=128, mel_scale="htk",
)
melspec = mel_spectrogram(waveform)

mfcc_transform = T.MFCC(
    sample_rate=sample_rate, n_mfcc=256,
    melkwargs={"n_fft": 2048, "n_mels": 256, "hop_length": 512, "mel_scale": "htk"},
)
mfcc = mfcc_transform(waveform)

lfcc_transform = T.LFCC(
    sample_rate=sample_rate, n_lfcc=256,
    speckwargs={"n_fft": 2048, "win_length": None, "hop_length": 512},
)
```

### Augmentations (SpecAugment)

`FrequencyMasking`, `TimeMasking`, `TimeStretch` — apply directly to a
spectrogram, typically chained via `nn.Sequential` as shown above.

### Loss

`RNNTLoss` — **deprecated**.

### Multi-channel (beamforming — see TTS-SEPARATION-ENHANCEMENT.md for full workflow)

`PSD` (cross-channel power spectral density), `MVDR`, `RTFMVDR`, `SoudenMVDR`
— Minimum Variance Distortionless Response beamforming variants.

## `torchaudio.functional` — stateless equivalents

Same operations as `transforms`, as plain functions (like `torch.nn` vs
`torch.nn.functional`). Notable groups:

- **Utility:** `amplitude_to_DB`, `DB_to_amplitude`, `melscale_fbanks`,
  `linear_fbanks`, `create_dct`, `mask_along_axis`, `mask_along_axis_iid`,
  `mu_law_encoding`, `mu_law_decoding`, `resample`, `loudness`, `convolve`,
  `fftconvolve`, `add_noise`, `preemphasis`, `deemphasis`, `speed`,
  `frechet_distance`.
- **Forced alignment:** `forced_align` (**deprecated**, still used internally
  by `pipelines.Wav2Vec2FABundle`), `merge_tokens`, `TokenSpan`.
- **Filtering (audio effects):** `allpass_biquad`, `band_biquad`,
  `bandpass_biquad`, `bandreject_biquad`, `bass_biquad`, `biquad`,
  `contrast`, `dcshift`, `deemph_biquad`, `dither`, `equalizer_biquad`,
  `filtfilt`, `flanger`, `gain`, `highpass_biquad`, `lfilter`,
  `lowpass_biquad`, `overdrive`, `phaser`, `riaa_biquad`, `treble_biquad`.
- **Feature extraction:** `vad`, `spectrogram`, `inverse_spectrogram`,
  `griffinlim`, `phase_vocoder`, `pitch_shift`, `compute_deltas`,
  `detect_pitch_frequency`, `sliding_window_cmn`, `spectral_centroid`.
- **Multi-channel:** `psd`, `mvdr_weights_souden`, `mvdr_weights_rtf`,
  `rtf_evd`, `rtf_power`, `apply_beamforming`.
- **Loss:** `rnnt_loss` (**deprecated**).
- **Metric:** `edit_distance` (word-level Levenshtein distance).

```python
resampled = F.resample(waveform, orig_freq, new_freq, lowpass_filter_width=6, rolloff=0.99)
pitch = F.detect_pitch_frequency(waveform, sample_rate)
```

`lowpass_filter_width` and `rolloff` control the bandlimited-sinc resampling
quality/cost tradeoff — higher `lowpass_filter_width` (e.g. 128 vs. the
default 6) reduces aliasing artifacts at a higher compute cost.

## `torchaudio.compliance.kaldi`

Kaldi-compatible feature extraction, matching Kaldi's exact numerics for
projects migrating from or interoperating with Kaldi pipelines:

```python
from torchaudio.compliance import kaldi

spec = kaldi.spectrogram(waveform)
fbank = kaldi.fbank(waveform)
mfcc = kaldi.mfcc(waveform)
```

Use these instead of `torchaudio.transforms`/`functional` only when you
specifically need Kaldi-identical output (e.g. reproducing a Kaldi-trained
model's feature pipeline); otherwise prefer the native `transforms`/
`functional` APIs above.
