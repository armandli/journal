# torchaudio.datasets

All datasets are `torch.utils.data.Dataset` subclasses (`__getitem__` +
`__len__`), so they drop straight into a `DataLoader`:

```python
import torchaudio
import torch

yesno_data = torchaudio.datasets.YESNO(".", download=True)
data_loader = torch.utils.data.DataLoader(
    yesno_data, batch_size=1, shuffle=True, num_workers=4
)
```

## Available datasets

| Dataset | Purpose |
|---|---|
| `CMUARCTIC` | CMU ARCTIC speech synthesis corpus. |
| `CMUDict` | CMU Pronouncing Dictionary. |
| `COMMONVOICE` | Mozilla CommonVoice multilingual speech. |
| `DR_VCTK` | Device Recorded VCTK (small subset). |
| `FluentSpeechCommands` | Spoken language understanding commands. |
| `GTZAN` | Music genre classification. |
| `IEMOCAP` | Emotion recognition (requires manual download/license). |
| `LibriMix` | Multi-speaker mixtures for source separation, built from LibriSpeech. |
| `LIBRISPEECH` | The standard LibriSpeech ASR corpus. |
| `LibriLightLimited` | Libri-Light subset used for HuBERT supervised fine-tuning. |
| `LIBRITTS` | Multi-speaker TTS corpus derived from LibriSpeech. |
| `LJSPEECH` | Single-speaker TTS corpus (used to train torchaudio's Tacotron2/WaveRNN bundles). |
| `MUSDB_HQ` | Music source separation (vocals/drums/bass/other stems) — used to train Hybrid Demucs. |
| `QUESST14` | Query-by-example spoken term detection. |
| `Snips` | Spoken language understanding (Snips voice assistant commands). |
| `SPEECHCOMMANDS` | Keyword spotting (short spoken commands). |
| `TEDLIUM` | TED talk transcription corpus (releases 1, 2, 3). |
| `VCTK_092` | Multi-speaker English corpus, v0.92. |
| `VoxCeleb1Identification` / `VoxCeleb1Verification` | Speaker ID / speaker verification tasks. |
| `YESNO` | Tiny toy dataset (yes/no in Hebrew) — good for smoke-testing a pipeline end-to-end. |

## Usage pattern

```python
from torchaudio.datasets import LIBRISPEECH

dataset = LIBRISPEECH(root="./data", url="train-clean-100", download=True)
waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[0]
```

Return signatures differ per dataset (check the specific dataset's
`__getitem__` docstring), but audio-plus-metadata tuples like the above are
the common shape. For a quick pipeline smoke test without downloading a large
corpus, start with `YESNO` — it's tiny and downloads fast.
