# wav2vec2 / HuBERT / WavLM — Feature Extraction, ASR, and Forced Alignment

## Feature extraction (SSL models — not fine-tuned for ASR)

```python
import torch
import torchaudio

bundle = torchaudio.pipelines.WAV2VEC2_BASE   # or HUBERT_BASE, WAVLM_BASE, ...
model = bundle.get_model().to(device)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
assert sample_rate == bundle.sample_rate   # resample first if this doesn't match

with torch.inference_mode():
    features, _ = model.extract_features(waveform)
    # `features` is a list of tensors, one per transformer layer
```

## ASR with a fine-tuned wav2vec2 bundle

```python
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
print("Sample Rate:", bundle.sample_rate)
print("Labels:", bundle.get_labels())
# ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', ...)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
with torch.inference_mode():
    emission, _ = model(waveform)   # logits, shape [batch, time, num_labels] — NOT probabilities
```

`emission` is a CTC-style logit sequence: `'-'` (blank) marks "repeat/no new
symbol" and `'|'` marks word boundary. Turning logits into text requires
**decoding** — the model doesn't do this for you.

### Greedy decoding (simplest, no external resources)

```python
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)          # [time]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])   # e.g. "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"
```

Greedy decoding picks the single best label per frame independently — fast,
but can't use surrounding context to disambiguate homophones (e.g.
"night"/"knight"). For that, use a proper beam-search decoder with a lexicon
and/or language model:

```python
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

files = download_pretrained_files("librispeech-4-gram")
decoder = ctc_decoder(
    lexicon=files.lexicon, tokens=files.tokens, lm=files.lm,
    nbest=1, beam_size=1500, lm_weight=3.23, word_score=-0.26,
)
results = decoder(emission)   # batched beam-search hypotheses
```

There's also `torchaudio.models.decoder.cuda_ctc_decoder` for GPU beam search
— **but it's deprecated**, so avoid it in new code without checking whether
it's still present in the installed version.

## Forced alignment (CTC)

Forced alignment answers "given audio + its exact known transcript, what are
the start/end times of each word/token?" — different from ASR, which
produces the transcript itself.

```python
import torchaudio.functional as F

DICTIONARY = {c: i for i, c in enumerate(LABELS) if c not in ("-", "|")}
tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)   # deprecated, still functional
    return alignments[0], scores[0].exp()

aligned_tokens, alignment_scores = align(emission, tokenized_transcript)
```

`torchaudio.functional.forced_align` is deprecated as part of torchaudio's
maintenance-phase refactor — expect a `UserWarning` at runtime. It still
works as of this writing and is what `Wav2Vec2FABundle`'s aligner uses
internally, but don't build new long-lived infrastructure assuming it stays.

### Preferred entry point: `Wav2Vec2FABundle` (handles tokenizing + aligning for you)

```python
from torchaudio.pipelines import MMS_FA as bundle

model = bundle.get_model().to(device)     # multilingual acoustic model, 1100+ languages
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

def compute_alignments(waveform, transcript_words):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript_words))
    return emission, token_spans   # token_spans: per-word list of TokenSpan(start, end, score)
```

`token_spans[i]` is a list of `TokenSpan` objects for word `i`; convert frame
indices to seconds via `ratio = waveform.size(1) / emission.size(1) / sample_rate`.

### Multilingual alignment — transcript normalization is on you

MMS_FA expects a **normalized** transcript (lowercase, romanized, stripped of
punctuation) — normalization is language-dependent and not automatic:

```python
# 1. Romanize with the external `uroman` tool (not part of torchaudio):
#    uroman/bin/uroman.pl < text.txt > text_romanized.txt

import re

def normalize_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()
```

Non-Latin-script languages (Chinese, Japanese, Korean) additionally need word
segmentation before romanization — segment first, then romanize/normalize
per the above.

```python
transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
```
