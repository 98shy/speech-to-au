# lipsync_7au_deploy

Self-contained speech -> facial Action Unit (AU) intensity inference package.
Successor to `lipsync_au25_26_deploy` (AU25/AU26 only) -- this package adds
AU10, AU12, AU14, AU17, AU23, bringing coverage to 7 of the 9 AUs originally
tried in this project.

## Quick start

```bash
pip install -r requirements.txt
python infer.py --wav path/to/speech.wav --out predictions.npz
```

```python
import infer
result = infer.predict("path/to/speech.wav")
# result: {"timestamp_sec": [...], "AU10": [...], "AU12": [...], "AU14": [...],
#          "AU17": [...], "AU23": [...], "AU25": [...], "AU26": [...]}
```

Each AU array is float32, clipped to OpenFace's [0, 5] intensity scale, one
value per 25fps frame.

## What's inside

- `audio_feature_extractor.py` -- raw 16kHz wav -> WavLM Base+ features
  (768-d, resampled to 25fps). This is the ONLY feature type used by every
  specialist in this package (see "Why WavLM-only" below). The module also
  still contains MFCC/combined extraction code inherited from the previous
  package version, but `infer.py` does not call it.
- `model/` -- the 3 architectures needed: TCN, Mamba (Selective SSM), and
  CauseRNN (CAUSE BiLSTM variant, Kameoka et al. Interspeech 2022).
- `checkpoints/` -- 4 files. Each predicts all 9 of its source dataset's
  target AUs internally; `infer.py` keeps only the AU(s) assigned to it (see
  table below) and discards the rest.

## Why WavLM-only (not MFCC+WavLM "combined")

Two reasons, checked empirically before deciding:
1. On GRID, AU25/AU26 performance barely changes between wavlm-only and
   combined (PCC gap <=0.013 in every model tried).
2. On CREMA-D, most AUs performed as well or *better* on wavlm alone than on
   combined.

Fixing the whole package to one feature extractor also simplifies real-time
inference: a single WavLM pass feeds all 4 specialists, no MFCC extraction
needed at all.

## Per-AU specialist routing

Every AU in this package is predicted by whichever (dataset, model)
combination had the best test-set metrics for that specific AU, compared
across 5 datasets (GRID-SpeechAU-training-v1.1, IEMOCAP-SpeechAU-v2,
HDTF-SpeechAU-v2, CREMAD-SpeechAU-v1, RAVDESS-SpeechAU-v1) x up to 7 model
architectures x up to 3 features (wavlm-only comparisons decided the final
pick; see `config.json`'s `selection_method` for the full priority order:
PCC 1st, ICC(3,1) 2nd, MAE/MSE 3rd as a tiebreaker only).

| AU | Source dataset | Model | Checkpoint | PCC | ICC(3,1) |
|---|---|---|---|---|---|
| AU10 | GRID | Mamba | `grid_mamba_wavlm.pt` | 0.452 | 0.442 |
| AU12 | CREMA-D | Mamba | `cremad_mamba_wavlm.pt` | 0.543 | 0.424 |
| AU14 | CREMA-D | Mamba | `cremad_mamba_wavlm.pt` | 0.471 | 0.366 |
| AU17 | CREMA-D | CauseRNN | `cremad_cause_rnn_wavlm.pt` | 0.586 | 0.502 |
| AU23 | CREMA-D | CauseRNN | `cremad_cause_rnn_wavlm.pt` | 0.493 | 0.335 |
| AU25 | GRID | Mamba | `grid_mamba_wavlm.pt` | 0.696 | 0.696 |
| AU26 | GRID | TCN | `grid_tcn_wavlm.pt` | 0.681 | 0.664 |

Note `grid_mamba_wavlm.pt` is loaded once and reused for both AU10 and AU25
(same checkpoint, two output channels kept); `cremad_mamba_wavlm.pt` is
likewise shared by AU12/AU14, and `cremad_cause_rnn_wavlm.pt` by AU17/AU23.
`infer.py` caches each checkpoint so it's only loaded/run once regardless of
how many AUs point at it.

**Each checkpoint stores its own mean/std** (computed from that specialist's
own training-set WavLM features) alongside its weights. `infer.py` always
normalizes with the checkpoint's own stats -- never share normalization
across specialists, since they were fit on different training-data
distributions.

## Why AU15 and AU20 are excluded

Both stayed at ICC(3,1) ~0-0.15 across every one of the ~30-100+ (dataset,
model, feature) combinations tried across all 5 datasets -- i.e.
statistically indistinguishable from not predicting anything. Shipping them
would be actively misleading for a rendering pipeline. Do not request them
from this package.

## Known limitations (read before using for anything load-bearing)

- **Domain mismatch risk (biggest open question, now quantified)**:
  AU10/12/14/17/23 are trained on CREMA-D -- short (~2.5s),
  professionally-acted, deliberately exaggerated emotional speech.
  AU10/25/26 are trained on GRID -- short (~3s), clean, single-speaker read
  speech. Neither matches natural conversational speech (nor whatever the
  eventual deployment domain's actual input distribution looks like). A
  cross-domain sanity check ran every checkpoint on the 4 datasets it was
  NOT trained on (full per-AU/per-dataset numbers in
  `cross_domain_sanity_check_results.csv`, this directory; the generating
  script lives in the main project repo, not shipped here since it needs
  the full multi-dataset training data to run) -- performance drops
  substantially outside the native domain, confirming this risk is real,
  not just theoretical:

  | Checkpoint | AU | Native PCC | Best cross-domain PCC | Worst cross-domain PCC |
  |---|---|---|---|---|
  | GRID/mamba | AU10 | 0.452 | 0.183 (RAVDESS) | **-0.037 (IEMOCAP)** |
  | GRID/mamba | AU25 | 0.696 | 0.490 (CREMA-D) | 0.204 (IEMOCAP) |
  | GRID/tcn | AU26 | 0.681 | 0.516 (CREMA-D) | 0.169 (IEMOCAP) |
  | CREMA-D/mamba | AU12 | 0.543 | 0.325 (IEMOCAP) | 0.149 (GRID) |
  | CREMA-D/mamba | AU14 | 0.471 | 0.260 (HDTF) | **-0.012 (GRID)** |
  | CREMA-D/CauseRNN | AU17 | 0.586 | 0.245 (HDTF) | 0.074 (IEMOCAP) |
  | CREMA-D/CauseRNN | AU23 | 0.493 | 0.248 (GRID) | 0.040 (RAVDESS) |

  Two patterns worth noting: (1) GRID- and CREMA-D-trained specialists
  transfer to *each other* better than to IEMOCAP or RAVDESS -- both are
  solo-speaker, fixed-frontal-camera, scripted-utterance recordings, so they
  are closer to each other than to IEMOCAP's natural dyadic conversation
  (the most different domain along nearly every axis: real dialogue,
  natural head movement, free-form content). (2) IEMOCAP is consistently
  the worst or near-worst transfer target across every checkpoint -- if the
  eventual deployment input looks more like natural conversation than
  scripted/acted delivery, expect performance closer to these cross-domain
  numbers than to the native ones reported above. Re-run
  `cross_domain_sanity_check.py` against real target-domain audio as soon as
  it's available.
- **Selection-bias caveat**: every specialist above was picked by comparing
  test metrics across a large number of candidates. The reported PCC/ICC are
  likely mildly optimistic versus true generalization performance for that
  reason. Recommend an independent validation pass (fresh data, not reused
  from the comparison pool) before treating these numbers as final.
- AU labels across all 5 source datasets are OpenFace 2.2.0 pseudo-labels,
  not expert-annotated FACS ground truth.
- Output is intensity only (0-5 continuous scale), not binary presence.
- No coverage of upper-face AUs (brow/eye/cheek) -- not present in these
  datasets' primary target sets.
