#!/usr/bin/env python3
"""End-to-end inference: raw audio (.wav) -> 7-AU intensity curve.

Usage:
    python infer.py --wav path/to/speech.wav --out predictions.npz

Pipeline: audio_feature_extractor.extract_wavlm() produces a single WavLM
Base+ feature stream from the raw waveform (no MFCC needed -- every
specialist in this package uses wavlm-only features by design, see
README.md). That one feature stream is fed to 4 specialist checkpoints (2
trained on GRID-SpeechAU-training-v1.1, 2 trained on CREMAD-SpeechAU-v1),
each of which was trained to jointly predict all 9 AUs -- but only a
specific 1-2 AU output channel is kept from each, per the routing table in
SPECIALISTS below. Every model's other output channels are computed but
discarded; this "route and discard" design was chosen deliberately over
retraining slimmer single-AU models, so that every reported test-set metric
stays backed by the exact same checkpoint being shipped (see project notes
on why: retraining narrower heads would have invalidated all prior
benchmarking for uncertain gain).

**Critical correctness point**: each checkpoint carries its OWN mean/std
(computed from that specialist's own training-set WavLM features) --
never share normalization stats across specialists. This is handled
automatically by load_specialist() below; do not bypass it.

Scope: AU10, AU12, AU14, AU17, AU23, AU25, AU26 (7 of the original 9-AU
target set). AU15 and AU20 are deliberately excluded -- both stayed at
ICC(3,1) ~0-0.15 across every dataset/model/feature combination tried
(GRID, IEMOCAP, HDTF, CREMA-D, RAVDESS), i.e. statistically indistinguishable
from not predicting anything. See README.md for full per-AU picking
rationale (PCC 1st, ICC 2nd, MAE/MSE 3rd priority).
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_feature_extractor as afe  # noqa: E402
from model import SpeechToAU, SpeechToAUMamba, CauseRNN  # noqa: E402

CKPT_DIR = os.path.join(HERE, "checkpoints")
TARGET_FPS = 25

# Each entry: which checkpoint file to load, which model class built it, and
# which AU(s) to keep from that checkpoint's 9-AU output. A checkpoint may
# be listed more than once (GRID mamba covers both AU10 and AU25).
SPECIALISTS = {
    "AU10": {"ckpt_file": "grid_mamba_wavlm.pt", "factory": SpeechToAUMamba},
    "AU25": {"ckpt_file": "grid_mamba_wavlm.pt", "factory": SpeechToAUMamba},
    "AU26": {"ckpt_file": "grid_tcn_wavlm.pt", "factory": SpeechToAU},
    "AU12": {"ckpt_file": "cremad_mamba_wavlm.pt", "factory": SpeechToAUMamba},
    "AU14": {"ckpt_file": "cremad_mamba_wavlm.pt", "factory": SpeechToAUMamba},
    "AU17": {"ckpt_file": "cremad_cause_rnn_wavlm.pt", "factory": CauseRNN},
    "AU23": {"ckpt_file": "cremad_cause_rnn_wavlm.pt", "factory": CauseRNN},
}

_CKPT_CACHE = {}


def _load_checkpoint(ckpt_file, factory, device):
    """Loads (and caches) one checkpoint -- each is loaded at most once even
    though multiple AUs may point at the same file."""
    key = (ckpt_file, device)
    if key in _CKPT_CACHE:
        return _CKPT_CACHE[key]

    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_file), map_location=device, weights_only=False)
    model = factory(input_dim=ckpt["input_dim"], num_aus=len(ckpt["target_aus"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    entry = (model, ckpt["mean"], ckpt["std"], ckpt["target_aus"])
    _CKPT_CACHE[key] = entry
    return entry


def predict(wav_path, device=None):
    """Returns dict with 'timestamp_sec' [T] plus one float32[T] array
    (clipped to the physically meaningful 0-5 OpenFace intensity range) per
    AU in SPECIALISTS."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats = afe.extract_wavlm(wav_path, device=device)  # [T, 768]
    T = feats.shape[0]
    timestamps = np.arange(T) / TARGET_FPS

    out = {"timestamp_sec": timestamps}
    for au, spec in SPECIALISTS.items():
        model, mean, std, target_aus = _load_checkpoint(spec["ckpt_file"], spec["factory"], device)
        au_index = target_aus.index(au)

        feats_norm = (feats - mean) / std
        x = torch.from_numpy(feats_norm.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).cpu().numpy()[:, au_index]
        out[au] = np.clip(pred, 0, 5).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a 16kHz (or resamplable) mono/stereo wav file")
    parser.add_argument("--out", default="predictions.npz", help="Output .npz path")
    args = parser.parse_args()

    result = predict(args.wav)
    np.savez(args.out, **result)
    aus = [k for k in result if k != "timestamp_sec"]
    print(f"wrote {args.out}: {len(result['timestamp_sec'])} frames at {TARGET_FPS} fps "
          f"({result['timestamp_sec'][-1]:.2f}s)")
    for au in aus:
        print(f"{au} range: [{result[au].min():.2f}, {result[au].max():.2f}]")


if __name__ == "__main__":
    main()
