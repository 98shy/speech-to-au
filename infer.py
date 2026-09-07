#!/usr/bin/env python3
"""End-to-end inference: raw audio (.wav) -> AU25/AU26 intensity curve.

Usage:
    python infer.py --wav path/to/speech.wav --out predictions.npz

Pipeline: audio_feature_extractor.extract_combined() reproduces the
GRID-SpeechAU-training-v1.1 "combined" (MFCC+WavLM) feature from a raw
16kHz waveform, then the AU25 specialist (Mamba+combined) and AU26
specialist (TCN+combined) checkpoints each predict their own AU only.

Scope: AU25 (lips part) and AU26 (jaw drop) ONLY -- the two lip-sync-relevant
AUs with validated predictive performance (test-set ICC(3,1) 0.70 / 0.69).
See README.md for why every other AU is deliberately excluded from this
package.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_feature_extractor as afe  # noqa: E402
from model import SpeechToAU, SpeechToAUMamba  # noqa: E402

CKPT_DIR = os.path.join(HERE, "checkpoints")
TARGET_FPS = 25

SPECIALISTS = {
    "AU25": {"model_name": "mamba", "ckpt_file": "mamba_combined.pt", "factory": SpeechToAUMamba},
    "AU26": {"model_name": "tcn", "ckpt_file": "tcn_combined.pt", "factory": SpeechToAU},
}


def load_specialist(au, device):
    spec = SPECIALISTS[au]
    ckpt = torch.load(os.path.join(CKPT_DIR, spec["ckpt_file"]), map_location=device, weights_only=False)
    num_aus = len(ckpt["target_aus"])
    model = spec["factory"](input_dim=ckpt["input_dim"], num_aus=num_aus).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    au_index = ckpt["target_aus"].index(au)
    return model, ckpt["feat_mean"], ckpt["feat_std"], au_index


def predict(wav_path, device=None):
    """Returns dict with 'timestamp_sec' [T], 'AU25' [T], 'AU26' [T]
    (clipped to the physically meaningful 0-5 intensity range)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats = afe.extract_combined(wav_path, device=device)  # [T, 781]
    T = feats.shape[0]
    timestamps = np.arange(T) / TARGET_FPS

    out = {"timestamp_sec": timestamps}
    for au in ("AU25", "AU26"):
        model, mean, std, au_index = load_specialist(au, device)
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
    print(f"wrote {args.out}: {len(result['timestamp_sec'])} frames at {TARGET_FPS} fps "
          f"({result['timestamp_sec'][-1]:.2f}s)")
    print(f"AU25 range: [{result['AU25'].min():.2f}, {result['AU25'].max():.2f}]")
    print(f"AU26 range: [{result['AU26'].min():.2f}, {result['AU26'].max():.2f}]")


if __name__ == "__main__":
    main()
