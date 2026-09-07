"""Standalone feature extractor for NEW audio (not from the GRID corpus),
reproducing GRID-SpeechAU-training-v1.1's exact feature spec
(config/training/speech_feature_benchmark_v2.yaml) so audio recorded outside
this project can be fed to the trained predictors. Everything the shipped
dataset only pre-computed for its own 32,752 utterances is recomputed here
from a raw waveform.

Spec (must match the dataset's config exactly, or predictions are invalid):
  sample_rate = 16000, target_fps = 25
  mfcc:   n_mfcc=13, n_fft=400, win_length=400, hop_length=640, center=True,
          pad_mode='constant' -- no deltas. hop_length=640 samples @16kHz is
          exactly 40ms = 1/25fps, so MFCC's native frame rate already equals
          the target video timeline (still explicitly resampled below for
          robustness against librosa's own edge-frame rounding).
  wavlm:  microsoft/wavlm-base-plus, revision 4c66d4806a428f2e922ccfa1a962776e232d487b,
          layer=final hidden state (NOT averaged across layers), frozen,
          linearly interpolated from its native ~20ms hop to the 25fps
          video timeline.
  combined: mfcc and wavlm concatenated along the feature axis after both
          are resampled to the same 25fps timeline, truncated to
          T = min(len(mfcc), len(wavlm)) (never padded/extended).
"""

import numpy as np
import torch

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
SAMPLE_RATE = 16000
TARGET_FPS = 25

_MODEL_CACHE = {}


def load_audio(path, sr=SAMPLE_RATE):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def resample_to_fps(feats, src_hop_seconds, target_fps):
    """Linear interpolation from a native frame rate to a fixed target_fps."""
    n_src = feats.shape[0]
    src_times = np.arange(n_src) * src_hop_seconds
    duration = src_times[-1] if n_src > 1 else 0.0
    n_tgt = max(1, int(round(duration * target_fps)) + 1)
    tgt_times = np.arange(n_tgt) / target_fps

    out = np.empty((n_tgt, feats.shape[1]), dtype=np.float32)
    for d in range(feats.shape[1]):
        out[:, d] = np.interp(tgt_times, src_times, feats[:, d])
    return out


def extract_mfcc(wav_path_or_array, sr=SAMPLE_RATE):
    """Returns [T, 13] float32, resampled to TARGET_FPS."""
    import librosa
    wav = load_audio(wav_path_or_array, sr) if isinstance(wav_path_or_array, str) else wav_path_or_array

    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr, n_mfcc=13, n_fft=400, win_length=400, hop_length=640,
        center=True, pad_mode="constant",
    )
    feats = mfcc.T.astype(np.float32)  # [T, 13]
    return resample_to_fps(feats, 640 / sr, TARGET_FPS)


def _get_wavlm(device):
    key = (WAVLM_MODEL_ID, WAVLM_REVISION, device)
    if key not in _MODEL_CACHE:
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_ID, revision=WAVLM_REVISION)
        model = WavLMModel.from_pretrained(WAVLM_MODEL_ID, revision=WAVLM_REVISION).to(device).eval()
        _MODEL_CACHE[key] = (extractor, model)
    return _MODEL_CACHE[key]


def extract_wavlm(wav_path_or_array, sr=SAMPLE_RATE, device=None, max_chunk_seconds=20):
    """Returns [T, 768] float32 (final hidden state only), resampled to TARGET_FPS."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor, model = _get_wavlm(device)
    wav = load_audio(wav_path_or_array, sr) if isinstance(wav_path_or_array, str) else wav_path_or_array

    chunk_samples = int(max_chunk_seconds * sr)
    chunks = [wav[i:i + chunk_samples] for i in range(0, len(wav), chunk_samples)] or [wav]

    outs = []
    for chunk in chunks:
        inputs = extractor(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.input_values.to(device), output_hidden_states=True)
        outs.append(out.hidden_states[-1].squeeze(0).cpu().numpy().astype(np.float32))
    feats = np.concatenate(outs, axis=0)  # [T_native, 768]

    native_hop_seconds = 320 / sr  # WavLM's fixed ~20ms conv-frontend hop
    return resample_to_fps(feats, native_hop_seconds, TARGET_FPS)


def extract_combined(wav_path, device=None):
    """Returns [T, 781] float32: mfcc(13) concatenated with wavlm(768),
    T = min(len(mfcc), len(wavlm))."""
    mfcc_feat = extract_mfcc(wav_path)
    wavlm_feat = extract_wavlm(wav_path, device=device)
    T = min(mfcc_feat.shape[0], wavlm_feat.shape[0])
    return np.concatenate([mfcc_feat[:T], wavlm_feat[:T]], axis=1)
