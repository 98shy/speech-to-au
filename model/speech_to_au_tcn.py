import torch.nn as nn

from .tcn import TCN


class SpeechToAU(nn.Module):
    """Speech (MFCC / WavLM / concatenated) -> per-frame AU time series.

    Backbone follows the TCN temporal-modeling stage from Yu et al. (CVPRW
    2025, ABAW8 AU-detection winner); the video branch (STN/ConvNeXt) and
    audio-visual cross-modal fusion from that paper are dropped since our
    task is audio-only. The classification head mirrors their MLP head
    (FC-512 + ReLU + dropout + output layer), sized to our 3 lip AUs
    (AU12/AU25/AU26) instead of their 12.

    Outputs raw logits of shape [B, T, num_aus] -- apply sigmoid for
    per-AU occurrence probabilities (BCEWithLogitsLoss during training),
    or treat as regression targets directly (MSELoss) if labels are
    continuous AU intensities rather than binary occurrence.
    """

    def __init__(self, input_dim, num_aus=3, tcn_channels=(256, 256, 256, 256),
                 kernel_size=3, tcn_dropout=0.2, head_hidden_dim=512, head_dropout=0.5):
        super().__init__()
        self.tcn = TCN(input_dim, list(tcn_channels), kernel_size=kernel_size, dropout=tcn_dropout)
        self.head = nn.Sequential(
            nn.Linear(tcn_channels[-1], head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_aus),
        )

    def forward(self, x):
        """x: [B, T, D] pre-extracted speech features -> [B, T, num_aus]"""
        x = x.transpose(1, 2)          # [B, D, T] for Conv1d
        feats = self.tcn(x)            # [B, C, T]
        feats = feats.transpose(1, 2)  # [B, T, C]
        return self.head(feats)        # [B, T, num_aus]
