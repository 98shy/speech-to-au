import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as wn


class CauseMLP(nn.Module):
    """Kameoka et al., "CAUSE: Crossmodal Action Unit Sequence Estimation
    from Speech" (Interspeech 2022) -- MLP variant. Ten frame-independent
    fully-connected layers; every frame in the sequence is processed with
    the same shared weights, with no mixing across time (a sanity-check
    architecture: if AU intensity depended only on the current frame, this
    should perform on par with the temporal architectures below)."""

    def __init__(self, input_dim, num_aus=3):
        super().__init__()
        sizes = [input_dim, 128, 128, 64, 64, 32, 32, 16, 16, 8, num_aus]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(wn(nn.Linear(sizes[i], sizes[i + 1])))
            if 0 < i < len(sizes) - 2:  # skip activation/dropout after first and last layers
                layers.append(nn.Dropout(0.1))
                layers.append(nn.LeakyReLU(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, num_aus]"""
        return self.net(x)


class CauseRNN(nn.Module):
    """CAUSE RNN variant: FC(128) -> 4-layer BiLSTM(64 hidden) -> FC(num_aus)."""

    def __init__(self, input_dim, num_aus=3, hidden=64, num_layers=4):
        super().__init__()
        self.in_proj = wn(nn.Linear(input_dim, 128))
        self.lstm = nn.LSTM(128, hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_proj = wn(nn.Linear(hidden * 2, num_aus))

    def forward(self, x):
        x = self.in_proj(x)
        x, _ = self.lstm(x)
        return self.out_proj(x)


class _GLUConvBlock(nn.Module):
    """Conv1d -> Dropout -> GLU. The conv outputs 2*out_channels so GLU's
    channel-halving gate lands on out_channels, matching the paper's stated
    per-layer channel counts (128, 64, 64, 32, 32, 16, 16, 8)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = wn(nn.Conv1d(in_ch, out_ch * 2, kernel_size, padding=padding, dilation=dilation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, C, T] -> [B, out_ch, T]"""
        x = self.conv(x)
        x = self.dropout(x)
        return nn.functional.glu(x, dim=1)


class _CauseConvNet(nn.Module):
    CHANNELS = [128, 64, 64, 32, 32, 16, 16, 8]

    def __init__(self, input_dim, num_aus, dilations):
        super().__init__()
        self.in_proj = wn(nn.Linear(input_dim, 128))
        blocks = []
        prev_ch = 128
        for ch, d in zip(self.CHANNELS, dilations):
            blocks.append(_GLUConvBlock(prev_ch, ch, kernel_size=3, dilation=d))
            prev_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = wn(nn.Linear(prev_ch, num_aus))

    def forward(self, x):
        """x: [B, T, D] -> [B, T, num_aus]"""
        x = self.in_proj(x)
        x = x.transpose(1, 2)  # [B, C, T]
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # [B, T, C]
        return self.out_proj(x)


class CauseCNN(_CauseConvNet):
    """CAUSE CNN variant: regular (non-dilated) 1D convs -- looks at a
    short, local receptive field. Reported as the best-performing
    architecture in the paper."""

    def __init__(self, input_dim, num_aus=3):
        super().__init__(input_dim, num_aus, dilations=[1] * 8)


class CauseDCNN(_CauseConvNet):
    """CAUSE DCNN variant: same as CNN but with dilations 1,3,9,27 repeated
    twice, giving a much larger receptive field for the same parameter
    count. The paper found this performed slightly worse than plain CNN,
    suggesting AU-predictive cues in speech are concentrated locally."""

    def __init__(self, input_dim, num_aus=3):
        super().__init__(input_dim, num_aus, dilations=[1, 3, 9, 27, 1, 3, 9, 27])
