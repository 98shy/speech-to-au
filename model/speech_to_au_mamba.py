import torch.nn as nn

from .mamba_block import MambaStack


class SpeechToAUMamba(nn.Module):
    """Same backbone-swap ablation as SpeechToAU (TCN) and SpeechToAUGPT2,
    but replacing the temporal backbone with a stack of Mamba (Selective
    State Space Model) blocks -- the core temporal-modeling idea from Yu
    et al., "Hierarchical Granularity Alignment and State Space Modeling
    for Robust Multimodal AU Detection in the Wild" (arXiv 2603.11306,
    10th ABAW), MINUS their Audio-Guided SSM cross-modal gating (which has
    no audio-only equivalent -- it requires a paired visual stream).

    Everything else (input projection width, head, dropout) mirrors
    SpeechToAU so the TCN-vs-Mamba comparison isolates the backbone choice.
    """

    def __init__(self, input_dim, num_aus=3, d_model=256, d_state=16,
                 n_layers=4, expand=2, conv_kernel=4,
                 head_hidden_dim=512, head_dropout=0.5):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.mamba = MambaStack(d_model, n_layers=n_layers, d_state=d_state,
                                 expand=expand, conv_kernel=conv_kernel)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_aus),
        )

    def forward(self, x):
        """x: [B, T, D] -> [B, T, num_aus]"""
        x = self.in_proj(x)
        x = self.mamba(x)
        return self.head(x)
