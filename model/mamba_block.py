import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model (Mamba's S6 scan), input-dependent
    discretization of Gu & Dao (2023). Sequential (non-parallel) scan --
    fine for our clip lengths (~75 frames), no custom CUDA kernel needed.

    A ∈ R^{d_inner x d_state} is a real, negative, per-channel evolution
    matrix (diagonal SSM per channel). Δ, B, C are all input-dependent
    ("selective"), following the standard (non audio-guided) Mamba
    formulation -- this is the piece of Yu et al.'s AG-SSM (arXiv
    2603.11306) that does NOT require a second modality; their
    audio-guided variant additionally conditions Δ/B/C on a paired
    visual+audio descriptor, which has no audio-only analogue.
    """

    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.delta_param = nn.Parameter(torch.zeros(d_inner))
        self.delta_proj = nn.Linear(d_inner, d_inner)
        self.b_proj = nn.Linear(d_inner, d_state)
        self.c_proj = nn.Linear(d_inner, d_state)

        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1)
        self.A_log = nn.Parameter(A_log)          # [d_inner, d_state]
        self.D = nn.Parameter(torch.ones(d_inner))  # skip term

    def forward(self, x):
        """x: [B, T, d_inner] -> [B, T, d_inner]"""
        B, T, _ = x.shape
        A = -torch.exp(self.A_log)                       # [d_inner, d_state], real & negative

        delta = F.softplus(self.delta_param + self.delta_proj(x))  # [B, T, d_inner]
        Bt = self.b_proj(x)                                        # [B, T, d_state]
        Ct = self.c_proj(x)                                        # [B, T, d_state]

        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))       # [B,T,d_inner,d_state]
        delta_B = delta.unsqueeze(-1) * Bt.unsqueeze(2)                              # [B,T,d_inner,d_state]

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)   # [B, d_inner, d_state]
            y_t = torch.einsum("bdn,bn->bd", h, Ct[:, t])                  # [B, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                         # [B, T, d_inner]
        return y + x * self.D


class MambaBlock(nn.Module):
    """One Mamba block: pre-norm -> in_proj (expand + gate split) ->
    causal depthwise conv -> SiLU -> selective SSM -> gate -> out_proj ->
    residual. Matches the standard Mamba block (Gu & Dao, 2023)."""

    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                               padding=conv_kernel - 1, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.conv_kernel = conv_kernel

    def forward(self, x):
        """x: [B, T, d_model] -> [B, T, d_model]"""
        residual = x
        x = self.norm(x)
        x, z = self.in_proj(x).chunk(2, dim=-1)     # each [B, T, d_inner]

        x = x.transpose(1, 2)                       # [B, d_inner, T]
        x = self.conv(x)[:, :, :-(self.conv_kernel - 1)]  # causal: drop right-side padding
        x = x.transpose(1, 2)                        # [B, T, d_inner]
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class MambaStack(nn.Module):
    def __init__(self, d_model, n_layers=4, d_state=16, expand=2, conv_kernel=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand=expand, conv_kernel=conv_kernel)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
