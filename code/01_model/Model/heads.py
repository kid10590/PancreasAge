"""
Model/heads.py — shared head / pooling modules
Shared by all regressors to avoid duplicate definitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# Classification head (outputs logits)
# ═══════════════════════════════════════════════
class ResidualHead(nn.Module):
    """
    Residual classification head: x -> Norm -> Linear -> GELU -> Drop -> Linear -> +x_skip
    Supports an optional hidden layer and a residual connection.
    """
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 0, drop_p: float = 0.5,
                 use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.drop = nn.Dropout(drop_p)

        if hidden and hidden > 0:
            self.mlp = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(hidden, num_classes),
            )
            self.proj = nn.Linear(feat_dim, num_classes) if use_residual else None
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, num_classes),
            )
            self.proj = None

    def forward(self, x):
        x_in = self.drop(x)
        out = self.mlp(x_in)
        if self.use_residual and self.proj is not None:
            out = out + self.proj(x_in)
        return out


# ═══════════════════════════════════════════════
# Scalar residual head (regresses an age correction)
# ═══════════════════════════════════════════════
class ResidualScalarHead(nn.Module):
    """
    Outputs a single scalar residual r from the global feature, with optional tanh scaling.
    """
    def __init__(self, feat_dim: int, hidden: int = 128, dropout: float = 0.1, scale: float = 20.0):
        super().__init__()
        self.scale = float(scale)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        r = self.mlp(feat).squeeze(-1)            # (B,)
        if self.scale > 0:
            r = self.scale * torch.tanh(r / max(self.scale, 1e-6))
        return r


# ═══════════════════════════════════════════════
# Evidential head (NIG parameterization)
# ═══════════════════════════════════════════════
class EvidentialHead(nn.Module):
    """
    Outputs μ, ν, α, β (NIG parameterization), marginal is Student-t.
    Constraints: ν>0, α>1, β>0
    """
    def __init__(self, feat_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),
        )

    def forward(self, feat):
        out = self.mlp(feat)                       # (B,4)
        mu = out[:, 0]
        nu = F.softplus(out[:, 1]) + 1e-6          # >0
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6 # >1
        beta = F.softplus(out[:, 3]) + 1e-6        # >0
        return mu, nu, alpha, beta


# ═══════════════════════════════════════════════
# Offset head (CE + bounded regression: one offset per bin)
# ═══════════════════════════════════════════════
class OffsetHead(nn.Module):
    """
    Predicts an offset ∈ [-half_width, +half_width] for each age bin.

    Usage:
      offsets = offset_head(feat)                  # (B, K) raw
      offsets = torch.tanh(offsets) * half_width    # constrain to [-W/2, W/2]
      pred = bin_center[class_k] + offsets[b, class_k]  # inference: top-1 bin
    """
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 64, dropout: float = 0.1,
                 half_width: float = 0.5):
        super().__init__()
        self.half_width = half_width
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),     # one offset for each of the K bins
        )

    def forward(self, feat):
        """Returns the constrained offsets: (B, K), each ∈ [-half_width, +half_width]"""
        raw = self.mlp(feat)
        return torch.tanh(raw) * self.half_width


# ═══════════════════════════════════════════════
# Token attention pooling (for ViT-like backbones)
# ═══════════════════════════════════════════════
class TokenAttentionPool(nn.Module):
    """
    Learnable attention pooling over the token dimension: (B, N, D) -> (B, D)
    """
    def __init__(self, dim: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.droplayer = nn.Dropout(dropout)
        if hidden and hidden > 0:
            self.score = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1, bias=False),
            )
        else:
            self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        # x: (B, N, D)
        s = self.score(x).squeeze(-1)          # (B, N)
        s = self.droplayer(s)
        w = torch.softmax(s, dim=1).unsqueeze(-1)  # (B, N, 1)
        pooled = (w * x).sum(dim=1)            # (B, D)
        return pooled, w
