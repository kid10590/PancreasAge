"""
Model/monai_swin3d_regressor.py — Swin3D backbone + age regressor

Architecture:
  Swin3DBackbone (pure backbone: 3D image -> feature vector)
    └── Swin3DAgeRegressor (BaseAgeRegressor subclass)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
from typing import Optional

from monai.networks.nets.swin_unetr import SwinTransformer

from .base_regressor import BaseAgeRegressor


# ═══════════════════════════════════════════════
# Swin3D Backbone (pure feature extraction)
# ═══════════════════════════════════════════════
class Swin3DBackbone(nn.Module):
    """
    MONAI SwinTransformer 3D backbone.

    forward: (B, C, H, W, D) -> (B, out_dim)
    Internally: multi-scale feature extraction → layer selection → global pooling → feature vector
    """
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: tuple = (7, 7, 7),
        patch_size: tuple = (2, 2, 2),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        use_v2: bool = True,
        freeze_stages: int = 0,
        freeze_all: bool = False,
        freeze_norm: bool = True,
        feat_index: int = -1,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.feat_index = int(feat_index)

        self.swin = SwinTransformer(
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=tuple(window_size),
            patch_size=tuple(patch_size),
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=False,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            downsample="merging",
            use_v2=use_v2,
        )

        # compute the output dimension: Swin returns len(depths)+1 levels, with channels embed_dim * 2^i
        n_levels = len(depths) + 1
        idx = self.feat_index % n_levels
        self.out_dim = int(embed_dim * (2 ** idx))

        # load pretrained weights
        if pretrained_path:
            self._load_swin_ssl_pretrain(pretrained_path, embed_dim, num_heads, window_size, patch_size)

        # freezing strategy
        if freeze_all:
            self._freeze_modules([self.swin], freeze_norm=freeze_norm)
        elif freeze_stages and freeze_stages > 0:
            self._freeze_some_stages(int(freeze_stages), freeze_norm=freeze_norm)

    # ── pretrained weight loading ──
    def _load_swin_ssl_pretrain(self, path: str, embed_dim: int, num_heads, window_size, patch_size):
        try:
            sd = torch.load(path, map_location="cpu")
            state = sd.get("model", sd.get("state_dict", sd))

            clean = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                if k.startswith("encoder."):
                    k = k[len("encoder."):]
                else:
                    continue
                k = re.sub(r"^layers([1-4])\.(\d+)\.(\d+)\.", r"layers\1.\2.", k)
                if "relative_position_index" in k:
                    continue
                clean[k] = v

            own = self.swin.state_dict()
            matched = {k: v for k, v in clean.items() if k in own and v.shape == own[k].shape}
            skipped = [k for k in clean if k not in matched]

            self.swin.load_state_dict(matched, strict=False)
            print(f"[pretrain] loaded {len(matched)}/{len(own)} keys from {os.path.basename(path)}")
            if skipped:
                print(f"[pretrain] skipped {len(skipped)} keys (name/shape mismatch).")
        except Exception as e:
            print(f"[pretrain] load failed: {e}")

    # ── freezing helpers ──
    @staticmethod
    def _freeze_modules(modules, freeze_norm=True):
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            if freeze_norm:
                m.eval()

    def _freeze_some_stages(self, k: int, freeze_norm=True):
        """Freeze the first k stages (patch_embed + layers1..layers{k})"""
        mods = []
        if hasattr(self.swin, "patch_embed"):
            mods.append(self.swin.patch_embed)
        # ponytail: MONAI Swin v2 stores layers as separate attributes layers1..layers4, not a ModuleList
        for i in range(1, k + 1):
            layer = getattr(self.swin, f"layers{i}", None)
            if layer is not None:
                mods.append(layer)
        self._freeze_modules(mods, freeze_norm=freeze_norm)
        if freeze_norm and hasattr(self.swin, "norm"):
            self._freeze_modules([self.swin.norm], freeze_norm=True)

    # ── forward ──
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W, D)  value range [-1, 1]
        Returns: (B, out_dim) global feature vector
        """
        feats = self.swin(x)                     # [x0, x1, x2, x3, x4] multi-scale features
        f = feats[self.feat_index]               # select the specified level

        if f.dim() == 5:                         # (B, C, H, W, D) → 3D pooling
            pooled = F.adaptive_avg_pool3d(f, 1).flatten(1)
        elif f.dim() == 3:                       # (B, N, D) → mean
            pooled = f.mean(dim=1)
        elif f.dim() == 2:                       # already (B, D)
            pooled = f
        else:
            raise RuntimeError(f"unknown Swin output shape: {tuple(f.shape)}")
        return pooled


# ═══════════════════════════════════════════════
# Swin3D age regressor
# ═══════════════════════════════════════════════
class Swin3DAgeRegressor(BaseAgeRegressor):
    """
    Age regression model with a Swin3D backbone + classification head.

    Usage:
        model = Swin3DAgeRegressor(num_classes=80, model_cfg={...})
        model.build_residual_head(hidden=128).build_evidential_head(hidden=128)
    """

    def __init__(self, num_classes: int, model_cfg: Optional[dict] = None):
        m = model_cfg or {}

        # ── build backbone ──
        backbone = Swin3DBackbone(
            in_chans=int(m.get("in_channels", 1)),
            embed_dim=int(m.get("embed_dim", 96)),
            depths=tuple(m.get("depths", [2, 2, 6, 2])),
            num_heads=tuple(m.get("num_heads", [3, 6, 12, 24])),
            window_size=tuple(m.get("window_size", [7, 7, 7])),
            patch_size=tuple(m.get("patch_size", [2, 2, 2])),
            mlp_ratio=float(m.get("mlp_ratio", 4.0)),
            drop_rate=float(m.get("drop_rate", 0.0)),
            attn_drop_rate=float(m.get("attn_drop_rate", 0.0)),
            drop_path_rate=float(m.get("drop_path_rate", 0.1)),
            use_checkpoint=bool(m.get("use_checkpoint", False)),
            use_v2=bool(m.get("use_v2", True)),
            freeze_stages=int(m.get("freeze_stages", 0)),
            freeze_all=bool(m.get("freeze_backbone", False)),
            freeze_norm=bool(m.get("freeze_norm", True)),
            feat_index=int(m.get("feat_index", -1)),
            pretrained_path=m.get("pretrained_path", None),
        )

        feat_dim = backbone.out_dim
        head_hidden = int(m.get("head_hidden", 64))
        head_dropout = float(m.get("head_dropout", 0.5))

        super().__init__(num_classes=num_classes, feat_dim=feat_dim,
                         head_hidden=head_hidden, head_dropout=head_dropout)
        self.backbone = backbone
