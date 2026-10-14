"""
Model/base_regressor.py — base class for age regressors

Logic shared by all backbones:
  - extract_feat / forward_with_feat / forward
  - param_groups (grouped optimizer parameters)
  - build_residual_head / build_evidential_head
  - predict_age (inference interface)

Subclasses only need to:
  1. Build self.backbone and set self.feat_dim in __init__
  2. Optionally override extract_feat (if backbone.forward does not directly return a (B,D) feature vector)
"""
import torch
import torch.nn as nn
from typing import Optional

from .heads import ResidualHead, ResidualScalarHead, EvidentialHead, OffsetHead


class BaseAgeRegressor(nn.Module):
    """
    Base class for age regressors.

    Args:
        num_classes: size of the age space (number of classes)
        feat_dim: dimension of the backbone output feature
        head_hidden: hidden layer size of the classification head (0 = no hidden layer)
        head_dropout: dropout of the classification head
    """

    def __init__(self, num_classes: int, feat_dim: int,
                 head_hidden: int = 0, head_dropout: float = 0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # ── classification head ──
        self.head = ResidualHead(feat_dim, num_classes,
                                 hidden=head_hidden, drop_p=head_dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # ── optional heads (built on demand during training) ──
        self.res_head: Optional[ResidualScalarHead] = None
        self.evi_head: Optional[EvidentialHead] = None
        self.offset_head: Optional[OffsetHead] = None

        # ── backbone is set by the subclass ──
        self.backbone: nn.Module = None  # type: ignore[assignment]

    # ═══════════════════════════════════════════
    # Forward
    # ═══════════════════════════════════════════
    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract a global feature vector (B, D) from the input image.
        By default calls backbone(x) directly; subclasses may override.
        """
        return self.backbone(x)

    def forward_with_feat(self, x: torch.Tensor):
        """
        Returns (log_prob, feat).
        log_prob: (B, K) LogSoftmax output
        feat:     (B, D) intermediate features, reused by res_head / evi_head
        """
        feat = self.extract_feat(x)
        logits = self.head(feat)
        log_prob = self.log_softmax(logits)
        return log_prob, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_prob, _ = self.forward_with_feat(x)
        return log_prob

    # ═══════════════════════════════════════════
    # Optimizer parameter grouping
    # ═══════════════════════════════════════════
    def param_groups(self, lr_head: float, lr_backbone: float,
                     wd_head: float, wd_backbone: float):
        """
        Returns the optimizer parameter group list.
        Subclasses may override to add extra parameter groups (e.g. a separate LR for token_pool).
        """
        head_params = list(self.head.parameters())
        if self.res_head is not None:
            head_params += list(self.res_head.parameters())
        if self.evi_head is not None:
            head_params += list(self.evi_head.parameters())
        if self.offset_head is not None:
            head_params += list(self.offset_head.parameters())

        groups = [{"params": head_params, "lr": lr_head, "weight_decay": wd_head}]

        bb_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if len(bb_params) > 0:
            groups.append({"params": bb_params, "lr": lr_backbone, "weight_decay": wd_backbone})
        return groups

    # ═══════════════════════════════════════════
    # Optional head construction
    # ═══════════════════════════════════════════
    def build_residual_head(self, hidden: int = 128, dropout: float = 0.1,
                            scale: float = 20.0):
        """Build the scalar residual prediction head, returns self to support chaining"""
        self.res_head = ResidualScalarHead(self.feat_dim, hidden=hidden,
                                           dropout=dropout, scale=scale)
        return self

    def build_evidential_head(self, hidden: int = 128, dropout: float = 0.1):
        """Build the Evidential (NIG) head, returns self to support chaining"""
        self.evi_head = EvidentialHead(self.feat_dim, hidden=hidden,
                                       dropout=dropout)
        return self

    def build_offset_head(self, hidden: int = 64, dropout: float = 0.1,
                          half_width: float = 0.5):
        """Build the Offset head (CE + bounded regression), returns self to support chaining"""
        self.offset_head = OffsetHead(
            self.feat_dim, self.num_classes,
            hidden=hidden, dropout=dropout, half_width=half_width,
        )
        return self

    # ═══════════════════════════════════════════
    # Inference (offset mode: top-1 bin)
    # ═══════════════════════════════════════════
    def predict_age_with_offset(self, x: torch.Tensor, age_space: torch.Tensor):
        """
        CE + Offset inference: argmax to pick a bin, add the offset.
        Returns: pred_age (B,), class_idx (B,), offsets (B,), logits (B,K)
        """
        logp, feat = self.forward_with_feat(x)
        logits = self.head(feat)  # (B, K) pre-softmax logits
        offsets = self.offset_head(feat)  # (B, K) ∈ [-half_width, half_width]

        class_idx = logits.argmax(dim=-1)  # (B,)
        age_space_dev = age_space.to(x.device)
        bin_centers = age_space_dev[class_idx]  # (B,)
        offset_selected = offsets[torch.arange(x.shape[0], device=x.device), class_idx]  # (B,)
        pred_age = bin_centers + offset_selected
        return pred_age, class_idx, offsets, logits

    # ═══════════════════════════════════════════
    # Inference
    # ═══════════════════════════════════════════
    def predict_age(self, x: torch.Tensor, age_space: torch.Tensor,
                    use_residual: bool = True):
        """
        Returns the expected age prediction.

        Returns:
            final_age: (B,)   final predicted age
            base_age:  (B,)   classification head base value
            p:         (B,K)  probability distribution
            r:         (B,)   residual value (may be None)
            evi:       tuple  (mu,nu,alpha,beta) or None
        """
        logp, feat = self.forward_with_feat(x)
        p = logp.exp()
        base_age = (p * age_space.unsqueeze(0)).sum(dim=-1)

        r = None
        if use_residual and self.res_head is not None:
            r = self.res_head(feat)
            final_age = base_age + r
        else:
            final_age = base_age

        evi = None
        if self.evi_head is not None:
            mu, nu, alpha, beta = self.evi_head(feat)
            evi = (mu, nu, alpha, beta)

        return final_age, base_age, p, r, evi
