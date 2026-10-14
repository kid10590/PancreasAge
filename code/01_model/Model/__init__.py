"""
Model package — pancreatic-age regressors.

Layers:
  heads.py                  — shared heads (ResidualHead, ResidualScalarHead, EvidentialHead, TokenAttentionPool)
  base_regressor.py         — BaseAgeRegressor (base class of all regressors)
  monai_swin3d_regressor.py — Swin3DBackbone + Swin3DAgeRegressor

The RadFM backbone variant (pancreas_age_model.py) is optional and not required
for the published Swin3D model. It is imported lazily only if available, so that
the public package does not hard-depend on `transformers`.

Usage:
  from Model import Swin3DAgeRegressor
"""
from .base_regressor import BaseAgeRegressor
from .heads import ResidualHead, ResidualScalarHead, EvidentialHead, TokenAttentionPool
from .monai_swin3d_regressor import Swin3DBackbone, Swin3DAgeRegressor

try:  # optional RadFM backbone (needs `transformers`)
    from .pancreas_age_model import RadFMAgeRegressor  # noqa: F401
except Exception:  # pragma: no cover
    RadFMAgeRegressor = None

__all__ = [
    "BaseAgeRegressor",
    "ResidualHead", "ResidualScalarHead", "EvidentialHead", "TokenAttentionPool",
    "Swin3DBackbone", "Swin3DAgeRegressor",
    "RadFMAgeRegressor",
]
