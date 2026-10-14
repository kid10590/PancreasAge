"""
Common training utilities: loss, evaluation, helper functions
Extracted from pancreas_age_train.py to eliminate redundancy
"""
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm


# ═══════════════════════════════════════════════
# Basic utilities
# ═══════════════════════════════════════════════
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)


# ═══════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════
def _r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("nan")
    am, bm = a - a.mean(), b - b.mean()
    denom = np.sqrt((am**2).sum()) * np.sqrt((bm**2).sum()) + 1e-12
    return float((am * bm).sum() / denom)


def _prep_target(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Convert to a (B,K) probability distribution + label smoothing"""
    t = t.float()
    if t.dim() > 2:
        B, K = t.shape[:2]
        t = t.view(B, K, -1).mean(-1)
    t = t.clamp_min(0)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if eps and eps > 0:
        K = t.shape[1]
        t = (1 - eps) * t + eps / K
        t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return t


def evidential_nll(mu, nu, alpha, beta, y, eps=1e-8):
    """Student-t NLL (NIG)"""
    term1 = 0.5 * (math.log(math.pi) - torch.log(nu + eps))
    term2 = -alpha * torch.log(2.0 * beta * (1.0 + nu) + eps)
    term3 = (alpha + 0.5) * torch.log(nu * (y - mu).pow(2) + 2.0 * beta * (1.0 + nu) + eps)
    return term1 + term2 + term3


def fit_delta_vs_age_torch(age_t: torch.Tensor, pred_t: torch.Tensor):
    """Linear regression of Δ = pred - age = α*age + β"""
    if age_t.numel() == 0:
        return 0.0, 0.0
    age = age_t.float().view(-1)
    pred = pred_t.float().view(-1)
    delta = pred - age
    age_mean = age.mean()
    delta_mean = delta.mean()
    var_age = ((age - age_mean) ** 2).mean()
    if float(var_age.item()) < 1e-12:
        return 0.0, float(delta_mean.item())
    cov = ((age - age_mean) * (delta - delta_mean)).mean()
    alpha = cov / (var_age + 1e-12)
    beta = delta_mean - alpha * age_mean
    return float(alpha.item()), float(beta.item())


def fit_delta_vs_age_lowess(age_np: np.ndarray, pred_np: np.ndarray,
                            frac: float = 0.3, max_samples: int = 3000):
    """
    LOWESS non-parametric regression of Δ = pred - age.

    First fit LOWESS Δ ~ age on the training data, and return an interpolator
    that can compute the bias offset for any age.

    Parameters:
        age_np, pred_np: (N,) numpy arrays
        frac: LOWESS local window fraction
        max_samples: if N > max_samples, randomly downsample to speed up

    Returns:
        offset_fn: callable, offset_fn(age_array) → offset_array (numpy)
        lowess_summary: dict with {'ages_sorted', 'deltas_smoothed', 'frac'}
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import interp1d

    age = np.asarray(age_np, dtype=np.float64).ravel()
    pred = np.asarray(pred_np, dtype=np.float64).ravel()
    delta = pred - age

    N = len(age)
    if N == 0:
        def _zero_offset(x):
            return np.zeros_like(x)
        return _zero_offset, {'ages_sorted': np.array([]), 'deltas_smoothed': np.array([]), 'frac': frac}

    # downsample to speed up LOWESS (O(N²) complexity)
    if N > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=max_samples, replace=False)
        age_fit = age[idx]
        delta_fit = delta[idx]
    else:
        age_fit = age
        delta_fit = delta

    try:
        lowess_result = lowess(delta_fit, age_fit, frac=frac, return_sorted=True)
        ages_sorted = lowess_result[:, 0]
        deltas_smoothed = lowess_result[:, 1]
    except Exception as e:
        print(f"[LOWESS] fit failed: {e}, falling back to a constant offset")
        mean_delta = float(np.mean(delta))
        ages_sorted = np.array([float(np.min(age)), float(np.max(age))])
        deltas_smoothed = np.array([mean_delta, mean_delta])

    # build interpolator: output an offset for any age
    offset_fn = interp1d(
        ages_sorted, deltas_smoothed,
        kind='linear',
        bounds_error=False,
        fill_value=(deltas_smoothed[0], deltas_smoothed[-1]),
    )

    summary = {
        'ages_sorted': ages_sorted,
        'deltas_smoothed': deltas_smoothed,
        'frac': frac,
    }
    return offset_fn, summary


# ═══════════════════════════════════════════════
# Evaluation functions
# ═══════════════════════════════════════════════
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    age_space_tensor,
    apply_bias=None,
    use_amp=False,
    autocast_dtype=torch.float16,
    return_details=False,
    cfg=None,
    kl_weight_now=1.0,
    reg_weight_now=0.0,
    evi_w_now=0.0,
    evi_reg_w=0.0,
):
    model.eval()
    klloss_fn = nn.KLDivLoss(reduction='batchmean')
    use_residual = bool(cfg.get("use_residual_head", False))
    use_evi = bool(cfg.get("use_evidential_head", False))
    reg_type = str(cfg.get("reg_type", "l1")).lower()

    mae_list, kl_list, reg_list = [], [], []
    evi_nll_list, evi_reg_list = [], []
    total_loss_list = []
    preds_all_t_raw, preds_all_t, gts_all_t = [], [], []

    eps = float(cfg.get("label_smoothing", 1e-3))
    logp_min = float(cfg.get("clip_logp_min", -50))
    is_cuda = (device != "cpu")

    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            tgt = _prep_target(batch["target"].to(device, non_blocking=True), eps=eps)

            with autocast(enabled=(use_amp and is_cuda), dtype=autocast_dtype, device_type=("cuda" if is_cuda else "cpu")):
                if use_residual or use_evi:
                    logp, feat = model.forward_with_feat(x)
                else:
                    logp = model(x)
                    feat = None
                if logp_min is not None:
                    logp = torch.clamp(logp, min=float(logp_min))

                loss_kl = klloss_fn(logp, tgt)
                p = logp.exp()
                base_age = (p * age_space_tensor.unsqueeze(0)).sum(dim=-1)

                r = None
                if use_residual and reg_weight_now > 0 and hasattr(model, "res_head") and model.res_head is not None:
                    r = model.res_head(feat)
                pred_age = base_age + r if r is not None else base_age
                pred_age_raw = pred_age

                gt_age = batch["age"].to(device, non_blocking=True).float()

                # bias correction
                if apply_bias is not None and isinstance(apply_bias, dict):
                    alpha = float(apply_bias.get("alpha", 0))
                    beta = float(apply_bias.get("beta", 0))
                    pred_age = pred_age - (alpha * gt_age + beta)

                loss_reg = pred_age.new_tensor(0.0)
                if reg_weight_now > 0:
                    loss_reg = F.mse_loss(pred_age, gt_age) if reg_type == "mse" else F.l1_loss(pred_age, gt_age)

                evi_nll = pred_age.new_tensor(0.0)
                evi_reg = pred_age.new_tensor(0.0)
                if use_evi and hasattr(model, "evi_head") and model.evi_head is not None and (evi_w_now > 0 or evi_reg_w > 0):
                    mu, nu, alpha, beta = model.evi_head(feat)
                    evi_nll = evidential_nll(mu, nu, alpha, beta, gt_age).mean()
                    evi_reg = ((gt_age - mu).abs() * (2.0 * nu + alpha)).mean()

                total_loss = kl_weight_now * loss_kl + reg_weight_now * loss_reg + evi_w_now * evi_nll + evi_reg_w * evi_reg

            mae = (pred_age - gt_age).abs().mean()
            mae_list.append(float(mae.item()))
            kl_list.append(float(loss_kl.item()))
            reg_list.append(float(loss_reg.item()))
            evi_nll_list.append(float(evi_nll.item()))
            evi_reg_list.append(float(evi_reg.item()))
            total_loss_list.append(float(total_loss.item()))

            if return_details:
                preds_all_t_raw.append(pred_age_raw.detach())
                preds_all_t.append(pred_age.detach())
                gts_all_t.append(gt_age.detach())

    results = {
        "mae": float(np.mean(mae_list)),
        "kl": float(np.mean(kl_list)),
        "reg": float(np.mean(reg_list)),
        "evi_nll": float(np.mean(evi_nll_list)),
        "evi_reg": float(np.mean(evi_reg_list)),
        "total_loss": float(np.mean(total_loss_list)),
    }
    if return_details:
        if preds_all_t:
            results["preds"] = torch.cat(preds_all_t, dim=0).cpu().numpy()
            results["preds_raw"] = torch.cat(preds_all_t_raw, dim=0).cpu().numpy()
            results["gts"] = torch.cat(gts_all_t, dim=0).cpu().numpy()
        else:
            results["preds"] = np.array([])
            results["preds_raw"] = np.array([])
            results["gts"] = np.array([])
    return results
