"""Evaluation metrics: MAE, PCC_age, PCC_gap, CS(α)"""
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, Optional


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float('nan')
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    std_x, std_y = np.std(x), np.std(y)
    if std_x < 1e-12 or std_y < 1e-12:
        return float('nan')
    r, _ = pearsonr(x, y)
    return float(r)


def compute_metrics(
    ages_gt: np.ndarray,
    ages_pred: np.ndarray,
    age_groups: Optional[Dict[str, np.ndarray]] = None,
) -> dict:
    gt = np.asarray(ages_gt, dtype=np.float64).ravel()
    pred = np.asarray(ages_pred, dtype=np.float64).ravel()
    n = len(gt)
    if n == 0:
        return {"mae": float('nan'), "pcc_age": float('nan'), "pcc_gap": float('nan'),
                "cs1": 0.0, "cs3": 0.0, "n": 0}

    abs_err = np.abs(pred - gt)
    mae = float(np.mean(abs_err))
    pcc_age = _safe_pearsonr(pred, gt)
    bag = pred - gt
    pcc_gap = _safe_pearsonr(bag, gt)
    cs1 = float(np.mean(abs_err <= 1.0)) * 100.0
    cs3 = float(np.mean(abs_err <= 3.0)) * 100.0

    result = {"mae": mae, "pcc_age": pcc_age, "pcc_gap": pcc_gap,
              "cs1": cs1, "cs3": cs3, "n": n}

    if age_groups:
        result["groups"] = {}
        for gname, mask in age_groups.items():
            if mask.sum() > 0:
                result["groups"][gname] = compute_metrics(gt[mask], pred[mask])
            else:
                result["groups"][gname] = {"mae": float('nan'), "n": 0}

    return result


def make_age_groups(ages: np.ndarray, train_min: float = 45.0, train_max: float = 83.0) -> Dict[str, np.ndarray]:
    ages = np.asarray(ages).ravel()
    return {
        f"<{train_min:.0f}": ages < train_min,
        f"{train_min:.0f}-{train_max:.0f}": (ages >= train_min) & (ages <= train_max),
        f">{train_max:.0f}": ages > train_max,
    }


def print_metrics(metrics: dict, prefix: str = ""):
    n = metrics.get("n", 0)
    mae = metrics.get("mae", float('nan'))
    pcc = metrics.get("pcc_age", float('nan'))
    gap = metrics.get("pcc_gap", float('nan'))
    cs1 = metrics.get("cs1", 0.0)
    cs3 = metrics.get("cs3", 0.0)
    print(f"{prefix}n={n:5d} | MAE={mae:.3f} | PCC_age={pcc:.4f} | "
          f"PCC_gap={gap:.4f} | CS1={cs1:.1f}% CS3={cs3:.1f}%")
    if "groups" in metrics:
        for gname, gm in metrics["groups"].items():
            print_metrics(gm, prefix=f"  [{gname}] ")
