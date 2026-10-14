import numpy as np
import matplotlib.pyplot as plt
import os
from utils.train_utils import _r2_np

def _get(hist, *names, default=None):
    for n in names:
        if n in hist:
            return hist[n]
    return default

def plot_curves(hist: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n_epochs = len(_get(hist, "train_loss", default=[]))
    if n_epochs == 0:
        print("[plot] empty history, skipping plot")
        return
    xs = np.arange(1, n_epochs + 1)

    # ----- Loss curves -----
    plt.figure(figsize=(20, 4.5))
    plt.subplot(1,4,1)
    if "train_loss" in hist:
        plt.plot(xs, hist["train_loss"], label="train loss", color="#2166AC", linewidth=1.2)
    if "val_loss" in hist:
        plt.plot(xs, hist["val_loss"], label="val loss", color="#1B7837", linewidth=1.2)
    if "test_loss" in hist and any(np.isfinite(np.array(hist["test_loss"], dtype=float))):
        plt.plot(xs, hist["test_loss"], label="test loss", color="#B2182B", linewidth=1.5, alpha=0.85)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(fontsize=7)

    # ── color scheme: 3 methods × 2 sets = 6 colors ──
    # raw:        blue family
    # linear_bc:  green family
    # lowess_bc:  red/orange family
    # train=dark, val=light, test=medium-bright

    # ----- MAE raw + two bias corrections -----
    train_raw = _get(hist, "train_mae_raw", "train_mae", default=[])
    val_raw   = _get(hist, "val_mae_raw", "val_mae", default=[])
    train_linear = _get(hist, "train_mae_linear", "train_mae_bc", default=None)
    val_linear   = _get(hist, "val_mae_linear", "val_mae_bc", default=None)
    train_lowess = _get(hist, "train_mae_lowess", default=None)
    val_lowess   = _get(hist, "val_mae_lowess", default=None)

    plt.subplot(1,4,2)
    if train_raw:
        plt.plot(xs, train_raw, label="train raw", color="#2166AC", linewidth=1.2)
    if val_raw:
        plt.plot(xs, val_raw, label="val raw", color="#92C5DE", linewidth=1.2)
    if train_linear:
        plt.plot(xs, train_linear, label="train linear", color="#1B7837", linewidth=1.2)
    if val_linear:
        plt.plot(xs, val_linear, label="val linear", color="#A6DBA0", linewidth=1.2)
    if train_lowess:
        plt.plot(xs, train_lowess, label="train LOWESS", color="#B2182B", linewidth=1.2)
    if val_lowess:
        plt.plot(xs, val_lowess, label="val LOWESS", color="#F4A582", linewidth=1.2)
    if "test_mae_linear" in hist and any(np.isfinite(np.array(hist["test_mae_linear"], dtype=float))):
        plt.plot(xs, hist["test_mae_linear"], label="test linear", color="#4DAF4A", linewidth=1.5, alpha=0.85)
    if "test_mae_lowess" in hist and any(np.isfinite(np.array(hist["test_mae_lowess"], dtype=float))):
        plt.plot(xs, hist["test_mae_lowess"], label="test LOWESS", color="#E41A1C", linewidth=1.5, alpha=0.85)
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("MAE")
    plt.legend(fontsize=7, ncol=2)

    # ----- bias-correction gain -----
    plt.subplot(1,4,3)
    if train_linear and train_raw:
        train_gain_linear = np.array(train_raw) - np.array(train_linear)
        plt.plot(xs, train_gain_linear, label="train Δ(linear)", color="#1B7837", linewidth=1.2)
    if val_linear and val_raw:
        val_gain_linear = np.array(val_raw) - np.array(val_linear)
        plt.plot(xs, val_gain_linear, label="val Δ(linear)", color="#A6DBA0", linewidth=1.2)
    if train_lowess and train_raw:
        train_gain_lowess = np.array(train_raw) - np.array(train_lowess)
        plt.plot(xs, train_gain_lowess, label="train Δ(LOWESS)", color="#B2182B", linewidth=1.2)
    if val_lowess and val_raw:
        val_gain_lowess = np.array(val_raw) - np.array(val_lowess)
        plt.plot(xs, val_gain_lowess, label="val Δ(LOWESS)", color="#F4A582", linewidth=1.2)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Epoch"); plt.ylabel("MAE Reduction"); plt.title("Bias Correction Gain")
    plt.legend(fontsize=7, ncol=2)
    # ----- R2 curves -----
    plt.subplot(1,4,4)
    r_tr_raw    = _get(hist, "train_r2_raw", default=[])
    r_tr_linear = _get(hist, "train_r2_linear", "train_r2_bc", default=[])
    r_tr_lowess = _get(hist, "train_r2_lowess", default=[])
    r_val_raw   = _get(hist, "val_r2_raw", default=[])
    r_val_linear = _get(hist, "val_r2_linear", "val_r2_bc", default=[])
    r_val_lowess = _get(hist, "val_r2_lowess", default=[])
    r_test_linear = _get(hist, "test_r2_linear", default=[])
    r_test_lowess = _get(hist, "test_r2_lowess", default=[])
    if r_tr_raw:    plt.plot(xs, r_tr_raw, label="train raw", color="#2166AC", linewidth=1.2)
    if r_tr_linear: plt.plot(xs, r_tr_linear, label="train linear", color="#1B7837", linewidth=1.2)
    if r_tr_lowess: plt.plot(xs, r_tr_lowess, label="train LOWESS", color="#B2182B", linewidth=1.2)
    if r_val_raw:   plt.plot(xs, r_val_raw, label="val raw", color="#92C5DE", linewidth=1.2)
    if r_val_linear: plt.plot(xs, r_val_linear, label="val linear", color="#A6DBA0", linewidth=1.2)
    if r_val_lowess: plt.plot(xs, r_val_lowess, label="val LOWESS", color="#F4A582", linewidth=1.2)
    if r_test_linear and any(np.isfinite(np.array(r_test_linear))):
        plt.plot(xs, r_test_linear, label="test linear", color="#4DAF4A", linewidth=1.5, alpha=0.85)
    if r_test_lowess and any(np.isfinite(np.array(r_test_lowess))):
        plt.plot(xs, r_test_lowess, label="test LOWESS", color="#E41A1C", linewidth=1.5, alpha=0.85)
    plt.xlabel("Epoch"); plt.ylabel("R²"); plt.title("R²")
    plt.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] saved: {out_path}")

def plot_scatter(y: np.ndarray, yhat: np.ndarray, save_path: str, title: str = ""):
    if y.size == 0:
        return
    mae = float(np.mean(np.abs(y - yhat)))
    r2 = float(_r2_np(y, yhat)) if y.size > 2 else float('nan')
    lim_min = float(min(y.min(), yhat.min())) - 2
    lim_max = float(max(y.max(), yhat.max())) + 2

    plt.figure(figsize=(5.5, 5.2))
    plt.scatter(y, yhat, s=10, c="#2166AC", alpha=0.45, edgecolors="none")
    # identity line
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color="#B2182B", lw=1.2,
             linestyle="--", alpha=0.7, label="y = x")
    # regression line
    k, c = _linreg_1d(y, yhat)
    xs_line = np.linspace(lim_min, lim_max, 100)
    plt.plot(xs_line, k * xs_line + c, color="#1B7837", lw=1.2, alpha=0.8,
             label=f"fit: y={k:.2f}x+{c:.1f}")
    plt.text(0.05, 0.88, f"MAE={mae:.2f}\n$R^2$={r2:.3f}",
             transform=plt.gca().transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.xlabel("Age (GT)"); plt.ylabel("Age (Pred)"); plt.title(title)
    plt.xlim(lim_min, lim_max); plt.ylim(lim_min, lim_max)
    plt.legend(fontsize=7, loc="lower right")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def _linreg_1d(x: np.ndarray, y: np.ndarray):
    """
    Closed-form linear regression y = k*x + c
    Returns (k, c)
    """
    if x.size == 0:
        return 0.0, 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xm = x.mean(); ym = y.mean()
    varx = np.mean((x - xm)**2)
    if varx < 1e-12:
        return 0.0, ym
    cov = np.mean((x - xm)*(y - ym))
    k = cov / (varx + 1e-12)
    c = ym - k * xm
    return float(k), float(c)

def plot_delta_scatter(age: np.ndarray,
                       pred_age_raw: np.ndarray,
                       save_path: str,
                       title: str = ""):
    """
    Plot a scatter plot of Δ versus age
    age: true age (y_true)
    pred_age_raw: bias-uncorrected predicted age
    """
    if age.size == 0 or pred_age_raw.size == 0:
        return
    age = age.astype(np.float64)
    pred = pred_age_raw.astype(np.float64)
    delta = pred - age  # Δ

    # regression Δ = α * Age + β
    alpha, beta = _linreg_1d(age, delta)

    # correlation coefficient
    age_center = age - age.mean()
    delta_center = delta - delta.mean()
    denom = (np.sqrt(np.sum(age_center**2)) * np.sqrt(np.sum(delta_center**2)) + 1e-12)
    r = float(np.sum(age_center * delta_center) / denom)

    # fitted line
    x_line = np.linspace(age.min(), age.max(), 100)
    y_line = alpha * x_line + beta

    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(age, delta, s=10, alpha=0.45, color="#6ab04c")
    plt.plot(x_line, y_line, 'k-', linewidth=1.5, label="regression line")
    plt.axhline(0.0, color='red', linestyle='--', linewidth=1.0, label="Δ=0")

    plt.xlabel("Chronological Age (years)")
    plt.ylabel("Δ (Pred Age - Chronological Age)")
    plt.title(title)
    plt.grid(alpha=0.25)

    # text box (shows the regression equation and r)
    formula_txt = f"y = {alpha:.2f}x + {beta:.2f}\nr = {r:.2f}"
    plt.legend(loc="upper right")
    plt.text(0.05, 0.95, formula_txt,
             transform=plt.gca().transAxes,
             va="top", ha="left",
             fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="#333333"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"[plot] saved: {save_path}")