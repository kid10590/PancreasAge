#!/usr/bin/env python
"""Figure 1 | Model performance and calibration of the pancreatic age regressor.

Panels:
  a. Observed vs predicted pancreatic age (healthy, out-of-fold, n=40,649),
     with identity line and density shading.
  b. Mean absolute error (MAE) by age decade (bias-corrected), internal healthy.
  c. Bland-Altman plot: age gap (pred - obs) vs mean age, healthy.
  d. MAE in internal (healthy / diseased) and external (ext part1 / part3) cohorts.

Exports: SVG (editable text) + PDF + PNG (600 dpi). Arial, 7 pt Nature style.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from paths import OUT_ROOT  # noqa: E402
RES = OUT_ROOT
OUT = os.path.join(OUT_ROOT, "figures_nature")

# ---- publication style (Nature-figure skill quick-start) ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelsize": 7.5,
})

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
GREY = "#767676"
RED = "#B64342"
TEAL = "#42949E"


def panel(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def save_fig(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    master = pd.read_csv(f"{RES}/infer_cv/master_ukb_predictions.csv")
    h = master[master["is_healthy"] == 1].copy()
    h["delta"] = h["age_pred_bc"] - h["age_gt"]

    ext1 = pd.read_csv(f"{RES}/infer_cv/predictions_ext_part1.csv")
    ext3 = pd.read_csv(f"{RES}/infer_cv/predictions_ext_part3.csv")
    unhealthy = pd.read_csv(f"{RES}/infer_cv/predictions_unhealthy.csv")

    ext1["mae"] = (ext1["age_pred_bc"] - ext1["age_gt"]).abs()
    ext3["mae"] = (ext3["age_pred_bc"] - ext3["age_gt"]).abs()
    unhealthy["mae"] = (unhealthy["age_pred_bc"] - unhealthy["age_gt"]).abs()

    # aggregate MAE for panel d
    d_rows = pd.DataFrame({
        "Cohort": ["Healthy (internal)", "Diseased (internal)",
                   "External cohort 1", "External cohort 2"],
        "MAE (yr)": [h["delta"].abs().mean(),
                     unhealthy["mae"].mean(),
                     ext1["mae"].mean(),
                     ext3["mae"].mean()],
        "n": [len(h), len(unhealthy), len(ext1), len(ext3)],
    })
    d_rows["se"] = [h["delta"].std() / np.sqrt(len(h)),
                    unhealthy["mae"].std() / np.sqrt(len(unhealthy)),
                    ext1["mae"].std() / np.sqrt(len(ext1)),
                    ext3["mae"].std() / np.sqrt(len(ext3))]

    r2 = np.corrcoef(h["age_gt"], h["age_pred_bc"])[0, 1] ** 2
    mae_all = h["delta"].abs().mean()
    pcc = h["age_gt"].corr(h["age_pred_bc"])

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.2))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.10,
                        wspace=0.45, hspace=0.55)

    # ---- a. observed vs predicted scatter ----
    ax = axes[0, 0]
    ax.scatter(h["age_gt"], h["age_pred_bc"], s=3, alpha=0.25,
               color=BLUE, edgecolors="none", rasterized=True)
    lo, hi = 44, 88
    ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=0.9, zorder=5)
    ax.text(0.05, 0.90, f"n = 40,649\nMAE = {mae_all:.2f} yr\nR\u00b2 = {r2:.2f}\n"
            f"Pearson r = {pcc:.2f}", transform=ax.transAxes, fontsize=6.5,
            va="top", ha="left")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (yr)")
    ax.set_ylabel("Predicted pancreatic age (yr)")
    ax.set_aspect("equal")
    panel(ax, "a")

    # ---- b. MAE by age decade ----
    ax = axes[0, 1]
    h["decade"] = pd.cut(h["age_gt"], bins=np.arange(40, 91, 5))
    grp = h.groupby("decade", observed=True)["delta"].agg(["mean", "std", "count"])
    grp["se"] = grp["std"] / np.sqrt(grp["count"])
    x = np.arange(len(grp))
    ax.plot(x, grp["mean"].abs(), color=BLUE, lw=1.4, marker="o", ms=3.2)
    ax.fill_between(x, (grp["mean"].abs() - grp["se"]),
                    (grp["mean"].abs() + grp["se"]), color=BLUE, alpha=0.15)
    labels = [f"{int(i.left)}-{int(i.right)}" for i in grp.index]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("|Age gap| (yr)")
    ax.set_xlabel("Age at imaging (yr)")
    panel(ax, "b")

    # ---- c. Bland-Altman ----
    ax = axes[1, 0]
    mean_age = (h["age_gt"] + h["age_pred_bc"]) / 2
    ax.scatter(mean_age, h["delta"], s=3, alpha=0.25, color=BLUE2,
               edgecolors="none", rasterized=True)
    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    bias = h["delta"].mean()
    sd = h["delta"].std()
    ax.axhline(bias, color=RED, lw=1.0)
    ax.axhline(bias + 1.96 * sd, color=RED, lw=0.7, ls=":")
    ax.axhline(bias - 1.96 * sd, color=RED, lw=0.7, ls=":")
    ax.text(0.03, 0.95, f"bias = {bias:+.2f} yr\n95% LoA = {bias-1.96*sd:+.2f}, "
            f"{bias+1.96*sd:+.2f}", transform=ax.transAxes, fontsize=6.5,
            va="top", ha="left")
    ax.set_xlabel("Mean of chronological and predicted age (yr)")
    ax.set_ylabel("Age gap (predicted \u2212 chronological, yr)")
    panel(ax, "c")

    # ---- d. MAE across cohorts ----
    ax = axes[1, 1]
    x = np.arange(len(d_rows))
    bars = ax.bar(x, d_rows["MAE (yr)"], width=0.6, color=BLUE,
                  yerr=d_rows["se"], error_kw=dict(elinewidth=0.8, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n = {n:,})" for c, n in zip(d_rows["Cohort"], d_rows["n"])],
                       fontsize=6.2)
    ax.set_ylabel("MAE (yr)")
    ax.set_ylim(0, max(d_rows["MAE (yr)"]) * 1.35)
    for b, v, se in zip(bars, d_rows["MAE (yr)"], d_rows["se"]):
        ax.text(b.get_x() + b.get_width() / 2, v + se + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.2)
    panel(ax, "d")

    save_fig(fig, "Fig1_model_performance")


if __name__ == "__main__":
    main()
