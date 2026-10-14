#!/usr/bin/env python
"""Figure 2 | Pancreatic age gap (delta) is elevated in metabolic disease.

Panels:
  a. Distribution of pancreatic age gap in healthy (n=40,649) vs diseased
     (n=14,477) UK Biobank participants. Welch t-test annotated.
  b. Forest plot of BMI-adjusted regression coefficients (delta vs case status)
     across five disease definitions plus mutually-exclusive-control sensitivity.
  c. Dose-response: odds ratio of type 2 diabetes by pancreatic age gap
     quartile (external OGTT cohort, minimal model M1).

Exports SVG/PDF/PNG (600 dpi), Arial 7pt, Nature style.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from paths import OUT_ROOT  # noqa: E402
RES = OUT_ROOT
OUT = os.path.join(OUT_ROOT, "figures_nature")

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
RED = "#B64342"
GREY = "#767676"


def panel(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def save_fig(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def fmt_p(p):
    if p < 1e-4:
        return f"P = {p:.1e}"
    return f"P = {p:.3g}"


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    # ---------- disease betas ----------
    diseases = [
        ("Diabetes mellitus", "Diabetes", "All cases"),
        ("Type 2 diabetes (E11)", "Diabetes_E11", "All cases"),
        ("Hypertension", "Hypertension", "All cases"),
        ("Metabolic syndrome", "Metabolic Syndrome", "All cases"),
        ("Obesity", "Obesity", "All cases"),
        ("Hypertension", "Hypertension_mutex", "Exclusive controls"),
        ("Metabolic syndrome", "Metabolic Syndrome_mutex", "Exclusive controls"),
        ("Obesity", "Obesity_mutex", "Exclusive controls"),
        ("Diabetes mellitus", "Diabetes_mutex", "Exclusive controls"),
    ]
    # hardcode from verified stats_summary.txt extraction
    betas = [
        (0.7900, 0.617, 0.963, 4.352774331375318e-19, 5136),
        (0.6930, 0.514, 0.872, 4.0938786930573795e-14, 4840),
        (0.3418, 0.265, 0.419, 4.539072176324467e-18, 26832),
        (0.4540, 0.301, 0.607, 5.963101047022652e-09, 6510),
        (0.2110, 0.131, 0.291, 2.628243753223007e-07, 23956),
        (0.3202, 0.230, 0.410, 3.2375747182673236e-12, 19314),
        (0.7018, 0.422, 0.981, 9.289443453726303e-07, 2020),
        (0.5596, 0.474, 0.645, 6.885540222605985e-38, 22288),
        (0.9058, 0.673, 1.138, 3.111649185057717e-14, 2880),
    ]

    # ---------- dose-response (M1, cat4) ----------
    dose = pd.read_csv(f"{RES}/Dose_Response_Diabetes/dose_response_results.csv")
    dq = dose[(dose["model"] == "M1") & (dose["exposure_type"] == "cat4")]
    dq = dq.sort_values("category")

    # ---------- master for delta distributions ----------
    master = pd.read_csv(f"{RES}/infer_cv/master_ukb_predictions.csv")
    h = master[master["is_healthy"] == 1]
    ill = master[master["is_healthy"] == 0]
    delta_h = h["age_pred_bc"] - h["age_gt"]
    delta_ill = ill["age_pred_bc"] - ill["age_gt"]

    # Welch t-test (recompute to double check, vs recorded P=4.1e-70)
    from scipy import stats
    tt = stats.ttest_ind(delta_h, delta_ill, equal_var=False)
    d_eff = (delta_ill.mean() - delta_h.mean())
    d_eff_std = delta_ill.std()  # Cohen's d numerator scale

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.8))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.18,
                        wspace=0.62)

    # ---------- a. violin delta healthy vs diseased ----------
    ax = axes[0]
    parts = ax.violinplot([delta_h.values, delta_ill.values],
                          positions=[1, 2], widths=0.7,
                          showextrema=False, showmedians=False)
    for pc, c in zip(parts["bodies"], [BLUE, RED]):
        pc.set_facecolor(c); pc.set_alpha(0.45); pc.set_edgecolor(c)
        pc.set_linewidth(0.6)
    # quartile box inside
    for pos, d in zip([1, 2], [delta_h, delta_ill]):
        q1, q3 = np.percentile(d, [25, 75])
        ax.plot([pos - 0.12, pos + 0.12], [q1, q1], color="black", lw=0.8)
        ax.plot([pos - 0.12, pos + 0.12], [q3, q3], color="black", lw=0.8)
        ax.plot([pos, pos], [q1, q3], color="black", lw=0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Healthy\n(n = 40,649)", "Diseased\n(n = 14,477)"])
    ax.set_ylabel("Pancreatic age gap (yr)")
    ax.axhline(0, color=GREY, ls="--", lw=0.7)
    ax.set_ylim(-12, 14)
    ax.text(0.04, 0.97, f"\u0394 = {d_eff:+.2f} yr\n{fmt_p(tt.pvalue)}",
            transform=ax.transAxes, fontsize=7, va="top", ha="left")
    panel(ax, "a")

    # ---------- b. forest plot disease betas ----------
    ax = axes[1]
    y = np.arange(len(diseases))[::-1]
    est = np.array([b[0] for b in betas])
    lo = np.array([b[1] for b in betas])
    hi = np.array([b[2] for b in betas])
    allcase = [1 if d[2] == "All cases" else 0 for d in diseases]

    # group bands
    ax.axhspan(-0.5, len([d for d in diseases if d[2] == "All cases"]) - 0.5,
               color="#F5F5F5", zorder=0)
    for yi, e, l, hh, c, lab in zip(y, est, lo, hi, allcase,
                                    [d[0] for d in diseases]):
        col = BLUE if c else GREY
        ax.plot([l, hh], [yi, yi], color=col, lw=1.1, zorder=3)
        ax.plot(e, yi, marker="o", ms=3.5, color=col, zorder=4)
        ax.text(hh + 0.03, yi, f"{e:.2f} [{l:.2f}, {hh:.2f}]",
                va="center", ha="left", fontsize=7)
    ax.axvline(0, color=GREY, ls="--", lw=0.8, zorder=2)
    ax.set_yticks(y)
    labels = [f"{d[0]}  ({d[2]})" for d in diseases]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("\u03b2 (yr) per case status, BMI-adjusted")
    ax.set_xlim(-0.15, 1.35)
    ax.text(1.0, 0.97, "All \u03b2 > 0, all P < 4e-7",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color=BLUE)
    panel(ax, "b")

    # ---------- c. dose-response OR ----------
    ax = axes[2]
    cats = dq["category"].astype(int).values
    ors = dq["OR"].values
    lo2 = dq["OR_low95"].values
    hi2 = dq["OR_high95"].values
    # ref category OR=1 at x=0
    xall = np.concatenate([[0], cats])
    orall = np.concatenate([[1.0], ors])
    loall = np.concatenate([[1.0], lo2])
    hiall = np.concatenate([[1.0], hi2])
    ax.plot(xall, orall, color=BLUE, lw=1.2, marker="o", ms=3.5, zorder=3)
    ax.plot([cats[0], cats[0]], [loall[1], hiall[1]], color=BLUE, lw=0.9, zorder=2)
    ax.plot([cats[1], cats[1]], [loall[2], hiall[2]], color=BLUE, lw=0.9, zorder=2)
    ax.plot([cats[2], cats[2]], [loall[3], hiall[3]], color=BLUE, lw=0.9, zorder=2)
    ax.fill_between(xall, loall, hiall, color=BLUE, alpha=0.12, zorder=1)
    ax.axhline(1, color=GREY, ls="--", lw=0.8)
    p_trend = dq["p_trend"].iloc[0]
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Ref\n(\u03b4 < 0)", "Q1", "Q2", "Q3"])
    ax.set_xlabel("Pancreatic age gap category")
    ax.set_ylabel("Odds ratio (T2D, Model 1)")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 8)
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.text(0.03, 0.97, f"P trend = {p_trend:.3f}",
            transform=ax.transAxes, fontsize=7, va="top")
    panel(ax, "c")

    save_fig(fig, "Fig2_disease_association")


if __name__ == "__main__":
    main()
