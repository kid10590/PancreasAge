#!/usr/bin/env python
"""Figure 6 | PheWAS of the APAG polygenic risk score in the independent test
cohort (n = 16,102; 568 ICD-10 three-character phenotypes, case >= 20).

Panels:
  a. Manhattan-style PheWAS plot by ICD-10 chapter; Bonferroni line
     (P < 0.05/568) and FDR significance threshold shown.
  b. Top-ten phenotypes by uncorrected P with OR and 95% CI.

Note: no phenotype survives Bonferroni or FDR correction (P_fdr_min = 0.81).

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


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv(f"{RES}/PheWAS_test_independent/phewas_results.csv")
    df = df[df["P"].notna() & (df["P"] > 0)].copy()
    df["-log10P"] = -np.log10(df["P"])
    n_pheno = len(df)

    # ---------- chapter layout for x-axis ----------
    chapters = df["chapter"].unique()
    offset = 0.0
    offsets, mids = {}, []
    for ch in chapters:
        sub = df[df["chapter"] == ch]
        offsets[ch] = offset
        offset += len(sub) + 2
        mids.append((offsets[ch] + offset - 2) / 2)
    df["x"] = df["chapter"].map(offsets) + np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.9),
                             gridspec_kw=dict(width_ratios=[2.4, 1.4]))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18,
                        wspace=0.5)

    # ---------- a. PheWAS Manhattan ----------
    ax = axes[0]
    bonf = 0.05 / n_pheno
    fdr_max_sig = df["P_fdr"].min()
    for ch, c in zip(chapters, plt.cm.tab20(np.linspace(0, 1, len(chapters)))):
        sub = df[df["chapter"] == ch]
        ax.scatter(sub["x"], sub["-log10P"], s=5, color=c, alpha=0.75,
                   edgecolors="none", rasterized=True)
    ax.axhline(-np.log10(bonf), color=RED, ls="--", lw=0.9)
    ax.text(0.99, -np.log10(bonf) + 0.02, f"Bonferroni\n(P = {bonf:.2e})",
            transform=ax.transData, ha="right", va="bottom", fontsize=6.5,
            color=RED)
    ax.set_xticks(mids)
    ax.set_xticklabels([ch.split(":")[0] for ch in chapters],
                       rotation=60, ha="right", fontsize=6.5)
    ax.set_xlim(-1, offset)
    ax.set_ylabel("$-\log_{10}(P)$")
    ax.set_xlabel("ICD-10 chapter")
    ax.set_ylim(0, df["-log10P"].max() * 1.15)
    ax.text(0.03, 0.97, f"568 phenotypes\nmin FDR P = {fdr_max_sig:.2f}",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left")
    panel(ax, "a")

    # ---------- b. top phenotypes ----------
    ax = axes[1]
    top = df.nsmallest(10, "P").sort_values("P", ascending=False)
    y = np.arange(len(top))
    lo = np.exp(np.log(top["OR"]) - 1.96 * top["SE"])
    hi = np.exp(np.log(top["OR"]) + 1.96 * top["SE"])
    for yi, e, l, h, p in zip(y, top["OR"], lo, hi, top["P"]):
        col = BLUE if p < 0.01 else GREY
        ax.plot([l, h], [yi, yi], color=col, lw=1.2)
        ax.plot(e, yi, marker="o", ms=3.5, color=col)
        ax.text(h + 0.02, yi, f"{e:.2f}  P = {p:.3g}",
                va="center", ha="left", fontsize=6)
    ax.axvline(1, color=GREY, ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} ({t_ch.split(':')[0]})"
                        for t, t_ch in zip(top["phenotype"], top["chapter"])],
                       fontsize=6)
    ax.set_xlabel("Odds ratio (per SD APAG PRS)")
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.text(0.03, 0.97, "Top 10 by uncorrected P",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left")
    panel(ax, "b")

    save_fig(fig, "Fig6_phewas")


if __name__ == "__main__":
    main()
