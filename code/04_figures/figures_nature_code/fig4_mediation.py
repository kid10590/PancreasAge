#!/usr/bin/env python
"""Figure 4 | Genetic mediation of the CTRB1/2 locus effect on type 2 diabetes
through the pancreatic age gap.

Panels:
  a. Mediation path diagram (rs72802342 -> delta -> T2D) with coefficients
     (a, b, c, c', OR), n = 4,984.
  b. Bootstrap distribution (5,000 iterations) of the indirect effect ab,
     with 95% percentile CI.
  c. PRS-based mediation: genome-wide APAG PRS (all 8 clumped SNPs) and
     CTRB1/2-excluded PRS, indirect effect ab with bootstrap 95% CI.

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
RED = "#B64342"
GREY = "#767676"
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

    boot = pd.read_csv(f"{RES}/Disease_Phenotype_Analysis/Mediation/mediation_bootstrap.csv")
    ab = boot["indirect_effect"].values

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.7))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.16,
                        wspace=0.55)

    # ---------- a. path diagram ----------
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, text, w=2.6, h=1.6, fc="#EEF2F8", ec=BLUE, fs=7):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.12",
                           facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, label=None, dx=0.2, color=GREY, lw=1.4,
              label_pos=None, fs=6.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=9))
        if label:
            lx, ly = label_pos if label_pos else ((x1+x2)/2, (y1+y2)/2 + dx)
            ax.text(lx, ly, label, ha="center", va="center", fontsize=fs,
                    color=color)

    box(1.6, 8.2, "rs72802342\n(A allele)")
    box(5.0, 5.0, "Pancreatic\nage gap (\u03b4)")
    box(8.4, 8.2, "Type 2\ndiabetes")
    # c (total)
    arrow(2.9, 8.2, 7.1, 8.2, label="c = \u22120.180\nOR = 0.835\nP = 0.024",
          dx=0.15, label_pos=(5.0, 9.15), color=RED)
    # a
    arrow(2.9, 7.45, 4.35, 5.8, label="a = \u22120.394\nP = 1.7\u00d710\u207b\u00b3",
          dx=0.15, label_pos=(3.2, 6.35), color=BLUE)
    # b
    arrow(5.65, 5.8, 7.1, 7.45, label="b = 0.085\nOR = 1.088\nP = 4.8\u00d710\u207b\u00b2\u2070",
          dx=0.15, label_pos=(7.0, 6.15), color=BLUE)
    # c'
    arrow(3.0, 8.0, 7.0, 7.5, label="c\u2032 = \u22120.150\nP = 0.062 (n.s.)",
          dx=-0.15, label_pos=(5.1, 7.15), color=GREY, lw=1.0)
    ax.text(5.0, 0.6,
            "Sobel P = 2.9\u00d710\u207b\u00b3 | ab = \u22120.033\n"
            "Bootstrap 95% CI [\u22120.056, \u22120.013]\n"
            "\u21d2 Full mediation (c\u2032 CI includes 0)\nn = 4,984",
            ha="center", fontsize=6.2, color="#333333")
    panel(ax, "a")

    # ---------- b. bootstrap distribution of ab ----------
    ax = axes[1]
    lo, hi = np.percentile(ab, [2.5, 97.5])
    med = np.median(ab)
    n, bins, patches = ax.hist(ab, bins=60, color=BLUE, alpha=0.7,
                               edgecolor="none")
    for p, b in zip(patches, bins[:-1]):
        if b < lo or b > hi:
            p.set_facecolor("#D9E2F2")
    ax.axvline(lo, color=RED, ls="--", lw=0.9)
    ax.axvline(hi, color=RED, ls="--", lw=0.9)
    ax.axvline(0, color=GREY, ls=":", lw=0.8)
    ax.text(0.03, 0.95, f"median = {med:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left")
    ax.set_xlabel("Indirect effect ab (SNP \u2192 \u03b4 \u2192 T2D)")
    ax.set_ylabel("Bootstrap count")
    panel(ax, "b")

    # ---------- c. PRS mediation ----------
    ax = axes[2]
    prs_rows = [
        ("PRS (8 SNPs)", 0.0190, 0.0111, 0.0278, BLUE),
        ("PRS \u2212 CTRB1/2", 0.0184, 0.0105, 0.0272, TEAL),
    ]
    y = np.arange(len(prs_rows))[::-1]
    for yi, (lab, e, l, h, c) in zip(y, prs_rows):
        ax.plot([l, h], [yi, yi], color=c, lw=2.0)
        ax.plot(e, yi, marker="o", ms=5, color=c)
        ax.text(h + 0.001, yi, f"  {e:.3f} [{l:.3f}, {h:.3f}]",
                va="center", ha="left", fontsize=6.2)
    ax.axvline(0, color=GREY, ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in prs_rows])
    ax.set_xlabel("Indirect effect ab (PRS \u2192 \u03b4 \u2192 T2D)")
    ax.set_xlim(-0.005, 0.045)
    ax.text(0.03, 0.95, "Bootstrap 95% CI", transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left")
    panel(ax, "c")

    save_fig(fig, "Fig4_mediation")


if __name__ == "__main__":
    main()
