#!/usr/bin/env python
"""Figure 3 | Genome-wide association study of the pancreatic age gap (delta).

Panels:
  a. Manhattan plot (full cohort, n=53,673), genome-wide significance line
     P < 5e-8. Lead loci labelled (CTRB1/2 chr16, ABO chr9).
  b. QQ plot with genomic inflation lambda.
  c. Regional association at the CTRB1/2 lead locus (rs72802342, chr16).
  d. Effect sizes (beta, A1) of the 8 independent lead SNPs.

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
GWAS = f"{RES}/gwas_plink_results"

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
SIGNIF = 5e-8


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

    # ---- load all chromosome summary stats (only needed cols) ----
    chrs = []
    for chrom in range(1, 23):
        f = f"{GWAS}/gwas_chr{chrom}.catb_delta_with_t1_bc_cole.glm.linear"
        if not os.path.exists(f):
            print(f"missing {f}"); continue
        c = pd.read_csv(f, sep="\t", usecols=["#CHROM", "POS", "ID", "P", "BETA", "A1"],
                        low_memory=False)
        c = c[["#CHROM", "POS", "ID", "P", "BETA", "A1"]]
        c.columns = ["CHROM", "POS", "ID", "P", "BETA", "A1"]
        c["P"] = pd.to_numeric(c["P"], errors="coerce")
        c["BETA"] = pd.to_numeric(c["BETA"], errors="coerce")
        chrs.append(c)
    gwas = pd.concat(chrs, ignore_index=True)
    gwas = gwas[gwas["P"].notna() & (gwas["P"] > 0)]
    print(f"total variants: {len(gwas):,}")

    # chromosome boundaries for Manhattan x-axis
    chr_lens = gwas.groupby("CHROM")["POS"].max()
    gaps = 5e6
    offset = 0.0
    offsets = {}
    for chrom in range(1, 23):
        offsets[chrom] = offset
        offset += chr_lens.get(chrom, 0) + gaps
    gwas["pos_cum"] = gwas["POS"] + gwas["CHROM"].map(offsets)
    gwas["neglogp"] = -np.log10(gwas["P"])

    # lead SNPs (8 independent, from PRS build)
    lead = pd.read_csv(f"{RES}/prs_analysis/prs_build/lead_full.csv")
    lead.columns = [c.strip() for c in lead.columns]

    fig = plt.figure(figsize=(7.09, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          left=0.09, right=0.97, top=0.96, bottom=0.08,
                          wspace=0.5, hspace=0.75)

    # ---------- a. Manhattan ----------
    ax = fig.add_subplot(gs[0, 0])
    chrom_ticks, chrom_labels = [], []
    for chrom in range(1, 23):
        sub = gwas[gwas["CHROM"] == chrom]
        if sub.empty:
            continue
        col = BLUE if chrom % 2 == 1 else BLUE2
        ax.scatter(sub["pos_cum"], sub["neglogp"], s=1.8, color=col,
                   alpha=0.55, edgecolors="none", linewidths=0, rasterized=True)
        mid = offsets[chrom] + chr_lens.get(chrom, 0) / 2
        chrom_ticks.append(mid); chrom_labels.append(str(chrom))
    ax.axhline(-np.log10(SIGNIF), color=RED, ls="--", lw=0.9)
    ax.axhline(-np.log10(5e-8 / 604), color=GREY, ls=":", lw=0.8)
    ax.text(0.99, -np.log10(SIGNIF) + 0.3, "P = 5\u00d710\u207b\u2078",
            transform=ax.transData, ha="right", va="bottom", fontsize=6,
            color=RED)
    # label top loci
    top = gwas.nsmallest(6, "P")
    for _, r in top.iterrows():
        lab = r["ID"]
        ax.annotate(lab, xy=(r["pos_cum"], r["neglogp"]),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=5.6, ha="center", color="black")
    ax.set_xticks(chrom_ticks); ax.set_xticklabels(chrom_labels, fontsize=5.5)
    ax.set_xlim(-gaps, offset)
    ax.set_ylim(0, gwas["neglogp"].max() * 1.12)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("$-\log_{10}(P)$")
    panel(ax, "a")

    # ---------- b. QQ ----------
    ax = fig.add_subplot(gs[0, 1])
    p = gwas["P"].sort_values().values
    n = len(p)
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    obs = -np.log10(p)
    # inflation lambda from median
    lam = np.median(p * 0 + np.where(p < 0.5, np.nan, 1))  # placeholder
    lam = np.median((np.where(p == 0, 1e-300, p)))
    # chi-sq based lambda
    from scipy.stats import chi2, norm
    chi2_obs = norm.ppf(1 - p / 2) ** 2
    lam = np.median(chi2_obs) / chi2.ppf(0.5, 1)
    # thin for plotting
    idx = np.unique(np.round(np.linspace(0, n - 1, 20000)).astype(int))
    ax.scatter(exp[idx], obs[idx], s=2, color=BLUE, alpha=0.5,
               edgecolors="none", rasterized=True)
    maxv = max(exp.max(), obs.max()) + 0.5
    ax.plot([0, maxv], [0, maxv], color=GREY, ls="--", lw=0.9)
    ax.set_xlim(0, maxv); ax.set_ylim(0, maxv)
    ax.set_xlabel("Expected $-\log_{10}(P)$")
    ax.set_ylabel("Observed $-\log_{10}(P)$")
    ax.text(0.05, 0.95, f"\u03bb = {lam:.2f}", transform=ax.transAxes,
            fontsize=7, va="top", ha="left")
    panel(ax, "b")

    # ---------- c. regional CTRB1/2 ----------
    ax = fig.add_subplot(gs[1, 0])
    lead16 = lead[(lead["CHROM"] == 16)]
    lead_snp = lead16.sort_values("P").iloc[0]
    center = lead_snp["POS"]
    win = 500000
    c16 = gwas[gwas["CHROM"] == 16]
    region = c16[(c16["POS"] >= center - win) & (c16["POS"] <= center + win)].copy()
    region = region[region["P"].notna() & (region["P"] > 0)]
    region["neglogp"] = -np.log10(region["P"])
    region["color"] = np.where(region["ID"] == lead_snp["ID"], RED,
                               np.where(region["P"] < 5e-8, BLUE2, GREY))
    ax.scatter(region["POS"], region["neglogp"], s=4,
               c=region["color"], alpha=0.8, edgecolors="none", rasterized=True)
    ax.scatter([center], [-np.log10(lead_snp["P"])], s=16, marker="*",
               color=RED, edgecolors="black", linewidths=0.4, zorder=5)
    ax.axhline(-np.log10(SIGNIF), color=RED, ls="--", lw=0.7)
    ax.set_xlabel(f"Chr16 position (Mb)")
    ax.set_ylabel("$-\log_{10}(P)$")
    ax.set_xticks(np.arange(center - win, center + win + 1, 250000))
    ax.set_xticklabels([f"{v/1e6:.2f}" for v in
                        np.arange(center - win, center + win + 1, 250000)])
    ax.text(0.03, 0.97, f"rs72802342\nP = 1.2\u00d710\u207b\u00b2\u00b2",
            transform=ax.transAxes, fontsize=6.2, va="top", ha="left")
    panel(ax, "c")

    # ---------- d. lead SNP effects ----------
    ax = fig.add_subplot(gs[1, 1])
    lead_s = lead.sort_values("P")
    y = np.arange(len(lead_s))[::-1]
    beta = lead_s["BETA"].values
    color = [RED if ":" in str(i) or i == "rs72802342" else BLUE for i in
             lead_s["ID"]]
    ax.barh(y, beta, height=0.62, color=color, edgecolor="black", linewidth=0.4)
    for yi, b, snp in zip(y, beta, lead_s["ID"]):
        ax.text(b + (0.01 if b > 0 else -0.01), yi,
                f"{snp}  \u03b2 = {b:.2f}", va="center",
                ha="left" if b > 0 else "right", fontsize=5.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("\u03b2 (yr) per effect allele")
    ax.set_ylim(-0.6, len(lead_s) - 0.4)
    panel(ax, "d")

    save_fig(fig, "Fig3_gwas")


if __name__ == "__main__":
    main()
