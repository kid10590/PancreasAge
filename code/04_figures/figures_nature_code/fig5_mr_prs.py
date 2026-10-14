#!/usr/bin/env python
"""Figure 5 | Genetic evidence: two-sample Mendelian randomization and
polygenic risk score analyses of the pancreatic age gap and type 2 diabetes.

Panels:
  a. Forest plot of MR estimates (OR per SD of APAG on T2D) across three
     T2D GWAS outcomes and five methods (IVW highlighted).
  b. APAG PRS prediction of the pancreatic age gap: beta, R2 base/full and
     delta-R2 in discovery vs independent test cohorts.
  c. PRS independence: OR per SD of APAG PRS and T2D PRS for T2D, and the
     lack of AUC gain from adding APAG PRS to T2D-PRS models.

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
TEAL = "#42949E"
VIO = "#9A4D8E"


def panel(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def save_fig(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


METHOD_ORDER = ["Inverse variance weighted",
                "Weighted median",
                "Weighted mode",
                "MR Egger",
                "Inverse variance weighted (multiplicative random effects)"]
METHOD_LABEL = {"Inverse variance weighted": "IVW",
                "Weighted median": "Weighted median",
                "Weighted mode": "Weighted mode",
                "MR Egger": "MR Egger",
                "Inverse variance weighted (multiplicative random effects)": "IVW (MRE)"}


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    # ---------- MR main results ----------
    mr_rows = []
    for trait, label in [("finn-b-T2D", "FinnGen T2D"),
                         ("ieu-a-23", "DIAMANTE T2D"),
                         ("ebi-a-GCST007515", "Mahajan 2018 T2D")]:
        df = pd.read_csv(f"{RES}/mr/{trait}/mr_main_results_{trait}.csv")
        df.columns = [c.strip('"') for c in df.columns]
        for _, r in df.iterrows():
            mr_rows.append(dict(outcome=label, method=r["method"].strip('"'),
                                or_=r["or"], lo=r["or_lci95"], hi=r["or_uci95"],
                                p=r["pval"], nsnp=r["nsnp"]))
    mr = pd.DataFrame(mr_rows)

    # ---------- PRS validation ----------
    prs = pd.read_csv(f"{RES}/prs_analysis/prs_apag_validation.csv")
    prs = prs[prs["split"].isin(["discovery", "test"])]

    # ---------- PRS diabetes association ----------
    assoc = pd.read_csv(f"{RES}/prs_analysis/diabetes_prs_association.csv")
    assoc = assoc[assoc["model"] == "full"]

    # ---------- independence ----------
    indep = pd.read_csv(f"{RES}/prs_analysis/prs_diabetes_independence.csv")

    fig = plt.figure(figsize=(7.09, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0],
                          left=0.15, right=0.97, top=0.95, bottom=0.09,
                          wspace=0.6, hspace=0.8)

    # ---------- a. MR forest ----------
    ax = fig.add_subplot(gs[0, :])
    # build y order: methods top, outcomes below (grouped)
    labels, ors, los, his, pvals, cols = [], [], [], [], [], []
    y = []
    yi = 0
    for method in METHOD_ORDER:
        for outcome in ["FinnGen T2D", "DIAMANTE T2D", "Mahajan 2018 T2D"]:
            r = mr[(mr["method"] == method) & (mr["outcome"] == outcome)]
            if r.empty:
                continue
            row = r.iloc[0]
            labels.append(f"{METHOD_LABEL.get(method, method)} \u00b7 {outcome}"
                          f"  (n SNP = {row['nsnp']})")
            ors.append(row["or_"]); los.append(row["lo"]); his.append(row["hi"])
            pvals.append(row["p"])
            y.append(yi)
            cols.append(RED if method == "Inverse variance weighted" else GREY)
            yi += 1
    y = np.array(y)[::-1]
    ax.axvspan(-0.5, len([m for m in METHOD_ORDER]) * 3 - 2.5,
               color="#F7F7F7", zorder=0)
    for yy, e, lo, hi, c in zip(y, ors, los, his, cols):
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.2, zorder=3)
        ax.plot(e, yy, marker="o", ms=4, color=c, zorder=4)
        ax.text(hi + 0.03, yy,
                f"{e:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="left",
                fontsize=6.5)
    ax.axvline(1, color=GREY, ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("OR per SD of pancreatic age gap on type 2 diabetes")
    ax.set_xlim(0.35, 2.6)
    ax.text(0.99, 0.97, "FinnGen: IVW OR = 1.15, P = 0.006",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=RED)
    panel(ax, "a")

    # ---------- b. PRS -> delta ----------
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(prs))
    beta = prs["beta_APAGPRS"].values
    se = prs["se"].values
    dr2 = prs["delta_R2"].values
    bars = ax.bar(x, beta, width=0.55, color=BLUE,
                  yerr=se, error_kw=dict(elinewidth=0.9, capsize=2.5))
    for xi, b, e, d in zip(x, beta, se, dr2):
        ax.text(xi, b + e + 0.005, f"\u03b2 = {b:.3f}\nP < 1e-10\n\u0394R\u00b2 = {d*100:.2f}%",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Discovery\n(n = 37,571)", "Independent test\n(n = 16,102)"])
    ax.set_ylabel("PRS \u03b2 on age gap (yr/SD)")
    ax.set_ylim(0, 0.4)
    panel(ax, "b")

    # ---------- c. PRS independence ----------
    ax = fig.add_subplot(gs[1, 1])
    r1 = indep[indep["APAG_PRS"] == "APAG_PRS_full"].iloc[0]
    r2 = indep[indep["APAG_PRS"] == "APAG_PRS_without_CTRB12"].iloc[0]
    rows = [
        ("T2D PRS (DIAMANTE)\nbase model", r1["T2D_OR_perSD_M1"], None, None, VIO),
        ("+ APAG PRS (full)", r1["APAG_OR_perSD_M2"],
         r1["APAG_OR_low95_M2"], r1["APAG_OR_high95_M2"], BLUE),
        ("T2D PRS (FinnGen)\nbase model", r2["T2D_OR_perSD_M1"], None, None, VIO),
        ("+ APAG PRS (no CTRB1/2)", r2["APAG_OR_perSD_M2"],
         r2["APAG_OR_low95_M2"], r2["APAG_OR_high95_M2"], TEAL),
    ]
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, e, lo, hi, c) in zip(yy, rows):
        if lo is None:
            ax.plot(e, yi, marker="o", ms=5, color=c)
            ax.text(e + 0.01, yi, f"{e:.2f}  (P < 1e-50)", va="center",
                    ha="left", fontsize=6.5)
        else:
            ax.plot([lo, hi], [yi, yi], color=c, lw=1.8)
            ax.plot(e, yi, marker="o", ms=5, color=c)
            ax.text(hi + 0.01, yi, f"{e:.2f} [{lo:.2f}, {hi:.2f}]",
                    va="center", ha="left", fontsize=6.5)
    ax.axvline(1, color=GREY, ls="--", lw=0.9)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("OR per SD")
    ax.set_xlim(0.8, 1.7)
    ax.text(0.03, 0.97, "APAG PRS adds no risk beyond T2D-PRS\n"
            "LRT P = 0.53\u20130.80, \u0394AUC \u2248 0",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left")
    panel(ax, "c")

    save_fig(fig, "Fig5_mr_prs")


if __name__ == "__main__":
    main()
