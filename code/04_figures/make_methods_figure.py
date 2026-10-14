#!/usr/bin/env python
"""Generate a professional study overview (methods figure) flowchart per prompt 2.
Output: figures_sci/methods_flowchart.pdf / .png / editable .svg
Style: Nature/Cell-style flat flowchart, three-color lanes, vector-editable."""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

R = OUT_ROOT
_ap = argparse.ArgumentParser(description="methods_flowchart (regenie/plink)")
_ap.add_argument("--method", choices=["regenie", "plink"], default="regenie")
_ap.add_argument("--out", default=None)
_args, _ = _ap.parse_known_args()
METHOD = _args.method
OUT = _args.out or os.path.join(R, f"figures_sci_{METHOD}")
ED = os.path.join(OUT, "editable")
os.makedirs(ED, exist_ok=True)
print(f"[make_methods_figure] method={METHOD} out={OUT}")

BLUE, BLUE2, TEAL, VIOLET, RED, NEUTRAL = "#0F4D92", "#3775BA", "#42949E", "#9A4D8E", "#B64342", "#CFCECE"
FILLS = {"blue": "#EAF1F9", "teal": "#E4F2F3", "violet": "#F1EBF6", "red": "#FBECEA"}
BORDERS = {"blue": BLUE, "teal": TEAL, "violet": VIOLET, "red": RED}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300,
})

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, title=None, lines=(), kind="blue", fs=7.5, title_fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                fc=FILLS[kind], ec=BORDERS[kind], lw=1.3,
                                transform=ax.transAxes, mutation_scale=1, zorder=2))
    top = y + h
    if title:
        ax.text(x + w / 2, top - 0.016, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=BORDERS[kind], transform=ax.transAxes, zorder=3)
        body_y = top - 0.036
    else:
        body_y = top - 0.018
    for i, line in enumerate(lines):
        ax.text(x + w / 2, body_y - i * 0.019, line, ha="center", va="center",
                fontsize=fs, color="#222222", transform=ax.transAxes, zorder=3)


def arrow(x1, y1, x2, y2, color="#555555", rad=0.0, lw=1.4, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, zorder=1, transform=ax.transAxes,
                                 connectionstyle=f"arc3,rad={rad}",
                                 linestyle="--" if dashed else "-"))


def divider(y):
    ax.plot([0.02, 0.98], [y, y], color="#CCCCCC", lw=1.0, transform=ax.transAxes)


# ────────────────────── Lane 1: imaging age model (blue) ──────────────────────
box(0.02, 0.81, 0.16, 0.14, "UKB abdominal MRI", ["(pancreas)", "n=40,649 healthy", "45–87 y"], "blue")
box(0.22, 0.81, 0.20, 0.14, "3D Swin-Transformer", ["age regression", "5-fold age-stratified CV", "(train/val/test ≈70/10/20)", "soft-label + bias correction"], "blue", fs=7)
box(0.46, 0.81, 0.18, 0.14, "APAG (Δ)", ["= Predicted age", "− Chronological age", "OOF: MAE≈2.6 y, R²≈0.85"], "blue", fs=7)
box(0.68, 0.81, 0.13, 0.14, "External set 1", ["(healthy)", "n=1,053", "MAE≈2.7 y"], "teal")
box(0.84, 0.81, 0.14, 0.14, "External set 2", ["(OGTT cohort)", "n=210", "MAE≈3.7 y"], "teal")
arrow(0.18, 0.88, 0.22, 0.88, BLUE)
arrow(0.42, 0.88, 0.46, 0.88, BLUE)
arrow(0.64, 0.90, 0.68, 0.90, TEAL, rad=0.0)
arrow(0.64, 0.86, 0.84, 0.86, TEAL, rad=0.0)

# ────────────────────── Lane 2: phenotype and genetics (teal/violet) ──────────────────────
box(0.02, 0.44, 0.30, 0.24, "Disease association", [
    "PSM 1:1 + OLS (age, age², sex, BMI)",
    "Diabetes (E10–E14): Δ +0.79 y, P=4×10⁻¹⁹",
    "Hypertension (BP≥140/90 or I10–I15): +0.34 y, P=4×10⁻¹⁸",
    "Obesity (BMI≥28 or E66): +0.21 y, P=3×10⁻⁷",
    "Healthy vs disease Δ −0.535 y, P=4×10⁻⁷⁰ (Welch t)",
], "teal", fs=7)
if METHOD == "regenie":
    gwas_title = "GWAS (REGENIE step-2)"
    gwas_lines = [
        "Full-sample N=53,673",
        "546 SNPs P<5×10⁻⁸ → 11 independent loci",
        "CTRB1/2 rs72802342 (P=9.4×10⁻¹⁶)",
        "ABO chr9 (P<5×10⁻⁸)",
    ]
    prs_lines = [
        "GWAS n≈37,571 → LD clump → 5 lead SNPs (β-weighted, ±CTRB1/2)",
        "Validation test β=+0.148, P=4×10⁻⁹, ΔR²=0.21%",
    ]
    mr_ivw = "IVW OR=1.195 [1.003–1.424], P=0.046"
else:
    gwas_title = "GWAS (PLINK2 linear)"
    gwas_lines = [
        "Full-sample N=53,673",
        "603 SNPs P<5×10⁻⁸ → 10 independent loci",
        "CTRB1/2 rs72802342 (P=1.2×10⁻²²)",
        "ABO rs550057 (P=2.3×10⁻¹⁷)",
    ]
    prs_lines = [
        "GWAS n≈37,571 → LD clump → 8 lead SNPs (β-weighted, ±CTRB1/2)",
        "Validation test β=+0.166, P=4×10⁻¹¹, ΔR²=0.27%",
    ]
    mr_ivw = "IVW OR=1.151 [1.041–1.273], P=0.006"

box(0.38, 0.50, 0.30, 0.18, gwas_title, gwas_lines, "violet", fs=7)
box(0.38, 0.43, 0.30, 0.06, "APAG PRS (discovery)", prs_lines, "blue", fs=6.5)
arrow(0.53, 0.50, 0.53, 0.49, VIOLET, dashed=True, rad=0.0)  # GWAS → PRS
arrow(0.17, 0.81, 0.17, 0.68, TEAL, rad=0.15)                  # APAG → disease
arrow(0.53, 0.81, 0.53, 0.68, VIOLET, rad=0.10)                # APAG → GWAS

# ────────────────────── Lane 3: causal and integrative analysis (red/violet) ──────────────────────
# Mediation: three-node path diagram
mx, my, mw, mh = 0.02, 0.06, 0.30, 0.34
box(mx, my, mw, mh, "Mediation: rs72802342 → APAG → T2D", [
    "a = −0.39   b = 0.085 (OR 1.09)",
    "ab = −0.033 (Bootstrap 95% CI excludes 0)",
    "→ complete mediation (18.5%)",
], "red", fs=7)
# Mediation internal nodes
node_w, node_h, node_y = 0.045, 0.045, my + 0.20
for nx, txt in [(mx + 0.05, "SNP"), (mx + 0.155, "APAG"), (mx + 0.26, "T2D")]:
    ax.add_patch(FancyBboxPatch((nx, node_y), node_w, node_h, boxstyle="round,pad=0.004",
                                fc="white", ec=RED, lw=1.0, transform=ax.transAxes, mutation_scale=1))
    ax.text(nx + node_w / 2, node_y + node_h / 2, txt, ha="center", va="center",
            fontsize=6.5, transform=ax.transAxes)
arrow(mx + 0.095, node_y + node_h / 2, mx + 0.155, node_y + node_h / 2, RED, lw=1.0)
arrow(mx + 0.20, node_y + node_h / 2, mx + 0.26, node_y + node_h / 2, RED, lw=1.0)
ax.plot([mx + 0.075, mx + 0.235], [node_y - 0.012, node_y - 0.012], color=RED, lw=0.8,
        transform=ax.transAxes)
ax.annotate("", xy=(mx + 0.235, node_y - 0.012), xytext=(mx + 0.075, node_y - 0.012),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8, mutation_scale=8))
ax.text(mx + 0.155, node_y - 0.022, "c′ = −0.15 (ns)", ha="center", fontsize=6,
        color=RED, transform=ax.transAxes)
ax.text(mx + 0.10, node_y + node_h + 0.008, "a", fontsize=6.5, color=RED, transform=ax.transAxes)
ax.text(mx + 0.215, node_y + node_h + 0.008, "b", fontsize=6.5, color=RED, transform=ax.transAxes)

box(0.38, 0.10, 0.26, 0.24, "Mendelian randomization", [
    "APAG → T2D (TwoSampleMR)",
    mr_ivw,
    "(FinnGen); DIAMANTE null",
    "Reverse MR null",
    "No pleiotropy / heterogeneity",
], "violet", fs=7)
box(0.67, 0.14, 0.16, 0.20, "PRS independence", [
    "T2D PRS OR=1.46 (P≈5×10⁻⁵³)",
    "APAG PRS: no independent",
    "contribution (LRT P≈0.6–0.8)",
], "blue", fs=7)
box(0.85, 0.14, 0.13, 0.20, "PheWAS", [
    f"{566 if METHOD=='regenie' else 568} ICD-10 phenotypes",
    "0 significant after",
    "Bonferroni / FDR",
], "red", fs=7)
# lane2 → lane3 arrows
arrow(0.17, 0.44, 0.17, 0.40, RED, rad=0.0)      # disease → mediation
arrow(0.53, 0.50, 0.53, 0.34, VIOLET, rad=0.0)   # GWAS → MR
arrow(0.53, 0.43, 0.75, 0.34, BLUE, rad=0.2)     # PRS → PRS independence
arrow(0.53, 0.50, 0.91, 0.34, RED, rad=-0.2)     # GWAS → PheWAS

# lane dividers
divider(0.77)
divider(0.40)

fig.text(0.5, 0.965, "Study design: imaging age model → phenotype & genetics → causal & integrative analysis",
         ha="center", fontsize=12, fontweight="bold")
fig.tight_layout(pad=0.6)

# Text collision self-check (excluding the figure title and intentional small labels)
fig.canvas.draw()
ren = fig.canvas.get_renderer()
from matplotlib.text import Text
texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
hits = 0
for i, a in enumerate(texts):
    for b in texts[i + 1:]:
        if a.get_window_extent(ren).overlaps(b.get_window_extent(ren)):
            hits += 1
print(f"text-overlap hits: {hits}")
if hits:
    print("  pairs:", [(x.get_text()[:22], y.get_text()[:22]) for i, x in enumerate(texts)
                       for y in texts[i+1:] if x.get_window_extent(ren).overlaps(y.get_window_extent(ren))][:6])

for name in ["methods_flowchart.pdf", "methods_flowchart.png", "editable/methods_flowchart.svg"]:
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.05)
print("saved", "methods_flowchart.pdf / .png / editable .svg")
