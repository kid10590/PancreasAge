#!/usr/bin/env python3
"""
make_sci_figures.py — draw 10 SCI main figures in the scientific-figure-making house style
Output: <OUT_ROOT>/figures_sci/
      Each figure is exported as both PDF (vector) + PNG (300 DPI)
"""
import os
import sys
import glob
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text

warnings.filterwarnings("ignore")

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

R = OUT_ROOT
_ap = argparse.ArgumentParser(description="make_sci_figures (regenie / plink)")
_ap.add_argument("--method", choices=["regenie", "plink"], default="regenie",
                 help="GWAS/MR/PRS/PheWAS data source")
_ap.add_argument("--out", default=None, help="output directory, default figures_sci_<method>")
_args, _ = _ap.parse_known_args()
METHOD = _args.method
OUT = _args.out or os.path.join(R, f"figures_sci_{METHOD}")
ED = os.path.join(OUT, "editable")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ED, exist_ok=True)
print(f"[make_sci_figures] method={METHOD} out={OUT}")

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE",
    "green_2": "#AADCA9", "green_3": "#8BCF8B", "red_1": "#F6CFCB",
    "red_2": "#E9A6A1", "red_strong": "#B64342", "neutral": "#CFCECE",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}
BLUE, BLUE2, GREEN, RED, TEAL, VIOLET, NEUTRAL = (
    PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["green_3"],
    PALETTE["red_strong"], PALETTE["teal"], PALETTE["violet"], PALETTE["neutral"])


def apply_style(font=10):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": font,
        "axes.edgecolor": "#333333", "axes.linewidth": 1.4,
        "axes.labelsize": font, "axes.titlesize": font,
        "xtick.labelsize": font - 1, "ytick.labelsize": font - 1,
        "legend.fontsize": font - 1, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    for s in ("top", "right"):
        plt.rcParams[f"axes.spines.{s}"] = False


def _report_text_overlaps(fig, name):
    """Report avoidable annotation/title collisions in the final layout."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    collisions = []
    for i, left in enumerate(texts):
        # Tick labels are intentionally dense in a few genomic axes; do not flag them.
        if left in left.axes.get_xticklabels() + left.axes.get_yticklabels() if left.axes else False:
            continue
        for right in texts[i + 1:]:
            if left.axes is not right.axes or right.axes is None:
                continue
            if right in right.axes.get_xticklabels() + right.axes.get_yticklabels():
                continue
            if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                collisions.append((left.get_text().replace("\n", " "), right.get_text().replace("\n", " ")))
    if collisions:
        labels = "; ".join(f"'{a}'/'{b}'" for a, b in collisions[:3])
        print(f"  [layout warning] {name}: {labels}")


def save(fig, name, panel_names=None):
    """Save a combined figure and vector/raster crops for each selectable panel.
    PDF (print) + PNG (300 dpi) + editable SVG (text kept as text)."""
    fig.tight_layout(pad=1.2)
    _report_text_overlaps(fig, name)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.05, dpi=300)
    fig.savefig(os.path.join(ED, name + ".svg"), bbox_inches="tight", pad_inches=0.05)
    if panel_names:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, panel_name in zip(np.ravel(fig.axes), panel_names):
            if not panel_name or not ax.get_visible():
                continue
            bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
            stem = os.path.join(OUT, f"{name}_{panel_name}")
            fig.savefig(stem + ".pdf", bbox_inches=bbox, pad_inches=0)
            fig.savefig(stem + ".png", bbox_inches=bbox, pad_inches=0, dpi=300)
            estem = os.path.join(ED, f"{name}_{panel_name}")
            fig.savefig(estem + ".svg", bbox_inches=bbox, pad_inches=0)
    plt.close(fig)
    print(f"  -> {name}.pdf / .png / editable {name}.svg")


def pval_star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def pval_label(p):
    """Explicit P-value annotation (instead of asterisks): P<0.001 / P<0.01 / P<0.05 / P=xx"""
    if p < 0.001:
        return "P<0.001"
    if p < 0.01:
        return "P<0.01"
    if p < 0.05:
        return "P<0.05"
    return f"P={p:.2f}"


# ════════════════════════════════════════════════════════
# Fig 1: Model performance
# ════════════════════════════════════════════════════════
def fig1_model_performance():
    print("Fig1 model performance")
    cv = pd.read_csv(f"{R}/infer_cv/cv_per_fold_metrics.csv")
    t = cv[cv["split"] == "test"].reset_index(drop=True)
    m = pd.read_csv(f"{R}/infer_cv/master_ukb_predictions.csv", low_memory=False)
    h = m[m["is_healthy"] == 1][["age_gt", "age_pred_bc"]].dropna()
    un = m[m["is_healthy"] == 0][["age_gt", "age_pred_bc_self"]].dropna()
    un.columns = ["age_gt", "age_pred_bc"]
    ep1 = pd.read_csv(f"{R}/infer_cv/ext_part1_per_fold_metrics.csv")
    ep3 = pd.read_csv(f"{R}/infer_cv/ext_part3_per_fold_metrics.csv")
    upf = pd.read_csv(f"{R}/infer_cv/unhealthy_per_fold_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6))

    # A: per-fold test MAE raw vs bc
    ax = axes[0, 0]
    x = np.arange(5)
    w = 0.38
    ax.bar(x - w / 2, t["raw_mae"], w, color=NEUTRAL, label="Raw")
    ax.bar(x + w / 2, t["bc_mae"], w, color=BLUE, label="Bias-corrected")
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_ylabel("MAE (years)")
    ax.set_title("(a) Test MAE")
    # Bars start above y=2.4; reserve the lower whitespace for the legend.
    ax.set_ylim(0, 7.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=6.5)

    # B: scatter pred vs true (hexbin) healthy OOF
    ax = axes[0, 1]
    hb = ax.hexbin(h["age_gt"], h["age_pred_bc"], gridsize=45, cmap="Blues", mincnt=1)
    lo = min(h["age_gt"].min(), h["age_pred_bc"].min())
    hi = max(h["age_gt"].max(), h["age_pred_bc"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color=RED, lw=1.2, label="Identity")
    ax.set_xlabel("Chronological age (y)"); ax.set_ylabel("Predicted age (y)")
    ax.set_title("(b) Prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.text(0.97, 0.04, f"MAE {h.age_gt.sub(h.age_pred_bc).abs().mean():.2f} y\nR² {np.corrcoef(h.age_gt,h.age_pred_bc)[0,1]**2:.3f}",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count", fontsize=8)

    # C: scatter, rather than hexbin. Pale low-count hexagons disappear in some PDF renderers.
    ax = axes[1, 0]
    d = h["age_pred_bc"] - h["age_gt"]
    mm = (h["age_gt"] + h["age_pred_bc"]) / 2
    ax.scatter(mm, d, s=1.6, color=RED, alpha=0.24, linewidths=0, rasterized=True)
    mean_d = d.mean(); sd_d = d.std()
    ax.axhline(mean_d, color=BLUE, lw=1.2)
    ax.axhline(mean_d + 1.96 * sd_d, color=RED, ls="--", lw=1)
    ax.axhline(mean_d - 1.96 * sd_d, color=RED, ls="--", lw=1)
    lim = max(abs(d.min()), abs(d.max()), abs(mean_d + 1.96 * sd_d), abs(mean_d - 1.96 * sd_d)) * 1.08
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Mean age (y)"); ax.set_ylabel("Predicted − true (y)")
    ax.set_title("(c) Bland-Altman")
    ax.text(0.97, 0.05, f"Bias {mean_d:.2f} y\nLoA [{mean_d - 1.96 * sd_d:.2f}, {mean_d + 1.96 * sd_d:.2f}]",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    # D: dataset MAE mean±std over folds
    ax = axes[1, 1]
    labs = ["UKB\ntest", "UKB\nunhealthy", "Ext\npart1", "Ext\npart3"]
    means = [t["bc_mae"].mean(), upf["bc_self_mae"].mean(), ep1["bc_mae"].mean(), ep3["bc_mae"].mean()]
    sds = [t["bc_mae"].std(), upf["bc_self_mae"].std(), ep1["bc_mae"].std(), ep3["bc_mae"].std()]
    ax.bar(range(4), means, yerr=sds, color=[BLUE, BLUE2, TEAL, VIOLET],
           capsize=4, error_kw=dict(lw=1.2, ecolor="#333"))
    for i, (m_, sd_) in enumerate(zip(means, sds)):
        ax.text(i, m_ + sd_ + 0.1, f"{m_:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("MAE (years)")
    ax.set_title("(d) External MAE")
    ax.set_ylim(0, max(np.array(means) + np.array(sds)) + 0.8)
    save(fig, "Fig1_model_performance", ["a_test_mae", "b_prediction", "c_bland_altman", "d_external_mae"])


# ════════════════════════════════════════════════════════
# Fig 2: Model calibration and distribution
# ════════════════════════════════════════════════════════
def fig2_calibration():
    print("Fig2 calibration and distribution")
    m = pd.read_csv(f"{R}/infer_cv/master_ukb_predictions.csv", low_memory=False)
    h = m[m["is_healthy"] == 1][["age_gt", "age_pred_bc"]].dropna()
    un = m[m["is_healthy"] == 0][["age_gt", "age_pred_bc"]].dropna()
    un.columns = ["age_gt", "age_pred_bc"]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2))

    ax = axes[0]
    bins = np.linspace(40, 92, 42)
    ax.hist(h["age_gt"], bins=bins, color=NEUTRAL, alpha=0.7, 
        label="True age", edgecolor='none')
    ax.hist(h["age_pred_bc"], bins=bins, color=BLUE, alpha=0.55, 
        label="Predicted age", edgecolor='none')
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
    ax.set_title("(a) Age distribution")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)

    ax = axes[1]
    h["age_bin"] = (h["age_gt"] // 5) * 5
    g = h.groupby("age_bin").apply(lambda d: np.abs(d.age_pred_bc - d.age_gt).mean())
    g = g[g.index >= 45]
    ax.bar(g.index, g.values, width=4, color=TEAL)
    ax.set_xlabel("Age group (y)"); ax.set_ylabel("MAE (y)")
    ax.set_title("(b) MAE by age")

    ax = axes[2]
    ax.hist(h.age_pred_bc - h.age_gt, bins=60, color=BLUE, alpha=0.6,
            label=f"Healthy (n={len(h)})", density=True)
    ax.hist(un.age_pred_bc - un.age_gt, bins=60, color=RED, alpha=0.5,
            label=f"Unhealthy (n={len(un)})", density=True)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Age gap Δ (y)"); ax.set_ylabel("Density")
    ax.set_title("(c) Δ distribution")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
    save(fig, "Fig2_calibration", ["a_age_distribution", "b_mae_by_age", "c_delta_distribution"])


# ════════════════════════════════════════════════════════
# Fig 3: Disease association
# ════════════════════════════════════════════════════════
def fig3_disease():
    print("Fig3 disease association")
    diseases = ["Diabetes", "Hypertension", "Obesity"]
    dcols = {"Diabetes": BLUE, "Hypertension": TEAL, "Obesity": VIOLET}

    # Recompute OLS coefficients from matched_data
    rows = []
    for d in diseases:
        df = pd.read_csv(f"{R}/Disease_Phenotype_Analysis/{d}/matched_data.csv", low_memory=False)
        if "age_use" not in df.columns:
            df["age_use"] = df["age_gt"]
        am = df["age_use"].mean()
        df["age_c"] = df.age_use - am
        df["age_c_sq"] = df.age_c ** 2
        import statsmodels.formula.api as smf
        bmi_term = "" if d == "Obesity" else " + bmi"
        mod = smf.ols(f"delta ~ is_case + age_c + age_c_sq + sex_code{bmi_term}", data=df).fit()
        b = mod.params["is_case"]; ci = mod.conf_int().loc["is_case"]
        rows.append({"disease": d, "beta": b, "lo": ci[0], "hi": ci[1], "p": mod.pvalues["is_case"],
                     "n": len(df), "color": dcols[d]})
        for sex, nm in [(0, "F"), (1, "M")]:
            sd = df[df["sex_code"] == sex].copy()
            sm2 = smf.ols(f"delta ~ is_case + age_c + age_c_sq{bmi_term}", data=sd).fit()
            ci2 = sm2.conf_int().loc["is_case"]
            rows.append({"disease": f"{d} ({nm})", "beta": sm2.params["is_case"], "lo": ci2[0],
                         "hi": ci2[1], "p": sm2.pvalues["is_case"], "n": len(sd), "color": dcols[d]})
    res = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})

    # A: three disease distributions
    ax = axes[0]
    pos = 0
    ticks = []
    for d in diseases:
        df = pd.read_csv(f"{R}/Disease_Phenotype_Analysis/{d}/matched_data.csv", low_memory=False)
        ctrl = df[df["is_case"] == 0]["delta"].dropna()
        case = df[df["is_case"] == 1]["delta"].dropna()
        vp = ax.violinplot([ctrl, case], positions=[pos, pos + 0.8], widths=0.7,
                           showmedians=True, showextrema=False)
        for pc, cc in zip(vp["bodies"], [NEUTRAL, dcols[d]]):
            pc.set_facecolor(cc); pc.set_alpha(0.55)
        ax.axhline(0, color="k", lw=0.7, ls=":", alpha=0.5)
        ticks += [pos + 0.4]; pos += 1.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Diabetes", "Hypertension", "Obesity"], fontsize=7.5, rotation=25, ha="right")
    ax.set_ylabel("Δ (years)")
    ax.set_title("(a) Δ by disease")
    ax.set_xlim(-0.5, pos - 0.5)

    # B: forest plot
    ax = axes[1]
    r2 = res.iloc[::-1].reset_index(drop=True)  # reverse so diabetes at top
    y = np.arange(len(r2))
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.errorbar(r2.beta, y, xerr=[r2.beta - r2.lo, r2.hi - r2.beta],
                fmt="o", ms=5, color=BLUE, ecolor="#555", elinewidth=1.3, capsize=3)
    for yi, row in zip(y, r2.itertuples()):
        ax.text(row.hi + 0.03, yi, f"{row.beta:.2f} [{row.lo:.2f}, {row.hi:.2f}] {pval_label(row.p)}",
                va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(r2.disease, fontsize=8)
    ax.set_xlabel("β (years)")
    ax.set_title("(b) Association with Δ")
    ax.set_ylim(-0.5, len(r2) - 0.5)
    save(fig, "Fig3_disease", ["a_delta_distribution", "b_association_forest"])


# ════════════════════════════════════════════════════════
# Fig 4: GWAS
# ════════════════════════════════════════════════════════
def _draw_manhattan(ax, g, plot_df, cum, chrom_max, lead, title, big=False):
    """Shared Manhattan renderer; both the combined panel and the standalone wide figure use it."""
    colors = ["#3775BA", "#A9C4E4"]
    s = 3.0 if big else 2.2
    for i, (c, grp) in enumerate(plot_df.groupby("CHROM")):
        ax.scatter(grp.pos_cum, grp.logp, s=s, c=colors[i % 2], lw=0, alpha=0.7, rasterized=True)
    ax.axhline(-np.log10(5e-8), color=RED, ls="--", lw=1)
    # Each LD-clumped lead sits on its own vertical level (Nature-style
    # staircase) with a short leader line, so neighbouring loci never collide.
    labels = g[g.ID.isin(lead.ID)].copy().sort_values(["CHROM", "POS"])
    if big:
        fs, step, base = 7.0, 1.4, 1.5
    else:
        fs, step, base = 5.2, 1.7, 1.0
    ymax = plot_df.logp.max()
    for i, (_, srow) in enumerate(labels.iterrows()):
        xp = cum[int(srow["CHROM"]) - 1] + srow["POS"]
        yp = -np.log10(srow["P"])
        ylab = ymax + base + step * i
        ax.plot([xp, xp], [yp, ylab], color="#888888", lw=0.6, zorder=3)
        ax.annotate(str(srow["ID"]), xy=(xp, ylab), xytext=(0, 2), textcoords="offset points",
                    fontsize=fs, ha="center", va="bottom", color=RED, clip_on=False,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    if big:
        xl = [cum[c - 1] + chrom_max.get(c, 0) / 2 for c in range(1, 23)]
        ax.set_xticks(xl); ax.set_xticklabels(range(1, 23), fontsize=8)
    else:
        xl = [cum[c - 1] + chrom_max.get(c, 0) / 2 for c in range(1, 23, 2)]
        ax.set_xticks(xl); ax.set_xticklabels(range(1, 23, 2), fontsize=6.5)
    ax.set_xlabel("Chromosome"); ax.set_ylabel("−log₁₀(P)")
    ax.set_ylim(0, ymax + base + step * max(len(labels) - 1, 0) + 1.5)
    ax.set_title(title, pad=12)


def _load_gwas():
    """Load full-sample GWAS + LD-clump lead loci according to METHOD."""
    if METHOD == "regenie":
        gdir = f"{R}/gwas_regenie_results"
        files = sorted(glob.glob(f"{gdir}/gwas_chr*_catb_delta_with_t1_bc_cole.regenie.gz"))
        dfs = []
        for f in files:
            d = pd.read_csv(f, sep=r"\s+", comment="#", low_memory=False)
            if "TEST" in d.columns:
                d = d[d["TEST"].astype(str) == "ADD"]
            dfs.append(d)
        g = pd.concat(dfs, ignore_index=True)
        g["CHROM"] = pd.to_numeric(g["CHROM"], errors="coerce")
        g["POS"] = pd.to_numeric(g["GENPOS"], errors="coerce")
        g["BETA"] = pd.to_numeric(g["BETA"], errors="coerce")
        g["SE"] = pd.to_numeric(g["SE"], errors="coerce")
        g["P"] = 10.0 ** (-pd.to_numeric(g["LOG10P"], errors="coerce"))
        g = g.dropna(subset=["CHROM", "POS", "BETA", "SE", "P"])
        lead_path = f"{gdir}/regenie_lead_loci.csv"
    else:
        gdir = f"{R}/gwas_plink_results"
        files = sorted(glob.glob(f"{gdir}/gwas_chr*.catb_delta_with_t1_bc_cole.glm.linear"))
        dfs = []
        for f in files:
            with open(f) as fh:
                hdr = fh.readline().strip().lstrip("#").split("\t")
            d = pd.read_csv(f, sep="\t", comment="#", header=None, low_memory=False)
            d.columns = hdr[: len(d.columns)]
            d = d[["CHROM", "POS", "ID", "P", "BETA", "SE"]].copy()
            d["CHROM"] = pd.to_numeric(d["CHROM"], errors="coerce")
            d["POS"] = pd.to_numeric(d["POS"], errors="coerce")
            d["P"] = pd.to_numeric(d["P"], errors="coerce")
            d["BETA"] = pd.to_numeric(d["BETA"], errors="coerce")
            d["SE"] = pd.to_numeric(d["SE"], errors="coerce")
            dfs.append(d)
        g = pd.concat(dfs, ignore_index=True).dropna(subset=["P"])
        lead_path = f"{gdir}/full_lead_loci.csv"
    g = g[g.P > 0]
    print(f"  [{METHOD}] GWAS variants: {len(g):,} (files={len(files)})")
    lead = pd.read_csv(lead_path).rename(columns={"lead": "ID", "CHR": "CHROM"})
    lead["ID"] = lead["ID"].astype(str)
    return g, lead


def fig4_gwas():
    print("Fig4 GWAS")
    # Manhattan plot shows full-sample GWAS (N=53,673); only PRS uses discovery.
    g, lead = _load_gwas()

    # Keep strong signals and downsample the background; cumulative positions are always computed from all variants.
    keep = g[g.P < 1e-3]
    rest = g[g.P >= 1e-3].sample(n=300_000, random_state=42)
    plot_df = pd.concat([keep, rest]).copy()
    plot_df["logp"] = -np.log10(plot_df["P"])
    plot_df = plot_df.sort_values(["CHROM", "POS"])
    chrom_max = g.groupby("CHROM")["POS"].max()
    cum = np.zeros(23, dtype=float)
    for c in range(1, 23):
        cum[c] = cum[c - 1] + chrom_max.get(c, 0)
    plot_df["pos_cum"] = plot_df.apply(lambda r: cum[int(r["CHROM"]) - 1] + r["POS"], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8))

    # A: Manhattan
    _draw_manhattan(axes[0, 0], g, plot_df, cum, chrom_max, lead, "(a) GWAS Manhattan", big=False)

    # B: QQ (same formula and display format as 3_3 to keep λ consistent)
    ax = axes[0, 1]
    pv = np.sort(g["P"].values)
    exp = -np.log10(np.linspace(1 / len(pv), 1, len(pv)))
    obs = -np.log10(pv)
    from scipy import stats as st
    lam = np.median(st.chi2.isf(pv, 1)) / 0.4549
    ax.scatter(exp, obs, s=2, c=BLUE, alpha=0.5, lw=0, rasterized=True)
    mx = max(exp.max(), obs.max())
    ax.plot([0, mx], [0, mx], color=RED, ls="--", lw=1)
    ax.set_xlabel("Expected −log₁₀(P)"); ax.set_ylabel("Observed −log₁₀(P)")
    ax.set_title(f"(b) QQ plot (λ = {lam:.3f})")

    # C: CTRB1/2 regional
    ax = axes[1, 0]
    c16 = g[(g["CHROM"] == 16) & (g["POS"].between(74_500_000, 76_000_000))]
    ax.scatter(c16.POS, -np.log10(c16.P), s=5, c=BLUE, lw=0)
    top = c16.loc[c16.P.idxmin()]
    ax.scatter([top.POS], [-np.log10(top.P)], s=30, c=RED, marker="*", zorder=5)
    ax.annotate(f"{top.ID}", xy=(top.POS, -np.log10(top.P)), xytext=(0, 9),
                textcoords="offset points", fontsize=7, ha="center", va="bottom",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5), clip_on=False)
    ax.axhline(-np.log10(5e-8), color="grey", ls=":", lw=1)
    ax.set_xlabel("chr16 position (Mb)"); ax.set_ylabel("−log₁₀(P)")
    ax.set_ylim(-1, (-np.log10(c16.P)).max() + 3.0)
    ax.set_title("(c) CTRB1/2 locus", pad=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1e6:.0f}"))

    # D: lead SNP effects
    ax = axes[1, 1]
    lsig = g[g.ID.isin(lead.ID)].copy()
    lsig = lsig.sort_values("P")
    y = np.arange(len(lsig))
    ax.errorbar(lsig.BETA, y, xerr=lsig.SE, fmt="o", ms=5, color=TEAL,
                ecolor="#555", elinewidth=1.2, capsize=3)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"chr{int(r.CHROM)}:{r.ID}" for r in lsig.itertuples()], fontsize=6)
    ax.set_xlabel("β (years/allele)")
    ax.set_title("(d) Lead SNP effects")
    save(fig, "Fig4_gwas", ["a_manhattan", "b_qq", "c_ctrb_locus", "d_lead_effects"])

    # Standalone wide Manhattan plot (overwrites the same-named cropped file) for separate use.
    fig_wide, ax_wide = plt.subplots(figsize=(11.5, 4.6))
    _draw_manhattan(ax_wide, g, plot_df, cum, chrom_max, lead, "(a) GWAS Manhattan", big=True)
    save(fig_wide, "Fig4_gwas_a_manhattan")


# ════════════════════════════════════════════════════════
# Fig 5: Mediation analysis
# ════════════════════════════════════════════════════════
def fig5_mediation():
    print("Fig5 mediation analysis")
    med_root = (f"{R}/Disease_Phenotype_Analysis/Mediation_regenie"
                if METHOD == "regenie" else f"{R}/Disease_Phenotype_Analysis/Mediation")
    boot = pd.read_csv(f"{med_root}/mediation_bootstrap.csv")
    txt = open(f"{med_root}/mediation_results.txt").read()
    paths = {}
    for lbl in ["a (SNP→APAG)", "b (APAG→DM)", "c (total effect)", "c' (direct effect)", "ab (indirect effect)"]:
        mm = re.search(rf"{re.escape(lbl)}.*?([-\d.]+)\s+\[\s*([-\d.]+),\s*([-\d.]+)\]", txt)
        if mm:
            paths[lbl] = (float(mm.group(1)), float(mm.group(2)), float(mm.group(3)))
    ab = boot["indirect_effect"].dropna()
    a, b_, c, cp = (boot["a_coef"].median(), boot["b_coef"].median(),
                    boot["c_coef"].median(), boot["cprime_coef"].median())

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))

    # A: clean three-node mediation diagram.
    ax = axes[0]; ax.axis("off")
    box = dict(boxstyle="round,pad=0.25", fc="#EAF1F9", ec=BLUE, lw=1.2)
    ax.text(0.09, 0.66, "rs72802342", ha="center", va="center", fontsize=6.8, bbox=box, transform=ax.transAxes)
    ax.text(0.50, 0.66, "APAG", ha="center", va="center", fontsize=8, bbox=box, transform=ax.transAxes)
    ax.text(0.91, 0.66, "T2D", ha="center", va="center", fontsize=8, bbox=box, transform=ax.transAxes)
    ax.annotate("", xy=(0.40, 0.66), xytext=(0.20, 0.66), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.annotate("", xy=(0.80, 0.66), xytext=(0.60, 0.66), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.annotate("", xy=(0.84, 0.38), xytext=(0.16, 0.38), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, connectionstyle="arc3,rad=-0.20"))
    ax.text(0.30, 0.80, f"a = {a:.2f}", color=BLUE, fontsize=7.2, ha="center", transform=ax.transAxes)
    ax.text(0.70, 0.80, f"b = {b_:.3f}", color=BLUE, fontsize=7.2, ha="center", transform=ax.transAxes)
    ax.text(0.50, 0.30, f"c' = {cp:.2f}", color=RED, fontsize=7.2, ha="center", transform=ax.transAxes)
    ax.text(0.50, 0.13, f"a*b = {ab.median():.3f}  (95% CI {ab.quantile(.025):.3f} to {ab.quantile(.975):.3f})",
            fontsize=6.5, ha="center", transform=ax.transAxes)
    ax.set_title("(a) Mediation", fontsize=9)

    # B: bootstrap ab distribution
    ax = axes[1]
    ax.hist(ab, bins=40, color=BLUE, alpha=0.7)
    lo95, hi95 = ab.quantile([0.025, 0.975])
    ax.axvspan(lo95, hi95, color=RED, alpha=0.12)
    for v in (lo95, hi95):
        ax.axvline(v, color=RED, ls="--", lw=1.2)
    ax.set_xlabel("Indirect effect (ab)")
    ax.set_ylabel("Bootstrap count")
    ax.set_title("(b) Bootstrap indirect effect")

    # C: path effects forest
    ax = axes[2]
    labels = list(paths.keys())
    med = np.array([paths[l][0] for l in labels])
    lo = np.array([paths[l][1] for l in labels])
    hi = np.array([paths[l][2] for l in labels])
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="o", ms=5,
                color=BLUE, ecolor="#555", elinewidth=1.3, capsize=3)
    ax.scatter([med[-1]], [y[-1]], s=40, color=RED, zorder=5)
    path_labels = ["a", "b", "c", "c'", "a*b"][:len(labels)]
    ax.set_yticks(y); ax.set_yticklabels(path_labels, fontsize=7)
    ax.set_xlabel("Effect")
    ax.set_title("(c) Path effects")
    save(fig, "Fig5_mediation", ["a_path_diagram", "b_bootstrap", "c_path_effects"])


# ════════════════════════════════════════════════════════
# Fig 6: MR
# ════════════════════════════════════════════════════════
def fig6_mr():
    print("Fig6 MR")
    mr_root = f"{R}/mr_regenie" if METHOD == "regenie" else f"{R}/mr"
    outs = [("ieu-a-23", "DIAMANTE"), ("finn-b-T2D", "FinnGen")]
    main_rows = []
    for oid, name in outs:
        df = pd.read_csv(f"{mr_root}/{oid}/mr_main_results_{oid}.csv")
        for _, r in df.iterrows():
            main_rows.append({"outcome": name, "method": r["method"], "nsnp": r["nsnp"],
                              "or": r["or"], "lo": r["or_lci95"], "hi": r["or_uci95"],
                              "p": r["pval"]})
    mrdf = pd.DataFrame(main_rows)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4))

    # A: forest IVW + weighted median
    ax = axes[0, 0]
    sel = mrdf[mrdf.method.isin(["Inverse variance weighted", "Weighted median"])]
    y = []
    lab = []
    for oi, (_, name) in enumerate(outs):
        for mi, meth in enumerate(["Inverse variance weighted", "Weighted median"]):
            sub = sel[(sel.outcome == name) & (sel.method == meth)]
            if len(sub):
                r = sub.iloc[0]
                ov = r["or"]; lo = r["lo"]; hi = r["hi"]; p = r["p"]
                ax.errorbar(ov, len(y), xerr=[[ov - lo], [hi - ov]], fmt="o",
                            ms=6, color=BLUE if meth.startswith("Inv") else TEAL, capsize=3, elinewidth=1.4)
                ax.text(hi + 0.02, len(y), f"{ov:.2f} {pval_label(p)}", va="center", fontsize=7)
                lab.append(f"{name} · {'IVW' if meth.startswith('Inv') else 'WMedian'} (n={int(r['nsnp'])})")
                y.append(len(y))
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("OR per SD APAG")
    ax.set_title("(a) MR estimates")

    # B: SNP scatter (3 colors in one panel)
    ax = axes[0, 1]
    for oi, (oid, name) in enumerate(outs):
        h = pd.read_csv(f"{mr_root}/{oid}/mr_harmonised_data_{oid}.csv", low_memory=False)
        ax.scatter(h["beta.exposure"], h["beta.outcome"], s=40, alpha=0.8,
                   label=f"{name} (n={len(h)})",
                   color=[BLUE, RED, TEAL][oi], edgecolor="white", lw=0.5)
    ax.axhline(0, color="k", lw=0.7, ls=":"); ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("SNP effect on APAG"); ax.set_ylabel("SNP effect on T2D")
    ax.set_title("(b) SNP effects")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # C: funnel (previously overlaid panel B by reusing axes[0, 1]).
    ax = axes[1, 1]
    for oi, (oid, name) in enumerate(outs):
        h = pd.read_csv(f"{mr_root}/{oid}/mr_harmonised_data_{oid}.csv", low_memory=False)
        ax.scatter(h["beta.outcome"], 1 / h["se.outcome"], s=30, alpha=0.8,
                   label=name, color=[BLUE, RED, TEAL][oi], edgecolor="white", lw=0.5)
    ax.set_xlabel("SNP effect on T2D"); ax.set_ylabel("1/SE")
    ax.set_title("(c) Funnel plot")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # D: F-statistics
    ax = axes[1, 0]
    fstat = []
    for oi, (oid, name) in enumerate(outs):
        f = pd.read_csv(f"{mr_root}/{oid}/mr_f_statistics_{oid}.csv")
        f["outcome"] = name
        fstat.append(f)
    fs = pd.concat(fstat, ignore_index=True)
    for oi, (oid, name) in enumerate(outs):
        sub = fs[fs.outcome == name]
        ax.barh([str(s) for s in sub.SNP], sub.F_stat, color=[BLUE, RED, TEAL][oi], alpha=0.8, height=0.55, label=name)
    ax.axvline(10, color=RED, ls="--", lw=1, label="F=10 (weak IV)")
    ax.set_xlabel("F-statistic")
    ax.set_title("(d) Instrument strength")
    ax.tick_params(axis="y", labelsize=5.5)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=5.5, loc="lower right", bbox_to_anchor=(1.1, 0), borderaxespad=0)
    save(fig, "Fig6_mr", ["a_mr_forest", "b_snp_effects", "d_instrument_strength", "c_funnel"])


# ════════════════════════════════════════════════════════
# Fig 7: PRS
# ════════════════════════════════════════════════════════
def fig7_prs():
    print("Fig7 PRS")
    prs_root = f"{R}/prs_analysis_regenie" if METHOD == "regenie" else f"{R}/prs_analysis"
    indep = pd.read_csv(f"{prs_root}/prs_diabetes_independence.csv")
    val = pd.read_csv(f"{prs_root}/prs_apag_validation.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    # A: three rows per T2D PRS source (T2D PRS M1 + APAG full/noCTRB M2), labels not repeated
    ax = axes[0]
    rd2 = []
    for t2d_src in indep["T2D_PRS"].unique():
        sub = indep[indep["T2D_PRS"] == t2d_src]
        r_t2d = sub.iloc[0]  # M1 excludes APAG; estimate is independent of APAG variants
        r_full = sub[sub["APAG_PRS"] == "APAG_PRS_full"].iloc[0]
        r_no = sub[sub["APAG_PRS"] == "APAG_PRS_without_CTRB12"].iloc[0]
        rd2.append({"src": t2d_src, "lab": "T2D PRS", "or": r_t2d["T2D_OR_perSD_M1"],
                    "lo": r_t2d["T2D_OR_low95_M1"], "hi": r_t2d["T2D_OR_high95_M1"], "p": r_t2d["T2D_P_M1"]})
        rd2.append({"src": t2d_src, "lab": "APAG full", "or": r_full["APAG_OR_perSD_M2"],
                    "lo": r_full["APAG_OR_low95_M2"], "hi": r_full["APAG_OR_high95_M2"], "p": r_full["APAG_P_M2"]})
        rd2.append({"src": t2d_src, "lab": "APAG noCTRB", "or": r_no["APAG_OR_perSD_M2"],
                    "lo": r_no["APAG_OR_low95_M2"], "hi": r_no["APAG_OR_high95_M2"], "p": r_no["APAG_P_M2"]})
    rd2 = pd.DataFrame(rd2).rename(columns={"or": "OR_val"})
    y = np.arange(len(rd2))
    ax.axvline(1, color="k", ls="--", lw=1)
    ov = rd2["OR_val"]; lo = rd2["lo"]; hi = rd2["hi"]
    ax.errorbar(ov, y, xerr=[ov - lo, hi - ov], fmt="none",
                color="#777", capsize=3, elinewidth=1.3)
    colmap = {"T2D PRS": NEUTRAL, "APAG full": BLUE, "APAG noCTRB": BLUE2}
    ax.scatter(ov, y, s=35,
               c=[colmap[l] for l in rd2.lab],
               edgecolor="white", lw=0.5, zorder=3)
    for yi, row in zip(y, rd2.itertuples()):
        ax.text(row.hi + 0.03, yi, f"{row.OR_val:.2f} ({pval_label(row.p)})", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.lab} · {r.src}" for r in rd2.itertuples()], fontsize=6.5)
    ax.set_xlabel("OR per SD")
    ax.set_title("(a) T2D PRS vs APAG PRS")

    # B: PRS validation beta
    ax = axes[1]
    for i, (_, r) in enumerate(val.iterrows()):
        b, se = r["beta_APAGPRS"], r["se"]
        ax.errorbar([i], [b], yerr=[se], fmt="o", ms=7, color=[BLUE, TEAL][i],
                    capsize=4, elinewidth=1.5)
        ax.text(i, b + se + 0.02, f"β={b:.2f}\nP={r['P']:.1e}", ha="center", fontsize=7)
    ax.set_xticks(range(2)); ax.set_xticklabels(["Discovery", "Test"])
    ax.set_xlim(-0.65, 1.65)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("β (PRS → APAG)")
    ymax = max(val["beta_APAGPRS"] + val["se"]) + 0.14
    ax.set_ylim(top=ymax)
    ax.set_title("(b) PRS prediction", pad=10)

    # C: delta R2
    ax = axes[2]
    ax.bar(range(2), val["delta_R2"] * 100, color=[BLUE, TEAL], width=0.5)
    for i, v in enumerate(val["delta_R2"] * 100):
        ax.text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=7)
    ax.set_xticks(range(2)); ax.set_xticklabels(["Discovery", "Test"])
    ax.set_ylabel("ΔR² (%)")
    ax.set_title("(c) Variance explained")
    save(fig, "Fig7_prs", ["a_prs_forest", "b_validation", "c_variance"])


# ════════════════════════════════════════════════════════
# Fig 8: Dose-response
# ════════════════════════════════════════════════════════
def fig8_dose_response():
    print("Fig8 dose-response")
    df = pd.read_csv(f"{R}/Dose_Response_Diabetes/dose_response_results.csv")
    # Main analysis is complete-case (scheme==cc); mean imputation/missing-indicator in sensitivity
    df = df[(df.exposure_type == "cat4") & (df.scheme == "cc")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    ax = axes[0]
    y = 0
    lab = []
    positions = []
    model_order = ["M1", "M2", "M3"]
    for m in model_order:
        sub = df[df.model == m]
        for r in sub.itertuples():
            ax.errorbar(r.OR, y, xerr=[[r.OR - r.OR_low95], [r.OR_high95 - r.OR]],
                        fmt="o", ms=6, color=BLUE if r.P < 0.05 else NEUTRAL, capsize=3, elinewidth=1.3)
            ax.text(r.OR_high95 + 0.02, y, f"{r.OR:.2f} [{r.OR_low95:.2f}, {r.OR_high95:.2f}]", va="center", fontsize=6)
            lab.append(f"{m} · cat{r.category}")
            positions.append(y)
            y += 1
        y += 1.1
    ax.axvline(1, color="k", ls="--", lw=0.9)
    ax.set_yticks(positions); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("OR")
    ax.set_title("(a) Dose-response")

    ax = axes[1]
    for m, c in zip(model_order, [BLUE, TEAL, VIOLET]):
        sub = df[df.model == m].sort_values("category")
        xs = np.arange(4)
        ors = [1] + list(sub.OR)
        lo = [1] + list(sub.OR_low95)
        hi = [1] + list(sub.OR_high95)
        ax.plot(xs, ors, marker="o", color=c, lw=1.4, label=f"{m}")
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["No aging", "Low", "Medium", "High"], fontsize=7)
    ax.set_ylabel("OR"); ax.set_xlabel("Δ category")
    ax.set_title("(b) Trend")
    ax.legend(fontsize=7)
    save(fig, "Fig8_dose_response", ["a_forest", "b_trend"])


# ════════════════════════════════════════════════════════
# Fig 9: PheWAS
# ════════════════════════════════════════════════════════
def fig9_phewas():
    print("Fig9 PheWAS")
    pw_root = (f"{R}/PheWAS_regenie_test_independent" if METHOD == "regenie"
               else f"{R}/PheWAS_test_independent")
    df = pd.read_csv(f"{pw_root}/phewas_results.csv")
    chapters = [
        ("I: Infections", "A", "B"), ("II: Neoplasms", "C", "D"), ("III: Blood", "D", "E"),
        ("IV: Endocrine", "E", "F"), ("V: Mental", "F", "G"), ("VI: Nervous", "G", "H"),
        ("VII: Eye", "H", "I"), ("VIII: Ear", "H", "J"), ("IX: Circulatory", "I", "J"),
        ("X: Respiratory", "J", "K"), ("XI: Digestive", "K", "L"), ("XII: Skin", "L", "M"),
        ("XIII: Musculoskeletal", "M", "N"), ("XIV: Genitourinary", "N", "O"),
        ("XV: Pregnancy", "O", "P"), ("XVI: Perinatal", "P", "Q"),
        ("XVII: Congenital", "Q", "R"), ("XVIII: Symptoms", "R", "S"),
        ("XIX: Injury", "S", "T"), ("XX: External", "V", "Y"), ("XXI: Health", "Z", "Z")]
    # Determine colors by sorting on chapter number
    df["chapter_rank"] = df["chapter"].map({c: i for i, (c, _, _) in enumerate(chapters)}).fillna(99)
    df = df.sort_values(["chapter_rank", "phenotype"])
    df["x"] = np.arange(len(df))
    n_phen = len(df)
    bonf_line = -np.log10(0.05 / n_phen)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.7))

    ax = axes[0]
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    seen = {}
    for i, (c, grp) in enumerate(df.groupby("chapter", sort=False)):
        ci = int(grp["chapter_rank"].iloc[0]) if grp["chapter_rank"].iloc[0] < 20 else 19
        cc = cmap[ci % 20] if ci < 20 else "#999999"
        ax.scatter(grp.x, -np.log10(grp.P), s=6, c=[cc], alpha=0.75, lw=0)
        seen[ci] = cc
    ax.axhline(bonf_line, color=RED, ls="--", lw=1)
    ax.text(0.98, 0.96, "Bonferroni", fontsize=6.5, ha="right", va="top",
            transform=ax.transAxes, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    # x ticks = chapter midpoints
    mids = []
    labs = []
    for c, (name, _, _) in enumerate(chapters):
        sub = df[df.chapter_rank == c]
        if len(sub):
            mids.append(sub.x.mean())
            labs.append(name.split(":")[0])
    # 21 ICD chapters cannot be read at publication width; color retains chapter information.
    ax.set_xticks([])
    ax.set_xlabel("ICD-10 phenotype")
    ax.set_ylabel("−log₁₀(P)")
    ax.set_title("(a) PheWAS Manhattan")

    ax = axes[1]
    ax.scatter(df.beta, -np.log10(df.P), s=8, c=BLUE, alpha=0.6, lw=0)
    ax.axhline(-np.log10(0.05), color="grey", ls=":", lw=0.8)
    ax.axvline(0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("β (per SD PRS)"); ax.set_ylabel("−log₁₀(P)")
    ax.set_title("(b) Volcano")

    ax = axes[2]
    top = df.head(20).iloc[::-1]
    y = np.arange(len(top))
    ax.errorbar(top.OR, y, xerr=[top.OR - np.exp(top.beta - 1.96 * top.SE),
                                 np.exp(top.beta + 1.96 * top.SE) - top.OR],
                fmt="o", ms=5, color=BLUE, ecolor="#555", capsize=2.5, elinewidth=1)
    ax.axvline(1, color="k", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(top.phenotype, fontsize=6)
    ax.set_xlabel("OR")
    ax.set_title("(c) Top nominal hits")
    save(fig, "Fig9_phewas", ["a_manhattan", "b_volcano", "c_top_hits"])


# ════════════════════════════════════════════════════════
# Fig 10: Study overview
# ════════════════════════════════════════════════════════
def fig10_overview():
    print("Fig10 study overview")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)

    def box(x, y, w, h, text, fc="#EAF1F9", ec=BLUE, fs=7.5, tc="#111"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           fc=fc, ec=ec, lw=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, color="#555", lw=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=color, lw=lw))

    box(0.4, 7.3, 3.2, 1.3, "UKB abdominal MRI\n(pancreas, n=40,649 healthy)", fc="#DDF3DE", ec="#3f9e4d")
    box(4.2, 7.3, 3.4, 1.3, "Swin3D age regression\n5-fold CV, MAE≈2.6 y, R²≈0.85", fc="#DDF3DE", ec="#3f9e4d")
    box(8.2, 7.3, 3.2, 1.3, "Pancreatic age gap Δ\n(APAG = pred − true)", fc="#EAF1F9")
    arrow(3.6, 7.95, 4.2, 7.95)
    arrow(7.6, 7.95, 8.2, 7.95)

    box(0.4, 4.6, 5.6, 1.7,
        "Disease association (PSM + OLS, Δ ~ case)\n"
        "Diabetes +0.79 y (P<0.001) | Hypertension +0.34 y (P<0.001)\n"
        "Obesity +0.21 y (P<0.001)", fc="#EAF1F9")
    box(6.4, 4.6, 5.0, 1.7,
        "External validation (n=1,053 / 210, <18 excl.)\n"
        "MAE≈2.7 / 3.7 y, R²≈0.92 / 0.86\n"
        "Dose-response: M1 P-trend=0.098 (CC)", fc="#EAF1F9")
    arrow(8.2, 7.3, 8.2, 6.3)
    arrow(4.0, 7.3, 4.0, 6.3)

    if METHOD == "regenie":
        gwas_txt = ("GWAS (N=53,673, REGENIE): 546 SNPs P<5e-8\n"
                    "11 independent lead loci (LD-clumped)\n"
                    "CTRB1/2 chr16 rs72802342 | ABO chr9")
        prs_txt = ("PRS (5 lead SNPs, REGENIE)\nPRS→APAG β=+0.15 (P=4e-9)\n"
                   "no independent T2D contribution")
        mr_txt = ("MR APAG→T2D (REGENIE)\nFinnGen OR=1.20 (P=0.046)\n"
                  "ieu-a-23 null")
    else:
        gwas_txt = ("GWAS (N=53,673, PLINK2): 603 SNPs P<5e-8\n"
                    "10 independent lead loci (LD-clumped)\n"
                    "CTRB1/2 chr16 rs72802342 | ABO chr9")
        prs_txt = ("PRS (8 lead SNPs, PLINK)\nPRS→APAG β=+0.17 (P=4e-11)\n"
                   "no independent T2D contribution")
        mr_txt = ("MR APAG→T2D (PLINK)\nFinnGen OR=1.15 (P=0.006)\n"
                  "ieu-a-23 null")

    box(0.4, 2.0, 6.2, 1.6, gwas_txt, fc="#F6CFCB", ec="#B64342")
    box(7.0, 2.0, 5.4, 1.6,
        "Mediation rs72802342→APAG→T2D\n"
        "ab (bootstrap CI excludes 0) → complete\nmediation; proportion ≈17%", fc="#F6CFCB", ec="#B64342")
    arrow(5.2, 4.6, 3.0, 3.6, color=RED)
    arrow(7.0, 4.6, 8.2, 3.6, color=RED)

    box(0.4, 0.15, 3.4, 1.3, prs_txt, fc="#F6CFCB", ec="#B64342", fs=6.8)
    box(4.0, 0.15, 3.4, 1.3, mr_txt, fc="#F6CFCB", ec="#B64342", fs=6.8)
    box(7.6, 0.15, 4.8, 1.3,
        "PheWAS (test N=16,102, 566 phenotypes)\n"
        "no phenotype significant after Bonferroni/FDR", fc="#F6CFCB", ec="#B64342", fs=6.8)
    arrow(3.0, 2.0, 1.5, 1.45, color=RED, lw=1.2)
    arrow(5.0, 2.0, 5.7, 1.45, color=RED, lw=1.2)
    arrow(8.0, 2.0, 10.0, 1.45, color=RED, lw=1.2)

    ax.set_title("Study overview", fontsize=10, pad=8)
    save(fig, "Fig10_overview")


if __name__ == "__main__":
    apply_style(11)
    fig1_model_performance()
    fig2_calibration()
    fig3_disease()
    fig4_gwas()
    fig5_mediation()
    fig6_mr()
    fig7_prs()
    fig8_dose_response()
    fig9_phewas()
    fig10_overview()
    print("\nAll 10 figures complete →", OUT)
