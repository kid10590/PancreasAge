#!/usr/bin/env python
"""
3_4_compare_regenie_plink.py — REGENIE vs PLINK2 GWAS cross-validation

After aligning allele directions, compare effect sizes and significance between the two GWAS:
  - Pearson / Spearman correlation of BETA, sign concordance rate
  - -log10(P) correlation
  - Genomic inflation factor λGC for each
  - Overlap / Jaccard of significant loci (P < threshold, default 5e-8)
  - Per-chromosome lead SNP comparison table

Output (--out-dir):
  comparison_stats.txt          overall statistics
  lead_snps_compare.csv         per-chromosome lead SNP comparison
  beta_scatter.png / p_scatter.png

Usage:
  python code/03_genetics/3_4_compare_regenie_plink.py \
      --regenie-dir <.../gwas_regenie_discovery> \
      --plink-dir   <.../gwas_plink_discovery> \
      --out-dir     <.../regenie_vs_plink_discovery>

  # Or auto-locate by mode:
  python code/03_genetics/3_4_compare_regenie_plink.py --mode discovery
  python code/03_genetics/3_4_compare_regenie_plink.py --mode full
"""
import os
import glob
import gzip
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

RES_ROOT = OUT_ROOT
PHENO_NAME = "catb_delta_with_t1_bc_cole"

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ─────────────────────────── Reading ───────────────────────────
def _read_header(path):
    """Return (list of column names, whether the header starts with #)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                return line.lstrip("#").split(), True
            return line.split(), False
    return None, False


def _read_table(path):
    header, hashed = _read_header(path)
    opener = gzip.open if path.endswith(".gz") else open
    # When the header starts with #, comment="#" skips it; otherwise manually skip the first row
    skiprows = 0 if hashed else 1
    df = pd.read_csv(opener(path, "rt"), sep=r"\s+", comment="#",
                     header=None, skiprows=skiprows, low_memory=False)
    if header and len(header) == df.shape[1]:
        df.columns = header
    return df


def load_regenie(d):
    files = sorted(glob.glob(os.path.join(d, f"gwas_chr*{PHENO_NAME}.regenie*")))
    if not files:
        files = sorted(glob.glob(os.path.join(d, "gwas_chr*.regenie*")))
    if not files:
        raise FileNotFoundError(f"No REGENIE results in {d}")
    dfs = []
    for f in files:
        df = _read_table(f)
        if "TEST" in df.columns:
            df = df[df["TEST"].astype(str) == "ADD"]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["CHROM"] = pd.to_numeric(df["CHROM"], errors="coerce")
    df["GENPOS"] = pd.to_numeric(df["GENPOS"], errors="coerce")
    df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
    df["LOG10P"] = pd.to_numeric(df["LOG10P"], errors="coerce")
    df["P"] = 10.0 ** (-df["LOG10P"])
    for c in ["ALLELE0", "ALLELE1", "A1FREQ"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper() if c.startswith("ALLELE") else pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["CHROM", "GENPOS", "BETA", "LOG10P"])
    print(f"  REGENIE: {len(files)} files, {len(df)} variants")
    return df


def load_plink(d):
    files = sorted(glob.glob(os.path.join(d, f"gwas_chr*.{PHENO_NAME}.glm.linear*")))
    if not files:
        files = sorted(glob.glob(os.path.join(d, "gwas_chr*.glm.linear*")))
    if not files:
        raise FileNotFoundError(f"No PLINK results in {d}")
    dfs = []
    for f in files:
        df = _read_table(f)
        if "TEST" in df.columns:
            df = df[df["TEST"].astype(str) == "ADD"]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"#CHROM": "CHROM"})
    df["CHROM"] = pd.to_numeric(df["CHROM"], errors="coerce")
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df["BETA"] = pd.to_numeric(df["BETA"], errors="coerce")
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    for c in ["REF", "ALT", "A1", "OMITTED"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    df = df.dropna(subset=["CHROM", "POS", "BETA", "P"])
    print(f"  PLINK2:  {len(files)} files, {len(df)} variants")
    return df


# ─────────────────────── Allele alignment ───────────────────────
def allele_sign(r_a1, r_a0, p_a1, p_ref):
    """Return +1/-1 (whether the regenie effect-allele direction needs flipping relative to plink), or 0 if it cannot be aligned."""
    if not all(isinstance(x, str) and x for x in (r_a1, r_a0, p_a1, p_ref)):
        return 0
    if r_a1 == p_a1 and r_a0 == p_ref:
        return 1
    if r_a1 == p_ref and r_a0 == p_a1:
        return -1
    c1 = COMPLEMENT.get(r_a1)
    c0 = COMPLEMENT.get(r_a0)
    if c1 is not None and c0 is not None:
        if c1 == p_a1 and c0 == p_ref:
            return 1
        if c1 == p_ref and c0 == p_a1:
            return -1
    return 0


def lambda_gc(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan
    return float(np.median(chi2.isf(p, 1)) / 0.4549364)


# ─────────────────────────── Main flow ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="REGENIE vs PLINK2 GWAS cross-validation")
    ap.add_argument("--regenie-dir", default=None)
    ap.add_argument("--plink-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mode", choices=["full", "discovery"], default=None,
                    help="Auto-locate directory: full / discovery")
    ap.add_argument("--p-threshold", type=float, default=5e-8)
    ap.add_argument("--save-merged", action="store_true",
                    help="Additionally save the merged variant-level csv (large)")
    args = ap.parse_args()

    if args.mode:
        reg_dir = os.path.join(RES_ROOT, "gwas_regenie_results" if args.mode == "full" else "gwas_regenie_discovery")
        plink_dir = os.path.join(RES_ROOT, "gwas_plink_results" if args.mode == "full" else "gwas_plink_discovery")
    else:
        reg_dir = args.regenie_dir
        plink_dir = args.plink_dir
    if not reg_dir or not plink_dir:
        ap.error("Requires --regenie-dir/--plink-dir, or --mode")

    tag = args.mode or "custom"
    out_dir = args.out_dir or os.path.join(RES_ROOT, f"regenie_vs_plink_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f" REGENIE vs PLINK2 cross-validation ({tag})")
    print(f"   REGENIE: {reg_dir}")
    print(f"   PLINK2 : {plink_dir}")
    print(f"   Output : {out_dir}")
    print("=" * 60)

    reg = load_regenie(reg_dir)
    plk = load_plink(plink_dir)

    # Merge key: prefer ID; if missing/duplicated use CHROM:POS
    if "ID" in reg.columns and "ID" in plk.columns:
        key = "ID"
    else:
        key = "loc"
        reg["loc"] = reg["CHROM"].astype("Int64").astype(str) + ":" + reg["GENPOS"].astype("Int64").astype(str)
        plk["loc"] = plk["CHROM"].astype("Int64").astype(str) + ":" + plk["POS"].astype("Int64").astype(str)

    reg = reg.drop_duplicates(subset=[key])
    plk = plk.drop_duplicates(subset=[key])

    m = pd.merge(reg, plk, on=key, suffixes=("_reg", "_plk"), how="inner")
    print(f"\n  Merged variants (inner join on {key}): {len(m)}")
    print(f"  REGENIE only: {len(reg) - len(m)} | PLINK2 only: {len(plk) - len(m)}")

    # After merging, CHROM becomes CHROM_reg/CHROM_plk; unify to CHROM
    if "CHROM_reg" in m.columns:
        m["CHROM"] = m["CHROM_reg"]

    # Allele alignment (plink2 non-effect allele is OMITTED, fall back to REF)
    r_a1, r_a0 = m["ALLELE1"], m["ALLELE0"]
    p_a1 = m["A1"]
    p_other = m["OMITTED"] if "OMITTED" in m.columns else m["REF"]

    m["SIGN"] = [
        allele_sign(a1, a0, b1, br)
        for a1, a0, b1, br in zip(r_a1, r_a0, p_a1, p_other)
    ]
    n_bad = int((m["SIGN"] == 0).sum())
    m = m[m["SIGN"] != 0].copy()
    m["BETA_REG_ALIGNED"] = m["BETA_reg"] * m["SIGN"]
    m["LOG10P_REG"] = pd.to_numeric(m["LOG10P"], errors="coerce")
    m["LOG10P_PLK"] = -np.log10(pd.to_numeric(m["P_plk"], errors="coerce"))
    m = m.dropna(subset=["BETA_REG_ALIGNED", "BETA_plk", "LOG10P_REG", "LOG10P_PLK"])
    print(f"  Alleles not alignable/dropped: {n_bad} | final compared: {len(m)}")

    if len(m) == 0:
        raise SystemExit("No variants available for comparison")

    # ── Statistics ──
    r_beta = pearsonr(m["BETA_REG_ALIGNED"], m["BETA_plk"])[0]
    rho_beta = spearmanr(m["BETA_REG_ALIGNED"], m["BETA_plk"])[0]
    r_logp = pearsonr(m["LOG10P_REG"], m["LOG10P_PLK"])[0]
    sign_conc = float(np.mean(np.sign(m["BETA_REG_ALIGNED"]) == np.sign(m["BETA_plk"])))
    lam_reg = lambda_gc(10.0 ** (-m["LOG10P_REG"]))
    lam_plk = lambda_gc(m["P_plk"])

    thr = args.p_threshold
    sig_reg = m[m["LOG10P_REG"] > -np.log10(thr)]
    sig_plk = m[m["P_plk"] < thr]
    set_reg, set_plk = set(sig_reg[key]), set(sig_plk[key])
    inter = set_reg & set_plk
    union = set_reg | set_plk
    jaccard = len(inter) / len(union) if union else np.nan

    lines = []
    lines.append("REGENIE vs PLINK2 GWAS cross-validation")
    lines.append(f"regenie_dir\t{reg_dir}")
    lines.append(f"plink_dir\t{plink_dir}")
    lines.append(f"merged_variants\t{len(m)}")
    lines.append(f"allele_mismatch_dropped\t{n_bad}")
    lines.append(f"beta_pearson_r\t{r_beta:.6f}")
    lines.append(f"beta_spearman_rho\t{rho_beta:.6f}")
    lines.append(f"log10p_pearson_r\t{r_logp:.6f}")
    lines.append(f"beta_sign_concordance\t{sign_conc:.6f}")
    lines.append(f"lambda_gc_regenie\t{lam_reg:.4f}")
    lines.append(f"lambda_gc_plink\t{lam_plk:.4f}")
    lines.append(f"p_threshold\t{thr}")
    lines.append(f"n_sig_regenie\t{len(sig_reg)}")
    lines.append(f"n_sig_plink\t{len(sig_plk)}")
    lines.append(f"n_sig_overlap\t{len(inter)}")
    lines.append(f"jaccard_sig\t{jaccard:.6f}")

    stats_txt = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "comparison_stats.txt"), "w") as fh:
        fh.write(stats_txt)
    print("\n" + stats_txt)

    # ── Per-chromosome lead SNP comparison ──
    def lead(df, pcol, beta_col):
        # pcol is -log10(P); larger is more significant, so use idxmax (idxmin would pick the least significant)
        idx = df.groupby("CHROM")[pcol].idxmax()
        return df.loc[idx, ["CHROM", key, pcol, beta_col]]

    lead_reg = lead(m, "LOG10P_REG", "BETA_REG_ALIGNED").rename(
        columns={"LOG10P_REG": "LOG10P_REG", "BETA_REG_ALIGNED": "BETA_REG"})
    lead_plk = lead(m, "LOG10P_PLK", "BETA_plk").rename(
        columns={"LOG10P_PLK": "LOG10P_PLK", "BETA_plk": "BETA_PLK"})
    lead_m = pd.merge(lead_reg, lead_plk, on=["CHROM", key], how="outer")
    lead_m.to_csv(os.path.join(out_dir, "lead_snps_compare.csv"), index=False)

    if args.save_merged:
        m.to_csv(os.path.join(out_dir, "merged_variants.csv.gz"), index=False, compression="gzip")

    # ── Scatter plots (downsampled) ──
    n = len(m)
    step = max(1, n // 200000)
    s = m.iloc[::step]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s["BETA_plk"], s["BETA_REG_ALIGNED"], s=2, alpha=0.2)
    lim = np.nanpercentile(np.abs(np.r_[s["BETA_plk"], s["BETA_REG_ALIGNED"]]), 99.9) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("PLINK2 BETA"); ax.set_ylabel("REGENIE BETA (aligned)")
    ax.set_title(f"BETA: r={r_beta:.4f}, sign={sign_conc:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "beta_scatter.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s["LOG10P_PLK"], s["LOG10P_REG"], s=2, alpha=0.2)
    lim = max(s["LOG10P_PLK"].max(), s["LOG10P_REG"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("PLINK2 -log10(P)"); ax.set_ylabel("REGENIE -log10(P)")
    ax.set_title(f"-log10(P): r={r_logp:.4f}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "p_scatter.png"), dpi=150); plt.close()

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
