#!/usr/bin/env python
"""
3_6_compare_regenie_plink.py — Summary comparison of PLINK vs REGENIE full-pipeline results

Summarizes the results of the two GWAS methods across downstream analyses:
  - GWAS level: discovery / full BETA correlation, λGC, significant-locus overlap (reads 3_4 output)
  - Lead loci: independent loci of full GWAS + discovery PRS
  - PRS validation / diabetes association / independent contribution
  - PheWAS
  - Mediation analysis (SNP + PRS)
  - MR (each outcome)

Output directory (--out-dir, default RES_ROOT/regenie_vs_plink_summary):
  gwas_comparison.csv
  lead_loci_comparison.csv
  prs_validation_comparison.csv
  prs_association_comparison.csv
  prs_independence_comparison.csv
  phewas_comparison.csv
  phewas_top_hits.csv
  mediation_comparison.csv
  mr_comparison.csv
  summary.txt

Usage:
  python code/03_genetics/3_6_compare_regenie_plink.py
"""
import os
import re
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

RES_ROOT = OUT_ROOT
PHENO_NAME = "catb_delta_with_t1_bc_cole"
MR_OUTCOMES = ["ieu-a-23", "finn-b-T2D"]


def _read(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def _read_text(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return fh.read()


def _parse_stats(path):
    txt = _read_text(path)
    if not txt:
        return {}
    out = {}
    for line in txt.splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            out[k.strip()] = v.strip()
    return out


def gwas_comparison(out_dir):
    rows = []
    for mode, d in [("discovery", "regenie_vs_plink_discovery"),
                    ("full", "regenie_vs_plink_full")]:
        s = _parse_stats(os.path.join(RES_ROOT, d, "comparison_stats.txt"))
        if not s:
            continue
        def f(k):
            try:
                return float(s.get(k, np.nan))
            except Exception:
                return np.nan
        rows.append({
            "mode": mode,
            "merged_variants": f("merged_variants"),
            "beta_pearson_r": f("beta_pearson_r"),
            "beta_spearman_rho": f("beta_spearman_rho"),
            "log10p_pearson_r": f("log10p_pearson_r"),
            "beta_sign_concordance": f("beta_sign_concordance"),
            "lambda_gc_regenie": f("lambda_gc_regenie"),
            "lambda_gc_plink": f("lambda_gc_plink"),
            "n_sig_regenie": f("n_sig_regenie"),
            "n_sig_plink": f("n_sig_plink"),
            "n_sig_overlap": f("n_sig_overlap"),
            "jaccard_sig": f("jaccard_sig"),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(os.path.join(out_dir, "gwas_comparison.csv"), index=False)
    return df


def lead_loci_comparison(out_dir):
    panels = [
        ("full_GWAS",
         os.path.join(RES_ROOT, "gwas_plink_results", "full_lead_loci.csv"),
         os.path.join(RES_ROOT, "gwas_regenie_results", "regenie_lead_loci.csv"),
         "lead"),
        ("discovery_PRS",
         os.path.join(RES_ROOT, "prs_analysis", "prs_build", "lead_full.csv"),
         os.path.join(RES_ROOT, "prs_analysis_regenie", "prs_build", "lead_full.csv"),
         "ID"),
    ]
    rows = []
    for panel, p_path, r_path, idcol in panels:
        p = _read(p_path)
        r = _read(r_path)
        p_ids = set(p[idcol].astype(str)) if p is not None and idcol in p.columns else set()
        r_ids = set(r[idcol].astype(str)) if r is not None and idcol in r.columns else set()
        rows.append({
            "panel": panel,
            "n_plink": len(p_ids),
            "n_regenie": len(r_ids),
            "n_shared": len(p_ids & r_ids),
            "plink_only": ";".join(sorted(p_ids - r_ids)),
            "regenie_only": ";".join(sorted(r_ids - p_ids)),
            "plink_ids": ";".join(sorted(p_ids)),
            "regenie_ids": ";".join(sorted(r_ids)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "lead_loci_comparison.csv"), index=False)
    return df


def prs_validation_comparison(out_dir):
    frames = []
    for method, d in [("plink", "prs_analysis"), ("regenie", "prs_analysis_regenie")]:
        df = _read(os.path.join(RES_ROOT, d, "prs_apag_validation.csv"))
        if df is None:
            continue
        df = df.copy()
        df.insert(0, "method", method)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "prs_validation_comparison.csv"), index=False)
    return out


def prs_association_comparison(out_dir):
    frames = []
    for method, d in [("plink", "prs_analysis"), ("regenie", "prs_analysis_regenie")]:
        df = _read(os.path.join(RES_ROOT, d, "diabetes_prs_association.csv"))
        if df is None:
            continue
        df = df[df["model"] == "full"].copy()
        df.insert(0, "method", method)
        frames.append(df[["method", "PRS_version", "OR_perSD", "OR_low95",
                          "OR_high95", "P", "AUC_test"]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "prs_association_comparison.csv"), index=False)
    return out


def prs_independence_comparison(out_dir):
    frames = []
    for method, d in [("plink", "prs_analysis"), ("regenie", "prs_analysis_regenie")]:
        df = _read(os.path.join(RES_ROOT, d, "prs_diabetes_independence.csv"))
        if df is None:
            continue
        df = df.copy()
        df.insert(0, "method", method)
        frames.append(df[["method", "APAG_PRS", "T2D_PRS",
                          "APAG_OR_perSD_M2", "APAG_OR_low95_M2", "APAG_OR_high95_M2",
                          "APAG_P_M2", "LRT_P", "AUC_delta_M2vsM1"]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "prs_independence_comparison.csv"), index=False)
    return out


def phewas_comparison(out_dir):
    frames = []
    top_frames = []
    for method, d in [("plink", "PheWAS_plink_matched"), ("regenie", "PheWAS_regenie")]:
        df = _read(os.path.join(RES_ROOT, d, "phewas_results.csv"))
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("P")
        top = df.head(1).iloc[0]
        frames.append({
            "method": method,
            "n_results": len(df),
            "n_bonferroni": int((df["P_bonferroni"] < 0.05).sum()),
            "n_fdr": int((df["P_fdr"] < 0.05).sum()),
            "top_phenotype": top["phenotype"],
            "top_OR": top["OR"],
            "top_P": top["P"],
            "top_P_bonferroni": top["P_bonferroni"],
        })
        t = df.head(10)[["phenotype", "chapter", "n_cases", "OR", "P", "P_bonferroni"]].copy()
        t.insert(0, "method", method)
        top_frames.append(t)
    summary = pd.DataFrame(frames)
    if len(summary):
        summary.to_csv(os.path.join(out_dir, "phewas_comparison.csv"), index=False)
    if top_frames:
        pd.concat(top_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "phewas_top_hits.csv"), index=False)
    return summary


def _parse_snp_mediation(path):
    txt = _read_text(path)
    if not txt:
        return None
    def g(pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else np.nan
    return {
        "a": g(r"Path a\s*\(SNP → APAG\):\s*(-?[\d.]+)"),
        "b": g(r"Path b\s*\(APAG → Diabetes\):\s*(-?[\d.]+)"),
        "c": g(r"Path c\s*\(SNP → Diabetes, total effect\):\s*(-?[\d.]+)"),
        "cp": g(r"Path c'\s*\(SNP → Diabetes, direct effect\):\s*(-?[\d.]+)"),
        "ab": g(r"ab = a × b = (-?[\d.]+)"),
        "sobel_p": g(r"Sobel P\s*=\s*([\d.eE+-]+)"),
        "ab_ci_low": g(r"ab \(indirect effect\)\s*-?[\d.]+\s*\[\s*(-?[\d.]+)"),
        "ab_ci_high": g(r"ab \(indirect effect\)\s*-?[\d.]+\s*\[\s*-?[\d.]+,\s*(-?[\d.]+)"),
        "conclusion": (re.search(r"Conclusion:\s*(.+)", txt).group(1).strip()
                       if re.search(r"Conclusion:\s*(.+)", txt) else ""),
    }


def _parse_prs_mediation(path):
    txt = _read_text(path)
    if not txt:
        return None
    def g(pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else np.nan
    ab = re.search(r"ab=(-?[\d.]+)\s+Sobel P=([\d.eE+-]+)\s*\|\s*Bootstrap 95%CI "
                   r"\[(-?[\d.]+),(-?[\d.]+)\]", txt)
    return {
        "a": g(r"a=(-?[\d.]+) P="),
        "b": g(r"b=(-?[\d.]+) OR="),
        "c": g(r"c=(-?[\d.]+) OR="),
        "cp": g(r"c'=(-?[\d.]+) P="),
        "ab": float(ab.group(1)) if ab else np.nan,
        "sobel_p": float(ab.group(2)) if ab else np.nan,
        "ab_ci_low": float(ab.group(3)) if ab else np.nan,
        "ab_ci_high": float(ab.group(4)) if ab else np.nan,
        "conclusion": (re.search(r"Classification:\s*(.+)", txt).group(1).strip()
                       if re.search(r"Classification:\s*(.+)", txt) else ""),
    }


def mediation_comparison(out_dir):
    D = os.path.join(RES_ROOT, "Disease_Phenotype_Analysis")
    specs = [
        ("SNP", os.path.join(D, "Mediation", "mediation_results.txt"),
         os.path.join(D, "Mediation_regenie", "mediation_results.txt"), _parse_snp_mediation),
        ("PRS", os.path.join(D, "Mediation_PRS", "prs_mediation_results.txt"),
         os.path.join(D, "Mediation_PRS_regenie", "prs_mediation_results.txt"), _parse_prs_mediation),
    ]
    rows = []
    for analysis, p_path, r_path, parser in specs:
        for method, path in [("plink", p_path), ("regenie", r_path)]:
            res = parser(path)
            if res is None:
                continue
            row = {"analysis": analysis, "method": method}
            row.update(res)
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(os.path.join(out_dir, "mediation_comparison.csv"), index=False)
    return df


def mr_comparison(out_dir):
    frames = []
    for method, root in [("plink", "mr"), ("regenie", "mr_regenie")]:
        for oc in MR_OUTCOMES:
            path = os.path.join(RES_ROOT, root, oc, f"mr_main_results_{oc}.csv")
            df = _read(path)
            if df is None:
                continue
            df = df.copy().rename(columns={"method": "method_mr"})
            df.insert(0, "method", method)
            df.insert(1, "outcome_id", oc)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    cols = ["method", "outcome_id", "method_mr", "nsnp", "b", "se", "pval",
            "or", "or_lci95", "or_uci95"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(os.path.join(out_dir, "mr_comparison.csv"), index=False)
    return out


def write_summary(out_dir, gwas, leads, pv, pa, pi, ph, med, mr):
    lines = []
    lines.append("PLINK vs REGENIE full-pipeline comparison summary")
    lines.append("=" * 60)

    lines.append("\n[1] GWAS level")
    if len(gwas):
        for _, r in gwas.iterrows():
            lines.append(f"  [{r['mode']}] merged={int(r['merged_variants']):,} "
                         f"beta_r={r['beta_pearson_r']:.4f} sign={r['beta_sign_concordance']:.4f} "
                         f"lambda(reg/plk)={r['lambda_gc_regenie']:.4f}/{r['lambda_gc_plink']:.4f} "
                         f"sig(reg/plk)={int(r['n_sig_regenie'])}/{int(r['n_sig_plink'])} "
                         f"overlap={int(r['n_sig_overlap'])} Jaccard={r['jaccard_sig']:.3f}")
    else:
        lines.append("  (none)")

    lines.append("\n[2] Lead loci")
    for _, r in leads.iterrows():
        lines.append(f"  [{r['panel']}] plink={r['n_plink']} regenie={r['n_regenie']} "
                     f"shared={r['n_shared']}")
        if r["plink_only"]:
            lines.append(f"    plink_only: {r['plink_only']}")
        if r["regenie_only"]:
            lines.append(f"    regenie_only: {r['regenie_only']}")

    lines.append("\n[3] PRS validation (PRS→Δ)")
    if len(pv):
        for _, r in pv.iterrows():
            lines.append(f"  [{r['method']}/{r['split']}] n={int(r['n'])} "
                         f"beta={r['beta_APAGPRS']:.4f} P={r['P']:.2e} dR2={r['delta_R2']:.4f}")
    else:
        lines.append("  (none)")

    lines.append("\n[4] APAG PRS diabetes association (test AUC)")
    if len(pa):
        for _, r in pa.iterrows():
            lines.append(f"  [{r['method']}/{r['PRS_version']}] "
                         f"OR={r['OR_perSD']:.3f} P={r['P']:.2e} AUC_test={r['AUC_test']:.4f}")
    else:
        lines.append("  (none)")

    lines.append("\n[5] APAG PRS independent contribution (APAG in M2)")
    if len(pi):
        for _, r in pi.iterrows():
            lines.append(f"  [{r['method']}/{r['APAG_PRS']} × {r['T2D_PRS']}] "
                         f"APAG_OR={r['APAG_OR_perSD_M2']:.3f} P={r['APAG_P_M2']:.2e} "
                         f"LRT_P={r['LRT_P']:.2e}")
    else:
        lines.append("  (none)")

    lines.append("\n[6] PheWAS")
    if len(ph):
        for _, r in ph.iterrows():
            lines.append(f"  [{r['method']}] n={int(r['n_results'])} "
                         f"Bonf={int(r['n_bonferroni'])} FDR={int(r['n_fdr'])} | "
                         f"top={r['top_phenotype']} OR={r['top_OR']:.3f} "
                         f"P={r['top_P']:.2e} P_bonf={r['top_P_bonferroni']:.3f}")
    else:
        lines.append("  (none)")

    lines.append("\n[7] Mediation analysis")
    if len(med):
        for _, r in med.iterrows():
            lines.append(f"  [{r['analysis']}/{r['method']}] a={r['a']:.4f} b={r['b']:.4f} "
                         f"c'={r['cp']:.4f} ab={r['ab']:.4f} "
                         f"CI=[{r['ab_ci_low']:.4f},{r['ab_ci_high']:.4f}] "
                         f"Sobel_P={r['sobel_p']:.2e} | {r['conclusion']}")
    else:
        lines.append("  (none)")

    lines.append("\n[8] MR")
    if len(mr):
        for _, r in mr.iterrows():
            lines.append(f"  [{r['method']}/{r['outcome_id']}] {r['method_mr']}: "
                         f"nsnp={int(r['nsnp'])} OR={r['or']:.3f} "
                         f"[{r['or_lci95']:.3f},{r['or_uci95']:.3f}] P={r['pval']:.3f}")
    else:
        lines.append("  (none, MR not yet completed or not run)")

    txt = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description="PLINK vs REGENIE full-pipeline comparison summary")
    ap.add_argument("--out-dir", default=os.path.join(RES_ROOT, "regenie_vs_plink_summary"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}\n")

    gwas = gwas_comparison(args.out_dir)
    leads = lead_loci_comparison(args.out_dir)
    pv = prs_validation_comparison(args.out_dir)
    pa = prs_association_comparison(args.out_dir)
    pi = prs_independence_comparison(args.out_dir)
    ph = phewas_comparison(args.out_dir)
    med = mediation_comparison(args.out_dir)
    mr = mr_comparison(args.out_dir)
    write_summary(args.out_dir, gwas, leads, pv, pa, pi, ph, med, mr)
    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
