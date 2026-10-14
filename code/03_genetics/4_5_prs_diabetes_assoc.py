#!/usr/bin/env python
"""
4_4_prs_diabetes_assoc.py — APAG PRS independent contribution + CTRB1/2 sensitivity analysis

Key question:
  Does APAG PRS still contribute independently to diabetes beyond known T2D PRS (DIAMANTE/FinnGen)?

Models (all fit on discovery, evaluated independently on test):
  M0: is_case ~ Age + Age² + Sex + Array + PC1-10          (baseline)
  M1: is_case ~ M0 + T2D_PRS                               (known T2D genetic risk)
  M2: is_case ~ M1 + APAG_PRS                              (independent contribution test)

Reports:
  - OR / 95%CI / P per 1 SD PRS for each model
  - LRT (nested model comparison): M1 vs M2
  - Test-set AUC increment
  - Full APAG PRS vs CTRB1/2-removed version (same cohort, same covariates)

Interpretation:
  If APAG PRS remains significant in M2 → the APAG genetic score has an association with diabetes
  (independent association; not a causal effect, only an association)

Usage: python code/03_genetics/4_5_prs_diabetes_assoc.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

RES_ROOT = OUT_ROOT
DEFAULT_PRS_DIR = os.path.join(RES_ROOT, "prs_analysis")
COVAR = ["age_gt", "age_gt2", "sex_code", "Array_bin"] + [f"PC{i}" for i in range(1, 11)]


def load_data(prs_dir, t2d_dir, cohort):
    df = pd.read_csv(cohort, low_memory=False)
    df["eid"] = df["eid"].astype(str)
    df["sex_code"] = df["Sex"].astype(int)
    df["age_gt2"] = df["age_gt"] ** 2
    df["Array_bin"] = df["Array"].astype(int)

    apag_prs_files = {
        "APAG_PRS_full": os.path.join(prs_dir, "APAG_PRS_raw.csv"),
        "APAG_PRS_without_CTRB12": os.path.join(prs_dir, "APAG_PRS_without_CTRB12_raw.csv"),
    }
    t2d_prs_files = {
        "DIAMANTE": os.path.join(t2d_dir, "T2D_PRS_ieu-a-23_raw.csv"),
        "FinnGen": os.path.join(t2d_dir, "T2D_PRS_finn-b-T2D_raw.csv"),
    }

    def _merge(df_in, path, col):
        p = pd.read_csv(path)
        p["IID"] = p["IID"].astype(str)
        score_col = [c for c in p.columns if c not in ("FID", "IID")][0]
        p = p.rename(columns={score_col: col})
        df_in = df_in.drop(columns=["IID"], errors="ignore")
        return df_in.merge(p[["IID", col]], left_on="eid", right_on="IID", how="left")

    for name, path in apag_prs_files.items():
        df = _merge(df, path, name)
    for name, path in t2d_prs_files.items():
        df = _merge(df, path, f"T2D_{name}")

    # Standardize per 1 SD (uniform across the full cohort for comparability)
    for c in list(apag_prs_files) + [f"T2D_{k}" for k in t2d_prs_files]:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df, list(apag_prs_files), list(t2d_prs_files)


def run_model(fit, evl, Xcols):
    X = sm.add_constant(fit[Xcols])
    m = sm.Logit(fit["is_case"], X).fit(disp=0)
    Xe = sm.add_constant(evl[Xcols])
    auc = roc_auc_score(evl["is_case"], m.predict(Xe))
    return m, auc


def extract_prs_row(m, prs_col):
    b, se, p = m.params[prs_col], m.bse[prs_col], m.pvalues[prs_col]
    return {"OR_perSD": float(np.exp(b)), "OR_low95": float(np.exp(b - 1.96 * se)),
            "OR_high95": float(np.exp(b + 1.96 * se)), "P": float(p)}


def lrt(m0, m1):
    """Likelihood ratio test (nested models)"""
    lr = 2 * (m1.llf - m0.llf)
    df = m1.df_model - m0.df_model
    p = 1 - stats.chi2.cdf(lr, df)
    return lr, df, p


def main():
    ap = argparse.ArgumentParser(description="APAG PRS independent contribution + CTRB1/2 sensitivity analysis")
    ap.add_argument("--prs-dir", default=DEFAULT_PRS_DIR,
                    help="APAG PRS directory, default prs_analysis")
    ap.add_argument("--t2d-dir", default=None,
                    help="T2D PRS directory, default same as --prs-dir")
    ap.add_argument("--cohort", default=None,
                    help="Analysis cohort csv, default <prs-dir>/analysis_cohort.csv")
    ap.add_argument("--out-dir", default=None, help="Output directory, default same as --prs-dir")
    args = ap.parse_args()
    prs_dir = args.prs_dir
    t2d_dir = args.t2d_dir or prs_dir
    cohort = args.cohort or os.path.join(prs_dir, "analysis_cohort.csv")
    if not os.path.exists(cohort):
        fallback = os.path.join(RES_ROOT, "prs_analysis", "analysis_cohort.csv")
        if os.path.exists(fallback):
            cohort = fallback
    out_dir = args.out_dir or prs_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(" 4_4 APAG PRS independent contribution + CTRB1/2 sensitivity analysis")
    print("=" * 70)
    print(f"  APAG PRS: {prs_dir}")
    print(f"  T2D  PRS: {t2d_dir}")
    df, apag_tags, t2d_tags = load_data(prs_dir, t2d_dir, cohort)
    df = df.dropna(subset=["is_case"] + COVAR + ["APAG_PRS_full", "T2D_DIAMANTE"])
    fit = df[df["cv_split"] == "discovery"]
    evl = df[df["cv_split"] == "test"]
    print(f"  discovery: {len(fit)} (cases {int((fit['is_case']==1).sum())})")
    print(f"  test:      {len(evl)} (cases {int((evl['is_case']==1).sum())})")

    rows = []
    for apag_tag in apag_tags:
        apag_col = apag_tag  # column name = key (APAG_PRS_full / APAG_PRS_without_CTRB12)
        for t2d_tag in t2d_tags:
            t2d_col = f"T2D_{t2d_tag}"  # column name = T2D_<key>
            print(f"\n── {apag_tag}  ×  T2D PRS: {t2d_tag} ──")
            # Fit models sequentially
            m0, auc0 = run_model(fit, evl, COVAR)
            m1, auc1 = run_model(fit, evl, COVAR + [t2d_col])
            m2, auc2 = run_model(fit, evl, COVAR + [t2d_col, apag_col])

            r_t2d = extract_prs_row(m1, t2d_col)
            r_apag = extract_prs_row(m2, apag_col)
            r_t2d_m2 = extract_prs_row(m2, t2d_col)
            lr, ddf, plrt = lrt(m1, m2)

            rows.append({
                "APAG_PRS": apag_tag, "T2D_PRS": t2d_tag,
                "T2D_OR_perSD_M1": r_t2d["OR_perSD"],
                "T2D_OR_low95_M1": r_t2d["OR_low95"],
                "T2D_OR_high95_M1": r_t2d["OR_high95"],
                "T2D_P_M1": r_t2d["P"],
                "APAG_OR_perSD_M2": r_apag["OR_perSD"],
                "APAG_OR_low95_M2": r_apag["OR_low95"],
                "APAG_OR_high95_M2": r_apag["OR_high95"],
                "APAG_P_M2": r_apag["P"],
                "T2D_OR_perSD_M2": r_t2d_m2["OR_perSD"],
                "T2D_OR_low95_M2": r_t2d_m2["OR_low95"],
                "T2D_OR_high95_M2": r_t2d_m2["OR_high95"],
                "T2D_P_M2": r_t2d_m2["P"],
                "LRT_stat": lr, "LRT_df": ddf, "LRT_P": plrt,
                "AUC_M0_test": auc0, "AUC_M1_test": auc1, "AUC_M2_test": auc2,
                "AUC_delta_M2vsM1": auc2 - auc1,
            })
            print(f"  M0 AUC_test={auc0:.4f} | M1(+T2D) AUC={auc1:.4f} | M2(+APAG) AUC={auc2:.4f}")
            print(f"  T2D PRS (M1): OR={r_t2d['OR_perSD']:.3f} "
                  f"[{r_t2d['OR_low95']:.3f}-{r_t2d['OR_high95']:.3f}] P={r_t2d['P']:.3e}")
            print(f"  APAG PRS (M2): OR={r_apag['OR_perSD']:.3f} "
                  f"[{r_apag['OR_low95']:.3f}-{r_apag['OR_high95']:.3f}] P={r_apag['P']:.3e}")
            print(f"  LRT (M1 vs M2): χ²={lr:.2f} P={plrt:.3e}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "prs_diabetes_independence.csv"), index=False)
    print(f"\n[Saved] {out_dir}/prs_diabetes_independence.csv")
    print("\nFull results:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
