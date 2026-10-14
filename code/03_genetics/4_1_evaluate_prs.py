#!/usr/bin/env python
"""
4_1_evaluate_prs.py — APAG PRS validation + diabetes association assessment

What it does (4 steps):
  A. Validate whether APAG PRS predicts measured APAG (delta) — discovery/test splits
     delta ~ Age + Age² + Sex + Array + PC1-10 + PRS
     → report β(PRS)/SE/P, R²_base vs R²_full, ΔR²
     (if the PRS cannot predict APAG, its diabetes association should not be emphasized further)
  B. Diabetes logistic baseline model (fit on discovery, evaluated on test)
     is_case ~ Age + Age² + Sex + Array + PC1-10
     → report AIC/BIC, AUC (discovery/test)
  C. Add APAG PRS
     is_case ~ baseline + APAG_PRS
     → report OR per 1 SD / 95%CI / P, LRT (baseline vs +PRS), AUC increment (test)
  D. Full version vs CTRB1/2-removed version comparison (same analysis cohort, same covariates)

Usage: python code/03_genetics/4_1_evaluate_prs.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

RES_ROOT = OUT_ROOT
DEFAULT_PRS_DIR = os.path.join(RES_ROOT, "prs_analysis")
COVAR = ["age_gt", "age_gt2", "sex_code", "Array_bin"] + [f"PC{i}" for i in range(1, 11)]


def load_data(prs_dir, cohort):
    df = pd.read_csv(cohort, low_memory=False)
    df["eid"] = df["eid"].astype(str)
    df["sex_code"] = df["Sex"].astype(int)
    df["age_gt2"] = df["age_gt"] ** 2
    df["Array_bin"] = df["Array"].astype(int)
    prs_raw = {
        "full": os.path.join(prs_dir, "APAG_PRS_raw.csv"),
        "without_ctrb12": os.path.join(prs_dir, "APAG_PRS_without_CTRB12_raw.csv"),
    }
    # Merge PRS
    for tag, path in prs_raw.items():
        prs = pd.read_csv(path)
        prs["IID"] = prs["IID"].astype(str)
        prs = prs.rename(columns={[c for c in prs.columns if c not in ("FID", "IID")][0]: f"PRS_{tag}"})
        df = df.merge(prs[["IID", f"PRS_{tag}"]], left_on="eid", right_on="IID", how="left")
    # Standardize (per 1 SD)
    for tag in prs_raw:
        df[f"PRS_{tag}_z"] = (df[f"PRS_{tag}"] - df[f"PRS_{tag}"].mean()) / df[f"PRS_{tag}"].std()
    return df


def validate_apag_prediction(df, split_tag):
    """Whether the PRS predicts measured delta"""
    sub = df[df["cv_split"] == split_tag].copy()
    base_cols = COVAR
    prs_col = [c for c in sub.columns if c.startswith("PRS_full_z")]
    if not prs_col:
        return None
    prs_col = prs_col[0]
    d = sub.dropna(subset=["delta"] + base_cols + [prs_col])
    if len(d) < 20:
        return None
    Xb = sm.add_constant(d[base_cols])
    rb = sm.OLS(d["delta"], Xb).fit()
    Xf = sm.add_constant(d[base_cols + [prs_col]])
    rf = sm.OLS(d["delta"], Xf).fit()
    b = rf.params[prs_col]
    se = rf.bse[prs_col]
    p = rf.pvalues[prs_col]
    return {
        "split": split_tag, "n": len(d),
        "beta_APAGPRS": b, "se": se, "P": p,
        "R2_base": rb.rsquared, "R2_full": rf.rsquared,
        "delta_R2": rf.rsquared - rb.rsquared,
    }


def run_logistic_models(df, split_fit, split_eval, prs_col):
    """Baseline vs +PRS diabetes logistic, fit on discovery and evaluated on test"""
    base_cols = COVAR
    d = df.dropna(subset=["is_case"] + base_cols).copy()
    fit = d[d["cv_split"] == split_fit]
    evl = d[d["cv_split"] == split_eval]

    def _auc(sub, Xcols, coef):
        X = sm.add_constant(sub[Xcols])
        p = 1 / (1 + np.exp(-X @ coef))
        return roc_auc_score(sub["is_case"], p)

    results = []
    for name, Xcols in [("base", base_cols), ("full", base_cols + [prs_col])]:
        X = sm.add_constant(fit[Xcols])
        m = sm.Logit(fit["is_case"], X).fit(disp=0)
        # Standardized OR
        if prs_col in Xcols:
            b = m.params[prs_col]
            se = m.bse[prs_col]
            p = m.pvalues[prs_col]
            or_ = np.exp(b)
            or_lo = np.exp(b - 1.96 * se)
            or_hi = np.exp(b + 1.96 * se)
            auc_fit = _auc(fit, Xcols, m.params)
            auc_eval = _auc(evl, Xcols, m.params)
            results.append({
                "model": name, "PRS": prs_col, "OR_perSD": or_,
                "OR_low95": or_lo, "OR_high95": or_hi, "P": p,
                "AIC": m.aic, "BIC": m.bic,
                "AUC_fit": auc_fit, "AUC_test": auc_eval,
                "n_fit": len(fit), "n_test": len(evl),
            })
        else:
            results.append({
                "model": name, "PRS": prs_col, "OR_perSD": np.nan,
                "OR_low95": np.nan, "OR_high95": np.nan, "P": np.nan,
                "AIC": m.aic, "BIC": m.bic,
                "AUC_fit": _auc(fit, Xcols, m.params), "AUC_test": _auc(evl, Xcols, m.params),
                "n_fit": len(fit), "n_test": len(evl),
            })
    return results


def main():
    ap = argparse.ArgumentParser(description="APAG PRS validation + diabetes association assessment")
    ap.add_argument("--prs-dir", default=DEFAULT_PRS_DIR,
                    help="PRS directory (containing APAG_PRS_raw.csv), default prs_analysis")
    ap.add_argument("--cohort", default=None,
                    help="Analysis cohort csv, default <prs-dir>/analysis_cohort.csv")
    ap.add_argument("--out-dir", default=None, help="Output directory, default same as --prs-dir")
    args = ap.parse_args()
    prs_dir = args.prs_dir
    cohort = args.cohort or os.path.join(prs_dir, "analysis_cohort.csv")
    if not os.path.exists(cohort):
        fallback = os.path.join(RES_ROOT, "prs_analysis", "analysis_cohort.csv")
        if os.path.exists(fallback):
            cohort = fallback
    out_dir = args.out_dir or prs_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(" 4_1 APAG PRS validation + diabetes association assessment")
    print("=" * 60)
    print(f"  PRS directory: {prs_dir}")
    df = load_data(prs_dir, cohort)
    print(f"  Analysis cohort: {len(df)}  (discovery={int((df['cv_split']=='discovery').sum())}, "
          f"test={int((df['cv_split']=='test').sum())})")

    # A. PRS predicts APAG
    print("\n--- A. APAG PRS validation (predicting measured delta) ---")
    val_rows = []
    for tag in ["discovery", "test"]:
        r = validate_apag_prediction(df, tag)
        if r:
            val_rows.append(r)
            print(f"  [{tag}] n={r['n']}  β={r['beta_APAGPRS']:.4f} P={r['P']:.3e} "
                  f"ΔR²={r['delta_R2']:.4f}")
    pd.DataFrame(val_rows).to_csv(os.path.join(out_dir, "prs_apag_validation.csv"), index=False)

    # B/C/D. Diabetes association + increment
    print("\n--- B/C/D. Diabetes logistic (fit on discovery, evaluated on test) ---")
    all_rows = []
    for tag, prs_col in [("full", "PRS_full_z"), ("without_ctrb12", "PRS_without_ctrb12_z")]:
        rows = run_logistic_models(df, "discovery", "test", prs_col)
        for r in rows:
            r["PRS_version"] = tag
            all_rows.append(r)
    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(out_dir, "diabetes_prs_association.csv"), index=False)

    # Print core results
    print("\n" + out[["PRS_version", "model", "OR_perSD", "OR_low95", "OR_high95",
                      "P", "AIC", "AUC_fit", "AUC_test"]].to_string(index=False))
    print(f"\n[Saved] {out_dir}/prs_apag_validation.csv, diabetes_prs_association.csv")


if __name__ == "__main__":
    main()
