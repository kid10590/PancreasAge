#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_4 Fuyi ext_part3 data table + baseline statistics

Inputs:
  - <DATA_ROOT>/obesity_cohort/metadata_norm.csv   (211 clinical records)
  - Results/swin3d_kl_reg_0801/infer_cv/predictions_ext_part3.csv    (5-fold average pancreatic age prediction)

Output (cohort_tables_pred/):
  - fuyi_part3_cohort.csv             all columns of metadata_norm + age_pred_raw + age_pred_bc
  - baseline_stats_fuyi_part3.csv     baseline statistics with the same structure as baseline_stats_ill_healthy.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT, OUT_ROOT, INFER_DIR  # noqa: E402

DEFAULT_META = os.path.join(DATA_ROOT, "obesity_cohort", "metadata_norm.csv")
DEFAULT_PRED = os.path.join(INFER_DIR, "predictions_ext_part3.csv")
DEFAULT_OUT = os.path.join(OUT_ROOT, "cohort_tables_pred")

CONT_VARS = [
    "age", "bmi",
    "pancreatic_head_fat_percent", "pancreatic_body_fat_percent", "pancreatic_tail_fat_percent",
    "mean_pancreatic_fat_percent", "anterior_subcutaneous_fat_thickness",
    "fasting_insulin", "hba1c", "fasting_c_peptide", "vitamin_d", "cea", "ca19_9",
    "high_sensitivity_crp", "alpha1_antitrypsin", "total_protein", "albumin", "globulin",
    "total_cholesterol", "triglyceride", "hdl_cholesterol", "ldl_cholesterol", "lipoprotein_a",
    "urea_nitrogen", "creatinine", "alanine_aminotransferase", "aspartate_aminotransferase",
]

CAT_VARS = ["sex", "mri_pancreatic_steatosis", "ogtt_2h_glucose"]


def _summary(vals: pd.Series) -> dict:
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if len(v) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                    min=np.nan, max=np.nan, missing=int(pd.to_numeric(vals, errors="coerce").isna().sum()))
    return dict(
        n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)),
        median=float(v.median()), q1=float(v.quantile(0.25)), q3=float(v.quantile(0.75)),
        min=float(v.min()), max=float(v.max()),
        missing=int(pd.to_numeric(vals, errors="coerce").isna().sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = pd.read_csv(args.meta)
    # Drop all-empty Unnamed columns
    meta = meta.loc[:, ~meta.columns.str.startswith("Unnamed")]
    meta["eid"] = meta["eid"].astype(str)
    # Consistent with external metric recomputation: exclude subjects <18 years old (see recompute_external_metrics.py)
    meta = meta[pd.to_numeric(meta.get("age"), errors="coerce") >= 18].reset_index(drop=True)

    pred = pd.read_csv(args.pred, usecols=["eid", "age_pred_raw", "age_pred_bc"])
    pred["eid"] = pred["eid"].astype(str)

    cohort = meta.merge(pred, on="eid", how="left")
    missing = cohort["age_pred_raw"].isna().sum()
    cohort.to_csv(os.path.join(args.out, "fuyi_part3_cohort.csv"), index=False, encoding="utf-8-sig")
    print(f"fuyi_part3_cohort.csv: {len(cohort)} rows x {cohort.shape[1]} cols, missing pred={missing}")

    # ── Baseline statistics (same structure as baseline_stats_ill_healthy.csv) ──
    rows = []
    for var in CONT_VARS:
        s = _summary(cohort[var])
        rows.append(dict(group="fuyi_part3", variable=var,
                         value_n=s["n"], value_mean=s["mean"], value_sd=s["sd"],
                         value_median=s["median"], value_q1=s["q1"], value_q3=s["q3"],
                         value_min=s["min"], value_max=s["max"], value_missing=s["missing"],
                         male=np.nan, male_pct=np.nan, female=np.nan, female_pct=np.nan))

    # Categorical variables: one row per level (value_n = number of subjects at that level)
    for var in CAT_VARS:
        missing_cat = int(cohort[var].isna().sum())
        vc = cohort[var].value_counts(dropna=False)
        for level, cnt in vc.items():
            rows.append(dict(group="fuyi_part3", variable=f"{var} = {level}",
                             value_n=int(cnt), value_mean=np.nan, value_sd=np.nan,
                             value_median=np.nan, value_q1=np.nan, value_q3=np.nan,
                             value_min=np.nan, value_max=np.nan, value_missing=missing_cat,
                             male=np.nan, male_pct=np.nan, female=np.nan, female_pct=np.nan))
        # Sex: 1=Male, 2=Female, 0=unfilled → statistics for male/female columns
        if var == "sex":
            se = pd.to_numeric(cohort["sex"], errors="coerce")
            n = int(se.notna().sum())
            male = int((se == 1).sum()); female = int((se == 2).sum())
            n_defined = male + female
            for r in rows:
                if r["variable"] == "sex = 1":
                    r.update(male=male, male_pct=100 * male / n_defined if n_defined else np.nan,
                             female=female, female_pct=100 * female / n_defined if n_defined else np.nan)
                if r["variable"] == "sex = 2":
                    r.update(male=male, male_pct=100 * male / n_defined if n_defined else np.nan,
                             female=female, female_pct=100 * female / n_defined if n_defined else np.nan)

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(args.out, "baseline_stats_fuyi_part3.csv"), index=False)
    print(f"baseline_stats_fuyi_part3.csv: {len(stats)} rows")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()