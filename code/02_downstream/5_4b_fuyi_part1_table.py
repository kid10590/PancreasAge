#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_4b_fuyi_part1_table.py — Fuyi ext_part1 data table + baseline statistics
(same structure as 5_4_fuyi_part3_table.py, exclude <18 years old, n=1053)

Inputs:
  - <OUT_ROOT>/cohort_tables/fuyidata_health_images.csv   (1057, eid/name/age/sex)
  - Results/swin3d_kl_reg_0801/infer_cv/predictions_ext_part1.csv      (5-fold average pancreatic age prediction, <18 already excluded)

Outputs:
  - fuyi_part1_cohort.csv            fuyidata + age_pred_raw + age_pred_bc
  - baseline_stats_fuyi_part1.csv    baseline statistics (age + sex)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT, INFER_DIR  # noqa: E402

DEFAULT_META = os.path.join(OUT_ROOT, "cohort_tables", "fuyidata_health_images.csv")
DEFAULT_PRED = os.path.join(INFER_DIR, "predictions_ext_part1.csv")
DEFAULT_OUT = os.path.join(OUT_ROOT, "cohort_tables_pred_recalc")
MIN_AGE = 18.0

CONT_VARS = ["age"]
CAT_VARS = ["sex"]


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
    os.makedirs(DEFAULT_OUT, exist_ok=True)
    meta = pd.read_csv(DEFAULT_META)
    meta["eid"] = meta["eid"].astype(str)
    # Exclude <18 years old (consistent with external metric recomputation)
    meta = meta[pd.to_numeric(meta["age"], errors="coerce") >= MIN_AGE].reset_index(drop=True)

    pred = pd.read_csv(DEFAULT_PRED, usecols=["eid", "age_pred_raw", "age_pred_bc"])
    pred["eid"] = pred["eid"].astype(str)
    cohort = meta.merge(pred, on="eid", how="inner")
    missing = cohort["age_pred_raw"].isna().sum()
    cohort.to_csv(os.path.join(DEFAULT_OUT, "fuyi_part1_cohort.csv"), index=False, encoding="utf-8-sig")
    print(f"fuyi_part1_cohort.csv: {len(cohort)} rows x {cohort.shape[1]} cols, missing pred={missing}")

    rows = []
    for var in CONT_VARS:
        s = _summary(cohort[var])
        rows.append(dict(group="fuyi_part1", variable=var,
                         value_n=s["n"], value_mean=s["mean"], value_sd=s["sd"],
                         value_median=s["median"], value_q1=s["q1"], value_q3=s["q3"],
                         value_min=s["min"], value_max=s["max"], value_missing=s["missing"],
                         male=np.nan, male_pct=np.nan, female=np.nan, female_pct=np.nan))

    for var in CAT_VARS:
        missing_cat = int(cohort[var].isna().sum())
        vc = cohort[var].value_counts(dropna=False)
        male = int((cohort[var] == "男").sum())
        female = int((cohort[var] == "女").sum())
        n_defined = male + female
        for level, cnt in vc.items():
            is_male = str(level) == "男"
            rows.append(dict(group="fuyi_part1", variable=f"{var} = {level}",
                             value_n=int(cnt), value_mean=np.nan, value_sd=np.nan,
                             value_median=np.nan, value_q1=np.nan, value_q3=np.nan,
                             value_min=np.nan, value_max=np.nan, value_missing=missing_cat,
                             male=male if is_male else np.nan,
                             male_pct=100 * male / n_defined if is_male and n_defined else np.nan,
                             female=female if is_male else np.nan,
                             female_pct=100 * female / n_defined if is_male and n_defined else np.nan))

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(DEFAULT_OUT, "baseline_stats_fuyi_part1.csv"), index=False, encoding="utf-8-sig")
    print(f"baseline_stats_fuyi_part1.csv: {len(stats)} rows")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()