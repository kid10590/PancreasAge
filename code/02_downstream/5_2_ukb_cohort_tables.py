#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_2 UKB diseased/healthy cohort tables + baseline statistics

Cohort definitions (all have MRI npy imaging):
  - Diseased cohort (ill):     icd-10-all_with_flags.csv is_healthy==0 ∩ ukb_healthy_npy/images_npy
  - Healthy cohort (healthy):  icd-10-all_with_flags.csv is_healthy==1 ∩ ukb_healthy_npy/images_npy

Baseline variables (age_sex_bmi_etal_participant.csv): sex, age, bmi, waist circumference (Waist circumference),
two systolic/diastolic blood pressures sbp_0/1, dbp_0/1

Output (cohort_tables/):
  ukb_ill_cohort.csv / ukb_healthy_cohort.csv      baseline + icd string per subject
  icd_code_counts_ill.csv                           number of subjects per ICD code (3-char/4-char) in the diseased cohort
  baseline_stats_ill.csv / baseline_stats_healthy.csv   mean±SD/median[IQR]/missing for each variable
  baseline_stats_compare.csv                        diseased vs healthy between-group comparison (t/Wilcoxon/chi-square)
"""
import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT, OUT_ROOT, UKB_NPY_DIR  # noqa: E402

DEFAULT_FLAGS = os.path.join(DATA_ROOT, "icd-10-all_with_flags.csv")
DEFAULT_NPY = UKB_NPY_DIR
DEFAULT_BASE = os.path.join(DATA_ROOT, "age_sex_bmi_etal_participant.csv")
DEFAULT_OUT = os.path.join(OUT_ROOT, "cohort_tables")

CONT_VARS = ["age", "bmi", "Waist circumference", "sbp_0", "sbp_1", "dbp_0", "dbp_1"]


def _icd_codes(s):
    if pd.isna(s):
        return [], []
    c3, c4 = [], []
    for it in str(s).split("|"):
        it = it.strip()
        m = re.match(r"([A-Z]\d{2}(?:\.\d)?)", it)
        if m:
            code = m.group(1)
            c4.append(code)
            c3.append(code[:3])
    return c3, c4


def _fmt_p(p):
    """p-value formatting: show <1e-300 when floating-point underflows to 0, otherwise scientific notation"""
    if pd.isna(p):
        return "—"
    if p == 0:
        return "<1e-300"
    return f"{p:.3e}"


def _summary(vals: pd.Series) -> dict:
    v = pd.to_numeric(vals, errors="coerce")
    v = v[v.notna()]
    if len(v) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan, min=np.nan, max=np.nan, missing=int(vals.isna().sum()))
    return dict(
        n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)),
        median=float(v.median()), q1=float(v.quantile(0.25)), q3=float(v.quantile(0.75)),
        min=float(v.min()), max=float(v.max()), missing=int(vals.isna().sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default=DEFAULT_FLAGS)
    ap.add_argument("--npy", default=DEFAULT_NPY)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Cohort definitions ---
    flags = pd.read_csv(args.flags, usecols=["eid", "icd-10", "exam_date", "is_healthy"], low_memory=False)
    flags["eid"] = flags["eid"].astype(str)
    npy_ids = set(f[:-4] for f in os.listdir(args.npy) if f.endswith(".npy"))
    flags["has_npy"] = flags["eid"].isin(npy_ids)

    ill = flags[(flags["is_healthy"] == 0) & (flags["has_npy"])].copy()
    healthy = flags[(flags["is_healthy"] == 1) & (flags["has_npy"])].copy()
    print(f"ill = {len(ill)}, healthy = {len(healthy)}")

    base = pd.read_csv(args.base)
    base["eid"] = base["eid"].astype(str)

    # --- Cohort tables ---
    for name, df in [("ill", ill), ("healthy", healthy)]:
        df = df.merge(base, on="eid", how="left")
        cols = ["eid", "age", "sex", "bmi", "Waist circumference", "sbp_0", "sbp_1", "dbp_0", "dbp_1", "exam_date", "icd-10", "is_healthy"]
        df = df[cols].sort_values("eid").reset_index(drop=True)
        df.to_csv(os.path.join(args.out, f"ukb_{name}_cohort.csv"), index=False)
        print(f"ukb_{name}_cohort.csv saved: {len(df)}")

    # --- ICD code counts (diseased cohort) ---
    cnt3, cnt4 = Counter(), Counter()
    for s in ill["icd-10"]:
        c3, c4 = _icd_codes(s)
        cnt3.update(set(c3))
        cnt4.update(set(c4))
    icd3 = pd.DataFrame(sorted(cnt3.items(), key=lambda x: -x[1]), columns=["icd3", "n_people"])
    icd4 = pd.DataFrame(sorted(cnt4.items(), key=lambda x: -x[1]), columns=["icd4", "n_people"])
    icd3.to_csv(os.path.join(args.out, "icd_code_counts_ill.csv"), index=False)
    icd4.to_csv(os.path.join(args.out, "icd_code_counts_ill_4char.csv"), index=False)
    print(f"icd_code_counts_ill.csv: {len(icd3)} distinct 3-char codes")

    # --- Baseline statistics ---
    ill_b = pd.read_csv(os.path.join(args.out, "ukb_ill_cohort.csv"))
    hea_b = pd.read_csv(os.path.join(args.out, "ukb_healthy_cohort.csv"))

    rows = []
    for grp, df in [("ill", ill_b), ("healthy", hea_b)]:
        for var in CONT_VARS:
            row = dict(group=grp, variable=var, **{f"value_{k}": v for k, v in _summary(df[var]).items()})
            rows.append(row)
        # Sex categories
        se = df["sex"].fillna("Unknown")
        n = len(se)
        male = int((se == "Male").sum())
        female = int((se == "Female").sum())
        rows.append(dict(group=grp, variable="sex (categorical)", value_n=n, value_mean=np.nan, value_sd=np.nan,
                         value_median=np.nan, value_q1=np.nan, value_q3=np.nan, value_min=np.nan, value_max=np.nan,
                         value_missing=int(se.isna().sum()), male=male, male_pct=100 * male / n, female=female, female_pct=100 * female / n))
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(args.out, "baseline_stats_ill_healthy.csv"), index=False, encoding="utf-8-sig")

    # --- Between-group comparison ---
    comp = []
    for var in CONT_VARS:
        a = pd.to_numeric(ill_b[var], errors="coerce").dropna()
        b = pd.to_numeric(hea_b[var], errors="coerce").dropna()
        t, pt = stats.ttest_ind(a, b, equal_var=False)
        _, pw = stats.mannwhitneyu(a, b, alternative="two-sided")
        comp.append(dict(variable=var, n_ill=len(a), n_healthy=len(b), mean_ill=a.mean(), mean_healthy=b.mean(),
                         delta_mean=b.mean() - a.mean(), t_p=_fmt_p(pt), wilcoxon_p=_fmt_p(pw),
                         chi2_p="—", chi2="—", dof="—"))
    ctab = pd.crosstab(
        pd.concat([ill_b["sex"], hea_b["sex"]], ignore_index=True),
        pd.Series(["ill"] * len(ill_b) + ["healthy"] * len(hea_b), name="group"))
    chi2, pchi, dof, _ = stats.chi2_contingency(ctab)
    comp.append(dict(variable="sex", n_ill=len(ill_b), n_healthy=len(hea_b), mean_ill=np.nan, mean_healthy=np.nan,
                     delta_mean=np.nan, t_p="—", wilcoxon_p="—", chi2_p=_fmt_p(pchi), chi2=float(chi2), dof=int(dof)))
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(os.path.join(args.out, "baseline_stats_compare.csv"), index=False,encoding="utf-8-sig")
    print("baseline_stats_ill_healthy.csv / baseline_stats_compare.csv saved")


if __name__ == "__main__":
    main()