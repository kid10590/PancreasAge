#!/usr/bin/env python3
"""
2_0_merge_results.py — Merge CV inference results with ICD-10/phenotype data, output tables for downstream analysis

Inputs:
  - master_ukb_predictions.csv   (output of 1_1_infer_all_cv.py, per-subject predictions)
  - icd-10-all_with_flags_1.csv  (ICD-10 codes + is_healthy)
  - age_sex_bmi_etal_participant.csv (age/sex/blood pressure/BMI/waist circumference)

Outputs:
  - merged_phenotype.csv          (full merged table, for 2_1~2_5 downstream analysis)
  - merged_phenotype_test.csv     (test + ill only, for phenotype analysis)

Usage:
  python code/02_downstream/2_0_merge_results.py
  python code/02_downstream/2_0_merge_results.py --master-csv /path/to/master_ukb_predictions.csv
"""
import os, sys, argparse
import pandas as pd
import numpy as np

# ── Paths ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT, OUT_ROOT, INFER_DIR  # noqa: E402
DEFAULT_MASTER = os.path.join(INFER_DIR, "master_ukb_predictions.csv")
DEFAULT_ICD10  = os.path.join(DATA_ROOT, "icd-10-all_with_flags_1.csv")
DEFAULT_PHENO  = os.path.join(DATA_ROOT, "age_sex_bmi_etal_participant.csv")
DEFAULT_OUT    = os.path.join(OUT_ROOT, "Disease_Phenotype_Analysis")


def main():
    parser = argparse.ArgumentParser(description="Merge CV inference results with phenotype/ICD data")
    parser.add_argument("--master-csv", type=str, default=DEFAULT_MASTER)
    parser.add_argument("--icd10-csv", type=str, default=DEFAULT_ICD10)
    parser.add_argument("--pheno-csv", type=str, default=DEFAULT_PHENO)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Read master_ukb_predictions ──
    print(f"[1/4] Reading master predictions: {args.master_csv}")
    df_master = pd.read_csv(args.master_csv, low_memory=False)
    df_master["eid"] = df_master["eid"].astype(str).str.strip()
    print(f"      {len(df_master):,} subjects  "
          f"(healthy={int((df_master['is_healthy']==1).sum()):,}, "
          f"unhealthy={int((df_master['is_healthy']==0).sum()):,})")

    # ── 2. Read ICD-10 ──
    print(f"[2/4] Reading ICD-10: {args.icd10_csv}")
    df_icd = pd.read_csv(args.icd10_csv, low_memory=False)
    df_icd["eid"] = df_icd["eid"].astype(str).str.strip()
    # Keep only the needed columns (deduplicate eid)
    icd_cols = ["eid", "date", "icd-10", "is_healthy"]
    icd_cols += [c for c in df_icd.columns if c.startswith("p41280_a")]
    icd_cols += ["_exam_date"]
    icd_cols = [c for c in icd_cols if c in df_icd.columns]
    df_icd = df_icd[icd_cols].drop_duplicates(subset="eid", keep="first")
    print(f"      {len(df_icd):,} subjects (ICD-10)")

    # ── 3. Read phenotype data ──
    print(f"[3/4] Reading phenotype: {args.pheno_csv}")
    df_pheno = pd.read_csv(args.pheno_csv, low_memory=False)
    df_pheno["eid"] = df_pheno["eid"].astype(str).str.strip()
    # If an age column exists and duplicates master, add a suffix to avoid conflict
    if "age" in df_pheno.columns:
        df_pheno = df_pheno.rename(columns={"age": "age_pheno"})
    df_pheno = df_pheno.drop_duplicates(subset="eid", keep="first")
    print(f"      {len(df_pheno):,} subjects (phenotype)")

    # ── 4. Merge ──
    print(f"[4/4] Merging...")
    # Step A: master × ICD-10 (inner join: keep only those with ICD info)
    df = df_master.merge(df_icd, on="eid", how="inner", suffixes=("", "_icd"))
    n1 = len(df)
    print(f"      master ∩ ICD-10: {n1:,} subjects")

    # Step B: × phenotype (left join: keep subjects even if phenotype is missing)
    df = df.merge(df_pheno, on="eid", how="left")
    n_missing_pheno = int(df["sex"].isna().sum())
    print(f"      + phenotype: {len(df):,} subjects (missing phenotype: {n_missing_pheno:,})")

    # ── 5. Tidy columns ──
    # split column: healthy→test, unhealthy→ukb_test_ill (for downstream scripts)
    if "is_healthy" in df.columns:
        df["split"] = df["is_healthy"].map({1: "ukb_test", 0: "ukb_test_ill"})
    # Ensure sex is in standard format
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1: "Male", 0: "Female", "Male": "Male", "Female": "Female",
                                     "M": "Male", "F": "Female"}).fillna(df["sex"])
    # Ensure is_healthy is int
    if "is_healthy" in df.columns:
        df["is_healthy"] = df["is_healthy"].astype(int)
    # age_use column (for subsequent analysis)
    df["age_use"] = df["age_gt"]
    # Use the bias correction fitted on the healthy training set consistently, to avoid inconsistent definitions from unhealthy self-fitted correction.
    df["delta"] = df["age_pred_bc"] - df["age_gt"]

    # ── 6. Save ──
    # Full
    full_path = os.path.join(args.out_dir, "merged_phenotype.csv")
    df.to_csv(full_path, index=False)
    print(f"\n  Full saved: {full_path} ({len(df):,} subjects)")

    # test + ill only (for phenotype analysis)
    df_analysis = df[df["split"].isin(["ukb_test", "ukb_test_ill"])].copy()
    analysis_path = os.path.join(args.out_dir, "merged_phenotype_test_bc_self.csv")
    df_analysis.to_csv(analysis_path, index=False)
    n_test = int((df_analysis["split"] == "ukb_test").sum())
    n_ill  = int((df_analysis["split"] == "ukb_test_ill").sum())
    print(f"  Analysis saved: {analysis_path} ({len(df_analysis):,} subjects, "
          f"test={n_test:,}, ill={n_ill:,})")

    # ── 7. Diagnostic print ──
    print(f"\n  === Key column statistics ===")
    for col in ["age_gt", "age_pred_bc_self", "delta", "bmi", "sbp_0", "sbp_1",
                "dbp_0", "dbp_1", "Waist circumference", "sex", "split", "is_healthy"]:
        if col in df.columns:
            n_nonnull = int(df[col].notna().sum())
            print(f"    {col}: {n_nonnull:,}/{len(df):,} non-null")
        else:
            print(f"    {col}: MISSING")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
