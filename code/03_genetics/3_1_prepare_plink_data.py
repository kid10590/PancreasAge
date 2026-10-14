#!/usr/bin/env python
"""
3_1_prepare_plink_data.py — Prepare data for APAG PRS independent-contribution analysis

What it does:
  1. Merge: master predictions (delta) + diabetes status + covariates (Sex/Age/Age2/Array/PC1-10)
  2. 7:3 split into discovery/test (stratified by diabetes status, fixed seed)
  3. Output:
     - analysis_cohort.csv           # full analysis cohort (with diabetes phenotype + covariates)
     - keep_discovery.id / keep_test.id / keep_all.id   # PLINK --keep (two columns FID IID)
     - pheno_discovery.csv / pheno_test.csv             # phenotypes for association
     - phenotypes.txt / covariates.cov                  # GWAS inputs (all samples)

Design notes:
  - GWAS and diabetes-assessment samples fully overlap → 7:3 split: discovery (70%) runs GWAS to build the PRS,
    test (30%) independently evaluates PRS association (avoids winner's curse)
  - Use the full cohort (no 1:1 matching) to preserve statistical power
  - Imaging center/genotyping batch fields are unavailable → covariates use Age/Sex/PC1-10/Array
  - MR uses the all-sample GWAS → this script does not touch gwas_plink_results (all samples), keeping MR inputs stable
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

RNG_SEED = 20260808

RES_ROOT = OUT_ROOT
GWAS_IN = os.path.join(RES_ROOT, "gwas_inputs")
PRS_DIR = os.path.join(RES_ROOT, "prs_analysis")
os.makedirs(PRS_DIR, exist_ok=True)

MASTER = os.path.join(RES_ROOT, "infer_cv", "master_ukb_predictions.csv")
DIABETES = os.path.join(RES_ROOT, "Disease_Phenotype_Analysis", "Diabetes", "organized_data.csv")


def main():
    # ── 1. Load master predictions (delta) ──
    print("[1/4] Loading master predictions + diabetes status + covariates ...")
    master = pd.read_csv(MASTER, low_memory=False)
    master["eid"] = master["eid"].astype(str)
    master["delta"] = pd.to_numeric(master["age_pred_bc"], errors="coerce") - pd.to_numeric(master["age_gt"], errors="coerce")
    keep_cols = ["eid", "age_gt", "age_pred_bc", "delta", "fold", "split", "is_healthy"]
    master = master[keep_cols]

    # Diabetes status (E10-E14 diagnosed before imaging)
    diab = pd.read_csv(DIABETES, usecols=["eid", "is_case"], low_memory=False)
    diab["eid"] = diab["eid"].astype(str)

    # Covariates (covariates.cov already has all samples)
    cov = pd.read_csv(os.path.join(GWAS_IN, "covariates.cov"), sep="\t", low_memory=False)
    cov = cov.rename(columns={"FID": "eid"})
    cov["eid"] = cov["eid"].astype(str)

    # ── 2. Merge ──
    df = master.merge(diab, on="eid", how="inner").merge(cov, on="eid", how="inner")
    df = df.dropna(subset=["delta", "is_case"])
    df = df[df["is_case"].isin([0, 1])]
    print(f"  Analysis cohort (genotype+APAG+diabetes): {len(df)}  cases: {int((df['is_case']==1).sum())} / controls: {int((df['is_case']==0).sum())}")

    # ── 3. 7:3 stratified split ──
    print("[2/4] 7:3 discovery/test stratified split ...")
    disc_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=RNG_SEED, stratify=df["is_case"]
    )
    df_disc = df.iloc[disc_idx].copy()
    df_test = df.iloc[test_idx].copy()
    print(f"  discovery: {len(df_disc)} (cases {int((df_disc['is_case']==1).sum())})")
    print(f"  test:      {len(df_test)} (cases {int((df_test['is_case']==1).sum())})")

    # ── 4. Save ──
    print("[3/4] Saving analysis cohort ...")
    df = df.copy()
    df["cv_split"] = "all"
    df.loc[disc_idx, "cv_split"] = "discovery"
    df.loc[test_idx, "cv_split"] = "test"
    df.to_csv(os.path.join(PRS_DIR, "analysis_cohort.csv"), index=False)

    for tag, sub in [("discovery", df_disc), ("test", df_test)]:
        sub[["eid", "delta", "is_case"]].to_csv(
            os.path.join(PRS_DIR, f"pheno_{tag}.csv"), index=False)

    # keep files (PLINK --keep requires two columns FID IID)
    for tag, sub in [("discovery", df_disc), ("test", df_test), ("all", df)]:
        keep = pd.DataFrame({"FID": sub["eid"], "IID": sub["eid"]})
        keep.to_csv(os.path.join(PRS_DIR, f"keep_{tag}.id"), sep="\t", index=False, header=False)

    # GWAS inputs (all-sample phenotypes/covariates, used by 3_2)
    print("[4/4] Saving GWAS inputs (all-sample phenotypes/covariates) ...")
    pheno = pd.DataFrame({
        "FID": df["eid"], "IID": df["eid"],
        "catb_delta_with_t1_bc_cole": df["delta"],
    })
    pheno.to_csv(os.path.join(GWAS_IN, "phenotypes.txt"), sep="\t", index=False, na_rep="NA")
    cov_out = df[["eid", "Sex", "Age", "Age2", "Array"] + [f"PC{i}" for i in range(1, 11)]].copy()
    cov_out = cov_out.rename(columns={"eid": "FID"})
    cov_out.insert(1, "IID", cov_out["FID"])
    cov_out.to_csv(os.path.join(GWAS_IN, "covariates.cov"), sep="\t", index=False)

    print("\n✓ Done!")
    print(f"  Analysis cohort: {PRS_DIR}/analysis_cohort.csv")
    print(f"  keep_discovery.id / keep_test.id / keep_all.id → {PRS_DIR}")
    print(f"  GWAS inputs (all samples): {GWAS_IN}/phenotypes.txt, covariates.cov")


if __name__ == "__main__":
    main()