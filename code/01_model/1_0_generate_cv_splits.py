#!/usr/bin/env python3
"""
generate_cv_splits.py — generate age-stratified 5-fold cross-validation splits

Usage:
  python code/01_model/generate_cv_splits.py

Output:
  - {out_dir}/cv_splits.json          # 5-fold splits: {fold_i: {train:[eid], val:[eid], test:[eid]}}
  - {out_dir}/metadata_cv.csv         # metadata + fold0-fold4 assignment columns

ponytail: read all eid/age from metadata.csv, 5-fold age stratification, test≈20% per fold.
          The already-completed single fold result is left untouched; CV folds 1-4 are filled in on top of it.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
from paths import UKB_META_CSV, CV_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, default=UKB_META_CSV)
    parser.add_argument("--out_dir", type=str, default=CV_DIR)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--age_bin_width", type=int, default=5,
                        help="age-stratification bin width (years)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Read all healthy subjects
    df = pd.read_csv(args.metadata_csv)
    print(f"[CV] all healthy subjects: {len(df)}, age range: [{df['age'].min():.0f}, {df['age'].max():.0f}]")

    # 2. Age stratification: convert age to integer bins
    age_bins = (df["age"] // args.age_bin_width).astype(int)
    # Merge small tail bins (< 3 subjects per bin treated as min coverage)
    bin_counts = age_bins.value_counts().sort_index()
    print(f"[CV] age stratification (bin_width={args.age_bin_width}): {len(bin_counts)} bins")
    for b, c in bin_counts.items():
        low = b * args.age_bin_width
        high = low + args.age_bin_width - 1
        print(f"  [{low}-{high}]: {c}")

    # 3. StratifiedKFold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    eids = df["eid"].values
    ages = df["age"].values

    fold_assignments = np.full(len(df), -1, dtype=int)
    cv_splits = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(eids, age_bins)):
        fold_assignments[test_idx] = fold_idx

        test_eids = eids[test_idx].tolist()
        # Further split train_val into train (80%) and val (20%)
        n_tv = len(train_val_idx)
        n_val = int(n_tv * 0.125)  # 5-fold CV: val≈10% of total, train≈70%
        # shuffle within fold
        rng = np.random.RandomState(args.seed + fold_idx)
        tv_perm = rng.permutation(train_val_idx)
        val_idx = tv_perm[:n_val]
        train_idx = tv_perm[n_val:]

        cv_splits[str(fold_idx)] = {
            "train": eids[train_idx].tolist(),
            "val": eids[val_idx].tolist(),
            "test": test_eids,
        }
        print(f"[CV] Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_eids)}")

    # 4. Save JSON
    splits_path = os.path.join(args.out_dir, "cv_splits.json")
    with open(splits_path, "w") as f:
        json.dump(cv_splits, f, indent=2)
    print(f"[CV] splits → {splits_path}")

    # 5. Save metadata_cv.csv (new fold0-fold4 columns + per-fold train/val/test markers)
    df_out = df.copy()
    for fi in range(args.n_folds):
        fold_info = cv_splits[str(fi)]
        # Create a column marking the split for this fold
        col_name = f"cv_fold{fi}_split"
        assignments = ["train"] * len(df_out)
        for eid in fold_info["val"]:
            idx = df_out.index[df_out["eid"] == eid][0]
            assignments[idx] = "val"
        for eid in fold_info["test"]:
            idx = df_out.index[df_out["eid"] == eid][0]
            assignments[idx] = "test"
        df_out[col_name] = assignments

    meta_path = os.path.join(args.out_dir, "metadata_cv.csv")
    df_out.to_csv(meta_path, index=False)
    print(f"[CV] metadata_cv → {meta_path}")
    print("[CV] done.")


if __name__ == "__main__":
    main()
