#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_1 Fuyi (affiliated hospital) healthy-subject intersection table

Task:
  - Input: <DATA_ROOT>/fuyidata_health.xlsx (eid, Name, age, sex)
         <DATA_ROOT>/fuyidata_health.csv (eid, age)
  - The eid sets of the two tables match (1373 subjects); take their merged intersection (supplement Name/sex)
  - Then intersect with imaging npy files: fuyi_ext_npy(1057)
  - Output the 1057 subjects that have imaging

Output (utf-8-sig encoding, for easy viewing of Chinese in Excel):
  <OUT_ROOT>/cohort_tables/fuyidata_health_merged_all.csv   (1373, full intersection)
  <OUT_ROOT>/cohort_tables/fuyidata_health_images.csv       (1057, with-imaging subset)
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT, OUT_ROOT  # noqa: E402

DEFAULT_XLSX = os.path.join(DATA_ROOT, "fuyidata_health.xlsx")
DEFAULT_CSV = os.path.join(DATA_ROOT, "fuyidata_health.csv")
DEFAULT_EXT_NPY = os.path.join(DATA_ROOT, "ext_healthy_npy")
DEFAULT_PART3_NPY = os.path.join(DATA_ROOT, "obesity_cohort")
DEFAULT_OUT = os.path.join(OUT_ROOT, "cohort_tables")


def _image_ids(npy_root: str):
    img = os.path.join(npy_root, "images_npy")
    ids = set()
    for f in os.listdir(img):
        if f.endswith(".npy") and f[:-4].strip().isdigit():
            ids.add(int(f[:-4]))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--ext_npy", default=DEFAULT_EXT_NPY)
    ap.add_argument("--part3_npy", default=DEFAULT_PART3_NPY)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    x = pd.read_excel(args.xlsx, sheet_name="Sheet3")
    c = pd.read_csv(args.csv)

    # Keep only valid numeric eid
    for df in (x, c):
        df["eid"] = pd.to_numeric(df["eid"], errors="coerce")
        df.dropna(subset=["eid"], inplace=True)
        df["eid"] = df["eid"].astype(int)

    # Merge intersection: use xlsx as the primary table (with Name/sex), validate against the csv eids
    merged = x[x["eid"].isin(set(c["eid"]))].copy()
    merged.rename(columns={"Name": "name", "性别": "sex"}, inplace=True)
    merged = merged[["eid", "name", "age", "sex"]].sort_values("eid").reset_index(drop=True)

    ext_ids = _image_ids(args.ext_npy)

    # Fuyi imaging split annotation
    meta = pd.read_csv(os.path.join(args.ext_npy, "metadata.csv"))
    meta["eid"] = meta["eid"].astype(int)
    meta = meta.rename(columns={"split": "ext_split"})[["eid", "ext_split"]]

    merged["has_ext_npy"] = merged["eid"].isin(ext_ids).astype(int)
    merged = merged.merge(meta, on="eid", how="left")

    images = merged[merged["has_ext_npy"] == 1].copy()

    merged.to_csv(os.path.join(args.out, "fuyidata_health_merged_all.csv"), index=False, encoding="utf-8-sig")
    images.to_csv(os.path.join(args.out, "fuyidata_health_images.csv"), index=False, encoding="utf-8-sig")

    print(f"xlsx valid eids: {x['eid'].nunique()}, csv valid eids: {c['eid'].nunique()}")
    print(f"merged intersection: {len(merged)}, with ext_npy images: {len(images)}")
    print(f"saved: {args.out}/fuyidata_health_merged_all.csv (1373)")
    print(f"saved: {args.out}/fuyidata_health_images.csv ({len(images)})")


if __name__ == "__main__":
    main()