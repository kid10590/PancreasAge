"""
Preprocess UKB NIfTI data (multiprocessing): save all samples as .npy, write only healthy subjects to metadata.csv

Pipeline:
  1. Read the CSV and filter healthy subjects with is_healthy == True
  2. Take all samples that have both an image and a mask (regardless of health status) for preprocessing
       Preprocessing: mask-centered crop/pad → (160,128,64), linear normalization inside the mask to [-1,1], background -1
  3. Preprocess all samples with multiprocessing, save .npy (images_npy/{eid}.npy)
  4. Split only healthy subjects into train/val/test at 7:1:2
  5. Save metadata.csv (healthy subjects only): eid, age, split

Usage:
  python code/00_data_prep/0_1_preprocess_ukb_healthy_to_npy.py
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT  # noqa: E402


def _mask_center(msk: np.ndarray):
    """Return the center of the mask bounding box; return the volume center if no mask"""
    idx = np.where(msk > 0)
    if idx[0].size == 0:
        H, W, D = msk.shape
        return (H // 2, W // 2, D // 2)
    xmin, xmax = idx[0].min(), idx[0].max()
    ymin, ymax = idx[1].min(), idx[1].max()
    zmin, zmax = idx[2].min(), idx[2].max()
    return ((xmin + xmax) // 2, (ymin + ymax) // 2, (zmin + zmax) // 2)


def _center_crop_or_pad(vol: np.ndarray, center_xyz, target_shape, pad_value):
    """Crop/pad to target_shape centered on center_xyz"""
    H, W, D = vol.shape
    th, tw, td = target_shape
    cx, cy, cz = center_xyz

    sh = cx - th // 2
    eh = sh + th
    sw = cy - tw // 2
    ew = sw + tw
    sd = cz - td // 2
    ed = sd + td

    dh0 = max(0, -sh)
    dh1 = max(0, eh - H)
    dw0 = max(0, -sw)
    dw1 = max(0, ew - W)
    dd0 = max(0, -sd)
    dd1 = max(0, ed - D)

    sh_clip = max(0, sh)
    eh_clip = min(H, eh)
    sw_clip = max(0, sw)
    ew_clip = min(W, ew)
    sd_clip = max(0, sd)
    ed_clip = min(D, ed)

    out = np.full((th, tw, td), pad_value, dtype=vol.dtype)
    out[dh0:th - dh1, dw0:tw - dw1, dd0:td - dd1] = (
        vol[sh_clip:eh_clip, sw_clip:ew_clip, sd_clip:ed_clip]
    )
    return out


def _normalize_inside_mask_to_neg1_pos1(img_c: np.ndarray, msk_c: np.ndarray):
    """Background is always -1; linearly normalize only inside the pancreas mask to [-1, 1]"""
    inside = img_c[msk_c > 0]
    img_norm = np.full_like(img_c, fill_value=-1.0, dtype=np.float32)
    if inside.size > 10:
        vmin, vmax = float(inside.min()), float(inside.max())
        if np.isfinite(vmax - vmin) and (vmax - vmin) > 1e-6:
            normed = (inside - vmin) / (vmax - vmin + 1e-6)
            normed = normed * 2.0 - 1.0
            img_norm[msk_c > 0] = normed.astype(np.float32)
        else:
            img_norm[msk_c > 0] = 0.0
    else:
        img_norm[msk_c > 0] = 0.0
    return img_norm


def _resample_to_spacing(img_nii, msk_nii, target_spacing):
    """Resample the NIfTI image and mask to the target spacing, return (img_fdata, msk_fdata)"""
    img_sitk = sitk.ReadImage(img_nii)
    msk_sitk = sitk.ReadImage(msk_nii)
    orig_sp = img_sitk.GetSpacing()
    orig_sz = img_sitk.GetSize()
    new_sz = [int(round(orig_sz[i] * orig_sp[i] / target_spacing[i])) for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(target_spacing); rs.SetSize(new_sz)
    rs.SetOutputDirection(img_sitk.GetDirection()); rs.SetOutputOrigin(img_sitk.GetOrigin())
    rs.SetTransform(sitk.Transform()); rs.SetDefaultPixelValue(0)
    rs.SetInterpolator(sitk.sitkLinear)
    img_r = rs.Execute(img_sitk)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    msk_r = rs.Execute(msk_sitk)
    return (sitk.GetArrayFromImage(img_r).astype(np.float32).transpose(2,1,0),
            sitk.GetArrayFromImage(msk_r).astype(np.float32).transpose(2,1,0))

def preprocess_one_sample(image_path: str, mask_path: str, crop_shape=(160, 128, 64), target_spacing=None):
    """
    Read NIfTI → preprocess → return (processed_image, processed_mask)
    The returned image is float32 and mask is uint8
    """
    # Optional resampling
    if target_spacing is not None:
        img, msk = _resample_to_spacing(image_path, mask_path, target_spacing)
        msk = (msk > 0).astype(np.uint8)
    else:
        img = nib.load(image_path).get_fdata().astype(np.float32)
        msk = (nib.load(mask_path).get_fdata().astype(np.float32) > 0).astype(np.uint8)

    cxyz = _mask_center(msk)
    img_c = _center_crop_or_pad(img, cxyz, crop_shape, pad_value=-1.0)
    msk_c = _center_crop_or_pad(msk, cxyz, crop_shape, pad_value=0)

    img_n = _normalize_inside_mask_to_neg1_pos1(img_c, msk_c)
    return img_n, msk_c


# ──────────────────────────────────────────────
# Splitting function (consistent with pancreas_age_dataset.py)
# ──────────────────────────────────────────────
def _split_data(ids, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """Split the id list by ratio, return (train_ids, val_ids, test_ids)"""
    from sklearn.model_selection import train_test_split
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_ids, temp_ids = train_test_split(
        ids, train_size=train_ratio, random_state=random_state
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_size, random_state=random_state
    )
    return train_ids, val_ids, test_ids


# ──────────────────────────────────────────────
# Multiprocessing worker (must be a top-level function for pickling)
# ──────────────────────────────────────────────
def _worker_process_sample(pid, images_dir, masks_dir, npy_dir, crop_shape, is_healthy, age, target_spacing=None):
    """Processing task for a single sample: read → preprocess → save .npy → return result"""
    try:
        ip = os.path.join(images_dir, f"{pid}_0000.nii.gz")
        mp = os.path.join(masks_dir, f"{pid}.nii.gz")
        img_n, _ = preprocess_one_sample(ip, mp, crop_shape, target_spacing)
        pid = int(pid)
        np.save(os.path.join(npy_dir, f"{pid}.npy"), img_n)
        return (pid, age, True, is_healthy)
    except Exception as e:
        return (pid, None, False, is_healthy)


def _worker_wrapper(args):
    """Unpack arguments to adapt to Pool.imap_unordered"""
    return _worker_process_sample(*args)


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess pancreas data and save as .npy")
    # UKB internal data (no resampling)
    parser.add_argument("--images-dir", default=os.path.join(DATA_ROOT, "images_ukb"))
    parser.add_argument("--masks-dir", default=os.path.join(DATA_ROOT, "masks_ukb"))
    parser.add_argument("--csv-path", default=os.path.join(DATA_ROOT, "icd-10-all_with_flags.csv"))
    parser.add_argument("--output-dir", default=os.path.join(DATA_ROOT, "ukb_healthy_npy"))
    parser.add_argument("--skip-ukb", action="store_true", help="Skip UKB (already processed)")
    # External test set (First Affiliated Hospital) — resample to 1.4x1.4x1.8
    parser.add_argument("--ext-images-dir", default=os.path.join(DATA_ROOT, "nifti", "external"))
    parser.add_argument("--ext-masks-dir", default=os.path.join(DATA_ROOT, "masks_external"))
    parser.add_argument("--ext-csv-path", default=os.path.join(DATA_ROOT, "external_clinical.csv"))
    parser.add_argument("--ext-output-dir", default=os.path.join(DATA_ROOT, "ext_healthy_npy"))
    parser.add_argument("--ext-target-spacing", type=float, nargs=3, default=[1.4, 1.4, 1.8],
                        help="Target spacing (H,W,D) for resampling the external test set; defaults to UKB original 1.4 1.4 1.8")
    parser.add_argument("--skip-ext", action="store_true", help="Skip the external test set")
    # Common
    parser.add_argument("--crop-shape", type=int, nargs=3, default=[160, 128, 64])
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--age-col", default="age")
    args = parser.parse_args()

    crop_shape = tuple(args.crop_shape)
    s_train, s_val, s_test = args.split_ratios
    num_workers = args.num_workers or cpu_count()
    num_workers = min(num_workers, 88)

    print("=" * 60)
    print("MRI preprocessing")
    print("=" * 60)
    print(f"  Crop shape (H,W,D): {crop_shape}")
    print(f"  Workers: {num_workers}")

    # # 1. UKB (no resampling)
    # if not args.skip_ukb:
    #     _process_dataset(
    #         images_dir=args.images_dir, masks_dir=args.masks_dir,
    #         csv_path=args.csv_path, output_dir=args.output_dir,
    #         crop_shape=crop_shape, num_workers=num_workers,
    #         split_ratios=(s_train, s_val, s_test), random_state=args.random_state,
    #         age_col=args.age_col, filter_healthy_only=True,
    #         fixed_split=None, target_spacing=None,
    #         label="UKB internal dataset",
    #     )
    # else:
    #     print("\n[skip] UKB internal dataset (--skip-ukb)")

    # 2. External test set (resample to 1.4x1.4x1.8 to match UKB voxels)
    if not args.skip_ext:
        ext_sp = tuple(args.ext_target_spacing)
        print(f"  External test set resample target spacing: {ext_sp}")
        _process_dataset(
            images_dir=args.ext_images_dir, masks_dir=args.ext_masks_dir,
            csv_path=args.ext_csv_path, output_dir=args.ext_output_dir,
            crop_shape=crop_shape, num_workers=num_workers,
            split_ratios=(1.0, 0.0, 0.0), random_state=args.random_state,
            age_col=args.age_col, filter_healthy_only=False,
            fixed_split="ext_test", target_spacing=ext_sp,
            label="external test set (First Affiliated Hospital, resampled to 1.4x1.4x1.8)",
        )
    else:
        print("\n[skip] external test set (--skip-ext)")

    print("\n" + "=" * 60)
    print("All preprocessing complete!")
    print("=" * 60)


def _process_dataset(images_dir, masks_dir, csv_path, output_dir,
                     crop_shape, num_workers, split_ratios, random_state,
                     age_col, filter_healthy_only, fixed_split, target_spacing, label):
    s_train, s_val, s_test = split_ratios
    print(f"\n{'='*60}")
    print(f"  [{label}]")
    print(f"  Images: {images_dir}")
    print(f"  Mask: {masks_dir}")
    print(f"  CSV:  {csv_path}")
    print(f"  Output: {output_dir}")
    if target_spacing is not None:
        print(f"  Resample spacing: {target_spacing}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["eid", age_col])
    df["eid"] = df["eid"].astype(str).str.strip()
    df = df[df["eid"] != ""]
    df['eid'] = df['eid'].apply(lambda x: f"{int(x):03d}") 
    if filter_healthy_only and "is_healthy" in df.columns:
        df = df[df["is_healthy"] == True].copy()

    target_ids = set(df["eid"])
    eid2age = dict(zip(df["eid"], df[age_col].astype(float)))
    print(f"  [1/5] CSV: {len(df)} rows, valid eid: {len(target_ids)}")

    nii_paths = sorted(glob.glob(os.path.join(images_dir, "*_0000.nii.gz")))
    nii_paths += sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    nii_paths += sorted(glob.glob(os.path.join(images_dir, "*.nii")))
    img_ids = set()
    for ip in nii_paths:
        b = os.path.basename(ip)
        img_ids.add(b.replace("_0000.nii.gz", "").replace(".nii.gz", "").replace(".nii", ""))

    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.nii.gz")))
    mask_paths += sorted(glob.glob(os.path.join(masks_dir, "*.nii")))
    mask_ids = set()
    for mp in mask_paths:
        mask_ids.add(os.path.basename(mp).replace(".nii.gz", "").replace(".nii", ""))

    all_avail = img_ids & mask_ids
    valid_ids = target_ids & all_avail if fixed_split is not None else (target_ids & all_avail if filter_healthy_only else all_avail)
    print(f"  [2/5] images: {len(img_ids)}, mask: {len(mask_ids)}, intersection: {len(all_avail)}, processing: {len(valid_ids)}")
    if len(valid_ids) == 0:
        print("  [ERROR] No valid samples, skipping."); return

    npy_dir = os.path.join(output_dir, "images_npy")
    os.makedirs(npy_dir, exist_ok=True)
    worker_args = []
    for pid in sorted(valid_ids):
        worker_args.append((pid, images_dir, masks_dir, npy_dir, crop_shape, True, eid2age.get(pid), target_spacing))
    print(f"  [3/5] {num_workers} workers x {len(worker_args)} samples")
    print(f"        npy -> {npy_dir}")

    processed, failed = [], 0
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(_worker_wrapper, worker_args), total=len(worker_args), desc=f"{label} preprocessing"):
            pid, age, success, _ = result
            if not success:
                failed += 1; continue
            if age is not None:
                processed.append({"eid": pid, "age": age})
    print(f"        success {len(worker_args) - failed}, failed {failed}, with age {len(processed)}")
    if len(processed) == 0:
        return

    if fixed_split is not None:
        print(f"  [4/5] split = '{fixed_split}'")
        for r in processed:
            r["split"] = fixed_split
    else:
        print(f"  [4/5] split {s_train}:{s_val}:{s_test} (seed={random_state})")
        id_list = [r["eid"] for r in processed]
        train_ids, val_ids, test_ids = _split_data(id_list, s_train, s_val, s_test, random_state=random_state)
        print(f"        Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        smap = {}
        for eid in train_ids: smap[eid] = "train"
        for eid in val_ids: smap[eid] = "val"
        for eid in test_ids: smap[eid] = "test"
        for r in processed: r["split"] = smap.get(r["eid"], "unknown")

    print(f"  [5/5] Saving metadata.csv")
    meta_path = os.path.join(output_dir, "metadata.csv")
    meta_df = pd.DataFrame(processed, columns=["eid", "age", "split"])
    meta_df.to_csv(meta_path, index=False)
    print(f"        -> {meta_path}")
    for sp in meta_df["split"].unique():
        print(f"        {sp}: {(meta_df['split'] == sp).sum()}")


if __name__ == "__main__":
    main()
