"""
Pure .npy dataset + training utilities
- Read eid/age/split from metadata.csv
- Filter train/val/test by split
- Load preprocessed 3D images from images_npy/{eid}.npy
"""
import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════
# Data splitting
# ═══════════════════════════════════════════════
def split_data(ids, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=random_state)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_size, random_state=random_state)
    return train_ids, val_ids, test_ids


# ═══════════════════════════════════════════════
# Gaussian soft label
# ═══════════════════════════════════════════════
def gaussian_label(age, all_ages, sigma=2.0):
    age_array = np.array(all_ages, dtype=np.float32)
    dist = np.exp(-0.5 * ((age_array - age) / sigma) ** 2)
    dist = dist / dist.sum()
    return dist.astype(np.float32)


# ═══════════════════════════════════════════════
# NpyDataset — read-only .npy + metadata.csv
# ═══════════════════════════════════════════════
class NpyDataset(Dataset):
    """
    Read preprocessed data from metadata.csv + .npy files.
    metadata.csv format: eid, age, split
    .npy format: (H, W, D) float32, value range [-1, 1]
    """
    def __init__(
        self,
        npy_dir: str,
        metadata_csv: str,
        split: str,                     # "train" | "val" | "test"
        age_space=None,
        label_sigma: float = 2.0,
        transform=None,                 # CPU augmentation (SimpleAug3D etc.)
    ):
        self.npy_dir = npy_dir
        self.label_sigma = float(label_sigma)
        self.transform = transform

        # read metadata
        df = pd.read_csv(metadata_csv)
        if "split" not in df.columns:
            # if metadata has no split column, use one split for everything
            df["split"] = split
        df = df[df["split"] == split].copy()
        self.df = df.reset_index(drop=True)

        # build sample list
        self.samples = []
        for _, row in df.iterrows():
            npy_path = os.path.join(npy_dir, f"{row['eid']}.npy")
            if os.path.exists(npy_path):
                self.samples.append((str(row["eid"]), npy_path, float(row["age"])))

        # age_space
        if age_space is None:
            ages_all = sorted({int(round(s[2])) for s in self.samples})
            self.age_space = ages_all
        else:
            self.age_space = sorted({int(a) for a in age_space})
        self.age_space_np = np.asarray(self.age_space, dtype=np.float32)
        self.K = len(self.age_space)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, npy_path, age = self.samples[idx]
        img = np.load(npy_path).astype(np.float32)  # (H, W, D), already preprocessed
        img = np.expand_dims(img, axis=0)  # (1, H, W, D)
        # CPU augmentation (optional; GPU augmentation is used by default)
        msk_dummy = np.zeros_like(img, dtype=np.uint8)  # npy does not store mask, fill a dummy
        if self.transform is not None:
            img, _ = self.transform(img, msk_dummy)

        # soft label
        sigma = self.label_sigma
        diffs = (self.age_space_np - float(age)) / (sigma + 1e-8)
        w = np.exp(-0.5 * diffs * diffs)
        w = w / w.sum() if w.sum() > 0 else (np.ones_like(w) / len(w))

        return {
            "pid": pid,
            "image": torch.from_numpy(img),
            "age": torch.tensor(int(round(age)), dtype=torch.long),
            "target": torch.from_numpy(w.astype(np.float32)),
        }
