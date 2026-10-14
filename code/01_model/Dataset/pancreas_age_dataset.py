
import os
import glob
import numpy as np
from scipy.ndimage import rotate as nd_rotate
_HAS_SCIPY = True
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from sklearn.model_selection import train_test_split
import random
import h5py
import math
import torch.nn.functional as F

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from paths import DATA_ROOT  # noqa: E402
def gpu_rand_affine_3d(x, max_deg=10.0, translate_frac=(0.1, 0.1, 0.1), p=0.8,
                        padding_mode="border", scale_range=None):
    """
    x: (B,C,H,W,D) CUDA tensor, value range [-1,1]
    - random rotation about the Z axis, independent translation on three axes (by fraction), optional random scaling
    - padding_mode='border' avoids padding the border with 0 (more natural when background is -1)
    - scale_range: (min_scale, max_scale); None means no scaling
    """
    if p <= 0.0 or (max_deg <= 0.0 and max(abs(t) for t in translate_frac) <= 0.0
                    and scale_range is None):
        return x
    if torch.rand(1, device=x.device).item() > p:
        return x
    B, C, H, W, D = x.shape
    deg = math.radians(float(max_deg))
    ang = (torch.rand(B, device=x.device) * 2 - 1) * deg
    cos, sin = torch.cos(ang), torch.sin(ang)

    # random scaling
    if scale_range is not None:
        s_min, s_max = float(scale_range[0]), float(scale_range[1])
        sx = s_min + (s_max - s_min) * torch.rand(B, device=x.device)
        sy = s_min + (s_max - s_min) * torch.rand(B, device=x.device)
        sz = s_min + (s_max - s_min) * torch.rand(B, device=x.device)
    else:
        sx = sy = sz = torch.ones(B, device=x.device)

    # translation on three axes (pixels -> normalized coords)
    tx_pix = (torch.rand(B, device=x.device) * 2 - 1) * (float(translate_frac[0]) * H)
    ty_pix = (torch.rand(B, device=x.device) * 2 - 1) * (float(translate_frac[1]) * W)
    tz_pix = (torch.rand(B, device=x.device) * 2 - 1) * (float(translate_frac[2]) * D)
    tx = 2.0 * tx_pix / max(H - 1, 1)
    ty = 2.0 * ty_pix / max(W - 1, 1)
    tz = 2.0 * tz_pix / max(D - 1, 1)

    A = torch.zeros(B, 3, 4, device=x.device, dtype=x.dtype)
    A[:, 0, 0] = cos * sx;  A[:, 0, 1] = -sin * sx; A[:, 0, 3] = tx
    A[:, 1, 0] = sin * sy;  A[:, 1, 1] =  cos * sy; A[:, 1, 3] = ty
    A[:, 2, 2] = sz;                              A[:, 2, 3] = tz

    grid = F.affine_grid(A, size=x.size(), align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return y

def gpu_rand_intensity(x, p=0.7, gamma_range=(0.9, 1.1), noise_std=0.02):
    """
    x: (B,C,H,W,D) CUDA tensor, value range [-1,1]
    - random gamma (per sample) + Gaussian noise
    """
    if p <= 0.0:
        return x
    if torch.rand(1, device=x.device).item() > p:
        return x
    B = x.shape[0]
    g0, g1 = float(gamma_range[0]), float(gamma_range[1])
    gamma = (g0 + (g1 - g0) * torch.rand(B, device=x.device)).view(B, 1, 1, 1, 1).to(dtype=x.dtype)

    x01 = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    x = torch.pow(x01, gamma).mul_(2.0).sub_(1.0)

    if noise_std and noise_std > 0:
        std = float(noise_std)
        x = x + torch.randn_like(x) * std
    return x

def _mask_center(msk: np.ndarray):
    # return the center of the mask bounding box (integer indices); if no mask, return the volume center
    idx = np.where(msk > 0)
    if idx[0].size == 0:
        H, W, D = msk.shape
        return (H // 2, W // 2, D // 2)
    xmin, xmax = idx[0].min(), idx[0].max()
    ymin, ymax = idx[1].min(), idx[1].max()
    zmin, zmax = idx[2].min(), idx[2].max()
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    cz = (zmin + zmax) // 2
    return (cx, cy, cz)

def _center_crop_or_pad(vol: np.ndarray, center_xyz, target_shape, pad_value):
    # crop to target_shape centered on center_xyz; constant-pad if not large enough
    H, W, D = vol.shape
    th, tw, td = target_shape
    cx, cy, cz = center_xyz
    # target start/end indices (left-closed, right-open)
    sh = cx - th // 2; eh = sh + th
    sw = cy - tw // 2; ew = sw + tw
    sd = cz - td // 2; ed = sd + td
    # compute the overlap between source and target regions, and pad as needed
    dh0 = max(0, -sh); dh1 = max(0, eh - H)
    dw0 = max(0, -sw); dw1 = max(0, ew - W)
    dd0 = max(0, -sd); dd1 = max(0, ed - D)
    sh_clip = max(0, sh); eh_clip = min(H, eh)
    sw_clip = max(0, sw); ew_clip = min(W, ew)
    sd_clip = max(0, sd); ed_clip = min(D, ed)
    out = np.full((th, tw, td), pad_value, dtype=vol.dtype)
    out[dh0:th-dh1, dw0:tw-dw1, dd0:td-dd1] = vol[sh_clip:eh_clip, sw_clip:ew_clip, sd_clip:ed_clip]
    return out
 
class SimpleAug3D:
    def __init__(self, rot_deg=10, scale_range=(0.95, 1.05),
                 translate_pix=None, translate_frac=0.10,
                 p_translate=None, p_rotate=None,
                p_intensity=0.7, gamma_range=(0.9, 1.1), noise_std=0.01,
                p_flip=0.3, flip_axes=(1,)):
        # geometry
        self.rot_deg = float(rot_deg)
        self.scale_range = scale_range
        self.p_translate = p_translate 
        self.p_rotate= p_rotate
        # parse scalar/triplet config
        def _to_tuple3(v, cast=int):
            if v is None: return None
            if isinstance(v, (list, tuple)) and len(v) == 3:
                return (cast(v[0]), cast(v[1]), cast(v[2]))
            return (cast(v), cast(v), cast(v))
        self.translate_pix3  = _to_tuple3(translate_pix, cast=int)
        self.translate_frac3 = _to_tuple3(translate_frac, cast=float)
        # intensity
        self.p_intensity = p_intensity
        self.gamma_range = gamma_range
        self.noise_std   = noise_std
        self.p_flip = p_flip
        self.flip_axes = flip_axes

    def __call__(self, img, msk):
        H, W, D = img.shape
        if random.random() < self.p_flip:
            for axis in self.flip_axes:
                img = np.flip(img, axis=axis).copy()
                msk = np.flip(msk, axis=axis).copy()

        # translation (independent uniform sampling for the H/W/D axes)
        if random.random() < self.p_translate:
            if self.translate_pix3 is not None:
                px_h, px_w, px_d = self.translate_pix3
                tx = int(np.random.randint(-px_h, px_h + 1))
                ty = int(np.random.randint(-px_w, px_w + 1))
                tz = int(np.random.randint(-px_d, px_d + 1))
            else:
                fh, fw, fd = self.translate_frac3 if self.translate_frac3 is not None else (0.0, 0.0, 0.0)
                tx = int(round(np.random.uniform(-fh, fh) * H))
                ty = int(round(np.random.uniform(-fw, fw) * W))
                tz = int(round(np.random.uniform(-fd, fd) * D))
            pad_img = ((max(tx,0), max(-tx,0)), (max(ty,0), max(-ty,0)), (max(tz,0), max(-tz,0)))
            pad_msk = ((max(tx,0), max(-tx,0)), (max(ty,0), max(-ty,0)), (max(tz,0), max(-tz,0)))
            img = np.pad(img, pad_img, mode="constant", constant_values=-1.0)
            msk = np.pad(msk, pad_msk, mode="constant", constant_values=0)
            img = img[max(tx,0):max(tx,0)+H, max(ty,0):max(ty,0)+W, max(tz,0):max(tz,0)+D]
            msk = msk[max(tx,0):max(tx,0)+H, max(ty,0):max(ty,0)+W, max(tz,0):max(tz,0)+D]

        # rotation (most time-consuming): consider lowering the probability or moving it to GPU
        if self.rot_deg > 0 and random.random() < self.p_rotate:
            if _HAS_SCIPY:
                ang = float(np.random.uniform(-self.rot_deg, self.rot_deg))
                img = nd_rotate(img, angle=ang, axes=(0,1), reshape=False,
                                order=1, mode="constant", cval=-1.0, prefilter=False)
                msk = nd_rotate(msk, angle=ang, axes=(0,1), reshape=False,
                                order=0, mode="constant", cval=0,    prefilter=False)
            # if SciPy is unavailable, skip rotation and keep translation and intensity augmentation
        # brightness perturbation
        if random.random() < self.p_intensity:
            gamma = random.uniform(*self.gamma_range)
            x = np.clip((img + 1.0) * 0.5, 0, 1) ** gamma
            img = x * 2.0 - 1.0
            if self.noise_std > 0:
                img += np.random.normal(0, self.noise_std, size=img.shape).astype(np.float32)
        return img.astype(np.float32), msk.astype(np.uint8)

def split_data(ids, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=random_state)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_size, random_state=random_state)
    return train_ids, val_ids, test_ids

class PancreasAgeDataset(Dataset):
    def __init__(
        self,
        images_dir=os.path.join(DATA_ROOT, "images_ukb"),
        masks_dir=os.path.join(DATA_ROOT, "masks_ukb"),
        csv_path=os.path.join(DATA_ROOT, "icd-10-healthy.csv"),
        patient_id_col="eid",
        age_col="age",
        do_normliza=True,
        subset_ids=None,
        age_space=None,          
        label_sigma=2.0,         
        transform=None,
        crop_to_shape=(160, 128, 64),
        save_norm_nii_dir=None,
        save_norm_nii_max=10,
        h5_files=None,
        return_mask: bool = False,      
        mask_zero_out: bool = False,    
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.csv_path = csv_path
        self.patient_id_col = patient_id_col
        self.age_col = age_col
        self.do_normliza = do_normliza
        self.label_sigma = float(label_sigma)
        self.transform = transform
        self.crop_to_shape = tuple(crop_to_shape) if crop_to_shape is not None else None
        self.save_norm_nii_dir = save_norm_nii_dir
        self.save_norm_nii_max = int(save_norm_nii_max) if save_norm_nii_max is not None else 0
        self._saved_norm_nii = 0
        self.h5_files = list(h5_files) if h5_files is not None else None
        self._h5_cache = {}   
        self.return_mask = bool(return_mask)
        self.mask_zero_out = bool(mask_zero_out)

        samples = []
        if self.h5_files:
            # read meta from H5
            for hf in self.h5_files:
                with h5py.File(hf, "r") as f:
                    pids = f["patient_ids"][:]
                    ages = f["ages"][:]
                    # h5py may return bytes
                    pids = [p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p) for p in pids]
                    for i, pid in enumerate(pids):
                        samples.append((pid, hf, i, float(ages[i])))  # (pid, h5_path, idx_in_file, age)
        else:
            # build from NIfTI (legacy path)
            df = pd.read_csv(self.csv_path)
            assert self.patient_id_col in df.columns, f"CSV is missing column {self.patient_id_col}"
            if self.age_col not in df.columns:
                for cand in ["Age", "AGE", "age_years"]:
                    if cand in df.columns:
                        self.age_col = cand
                        break
            assert self.age_col in df.columns, f"CSV is missing age column {self.age_col}"
            id2age = dict(zip(df[self.patient_id_col].astype(str), df[self.age_col].astype(float)))
            nii_paths = sorted(glob.glob(os.path.join(self.images_dir, "*_0000.nii.gz")))
            for ip in nii_paths:
                pid = os.path.basename(ip).replace("_0000.nii.gz", "")
                mp = os.path.join(self.masks_dir, f"{pid}.nii.gz")
                if os.path.exists(mp) and pid in id2age:
                    samples.append((pid, ip, mp, float(id2age[pid])))

        if subset_ids is not None:
            subset_ids = set(map(str, subset_ids))

            samples = [s for s in samples if str(s[0]) in subset_ids]
        self.samples = samples

        if age_space is None:
            ages_all = sorted({int(round(s[3])) for s in self.samples})
            self.age_space = ages_all
        else:
            self.age_space = sorted({int(a) for a in age_space})
        self.age_space_np = np.asarray(self.age_space, dtype=np.float32)
        self.K = len(self.age_space)
        self.age2idx = {age: idx for idx, age in enumerate(self.age_space)}
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _get_from_h5(self, h5_path: str, idx: int):
        if h5_path not in self._h5_cache:
            # enlarge the raw-data cache (rdcc_nbytes defaults to ~1MB); set it to 256MB here
            self._h5_cache[h5_path] = h5py.File(h5_path, "r", rdcc_nbytes=256*1024*1024, rdcc_nslots=1_000_003)
        f = self._h5_cache[h5_path]
        img = f["images"][idx]        # (H,W,D) float32, cropped and normalized
        msk = f["masks"][idx].astype(np.uint8)  # (H,W,D)
        pid = f["patient_ids"][idx]
        if isinstance(pid, (bytes, bytearray)):
            pid = pid.decode("utf-8")
        age = float(f["ages"][idx])
        return pid, img, msk, age

    def __getitem__(self, idx):

        rec = self.samples[idx]
        if self.h5_files:
            pid, img_c, msk_c, age = self._get_from_h5(rec[1], rec[2])
            from_h5 = True
        else:
            pid, ipath, mpath, age = rec
            img_nii = nib.load(ipath)
            msk_nii = nib.load(mpath)
            img = img_nii.get_fdata().astype(np.float32)   # (H,W,D)
            msk = (msk_nii.get_fdata().astype(np.float32) > 0).astype(np.uint8)
            # mask-centered crop → background -1
            if self.crop_to_shape is not None:
                cxyz = _mask_center(msk)
                img_c = _center_crop_or_pad(img, cxyz, self.crop_to_shape, pad_value=-1.0)
                msk_c = _center_crop_or_pad(msk, cxyz, self.crop_to_shape, pad_value=0)
            if self.mask_zero_out and msk_c.sum() > 0:
                img_c = img_c * (msk_c > 0)
            from_h5 = False


        # H5 is already "normalized inside pancreas only + background -1", no need to normalize again
        if self.do_normliza and not self.h5_files:
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
            img_c = img_norm
        # if self.save_norm_nii_dir is not None and not from_h5 and (self.save_norm_nii_max <= 0 or self._saved_norm_nii < self.save_norm_nii_max):
        #     os.makedirs(self.save_norm_nii_dir, exist_ok=True)
        #     save_path = os.path.join(self.save_norm_nii_dir, f"{pid}.nii.gz")
        #     img_save_nii = nib.Nifti1Image(img_c, affine=img_nii.affine, header=img_nii.header)
        #     nib.save(img_save_nii, save_path)
        #     self._saved_norm_nii += 1
        if self.transform is not None:
            img_c, msk_c = self.transform(img_c, msk_c)
        
        img_p = np.stack([img_c, img_c, img_c], axis=0)  # (3,H,W,D)

        # soft label
        mu = float(age)
        sigma = self.label_sigma
        diffs = (self.age_space_np - mu) / (sigma + 1e-8)
        w = np.exp(-0.5 * diffs * diffs)
        w_sum = w.sum()
        w = (w / w_sum) if w_sum > 0 else (np.ones_like(w) / len(w))
        target = w.astype(np.float32)
        age_int = int(round(age))
        sample = {
            "pid": pid,
            "image": torch.from_numpy(img_p),
            "age": torch.tensor(age_int, dtype=torch.long),
            "target": torch.from_numpy(target),
        }
        if self.return_mask:
            sample["mask"] = torch.from_numpy(msk_c)
        return sample

def gaussian_label(age, all_ages, sigma=2.0):
    """
    Convert the true age into a Gaussian soft label
    """
    age_array = np.array(all_ages)
    dist = np.exp(-0.5 * ((age_array - age) / sigma) ** 2)
    dist = dist / dist.sum()  # normalize
    return dist.astype(np.float32)