#!/usr/bin/env python3
"""
1_1_infer_all_cv.py — 5-fold cross-validation full inference script

For each fold:
  1. Load best_val_mae.pt → read age_space / bias / model weights
  2. Rebuild train/val/test splits from cv_splits.json, run inference and record metrics
  3. Save per-sample CSV for each fold

Summary:
  - mean ± std metrics for train/val over the 5 folds
  - 5-fold test OOF combined into the full UKB healthy result
  - per-fold inference on unhealthy / external → 5-fold averaged predictions + averaged metrics
  - merge 5-fold test (OOF) + unhealthy (5-fold average) → master UKB table
  - external datasets fuyi_ext_npy / fuyi_ext_part3_data averaged over 5 folds each

Usage:
  python code/01_model/1_1_infer_all_cv.py
  python code/01_model/1_1_infer_all_cv.py --device cuda:0 --n-folds 5
"""
import os, sys, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
from paths import (UKB_NPY_DIR, UKB_META_CSV, EXT_NPY_DIR, EXT_META_CSV,
                   OBESITY_NPY_DIR, OBESITY_META_CSV, CV_DIR, INFER_DIR, DATA_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as _DS
from torch.amp import autocast
from tqdm import tqdm

from Dataset.npy_dataset import NpyDataset
from Model.monai_swin3d_regressor import Swin3DAgeRegressor
from utils.train_utils import ensure_dir, set_seed, _r2_np, _rmse
from utils.metrics import compute_metrics, make_age_groups, print_metrics


# ═══════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════
@torch.no_grad()
def run_inference(model, loader, device, age_space_tensor, use_amp=True):
    """Run inference on one DataLoader, returns (ids, gts, preds_raw)"""
    model.eval()
    all_ids, all_gts, all_preds = [], [], []
    amp_dtype = torch.float16
    for batch in tqdm(loader, desc="Infer", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        with autocast(enabled=use_amp, dtype=amp_dtype, device_type="cuda"):
            logp = model(x)
            p = logp.exp()
            pred_age = (p * age_space_tensor.unsqueeze(0)).sum(dim=-1)
        gt_age = batch["age"].to(device, non_blocking=True).float()
        all_preds.append(pred_age.cpu().numpy())
        all_gts.append(gt_age.cpu().numpy())
        all_ids.extend(batch.get("pid", [str(i) for i in range(len(gt_age))]))
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    gts = np.concatenate(all_gts) if all_gts else np.array([])
    return all_ids, gts, preds


def apply_bias(preds, gts, alpha, beta):
    """pred_bc = pred - (alpha * gt + beta)"""
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return preds.copy()
    return preds - (alpha * gts + beta)


def fit_bias(preds, gts):
    """delta = pred - age = alpha * age + beta → returns (alpha, beta)"""
    preds = np.asarray(preds).ravel()
    gts = np.asarray(gts).ravel()
    if len(gts) < 2:
        return 0.0, 0.0
    delta = preds - gts
    age_m = gts.mean(); delta_m = delta.mean()
    var_a = ((gts - age_m) ** 2).mean()
    if var_a < 1e-12:
        return 0.0, float(delta_m)
    alpha = float(((gts - age_m) * (delta - delta_m)).mean() / var_a)
    beta = float(delta_m - alpha * age_m)
    if not np.isfinite(alpha) or not np.isfinite(beta) or abs(alpha) >= 0.9:
        return 0.0, float(delta_m)
    return alpha, beta


def metrics_to_dict(gts, preds, prefix=""):
    """compute_metrics + r2/rmse, flattened into a dict with a prefix"""
    m = compute_metrics(gts, preds)
    return {
        f"{prefix}mae": m["mae"],
        f"{prefix}pcc_age": m["pcc_age"],
        f"{prefix}pcc_gap": m["pcc_gap"],
        f"{prefix}cs1": m["cs1"],
        f"{prefix}cs3": m["cs3"],
        f"{prefix}r2": _r2_np(gts, preds),
        f"{prefix}rmse": _rmse(gts, preds),
        f"{prefix}n": m["n"],
    }


def _make_loader(ds, bs, nw):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=True, persistent_workers=(nw > 0))


def load_model(ckpt_path, device):
    """Load checkpoint → model, age_space, bias_alpha, bias_beta, label_sigma"""
    print(f"  [load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    age_space = ckpt.get("age_space")
    if age_space is None:
        cfg_path = os.path.join(os.path.dirname(ckpt_path), "config_used.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                train_cfg = json.load(f)
            tmp_ds = NpyDataset(train_cfg["npy_dir"], train_cfg["metadata_csv"],
                                split="train", label_sigma=train_cfg.get("label_sigma", 2.0))
            age_space = tmp_ds.age_space
        else:
            raise RuntimeError("checkpoint is missing age_space")
    K = len(age_space)

    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config_used.json")
    model_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            model_cfg = json.load(f).get("model", {})

    model = Swin3DAgeRegressor(num_classes=K, model_cfg=model_cfg or None).to(device)
    model.head.load_state_dict(ckpt["model_head"], strict=True)
    if ckpt.get("model_backbone"):
        model.backbone.load_state_dict(ckpt["model_backbone"], strict=False)
    model.eval()

    bias = ckpt.get("bias") or {}
    bias_alpha = float(bias.get("alpha", 0.0))
    bias_beta = float(bias.get("beta", 0.0))

    label_sigma = 2.0
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            label_sigma = float(json.load(f).get("label_sigma", 2.0))

    return model, age_space, bias_alpha, bias_beta, label_sigma


def build_unhealthy_dataset(unhealthy_csv, npy_dir, age_space, label_sigma):
    """Extract samples with is_healthy=0 and a matching .npy from the ICD-10 CSV"""
    if not os.path.exists(unhealthy_csv):
        print(f"  [skip] unhealthy CSV does not exist: {unhealthy_csv}")
        return None, None

    df = pd.read_csv(unhealthy_csv, low_memory=False)
    if "is_healthy" not in df.columns or "eid" not in df.columns:
        print("  [skip] unhealthy CSV is missing is_healthy/eid columns")
        return None, None

    df_u = df[df["is_healthy"] == 0].copy()
    df_u["eid"] = df_u["eid"].astype(str).str.strip()
    df_u = df_u[df_u["eid"] != ""]

    valid = []
    for _, row in df_u.iterrows():
        eid = row["eid"]
        npy_p = os.path.join(npy_dir, f"{eid}.npy")
        if os.path.exists(npy_p):
            age_val = float(row.get("age", float('nan')))
            if not np.isnan(age_val):
                valid.append((eid, npy_p, age_val))

    if not valid:
        print("  [skip] no valid .npy samples for unhealthy")
        return None, None

    print(f"  unhealthy valid samples: {len(valid)}/{len(df_u)}")
    asp_np = np.array(age_space, dtype=np.float32)
    sigma = float(label_sigma)

    class _UnhealthyDS(_DS):
        def __init__(self, samples, asp, s):
            self.samples, self.asp, self.s = samples, asp, s
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            pid, npy_p, age = self.samples[idx]
            img = np.expand_dims(np.load(npy_p).astype(np.float32), axis=0)
            diffs = (self.asp - age) / (self.s + 1e-8)
            w = np.exp(-0.5 * diffs * diffs)
            w /= w.sum() if w.sum() > 0 else 1
            return {"pid": pid, "image": torch.from_numpy(img),
                    "age": torch.tensor(int(round(age)), dtype=torch.long),
                    "target": torch.from_numpy(w.astype(np.float32))}

    return _UnhealthyDS(valid, asp_np, sigma), valid


def _compute_avg(per_fold_dict, metric_keys_list):
    """per_fold_dict: {fold_key: {metric: value}} → avg dict with _std"""
    vals = {k: [] for k in metric_keys_list}
    for r in per_fold_dict.values():
        for k in metric_keys_list:
            vals[k].append(r.get(k, float('nan')))
    avg = {}
    for k in metric_keys_list:
        arr = np.array(vals[k])
        arr = arr[np.isfinite(arr)]
        avg[k] = float(np.mean(arr)) if len(arr) > 0 else float('nan')
        avg[f"{k}_std"] = float(np.std(arr)) if len(arr) > 1 else 0.0
    avg["n_folds"] = len(per_fold_dict)
    return avg


# ═══════════════════════════════════════
# Main function
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="5-fold CV full inference")
    parser.add_argument("--cv-dir", type=str, default=CV_DIR)
    parser.add_argument("--cv-splits", type=str,
                        default=os.path.join(CV_DIR, "cv_splits.json"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=INFER_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    # ── data paths ──
    parser.add_argument("--ukb-npy-dir", type=str, default=UKB_NPY_DIR)
    parser.add_argument("--ukb-meta-csv", type=str, default=UKB_META_CSV)
    parser.add_argument("--unhealthy-csv", type=str,
                        default=os.path.join(DATA_ROOT, "icd-10-all_with_flags_1.csv"))
    parser.add_argument("--ext-npy-dir", type=str, default=EXT_NPY_DIR)
    parser.add_argument("--ext-meta-csv", type=str, default=EXT_META_CSV,
                        help="external data 1 metadata (used fully as test set)")
    parser.add_argument("--ext-part3-npy-dir", type=str, default=OBESITY_NPY_DIR)
    parser.add_argument("--ext-part3-meta-csv", type=str, default=OBESITY_META_CSV)

    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-end", type=int, default=None)
    parser.add_argument("--bias-split", type=str, default="train",
                        choices=["train", "val"],
                        help="dataset used to fit bias-correction parameters (train or val)")
    args = parser.parse_args()

    if args.fold_end is None:
        args.fold_end = args.n_folds - 1

    ensure_dir(args.out_dir)
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bs, nw = args.batch_size, args.num_workers
    print(f"[Infer CV] device={device}, folds={args.fold_start}..{args.fold_end}")

    with open(args.cv_splits, "r") as f:
        cv_all = json.load(f)

    # ═══════════════════════════════════════
    #   Per-fold inference on UKB healthy
    # ═══════════════════════════════════════
    all_fold_results = {}
    fold_dfs = []
    fold_biases = {}  # fold_key → (alpha, beta) fitted from the training set

    for fold in range(args.fold_start, args.fold_end + 1):
        fold_key = str(fold)
        if fold_key not in cv_all:
            print(f"\n[skip] fold {fold_key} not in cv_splits")
            continue

        fold_dir = os.path.join(args.cv_dir, f"fold_{fold_key}")
        ckpt_path = os.path.join(fold_dir, "best_val_mae.pt")
        if not os.path.exists(ckpt_path):
            print(f"\n[skip] fold {fold_key} — no best_val_mae.pt")
            continue

        print(f"\n{'='*60}")
        print(f"  Fold {fold_key}")
        print(f"{'='*60}")

        model, age_space, _, _, label_sigma = load_model(ckpt_path, device)
        age_space_tensor = torch.tensor(age_space, dtype=torch.float32, device=device)
        print(f"  age_space K={len(age_space)}: [{age_space[0]}..{age_space[-1]}]")

        # rebuild fold splits
        fold_info = cv_all[fold_key]
        df_meta = pd.read_csv(args.ukb_meta_csv)
        split_map = {}
        for e in fold_info["train"]: split_map[e] = "train"
        for e in fold_info["val"]:   split_map[e] = "val"
        for e in fold_info["test"]:  split_map[e] = "test"
        df_meta["split"] = df_meta["eid"].map(split_map).fillna("unused")
        df_meta = df_meta[df_meta["split"] != "unused"].copy()
        fold_meta_csv = os.path.join(args.out_dir, f"_fold_{fold_key}_meta.csv")
        df_meta.to_csv(fold_meta_csv, index=False)

        # ── 1. Inference on the bias-fitting dataset → fit this fold's bias parameters ──
        ds_bias = NpyDataset(args.ukb_npy_dir, fold_meta_csv, split=args.bias_split,
                             age_space=age_space, label_sigma=label_sigma)
        loader_bias = _make_loader(ds_bias, bs, nw)
        _, gts_bias, preds_bias = run_inference(model, loader_bias, device, age_space_tensor)
        alpha_fold, beta_fold = fit_bias(preds_bias, gts_bias)
        fold_biases[fold_key] = (alpha_fold, beta_fold)
        print(f"  [{args.bias_split}] n={len(gts_bias):,} → fitted bias: alpha={alpha_fold:.4f}, beta={beta_fold:.4f}")

        # ── 2. Inference on all splits + bias correction ──
        fold_results = {}
        for split in ["train", "val", "test"]:
            if split == args.bias_split:
                # already inferred during bias fitting, reuse directly
                ids = [str(s[0]) for s in ds_bias.samples]
                gts, preds_raw = gts_bias, preds_bias
            else:
                ds = NpyDataset(args.ukb_npy_dir, fold_meta_csv, split=split,
                                age_space=age_space, label_sigma=label_sigma)
                loader = _make_loader(ds, bs, nw)
                ids, gts, preds_raw = run_inference(model, loader, device, age_space_tensor)

            preds_bc = apply_bias(preds_raw, gts, alpha_fold, beta_fold)

            raw_m = metrics_to_dict(gts, preds_raw, prefix="raw_")
            bc_m  = metrics_to_dict(gts, preds_bc,  prefix="bc_")
            fold_results[split] = {**raw_m, **bc_m, "total_n": int(len(gts)),
                                   "alpha": alpha_fold, "beta": beta_fold}

            print(f"  [{split}] n={len(gts):,}  "
                  f"raw MAE={raw_m['raw_mae']:.3f} R²={raw_m['raw_r2']:.4f} | "
                  f"bc MAE={bc_m['bc_mae']:.3f} R²={bc_m['bc_r2']:.4f}")

            df_split = pd.DataFrame({
                "eid": ids, "age_gt": gts.astype(np.float32),
                "age_pred_raw": preds_raw.astype(np.float32),
                "age_pred_bc": preds_bc.astype(np.float32),
                "fold": fold, "split": split,
            })
            df_split.to_csv(os.path.join(args.out_dir, f"fold{fold}_{split}_predictions.csv"), index=False)

            if split == "test":
                df_split["is_healthy"] = 1
                fold_dfs.append(df_split)

        all_fold_results[fold_key] = fold_results
        del model; torch.cuda.empty_cache()

    # ═══════════════════════════════════════
    #   Summary: UKB averaged metrics
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("  5-fold UKB summary")
    print(f"{'='*60}")

    avg_results = {}
    metric_keys = ["raw_mae", "raw_r2", "raw_rmse", "raw_pcc_age", "raw_pcc_gap",
                   "raw_cs1", "raw_cs3",
                   "bc_mae", "bc_r2", "bc_rmse", "bc_pcc_age", "bc_pcc_gap",
                   "bc_cs1", "bc_cs3"]

    for split in ["train", "val", "test"]:
        vals = {k: [] for k in metric_keys}
        ns = []
        for fold_key in sorted(all_fold_results.keys()):
            if split in all_fold_results[fold_key]:
                r = all_fold_results[fold_key][split]
                ns.append(r["total_n"])
                for k in metric_keys:
                    vals[k].append(r.get(k, float('nan')))
        if vals[metric_keys[0]]:
            avg = {}
            for k in metric_keys:
                arr = np.array(vals[k]); arr = arr[np.isfinite(arr)]
                avg[k] = float(np.mean(arr)) if len(arr) > 0 else float('nan')
                avg[f"{k}_std"] = float(np.std(arr)) if len(arr) > 1 else 0.0
            avg["total_n"] = int(np.sum(ns))
            avg["n_folds"] = len(vals[metric_keys[0]])
            avg_results[split] = avg

            print(f"\n  [{split}] (avg over {avg['n_folds']} folds, total_n={avg['total_n']:,})")
            print(f"    Raw:  MAE={avg['raw_mae']:.3f}±{avg['raw_mae_std']:.3f}  "
                  f"R²={avg['raw_r2']:.4f}±{avg['raw_r2_std']:.4f}  "
                  f"CS1={avg['raw_cs1']:.1f}±{avg['raw_cs1_std']:.1f}%")
            print(f"    BC:   MAE={avg['bc_mae']:.3f}±{avg['bc_mae_std']:.3f}  "
                  f"R²={avg['bc_r2']:.4f}±{avg['bc_r2_std']:.4f}  "
                  f"CS1={avg['bc_cs1']:.1f}±{avg['bc_cs1_std']:.1f}%")

    rows = []
    for fold_key in sorted(all_fold_results.keys()):
        for split in ["train", "val", "test"]:
            if split in all_fold_results[fold_key]:
                row = {"fold": int(fold_key), "split": split}
                row.update(all_fold_results[fold_key][split])
                rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "cv_per_fold_metrics.csv"), index=False)
        print(f"\n[save] cv_per_fold_metrics.csv ({len(rows)} rows)")

    # ═══════════════════════════════════════
    #   Unhealthy: per-fold inference → 5-fold averaged predictions + metrics computed on the average
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Unhealthy inference (per fold → 5-fold average)")
    print(f"{'='*60}")

    u_per_fold = {}
    u_all_preds = []          # collect per-person predictions from each fold, for ensembling

    for fold in range(args.fold_start, args.fold_end + 1):
        fold_key = str(fold)
        ckpt_path = os.path.join(args.cv_dir, f"fold_{fold_key}", "best_val_mae.pt")
        if not os.path.exists(ckpt_path):
            continue

        model_u, age_u, _, _, ls_u = load_model(ckpt_path, device)
        age_t_u = torch.tensor(age_u, dtype=torch.float32, device=device)

        # use the bias parameters fitted on this fold's training set (not the ones in the ckpt)
        ba_u, bb_u = fold_biases.get(fold_key, (0.0, 0.0))

        u_ds, _ = build_unhealthy_dataset(args.unhealthy_csv, args.ukb_npy_dir, age_u, ls_u)
        if u_ds is None:
            del model_u; torch.cuda.empty_cache(); continue

        u_loader = _make_loader(u_ds, bs, nw)
        ids_u, gts_u, preds_raw_u = run_inference(model_u, u_loader, device, age_t_u)
        preds_bc_tr = apply_bias(preds_raw_u, gts_u, ba_u, bb_u)
        a_self, b_self = fit_bias(preds_raw_u, gts_u)
        preds_bc_self = apply_bias(preds_raw_u, gts_u, a_self, b_self)

        # record this fold's per-fold metrics
        raw_u = metrics_to_dict(gts_u, preds_raw_u, prefix="raw_")
        bc_tr_u = metrics_to_dict(gts_u, preds_bc_tr, prefix="bc_tr_")
        bc_self_u = metrics_to_dict(gts_u, preds_bc_self, prefix="bc_self_")
        u_per_fold[fold_key] = {**raw_u, **bc_tr_u, **bc_self_u,
                                "alpha_self": a_self, "beta_self": b_self}
        print(f"  [fold {fold_key}] n={len(gts_u):,}  "
              f"raw MAE={raw_u['raw_mae']:.3f}  bc_tr MAE={bc_tr_u['bc_tr_mae']:.3f}  "
              f"bc_self MAE={bc_self_u['bc_self_mae']:.3f}")

        # collect per-person predictions for ensembling
        u_all_preds.append(pd.DataFrame({
            "eid": ids_u,
            "age_gt": gts_u.astype(np.float32),
            "pred_raw": preds_raw_u.astype(np.float32),
            "pred_bc_tr": preds_bc_tr.astype(np.float32),
            "pred_bc_self": preds_bc_self.astype(np.float32),
            "fold": int(fold_key),
        }))

        del model_u; torch.cuda.empty_cache()

    # ── 5-fold average + final metrics ──
    df_u_avg = None
    if u_all_preds:
        df_u_all = pd.concat(u_all_preds, ignore_index=True)
        # ensemble average per eid
        df_u_avg = df_u_all.groupby("eid").agg(
            age_gt=("age_gt", "first"),
            age_pred_raw=("pred_raw", "mean"),
            age_pred_bc=("pred_bc_tr", "mean"),
            age_pred_bc_self_raw=("pred_bc_self", "mean"),
        ).reset_index()
        # re-apply self-bias correction to the averaged pred_bc_self_raw
        a_self_ens, b_self_ens = fit_bias(df_u_avg["age_pred_raw"].values,
                                           df_u_avg["age_gt"].values)
        df_u_avg["age_pred_bc_self"] = apply_bias(
            df_u_avg["age_pred_raw"].values, df_u_avg["age_gt"].values,
            a_self_ens, b_self_ens)

        # compute final metrics from the averaged predictions
        gts_u_ens = df_u_avg["age_gt"].values
        raw_u_ens  = df_u_avg["age_pred_raw"].values
        bc_tr_u_ens  = df_u_avg["age_pred_bc"].values
        bc_self_u_ens = df_u_avg["age_pred_bc_self"].values

        m_raw_ens = compute_metrics(gts_u_ens, raw_u_ens)
        m_bc_tr_ens = compute_metrics(gts_u_ens, bc_tr_u_ens)
        m_bc_self_ens = compute_metrics(gts_u_ens, bc_self_u_ens)
        print(f"\n  [unhealthy 5-fold average] n={len(df_u_avg):,}")
        print_metrics(m_raw_ens, prefix="    Raw       | ")
        print_metrics(m_bc_tr_ens, prefix="    BC(train)  | ")
        print_metrics(m_bc_self_ens, prefix="    BC(self)   | ")
        print(f"    self-fitted bias (ensemble): alpha={a_self_ens:.4f}, beta={b_self_ens:.4f}")

        df_u_avg["fold"] = -1
        df_u_avg["split"] = "unhealthy"
        df_u_avg["is_healthy"] = 0
        df_u_avg.to_csv(os.path.join(args.out_dir, "predictions_unhealthy.csv"), index=False)

    if u_per_fold:
        u_keys = [k for k in next(iter(u_per_fold.values())).keys()
                  if not k.startswith("alpha_") and not k.startswith("beta_")]
        u_avg = _compute_avg(u_per_fold, u_keys)
        print(f"\n  [unhealthy per-fold avg over {u_avg.get('n_folds',0)} folds]")
        print(f"    Raw:    MAE={u_avg.get('raw_mae',float('nan')):.3f}±{u_avg.get('raw_mae_std',0):.3f}")
        print(f"    BC(tr): MAE={u_avg.get('bc_tr_mae',float('nan')):.3f}±{u_avg.get('bc_tr_mae_std',0):.3f}")
        print(f"    BC(self):MAE={u_avg.get('bc_self_mae',float('nan')):.3f}±{u_avg.get('bc_self_mae_std',0):.3f}")

        u_rows = [{"fold": int(fk), "dataset": "unhealthy", **v}
                  for fk, v in u_per_fold.items()]
        pd.DataFrame(u_rows).to_csv(os.path.join(args.out_dir, "unhealthy_per_fold_metrics.csv"), index=False)

    # ═══════════════════════════════════════
    #  Master UKB table (healthy=OOF, unhealthy=5-fold average)
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Building Master UKB table")
    print(f"{'='*60}")

    master_parts = list(fold_dfs)
    if df_u_avg is not None:
        master_parts.append(df_u_avg)

    if master_parts:
        df_master = pd.concat(master_parts, ignore_index=True)
        mask_healthy = df_master["is_healthy"] == 1
        if "age_pred_bc_self" not in df_master.columns:
            df_master["age_pred_bc_self"] = np.nan
        df_master.loc[mask_healthy, "age_pred_bc_self"] = df_master.loc[mask_healthy, "age_pred_bc"]
        df_master = df_master.sort_values(["is_healthy", "eid"]).reset_index(drop=True)

        master_path = os.path.join(args.out_dir, "master_ukb_predictions.csv")
        df_master.to_csv(master_path, index=False)
        n_healthy = int(mask_healthy.sum())
        n_unhealthy = int((~mask_healthy).sum())
        print(f"  Master table: {len(df_master):,} subjects ({n_healthy:,} healthy + {n_unhealthy:,} unhealthy)")

        for label, mask in [("Healthy", mask_healthy), ("Unhealthy", ~mask_healthy)]:
            sub = df_master[mask]
            if len(sub) > 0:
                m_all = compute_metrics(sub["age_gt"].values, sub["age_pred_bc_self"].values)
                print(f"  [{label}] n={len(sub):,}  BC MAE={m_all['mae']:.3f}  "
                      f"PCC_age={m_all['pcc_age']:.4f}  CS1={m_all['cs1']:.1f}%")

    # ═══════════════════════════════════════
    #   External data 1: fuyi_ext_npy (full test, per fold→5-fold average)
    # ═══════════════════════════════════════
    ext1_per_fold = {}
    ext1_all_preds = []
    if os.path.exists(args.ext_meta_csv) and os.path.exists(args.ext_npy_dir):
        print(f"\n{'='*60}")
        print("  External data 1: fuyi_ext_npy (per fold → 5-fold average)")
        print(f"{'='*60}")

        for fold in range(args.fold_start, args.fold_end + 1):
            fold_key = str(fold)
            ckpt_path = os.path.join(args.cv_dir, f"fold_{fold_key}", "best_val_mae.pt")
            if not os.path.exists(ckpt_path):
                continue

            model_e1, age_e1, _, _, ls_e1 = load_model(ckpt_path, device)
            ba_e1, bb_e1 = fold_biases.get(fold_key, (0.0, 0.0))
            age_t_e1 = torch.tensor(age_e1, dtype=torch.float32, device=device)

            ds_e1 = NpyDataset(args.ext_npy_dir, args.ext_meta_csv, split="ext_test",
                               age_space=age_e1, label_sigma=ls_e1)
            loader_e1 = _make_loader(ds_e1, bs, nw)
            ids_e1, gts_e1, preds_raw_e1 = run_inference(model_e1, loader_e1, device, age_t_e1)
            preds_bc_e1 = apply_bias(preds_raw_e1, gts_e1, ba_e1, bb_e1)

            raw_e1 = metrics_to_dict(gts_e1, preds_raw_e1, prefix="raw_")
            bc_e1  = metrics_to_dict(gts_e1, preds_bc_e1,  prefix="bc_")
            ext1_per_fold[fold_key] = {**raw_e1, **bc_e1, "n": int(len(gts_e1))}
            print(f"  [fold {fold_key}] n={len(gts_e1):,}  "
                  f"raw MAE={raw_e1['raw_mae']:.3f}  bc MAE={bc_e1['bc_mae']:.3f}")

            ext1_all_preds.append(pd.DataFrame({
                "eid": ids_e1, "age_gt": gts_e1.astype(np.float32),
                "pred_raw": preds_raw_e1.astype(np.float32),
                "pred_bc": preds_bc_e1.astype(np.float32),
                "fold": int(fold_key),
            }))

            del model_e1; torch.cuda.empty_cache()

        # ── 5-fold average ──
        if ext1_all_preds:
            df_e1_all = pd.concat(ext1_all_preds, ignore_index=True)
            df_e1_avg = df_e1_all.groupby("eid").agg(
                age_gt=("age_gt", "first"),
                age_pred_raw=("pred_raw", "mean"),
                age_pred_bc=("pred_bc", "mean"),
            ).reset_index()

            # compute final metrics from the averaged predictions
            gts_e1_ens = df_e1_avg["age_gt"].values
            raw_e1_ens = df_e1_avg["age_pred_raw"].values
            bc_e1_ens  = df_e1_avg["age_pred_bc"].values
            m_raw_e1 = compute_metrics(gts_e1_ens, raw_e1_ens)
            m_bc_e1 = compute_metrics(gts_e1_ens, bc_e1_ens)
            print(f"\n  [ext1 5-fold average] n={len(df_e1_avg):,}")
            print_metrics(m_raw_e1, prefix="    Raw | ")
            print_metrics(m_bc_e1,  prefix="    BC  | ")

            df_e1_avg.to_csv(os.path.join(args.out_dir, "predictions_ext_part1.csv"), index=False)

        if ext1_per_fold:
            e1_keys = [k for k in next(iter(ext1_per_fold.values())).keys() if k != "n"]
            e1_avg = _compute_avg(ext1_per_fold, e1_keys)
            print(f"\n  [ext1 per-fold avg over {e1_avg.get('n_folds',0)} folds]")
            print(f"    Raw: MAE={e1_avg.get('raw_mae',float('nan')):.3f}±{e1_avg.get('raw_mae_std',0):.3f}")
            print(f"    BC:  MAE={e1_avg.get('bc_mae',float('nan')):.3f}±{e1_avg.get('bc_mae_std',0):.3f}")

            e1_rows = [{"fold": int(fk), "dataset": "ext_part1", **v}
                       for fk, v in ext1_per_fold.items()]
            pd.DataFrame(e1_rows).to_csv(os.path.join(args.out_dir, "ext_part1_per_fold_metrics.csv"), index=False)

    # ═══════════════════════════════════════
    #   External data 2: fuyi_ext_part3_data (per fold→5-fold average)
    # ═══════════════════════════════════════
    ext2_per_fold = {}
    ext2_all_preds = []
    if os.path.exists(args.ext_part3_meta_csv) and os.path.exists(args.ext_part3_npy_dir):
        print(f"\n{'='*60}")
        print("  External data 2: fuyi_ext_part3_data (per fold → 5-fold average)")
        print(f"{'='*60}")

        for fold in range(args.fold_start, args.fold_end + 1):
            fold_key = str(fold)
            ckpt_path = os.path.join(args.cv_dir, f"fold_{fold_key}", "best_val_mae.pt")
            if not os.path.exists(ckpt_path):
                continue

            model_e2, age_e2, _, _, ls_e2 = load_model(ckpt_path, device)
            ba_e2, bb_e2 = fold_biases.get(fold_key, (0.0, 0.0))
            age_t_e2 = torch.tensor(age_e2, dtype=torch.float32, device=device)

            ds_e2 = NpyDataset(args.ext_part3_npy_dir, args.ext_part3_meta_csv,
                               split="ext_test", age_space=age_e2, label_sigma=ls_e2)
            loader_e2 = _make_loader(ds_e2, bs, nw)
            ids_e2, gts_e2, preds_raw_e2 = run_inference(model_e2, loader_e2, device, age_t_e2)
            preds_bc_e2 = apply_bias(preds_raw_e2, gts_e2, ba_e2, bb_e2)

            raw_e2 = metrics_to_dict(gts_e2, preds_raw_e2, prefix="raw_")
            bc_e2  = metrics_to_dict(gts_e2, preds_bc_e2,  prefix="bc_")
            ext2_per_fold[fold_key] = {**raw_e2, **bc_e2, "n": int(len(gts_e2))}
            print(f"  [fold {fold_key}] n={len(gts_e2):,}  "
                  f"raw MAE={raw_e2['raw_mae']:.3f}  bc MAE={bc_e2['bc_mae']:.3f}")

            ext2_all_preds.append(pd.DataFrame({
                "eid": ids_e2, "age_gt": gts_e2.astype(np.float32),
                "pred_raw": preds_raw_e2.astype(np.float32),
                "pred_bc": preds_bc_e2.astype(np.float32),
                "fold": int(fold_key),
            }))

            del model_e2; torch.cuda.empty_cache()

        # ── 5-fold average ──
        if ext2_all_preds:
            df_e2_all = pd.concat(ext2_all_preds, ignore_index=True)
            df_e2_avg = df_e2_all.groupby("eid").agg(
                age_gt=("age_gt", "first"),
                age_pred_raw=("pred_raw", "mean"),
                age_pred_bc=("pred_bc", "mean"),
            ).reset_index()

            gts_e2_ens = df_e2_avg["age_gt"].values
            raw_e2_ens = df_e2_avg["age_pred_raw"].values
            bc_e2_ens  = df_e2_avg["age_pred_bc"].values
            m_raw_e2 = compute_metrics(gts_e2_ens, raw_e2_ens)
            m_bc_e2 = compute_metrics(gts_e2_ens, bc_e2_ens)
            print(f"\n  [ext_part3 5-fold average] n={len(df_e2_avg):,}")
            print_metrics(m_raw_e2, prefix="    Raw | ")
            print_metrics(m_bc_e2,  prefix="    BC  | ")

            # in/out range analysis
            # The training range is determined by this fold checkpoint's age_space; the external test set's own range must not be used.
            train_min = float(min(age_e2)); train_max = float(max(age_e2))
            in_mask = (gts_e2_ens >= train_min) & (gts_e2_ens <= train_max)
            n_in = int(in_mask.sum()); n_out = int((~in_mask).sum())
            if n_out > 0:
                m_in  = compute_metrics(gts_e2_ens[in_mask],  bc_e2_ens[in_mask])
                m_out = compute_metrics(gts_e2_ens[~in_mask], bc_e2_ens[~in_mask])
                print(f"    In-range  [{train_min:.0f}-{train_max:.0f}] n={n_in}:  BC MAE={m_in['mae']:.3f}")
                print(f"    Out-range n={n_out}:  BC MAE={m_out['mae']:.3f}")

            df_e2_avg["in_train_range"] = in_mask.astype(int)
            df_e2_avg.to_csv(os.path.join(args.out_dir, "predictions_ext_part3.csv"), index=False)

        if ext2_per_fold:
            e2_keys = [k for k in next(iter(ext2_per_fold.values())).keys() if k != "n"]
            e2_avg = _compute_avg(ext2_per_fold, e2_keys)
            print(f"\n  [ext_part3 per-fold avg over {e2_avg.get('n_folds',0)} folds]")
            print(f"    Raw: MAE={e2_avg.get('raw_mae',float('nan')):.3f}±{e2_avg.get('raw_mae_std',0):.3f}")
            print(f"    BC:  MAE={e2_avg.get('bc_mae',float('nan')):.3f}±{e2_avg.get('bc_mae_std',0):.3f}")

            e2_rows = [{"fold": int(fk), "dataset": "ext_part3", **v}
                       for fk, v in ext2_per_fold.items()]
            pd.DataFrame(e2_rows).to_csv(os.path.join(args.out_dir, "ext_part3_per_fold_metrics.csv"), index=False)

    # ═══════════════════════════════════════
    #   Final summary JSON
    # ═══════════════════════════════════════
    summary = {
        "cv_dir": os.path.abspath(args.cv_dir),
        "n_folds": len(all_fold_results),
        "folds_processed": sorted(all_fold_results.keys()),
        "per_fold": all_fold_results,
        "avg": avg_results,
        "unhealthy": {"per_fold": u_per_fold},
        "ext_part1": {"per_fold": ext1_per_fold},
        "ext_part3": {"per_fold": ext2_per_fold},
    }
    summary_path = os.path.join(args.out_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  CV full inference complete!")
    print(f"  Results dir: {args.out_dir}")
    print(f"  Summary JSON: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
