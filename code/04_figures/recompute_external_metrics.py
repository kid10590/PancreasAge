#!/usr/bin/env python3
"""
recompute_external_metrics.py — recompute external-center metrics (excluding age<18, no retraining)

Background: metrics for the external centers (fuyi_ext_npy / fuyi_ext_part3_data) must be
recomputed after excluding subjects aged <18. Model weights are unchanged (UKB is not
re-inferred); only the two external sets are inferred fold by fold (fast), using exactly
the same procedure as 1_1_infer_all_cv.py:
  - per-fold bias (alpha, beta) recovered precisely by linear regression from fold{fold}_train_predictions.csv
  - per-fold inference on the external sets → apply_bias → exclude age_gt<18 → per-fold metrics
  - average over 5 folds (ensemble) → prediction files + ensemble metrics

Output (overwrite, back up first):
  infer_cv/predictions_ext_part1.csv / predictions_ext_part3.csv
  infer_cv/ext_part1_per_fold_metrics.csv / ext_part3_per_fold_metrics.csv

Usage: python code/04_figures/recompute_external_metrics.py [--device cuda:0]
"""
import os, sys, json, argparse, importlib.util
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT, OUT_ROOT  # noqa: E402

R = OUT_ROOT
MIN_AGE = 18.0

# Load the 1_1 module to reuse exactly matching functions
_1_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_model", "1_1_infer_all_cv.py")
spec = importlib.util.spec_from_file_location("m11", _1_1)
m11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11)


def recover_fold_biases():
    """Recover each fold's bias precisely from fold{fold}_train_predictions.csv.
    pred_bc = pred_raw - (alpha*gt + beta) → regress (pred_raw - pred_bc) ~ gt"""
    biases = {}
    for fold in range(5):
        f = os.path.join(R, "infer_cv", f"fold{fold}_train_predictions.csv")
        t = pd.read_csv(f)
        y = t["age_pred_raw"].values - t["age_pred_bc"].values
        x = t["age_gt"].values
        A = np.vstack([np.ones_like(x), x]).T
        beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        biases[str(fold)] = (float(alpha), float(beta))
    return biases


def infer_ext(model, device, npy_dir, meta_csv, age_space, label_sigma, bs, nw, age_t):
    ds = m11.NpyDataset(npy_dir, meta_csv, split="ext_test",
                        age_space=age_space, label_sigma=label_sigma)
    loader = m11._make_loader(ds, bs, nw)
    ids, gts, preds_raw = m11.run_inference(model, loader, device, age_t)
    return ids, np.asarray(gts, dtype=np.float64), np.asarray(preds_raw, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[recompute] device={device}, MIN_AGE={MIN_AGE}")

    biases = recover_fold_biases()
    print("Recovered per-fold bias:", {k: (round(a,4), round(b,4)) for k,(a,b) in biases.items()})

    ext_cfgs = {
        "ext_part1": ("ext_healthy_npy", os.path.join(DATA_ROOT, "ext_healthy_npy", "metadata.csv"),
                      os.path.join(DATA_ROOT, "ext_healthy_npy", "images_npy")),
        "ext_part3": ("obesity_cohort", os.path.join(DATA_ROOT, "obesity_cohort", "metadata.csv"),
                      os.path.join(DATA_ROOT, "obesity_cohort", "images_npy")),
    }

    for tag, (name, meta_csv, npy_dir) in ext_cfgs.items():
        per_fold = []
        all_rows = []
        train_min, train_max = None, None
        for fold in range(5):
            fold_key = str(fold)
            ckpt = os.path.join(R, "cv", f"fold_{fold_key}", "best_val_mae.pt")
            if not os.path.exists(ckpt):
                print(f"  [skip] fold {fold_key} has no checkpoint"); continue
            model, age_space, _, _, label_sigma = m11.load_model(ckpt, device)
            if train_min is None:
                train_min, train_max = float(min(age_space)), float(max(age_space))
            age_t = torch.tensor(age_space, dtype=torch.float32, device=device)
            ba, bb = biases[fold_key]
            ids, gts, preds_raw = infer_ext(model, device, npy_dir, meta_csv,
                                            age_space, label_sigma, args.batch_size,
                                            args.num_workers, age_t)
            preds_bc = m11.apply_bias(preds_raw, gts, ba, bb)

            keep = gts >= MIN_AGE
            ids, gts = np.asarray(ids)[keep], gts[keep]
            preds_raw, preds_bc = preds_raw[keep], preds_bc[keep]

            raw = m11.metrics_to_dict(gts, preds_raw, prefix="raw_")
            bc = m11.metrics_to_dict(gts, preds_bc, prefix="bc_")
            per_fold.append({"fold": fold, "dataset": tag, **raw, **bc, "n": int(len(gts))})
            print(f"  [{tag}] fold {fold} n={len(gts)}  raw MAE={raw['raw_mae']:.3f}  bc MAE={bc['bc_mae']:.3f}")

            for i, eid in enumerate(ids):
                all_rows.append({"eid": str(eid), "age_gt": float(gts[i]),
                                 "age_pred_raw": float(preds_raw[i]),
                                 "age_pred_bc": float(preds_bc[i]),
                                 "fold": fold})
            del model; torch.cuda.empty_cache()

        pd.DataFrame(per_fold).to_csv(os.path.join(R, "infer_cv", f"{tag}_per_fold_metrics.csv"), index=False)
        print(f"  → {tag}_per_fold_metrics.csv ({len(per_fold)} folds, n={per_fold[0]['n']})")

        dfall = pd.DataFrame(all_rows)
        ens = dfall.groupby("eid").agg(age_gt=("age_gt", "first"),
                                       age_pred_raw=("age_pred_raw", "mean"),
                                       age_pred_bc=("age_pred_bc", "mean")).reset_index()
        ens = ens.sort_values("eid").reset_index(drop=True)
        if tag == "ext_part3":
            ens["in_train_range"] = ((ens["age_gt"] >= train_min) & (ens["age_gt"] <= train_max)).astype(int)
        ens.to_csv(os.path.join(R, "infer_cv", f"predictions_{tag}.csv"), index=False)
        g = ens["age_gt"].values; pr = ens["age_pred_raw"].values; pb = ens["age_pred_bc"].values
        mr = m11.compute_metrics(g, pr); mb = m11.compute_metrics(g, pb)
        print(f"  → predictions_{tag}.csv n={len(ens)}  ensemble bc MAE={mb['mae']:.4f} R2={mb.get('r2', np.corrcoef(g,pb)[0,1]**2):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
