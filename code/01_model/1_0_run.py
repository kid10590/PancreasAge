"""
pancreas_age_train_npy_ddp.py — DDP distributed training script based on .npy + metadata.csv

Usage:
  torchrun --nproc_per_node=4 pancreas_age_train_npy_ddp.py --config configs/pancreas_age_config_npy.json

Pipeline:
  1. Initialize DDP (NCCL backend)
  2. Load config JSON
  3. Read split from metadata.csv, build train/val/test NpyDataset
  4. Distributed training + validation + test
  5. Only rank 0 saves checkpoints and plots
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # modify as needed
import sys
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
# ── project imports (Model/ Dataset/ code/01_model/ are in the same dir as this script) ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
from paths import UKB_NPY_DIR, UKB_META_CSV, OUT_ROOT, PROJ_ROOT  # noqa: E402
from Dataset.npy_dataset import NpyDataset
from Dataset.pancreas_age_dataset import gpu_rand_affine_3d, gpu_rand_intensity
from Model.monai_swin3d_regressor import Swin3DAgeRegressor
try:  # optional RadFM backbone (requires `transformers`)
    from Model.pancreas_age_model import RadFMAgeRegressor
except Exception:  # pragma: no cover
    RadFMAgeRegressor = None
from utils.train_utils import (
    ensure_dir, set_seed, load_config,
    _r2_np, _rmse, _pearson, _prep_target, evidential_nll, fit_delta_vs_age_torch,
)
from utils.plot_figure import plot_curves, plot_scatter


# ═══════════════════════════════════════════════
# DDP utility functions
# ═══════════════════════════════════════════════
def is_rank_zero():
    """Whether this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    """Print only on rank 0"""
    if is_rank_zero():
        print(*args, **kwargs)


def gather_tensors(tensor, world_size=None):
    """
    Gather tensors across all ranks (last dim may differ in length),
    returns a tensor concatenated in rank order.
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if not dist.is_initialized() or world_size <= 1:
        return tensor

    # each rank first sends its length, then the data
    local_size = torch.tensor([tensor.numel()], dtype=torch.long, device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)

    # Pad to max_size
    if local_size < max_size:
        pad = torch.zeros(max_size - int(local_size.item()), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor.flatten(), pad], dim=0)
    else:
        tensor = tensor.flatten()

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    # slice back to original lengths
    result = torch.cat([g[:s] for g, s in zip(gathered, sizes)], dim=0)
    return result


def ddp_evaluate(
    model,
    loader,
    device,
    age_space_tensor,
    cfg,
    kl_weight_now=1.0,
    reg_weight_now=0.0,
    evi_w_now=0.0,
    evi_reg_w=0.0,
    apply_bias=None,
    return_details=False,
):
    """
    DDP version of evaluate: each rank runs forward independently, gathering predictions only when needed.
    model can be DDP-wrapped or a plain nn.Module.
    """
    model.eval()
    # get the underlying model (compatible with both DDP and plain mode)
    base_model = model.module if hasattr(model, "module") else model

    klloss_fn = nn.KLDivLoss(reduction='batchmean')
    use_residual = bool(cfg.get("use_residual_head", False))
    use_evi = bool(cfg.get("use_evidential_head", False))
    reg_type = str(cfg.get("reg_type", "l1")).lower()
    use_amp = bool(cfg.get("use_amp", True))
    amp_dtype_cfg = str(cfg.get("amp_dtype", "fp16")).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype_cfg in ("bf16", "bfloat16") else torch.float16

    mae_list, kl_list, reg_list = [], [], []
    evi_nll_list, evi_reg_list = [], []
    total_loss_list = []
    preds_all_raw, preds_all, gts_all = [], [], []
    pids_all = []  # collect eids in CV mode

    eps = float(cfg.get("label_smoothing", 1e-3))
    logp_min = float(cfg.get("clip_logp_min", -50))
    is_cuda = (device != "cpu")

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            tgt = _prep_target(batch["target"].to(device, non_blocking=True), eps=eps)

            with autocast(enabled=(use_amp and is_cuda), dtype=autocast_dtype, device_type=("cuda" if is_cuda else "cpu")):
                if use_residual or use_evi:
                    logp, feat = base_model.forward_with_feat(x)
                else:
                    logp = model(x)
                    feat = None
                if logp_min is not None:
                    logp = torch.clamp(logp, min=float(logp_min))

                loss_kl = klloss_fn(logp, tgt)
                p = logp.exp()
                base_age = (p * age_space_tensor.unsqueeze(0)).sum(dim=-1)

                r = None
                if use_residual and reg_weight_now > 0 and hasattr(base_model, "res_head") and base_model.res_head is not None:
                    r = base_model.res_head(feat)
                pred_age = base_age + r if r is not None else base_age
                pred_age_raw = pred_age

                gt_age = batch["age"].to(device, non_blocking=True).float()

                if apply_bias is not None and isinstance(apply_bias, dict):
                    alpha = float(apply_bias.get("alpha", 0))
                    beta = float(apply_bias.get("beta", 0))
                    pred_age = pred_age - (alpha * gt_age + beta)

                loss_reg = pred_age.new_tensor(0.0)
                if reg_weight_now > 0:
                    loss_reg = F.mse_loss(pred_age, gt_age) if reg_type == "mse" else F.l1_loss(pred_age, gt_age)

                evi_nll = pred_age.new_tensor(0.0)
                evi_reg = pred_age.new_tensor(0.0)
                if use_evi and hasattr(base_model, "evi_head") and base_model.evi_head is not None and (evi_w_now > 0 or evi_reg_w > 0):
                    mu, nu, alpha, beta = base_model.evi_head(feat)
                    evi_nll = evidential_nll(mu, nu, alpha, beta, gt_age).mean()
                    evi_reg = ((gt_age - mu).abs() * (2.0 * nu + alpha)).mean()

                total_loss = kl_weight_now * loss_kl + reg_weight_now * loss_reg + evi_w_now * evi_nll + evi_reg_w * evi_reg

            mae_list.append(float((pred_age - gt_age).abs().mean().item()))
            kl_list.append(float(loss_kl.item()))
            reg_list.append(float(loss_reg.item()))
            evi_nll_list.append(float(evi_nll.item()))
            evi_reg_list.append(float(evi_reg.item()))
            total_loss_list.append(float(total_loss.item()))

            if return_details:
                preds_all_raw.append(pred_age_raw.detach())
                preds_all.append(pred_age.detach())
                gts_all.append(gt_age.detach())
                pids_all.extend(batch["pid"])  # list of strings

    # average within each rank, then average across ranks
    def _reduce_mean(val_list):
        if not val_list:
            return 0.0
        local_mean = torch.tensor([float(np.mean(val_list))], device=device)
        if dist.is_initialized():
            dist.all_reduce(local_mean, op=dist.ReduceOp.SUM)
            local_mean /= dist.get_world_size()
        return float(local_mean.item())

    results = {
        "mae": _reduce_mean(mae_list),
        "kl": _reduce_mean(kl_list),
        "reg": _reduce_mean(reg_list),
        "evi_nll": _reduce_mean(evi_nll_list),
        "evi_reg": _reduce_mean(evi_reg_list),
        "total_loss": _reduce_mean(total_loss_list),
    }

    if return_details:
        if preds_all:
            local_preds = torch.cat(preds_all, dim=0)
            local_preds_raw = torch.cat(preds_all_raw, dim=0)
            local_gts = torch.cat(gts_all, dim=0)
            # Gather across ranks
            results["preds"] = gather_tensors(local_preds).cpu().numpy()
            results["preds_raw"] = gather_tensors(local_preds_raw).cpu().numpy()
            results["gts"] = gather_tensors(local_gts).cpu().numpy()
            # Gather pids (strings): collect the lists of all ranks onto rank 0
            if dist.is_initialized() and dist.get_world_size() > 1:
                all_pids = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(all_pids, pids_all)
                results["pids"] = [p for sub in all_pids for p in sub]
            else:
                results["pids"] = pids_all
        else:
            results["preds"] = np.array([])
            results["preds_raw"] = np.array([])
            results["gts"] = np.array([])
            results["pids"] = []
    return results


def main(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "..", "..", "configs",
                                                                   "pancreas_age_config_npy.json"))
    # ── path overrides (defaults come from configs/*.json or code/common/paths.py) ──
    parser.add_argument("--npy_dir", type=str, default=None, help="override npy_dir in config")
    parser.add_argument("--metadata_csv", type=str, default=None, help="override metadata_csv in config")
    parser.add_argument("--out_dir", type=str, default=None, help="override out_dir in config")
    # DDP args: torchrun automatically sets env vars such as LOCAL_RANK, RANK, WORLD_SIZE
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank (passed automatically by torchrun)")
    # ── cross-validation ──
    parser.add_argument("--cv_splits", type=str, default="",
                        help="path to cv_splits.json (enables CV mode)")
    parser.add_argument("--cv_fold", type=int, default=-1,
                        help="CV fold index (0-4), used together with --cv_splits")
    # ── resume ──
    parser.add_argument("--resume", action="store_true", default=True,
                        help="automatically resume training from out_dir/last.pt (enabled by default)")
    parser.add_argument("--no-resume", action="store_true", default=False,
                        help="force training from scratch, ignoring existing checkpoint")
    args = parser.parse_args(args=args_list)

    # ── DDP initialization ──
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    ddp_enabled = world_size > 1

    if ddp_enabled:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load config ──
    cfg = load_config(args.config)

    # ── environment ──
    if cfg.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = cfg["hf_endpoint"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)  # each rank uses a different seed

    print_rank0(f"[DDP] rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}")

    # ── paths (CLI > config > paths.py defaults) ──
    def _resolve(p, default):
        p = p or default
        return p if os.path.isabs(p) else os.path.join(PROJ_ROOT, p)

    npy_dir = _resolve(args.npy_dir or cfg.get("npy_dir"), UKB_NPY_DIR)
    metadata_csv = _resolve(args.metadata_csv or cfg.get("metadata_csv"), UKB_META_CSV)
    ckpt_path = cfg.get("ckpt_path", "")
    out_dir = _resolve(args.out_dir or cfg.get("out_dir"), OUT_ROOT)

    # ── cross-validation: override metadata_csv and out_dir ──
    cv_mode = bool(args.cv_splits and args.cv_fold >= 0)
    if cv_mode:
        with open(args.cv_splits, "r") as f:
            cv_all = json.load(f)
        fold_key = str(args.cv_fold)
        if fold_key not in cv_all:
            raise ValueError(f"no fold {fold_key} in CV splits, available: {list(cv_all.keys())}")
        fold_info = cv_all[fold_key]
        out_dir = os.path.join(out_dir, "cv", f"fold_{fold_key}")
        # generate a temporary metadata_csv: set the split column per fold
        _df_meta = pd.read_csv(metadata_csv)
        _split_map = {}
        for _e in fold_info["train"]:
            _split_map[_e] = "train"
        for _e in fold_info["val"]:
            _split_map[_e] = "val"
        for _e in fold_info["test"]:
            _split_map[_e] = "test"
        _df_meta["split"] = _df_meta["eid"].map(_split_map).fillna("unused")
        _df_meta = _df_meta[_df_meta["split"] != "unused"].copy()
        metadata_csv = os.path.join(out_dir, "metadata_fold.csv")
        if is_rank_zero():
            ensure_dir(out_dir)
            _df_meta.to_csv(metadata_csv, index=False)
        # wait for rank 0 to finish writing the CSV
        if ddp_enabled:
            dist.barrier()
        print_rank0(f"[CV] fold={fold_key} | out_dir={out_dir} | n={len(_df_meta)} "
                     f"(train={len(fold_info['train'])}, val={len(fold_info['val'])}, test={len(fold_info['test'])})")

    if is_rank_zero():
        ensure_dir(out_dir)

    # ── log file (rank 0 only, output to both terminal and .log file) ──
    if is_rank_zero():
        import datetime as _dt
        log_path = os.path.join(out_dir, f"train_{_dt.datetime.now().strftime('%m%d-%H%M%S')}.log")
        # use a Tee class to write to stdout and file at the same time
        class _Tee:
            def __init__(self, f1, f2):
                self.f1 = f1; self.f2 = f2
            def write(self, s):
                self.f1.write(s); self.f2.write(s)
            def flush(self):
                self.f1.flush(); self.f2.flush()
        _log_f = open(log_path, 'a', buffering=1)
        sys.stdout = _Tee(sys.stdout, _log_f)
        sys.stderr = _Tee(sys.stderr, _log_f)
        print(f"[Log] log saved to: {log_path}")
        # also print the full config
        print(f"[Config] {json.dumps(cfg, indent=2, ensure_ascii=False)}")

    # ── label / loss params ──
    label_sigma = float(cfg.get("label_sigma", 2.0))
    kl_weight = float(cfg.get("kl_weight", 1.0))
    reg_weight = float(cfg.get("reg_weight", 0.2))
    reg_type = str(cfg.get("reg_type", "l1")).lower()

    # ── build datasets ──
    print_rank0(f"[Data] npy: {npy_dir}")
    print_rank0(f"[Data] metadata: {metadata_csv}")

    train_set = NpyDataset(npy_dir, metadata_csv, split="train", label_sigma=label_sigma)
    val_set = NpyDataset(npy_dir, metadata_csv, split="val", label_sigma=label_sigma, age_space=train_set.age_space)
    test_set = NpyDataset(npy_dir, metadata_csv, split="test", label_sigma=label_sigma, age_space=train_set.age_space)

    age_space = train_set.age_space
    print_rank0(f"  Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print_rank0(f"  age_space (K={len(age_space)}): [{age_space[0]}, ..., {age_space[-1]}]")

    # ── Distributed Sampler + DataLoader ──
    bs_tr = int(cfg.get("batch_size_train", 2))
    bs_ev = int(cfg.get("batch_size_eval", 2))
    nw_tr = int(cfg.get("num_workers_train", 4))
    nw_ev = int(cfg.get("num_workers_eval", 2))
    pf = int(cfg.get("prefetch_factor", 2))

    if ddp_enabled:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
        shuf_tr = False  # sampler already shuffles
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuf_tr = True

    train_loader = DataLoader(train_set, batch_size=bs_tr, shuffle=shuf_tr,
                              sampler=train_sampler,
                              num_workers=nw_tr, pin_memory=True,
                              persistent_workers=(nw_tr > 0), prefetch_factor=pf,
                              drop_last=ddp_enabled)  # drop the last incomplete batch under DDP
    val_loader = DataLoader(val_set, batch_size=bs_ev, shuffle=False,
                            sampler=val_sampler,
                            num_workers=nw_ev, pin_memory=True,
                            persistent_workers=(nw_ev > 0), prefetch_factor=pf)
    test_loader = DataLoader(test_set, batch_size=bs_ev, shuffle=False,
                             sampler=test_sampler,
                             num_workers=nw_ev, pin_memory=True,
                             persistent_workers=(nw_ev > 0), prefetch_factor=pf)

    # ── TF32 ──
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── model ──
    model_type = str(cfg.get("model_type", "radfm")).lower()
    if model_type == "swin3d":
        model_cfg = dict(cfg.get("model", {}))
        model = Swin3DAgeRegressor(num_classes=len(age_space), model_cfg=model_cfg).to(device)
    else:
        if RadFMAgeRegressor is None:
            raise ImportError(
                "model_type != 'swin3d' requires the optional RadFM backbone "
                "(Model/pancreas_age_model.py) and the `transformers` package. "
                "The published model uses model_type='swin3d'.")
        model = RadFMAgeRegressor(
            num_classes=len(age_space), ckpt_path=ckpt_path,
            finetune_patch=bool(cfg.get("finetune_patch", True)),
            unfreeze_last_n=int(cfg.get("unfreeze_last_n", 2)),
        ).to(device)

    # optional residual head / Evidential head
    _regressor_types = (Swin3DAgeRegressor,) + ((RadFMAgeRegressor,) if RadFMAgeRegressor else ())
    if isinstance(model, _regressor_types):
        if bool(cfg.get("use_residual_head", False)):
            model.build_residual_head(
                hidden=int(cfg.get("residual_hidden", 256)),
                dropout=float(cfg.get("residual_dropout", 0.2)),
                scale=float(cfg.get("residual_scale", 10.0)),
            ).to(device)
        if bool(cfg.get("use_evidential_head", False)):
            model.build_evidential_head(
                hidden=int(cfg.get("evi_hidden", 256)),
                dropout=float(cfg.get("evi_dropout", 0.2)),
            ).to(device)

    age_space_tensor = torch.tensor(age_space, dtype=torch.float32, device=device)
    if hasattr(model, "head") and hasattr(model.head, 'drop'):
        model.head.drop.p = float(cfg.get("head_dropout", cfg.get("model", {}).get("head_dropout", 0.5)))

    def _set_backbone_requires_grad(model, requires_grad, freeze_norm=True):
        bb = getattr(model, "backbone", None)
        if bb is None:
            return
        for p in bb.parameters():
            p.requires_grad = requires_grad
        if freeze_norm:
            for m in bb.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()

    # ── optimizer ──
    lr = float(cfg.get("lr", 5e-3))
    wd = float(cfg.get("weight_decay", 5e-4))
    bb_lr = float(cfg.get("backbone_lr", 2e-5))
    bb_wd = float(cfg.get("backbone_weight_decay", 1e-4))
    base_head_lr = lr
    base_bb_lr = bb_lr

    if bool(cfg.get("scale_lr", False)):
        base_bs = int(cfg.get("base_batch_size", 32))
        scale = max(1.0, float(bs_tr * world_size) / max(1, base_bs))
        lr *= scale
        bb_lr *= scale
        print_rank0(f"[lr-scale] batch={bs_tr * world_size}, base={base_bs}, scale={scale:.2f} -> lr={lr:.3e}, bb_lr={bb_lr:.3e}")

    optimizer = torch.optim.AdamW(
        model.param_groups(lr_head=lr, lr_backbone=bb_lr, wd_head=wd, wd_backbone=bb_wd)
    )

    # ── learning rate schedule ──
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    total_epochs = int(cfg.get("epochs", 100))
    min_lr = float(cfg.get("min_lr", lr * 0.1))
    min_ratio = max(1e-6, min_lr / max(lr, 1e-12))

    def lr_lambda(e):
        e0 = e - 1
        if e0 < warmup_epochs:
            return (e0 + 1) / max(1, warmup_epochs)
        t = (e0 - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── AMP ──
    use_amp = bool(cfg.get("use_amp", True))
    amp_dtype_cfg = str(cfg.get("amp_dtype", "fp16")).lower()
    autocast_dtype = torch.bfloat16 if amp_dtype_cfg in ("bf16", "bfloat16") else torch.float16
    scaler = GradScaler(enabled=(use_amp and device != "cpu"))

    # ── DDP wrap model ──
    # use find_unused_parameters=True because some heads (res/evi) may be unused temporarily
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
        # synchronize BatchNorm (if the model has BN layers)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ── history (only rank 0 keeps the full history) ──
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    plot_every = int(cfg.get("plot_every", 1))
    early_stop_patience = int(cfg.get("early_stop_patience", 0))
    hist = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_mae_raw": [], "train_mae_bc": [],
        "val_mae_raw": [], "val_mae_bc": [],
        "train_kl": [], "train_reg": [],
        "val_kl": [], "val_reg": [],
        "train_evi_nll": [], "train_evi_reg": [],
        "val_evi_nll": [], "val_evi_reg": [],
        "train_r2_raw": [], "train_r2_bc": [],
        "val_r2_raw": [], "val_r2_bc": [], "test_r2": [],
        "train_rmse_raw": [], "train_rmse_bc": [],
        "train_r_raw": [], "train_r_bc": [],
        "val_rmse_raw": [], "val_rmse_bc": [],
        "val_r_raw": [], "val_r_bc": [],
        "test_rmse_raw": [], "test_rmse_bc": [],
        "test_r_raw": [], "test_r_bc": [],
    }

    klloss = nn.KLDivLoss(reduction='batchmean')
    epochs = int(cfg.get("epochs", 30))
    bias_record = None
    start_epoch = 1

    # ── save config (rank 0 only) ──
    if is_rank_zero():
        with open(os.path.join(out_dir, "config_used.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # ═══════════════════════════════════════════
    #   Resume: restore training state from last.pt
    # ═══════════════════════════════════════════
    do_resume = args.resume and not args.no_resume
    resume_path = os.path.join(out_dir, "last.pt")
    if do_resume and os.path.exists(resume_path):
        print_rank0(f"[Resume] found checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        base_model = model.module if ddp_enabled else model

        # ── model weights ──
        if ckpt.get("model_head"):
            base_model.head.load_state_dict(ckpt["model_head"])
        if ckpt.get("model_backbone") and hasattr(base_model, "backbone") and base_model.backbone is not None:
            base_model.backbone.load_state_dict(ckpt["model_backbone"])
        if ckpt.get("model_res_head") and hasattr(base_model, "res_head") and base_model.res_head is not None:
            base_model.res_head.load_state_dict(ckpt["model_res_head"])
        if ckpt.get("model_evi_head") and hasattr(base_model, "evi_head") and base_model.evi_head is not None:
            base_model.evi_head.load_state_dict(ckpt["model_evi_head"])

        # ── optimizer & scheduler ──
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])

        # ── training state ──
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_mae = ckpt.get("val_mae", best_val_mae)
        best_val_loss = ckpt.get("val_loss", best_val_loss)
        if ckpt.get("hist"):
            hist.update(ckpt["hist"])
        if ckpt.get("bias"):
            bias_record = ckpt["bias"]

        print_rank0(f"[Resume] resuming training from epoch {start_epoch}, "
                     f"best_val_mae={best_val_mae:.4f}, best_val_loss={best_val_loss:.4f}")
    elif do_resume:
        print_rank0(f"[Resume] no checkpoint found ({resume_path}), training from scratch.")

    # ═══════════════════════════════════════════
    #   training loop
    # ═══════════════════════════════════════════
    for epoch in range(start_epoch, epochs + 1):
        # set sampler epoch at the start of each epoch (ensures correct shuffle)
        if ddp_enabled and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── freeze/unfreeze ──
        freeze_until = int(cfg.get("freeze_until_epoch", 0))
        base_model = model.module if ddp_enabled else model
        if freeze_until > 0:
            if epoch <= freeze_until:
                _set_backbone_requires_grad(base_model, requires_grad=False, freeze_norm=bool(cfg.get("freeze_norm", True)))
            elif epoch == freeze_until + 1:
                _set_backbone_requires_grad(base_model, requires_grad=True, freeze_norm=False)

        model.train()
        eps = float(cfg.get("label_smoothing", 1e-3))
        logp_min = float(cfg.get("clip_logp_min", -50))
        grad_clip = float(cfg.get("grad_clip_norm", 1.0))

        # ── dynamic weights ──
        dual_anneal = bool(cfg.get("dual_anneal", False))
        anneal_epochs = int(cfg.get("kl_reg_anneal_epochs", max(5, epochs // 3)))
        kl_min_ratio = float(cfg.get("kl_min_ratio", 0.2))
        reg_start_ratio = float(cfg.get("reg_start_ratio", 0.1))

        progress = min(1.0, max(0.0, (epoch - 1) / max(1, anneal_epochs)))
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        if dual_anneal:
            kl_weight_now = kl_weight * (kl_min_ratio + (1.0 - kl_min_ratio) * cos_decay)
            reg_weight_now = reg_weight * (reg_start_ratio + (1.0 - reg_start_ratio) * (1.0 - cos_decay))
        else:
            reg_warmup = int(cfg.get("reg_warmup_epochs", 5))
            kl_weight_now = kl_weight
            reg_weight_now = reg_weight * min(1.0, epoch / max(1, reg_warmup))

        evi_w_base = float(cfg.get("evi_weight", 0.0))
        evi_reg_w = float(cfg.get("evi_reg_weight", 1e-3))
        evi_start_e = int(cfg.get("evi_start_epoch", 3))
        evi_warmup_e = int(cfg.get("evi_warmup_epochs", 5))
        if epoch < evi_start_e:
            evi_w_now = 0.0
        else:
            evi_prog = min(1.0, (epoch - evi_start_e + 1) / max(1, evi_warmup_e))
            evi_w_now = evi_w_base * evi_prog

        if epoch == start_epoch or epoch % int(cfg.get("print_weight_every", 1)) == 0:
            print_rank0(f"[weight] epoch {epoch}: KL={kl_weight_now:.3f}, REG={reg_weight_now:.3f}, EVI={evi_w_now:.3f}")

        # ── backbone unfreeze warm start ──
        bb_unfreeze_warmup = int(cfg.get("bb_unfreeze_warmup_epochs", 5))
        bb_start_ratio = float(cfg.get("bb_unfreeze_start_ratio", 0.1))
        sched_factor = lr_lambda(epoch)
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[0]["lr"] = base_head_lr * sched_factor
            if epoch <= int(cfg.get("freeze_until_epoch", 0)):
                optimizer.param_groups[1]["lr"] = 0.0
            else:
                step = max(0, epoch - int(cfg.get("freeze_until_epoch", 0)))
                warm_ratio = min(1.0, bb_start_ratio + (1.0 - bb_start_ratio) * (step / max(1, bb_unfreeze_warmup)))
                optimizer.param_groups[1]["lr"] = base_bb_lr * sched_factor * warm_ratio

        # ═══════════ TRAIN ═══════════
        use_residual = bool(cfg.get("use_residual_head", False))
        use_evi = bool(cfg.get("use_evidential_head", False))
        losses, maes, kls, regs = [], [], [], []
        evi_nll_list, evi_reg_list = [], []
        preds_tr_t, gts_tr_t = [], []

        train_bar = train_loader
        if is_rank_zero():
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Training")

        for batch in train_bar:
            x = batch["image"].to(device, non_blocking=True)

            # GPU augmentation
            if bool(cfg.get("gpu_aug", False)):
                if x.dtype != torch.float32:
                    x = x.float()
                with torch.no_grad():
                    x = gpu_rand_affine_3d(x,
                        max_deg=float(cfg.get("gpu_aug_rot_deg", 0.0)),
                        translate_frac=tuple(cfg.get("gpu_aug_translate_frac", [0.0, 0.0, 0.0])),
                        p=float(cfg.get("gpu_aug_p", 0.0)),
                        padding_mode=str(cfg.get("gpu_aug_padding_mode", "border")))
                    x = gpu_rand_intensity(x,
                        p=float(cfg.get("gpu_aug_p_intensity", 0.0)),
                        gamma_range=tuple(cfg.get("gpu_aug_gamma_range", [0.9, 1.1])),
                        noise_std=float(cfg.get("gpu_aug_noise_std", 0.0)))
                    x = x.contiguous()

            tgt = _prep_target(batch["target"].to(device, non_blocking=True), eps=eps)

            with autocast(enabled=(use_amp and device != "cpu"), dtype=autocast_dtype, device_type=("cuda" if device != "cpu" else "cpu")):
                if use_residual or use_evi:
                    logp, feat = base_model.forward_with_feat(x)
                else:
                    logp = model(x); feat = None
                if logp_min is not None:
                    logp = torch.clamp(logp, min=logp_min)

                loss_kl = klloss(logp, tgt)
                p = logp.exp()
                base_age = (p * age_space_tensor.unsqueeze(0)).sum(dim=-1)
                pred_age = base_age

                if use_residual and (reg_weight_now > 0 or evi_w_now > 0) and hasattr(base_model, "res_head") and base_model.res_head is not None:
                    r = base_model.res_head(feat)
                    pred_age = pred_age + r

                gt_age = batch["age"].to(device, non_blocking=True).float()
                loss_reg = pred_age.new_tensor(0.0)
                if reg_weight_now > 0:
                    loss_reg = F.mse_loss(pred_age, gt_age) if reg_type == "mse" else F.l1_loss(pred_age, gt_age)

                evi_nll = pred_age.new_tensor(0.0)
                evi_reg = pred_age.new_tensor(0.0)
                if use_evi and hasattr(base_model, "evi_head") and base_model.evi_head is not None and (evi_w_now > 0 or evi_reg_w > 0):
                    feat_evi = feat.detach() if bool(cfg.get("evi_detach_backbone", True)) else feat
                    mu, nu, alpha, beta = base_model.evi_head(feat_evi)
                    evi_nll = evidential_nll(mu, nu, alpha, beta, gt_age).mean()
                    evi_reg = ((gt_age - mu).abs() * (2.0 * nu + alpha)).mean()

                loss = kl_weight_now * loss_kl + reg_weight_now * loss_reg + evi_w_now * evi_nll + evi_reg_w * evi_reg

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            with torch.no_grad():
                maes.append((pred_age - gt_age).abs().mean().item())
                preds_tr_t.append(pred_age.detach())
                gts_tr_t.append(gt_age.detach())
            losses.append(loss.item())
            kls.append(loss_kl.item())
            regs.append(loss_reg.item())
            if use_evi and (evi_w_now > 0 or evi_reg_w > 0):
                evi_nll_list.append(float(evi_nll.item()))
                evi_reg_list.append(float(evi_reg.item()))

            if is_rank_zero():
                train_bar.set_postfix({
                    "Loss": f"{np.mean(losses):.4f}",
                    "MAE": f"{np.mean(maes):.4f}",
                })

        # ── Gather train metrics across ranks ──
        def _reduce_mean_list(val_list):
            if not val_list:
                return 0.0
            t = torch.tensor([np.mean(val_list)], device=device)
            if dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t /= world_size
            return float(t.item())

        train_loss = _reduce_mean_list(losses)
        train_mae_raw = _reduce_mean_list(maes)
        train_kl = _reduce_mean_list(kls)
        train_reg = _reduce_mean_list(regs)
        train_evi_nll = _reduce_mean_list(evi_nll_list)
        train_evi_reg = _reduce_mean_list(evi_reg_list)

        # ── Gather train preds/GTs across all ranks (for bias computation) ──
        # note: gather_tensors is used to collect everything at once
        # since during training each rank's pred needs gather_tensors, but here we first use the local ones for bias
        # a more accurate approach is to gather preds/gts from all ranks before computing bias
        preds_tr_t_local = torch.cat(preds_tr_t, dim=0) if preds_tr_t else torch.tensor([], device=device)
        gts_tr_t_local = torch.cat(gts_tr_t, dim=0) if gts_tr_t else torch.tensor([], device=device)

        # Gather across all ranks
        if ddp_enabled:
            preds_tr_t_cat = gather_tensors(preds_tr_t_local, world_size)
            gts_tr_t_cat = gather_tensors(gts_tr_t_local, world_size)
        else:
            preds_tr_t_cat = preds_tr_t_local
            gts_tr_t_cat = gts_tr_t_local

        # ═══════════ VAL ═══════════
        val_bar = val_loader
        if is_rank_zero():
            val_bar = tqdm(val_loader, desc="Validating")
        val_res = ddp_evaluate(model, val_bar, device, age_space_tensor,
                               cfg=cfg,
                               kl_weight_now=kl_weight_now, reg_weight_now=reg_weight_now,
                               evi_w_now=evi_w_now, evi_reg_w=evi_reg_w,
                               apply_bias=None, return_details=True)
        val_mae = val_res["mae"]
        val_total_loss = val_res["total_loss"]
        val_preds = val_res["preds"]
        val_gts = val_res["gts"]

        # ═══════════ BIAS ═══════════
        if preds_tr_t_cat.numel() > 0 and gts_tr_t_cat.numel() > 0:
            alpha_hat, beta_hat = fit_delta_vs_age_torch(gts_tr_t_cat, preds_tr_t_cat)
        else:
            alpha_hat, beta_hat = 0.0, 0.0

        train_preds_raw_np = preds_tr_t_cat.cpu().numpy()
        train_gts_np = gts_tr_t_cat.cpu().numpy()

        # bias stability check
        bias_min_epoch = int(cfg.get("bias_min_epoch", 10))
        den = 1.0 + float(alpha_hat)
        corr_ad = float(np.corrcoef(train_gts_np, train_preds_raw_np - train_gts_np)[0, 1]) if train_gts_np.size > 1 else 0.0
        stable_bias = (
            epoch >= bias_min_epoch
            and np.isfinite(den) and abs(den) > float(cfg.get("bias_den_eps", 1e-3))
            and abs(alpha_hat) < float(cfg.get("bias_max_abs_alpha", 0.9))
            and abs(corr_ad) >= float(cfg.get("bias_min_corr", 0.3))
        )

        train_mae_bc = train_mae_raw
        if stable_bias and preds_tr_t_cat.numel() > 0:
            train_pred_bc_t = preds_tr_t_cat - (alpha_hat * gts_tr_t_cat + beta_hat)
            train_mae_bc = float(torch.mean(torch.abs(train_pred_bc_t - gts_tr_t_cat)).item())

        val_mae_bc = val_mae
        val_pred_bc_np = val_preds.copy() if val_preds.size > 0 else val_preds
        if stable_bias and val_preds.size > 0:
            val_pred_bc_np = val_preds - (alpha_hat * val_gts + beta_hat)
            val_mae_bc = float(np.mean(np.abs(val_pred_bc_np - val_gts)))

        use_bias_now = stable_bias and (val_mae_bc < val_mae)

        # ═══════════ TEST ═══════════
        test_bias = ({"alpha": alpha_hat, "beta": beta_hat} if use_bias_now else None)
        test_bar = test_loader
        if is_rank_zero():
            test_bar = tqdm(test_loader, desc="Testing", leave=False)
        test_res = ddp_evaluate(model, test_bar, device, age_space_tensor,
                                cfg=cfg,
                                kl_weight_now=kl_weight_now, reg_weight_now=reg_weight_now,
                                evi_w_now=evi_w_now, evi_reg_w=evi_reg_w,
                                apply_bias=test_bias, return_details=True)
        test_loss_epoch = float(test_res["total_loss"])
        test_mae_epoch = float(test_res["mae"])
        test_preds_bc = test_res.get("preds", np.array([]))
        test_preds_raw = test_res.get("preds_raw", test_preds_bc)
        test_gts = test_res.get("gts", np.array([]))

        # ── compute metrics ──
        train_r2_raw = _r2_np(train_gts_np, train_preds_raw_np)
        val_r2_raw = _r2_np(val_gts, val_preds)
        test_r2 = _r2_np(test_gts, test_preds_bc)

        train_pred_bc_np = train_preds_raw_np
        train_r2_bc = train_r2_raw
        val_r2_bc = val_r2_raw
        if use_bias_now:
            train_pred_bc_np = train_preds_raw_np - (alpha_hat * train_gts_np + beta_hat)
            train_r2_bc = _r2_np(train_gts_np, train_pred_bc_np)
            val_r2_bc = _r2_np(val_gts, val_pred_bc_np)

        # ── early stop & best model (all ranks compute improved_mae, then broadcast early stop) ──
        improved_mae = (val_mae < best_val_mae)
        improved_loss = (val_total_loss < best_val_loss)

        # ponytail: all ranks update best_val_mae / epochs_no_improve first
        if improved_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
        elif early_stop_patience > 0:
            epochs_no_improve += 1

        if improved_loss:
            best_val_loss = val_total_loss

        # ponytail: broadcast early-stop signal (rank 0 → all ranks), must happen before break
        if ddp_enabled:
            stop_tensor = torch.tensor([epochs_no_improve], dtype=torch.int, device=device)
            dist.broadcast(stop_tensor, src=0)
            epochs_no_improve = stop_tensor.item()

        # ── save checkpoint (rank 0 only) ──
        if is_rank_zero():
            base_model = model.module if ddp_enabled else model
            ckpt_data = {
                "model_head": base_model.head.state_dict(),
                "model_res_head": (base_model.res_head.state_dict() if hasattr(base_model, "res_head") and base_model.res_head is not None else None),
                "model_evi_head": (base_model.evi_head.state_dict() if hasattr(base_model, "evi_head") and base_model.evi_head is not None else None),
                "model_backbone": (base_model.backbone.state_dict() if hasattr(base_model, "backbone") and base_model.backbone is not None else None),
                "age_space": age_space,
                "bias": ({"alpha": alpha_hat, "beta": beta_hat} if use_bias_now else None),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
                "hist": hist,
                "val_mae": best_val_mae,
                "val_loss": best_val_loss,
            }

            if improved_mae:
                torch.save(ckpt_data, os.path.join(out_dir, "best_val_mae.pt"))
                print_rank0(f"  >>> Best MAE: {best_val_mae:.4f} (raw={val_mae:.4f}, bc={val_mae_bc:.4f})")

            if improved_loss:
                torch.save(ckpt_data, os.path.join(out_dir, "best_val_loss.pt"))
                print_rank0(f"  >>> Best Loss: {best_val_loss:.4f}")

            torch.save(ckpt_data, os.path.join(out_dir, "last.pt"))

        # ponytail: all ranks break uniformly (after the broadcast)
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print_rank0(f"[early-stop] no improvement for {epochs_no_improve} epochs, stopping.")
            break

        scheduler.step()

        # ── record history (consistent across all ranks, but recorded only on rank 0) ──
        if is_rank_zero():
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_total_loss)
            hist["test_loss"].append(test_loss_epoch)
            hist["train_mae_raw"].append(train_mae_raw)
            hist["train_mae_bc"].append(train_mae_bc)
            hist["val_mae_raw"].append(val_mae)
            hist["val_mae_bc"].append(val_mae_bc)
            hist["train_kl"].append(train_kl)
            hist["train_reg"].append(train_reg)
            hist["val_kl"].append(val_res["kl"])
            hist["val_reg"].append(val_res["reg"])
            hist["train_r2_raw"].append(train_r2_raw)
            hist["train_r2_bc"].append(train_r2_bc)
            hist["val_r2_raw"].append(val_r2_raw)
            hist["val_r2_bc"].append(val_r2_bc)
            hist["test_r2"].append(test_r2)
            hist["train_evi_nll"].append(train_evi_nll)
            hist["train_evi_reg"].append(train_evi_reg)
            hist["val_evi_nll"].append(val_res["evi_nll"])
            hist["val_evi_reg"].append(val_res["evi_reg"])
            hist.setdefault("train_rmse_raw", []).append(_rmse(train_gts_np, train_preds_raw_np))
            hist.setdefault("train_rmse_bc", []).append(_rmse(train_gts_np, train_pred_bc_np))
            hist.setdefault("train_r_raw", []).append(_pearson(train_gts_np, train_preds_raw_np))
            hist.setdefault("train_r_bc", []).append(_pearson(train_gts_np, train_pred_bc_np))
            hist.setdefault("val_rmse_raw", []).append(_rmse(val_gts, val_preds))
            hist.setdefault("val_rmse_bc", []).append(_rmse(val_gts, val_pred_bc_np))
            hist.setdefault("val_r_raw", []).append(_pearson(val_gts, val_preds))
            hist.setdefault("val_r_bc", []).append(_pearson(val_gts, val_pred_bc_np))
            hist.setdefault("test_rmse_raw", []).append(_rmse(test_gts, test_preds_raw))
            hist.setdefault("test_rmse_bc", []).append(_rmse(test_gts, test_preds_bc))
            hist.setdefault("test_r_raw", []).append(_pearson(test_gts, test_preds_raw))
            hist.setdefault("test_r_bc", []).append(_pearson(test_gts, test_preds_bc))

            print(f"Epoch {epoch:3d} | Train Loss={train_loss:.4f} MAE={train_mae_raw:.3f}(bc={train_mae_bc:.3f}) "
                  f"| Val Loss={val_total_loss:.4f} MAE={val_mae:.3f}(bc={val_mae_bc:.3f}) "
                  f"| Test MAE={test_mae_epoch:.3f} R²={test_r2:.3f}")

            # ── plotting (rank 0 only) ──
            if epoch % plot_every == 0 and epoch > 1:
                try:
                    plot_curves(hist, out_dir)
                    scat_path = os.path.join(out_dir, f"scatter_epoch_{epoch}.png")
                    plot_scatter(val_gts, val_pred_bc_np if use_bias_now else val_preds,
                                 scat_path, title=f"Val Epoch {epoch}")
                except Exception as e:
                    print_rank0(f"[plot] plotting failed: {e}")

    # ═══════════════════════════════════════════
    #   Final test evaluation (all ranks take part in collective communication, only rank 0 saves)
    # ═══════════════════════════════════════════
    # ponytail: ddp_evaluate contains all_reduce/all_gather, so it must be called by all ranks
    final_test_res = ddp_evaluate(model, test_loader, device, age_space_tensor,
                                  cfg=cfg, kl_weight_now=kl_weight_now,
                                  reg_weight_now=reg_weight_now,
                                  evi_w_now=evi_w_now, evi_reg_w=evi_reg_w,
                                  apply_bias=test_bias, return_details=True)

    if is_rank_zero():
        print(f"\nTraining complete, best Val MAE={best_val_mae:.4f}")
        print_rank0("\n" + "=" * 50)
        print_rank0("Final test set evaluation & saving results")
        print_rank0("=" * 50)
        try:
            import pandas as _pd
            _preds_final = final_test_res.get("preds", np.array([]))
            _preds_raw   = final_test_res.get("preds_raw", _preds_final)
            _gts_final   = final_test_res.get("gts", np.array([]))
            _pids_final   = final_test_res.get("pids", [])
            _mae_final   = float(np.mean(np.abs(_preds_final - _gts_final))) if _gts_final.size > 0 else float('nan')
            _r2_final    = _r2_np(_gts_final, _preds_final) if _gts_final.size > 0 else float('nan')

            if _gts_final.size > 0:
                _df = pd.DataFrame({
                    "eid": _pids_final if len(_pids_final) == len(_gts_final) else [""] * len(_gts_final),
                    "age_gt": _gts_final,
                    "age_pred_raw": _preds_raw,
                    "age_pred_bc": _preds_final,
                })
                _csv_path = os.path.join(out_dir, "final_test_predictions.csv")
                _df.to_csv(_csv_path, index=False)
                print_rank0(f"[save] test predictions CSV ({len(_df)} rows) → {_csv_path}")

            _summary_path = os.path.join(out_dir, "final_summary.txt")
            with open(_summary_path, "w") as _f:
                _f.write(f"best_val_mae={best_val_mae:.4f}\n")
                _f.write(f"best_val_loss={best_val_loss:.4f}\n")
                _f.write(f"final_test_mae={_mae_final:.4f}\n")
                _f.write(f"final_test_r2={_r2_final:.4f}\n")
                _f.write(f"bias_alpha={alpha_hat:.6f}\n")
                _f.write(f"bias_beta={beta_hat:.6f}\n")
                _f.write(f"bias_stable={use_bias_now}\n")
                _f.write(f"epochs_completed={epoch}\n")
                _f.write(f"train_samples={len(train_set)}\n")
                _f.write(f"val_samples={len(val_set)}\n")
                _f.write(f"test_samples={len(test_set)}\n")
            print_rank0(f"[save] final summary → {_summary_path}")
        except Exception as e:
            print_rank0(f"[final eval] error: {e}")

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
