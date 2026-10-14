#!/bin/bash
#
# run_cv_folds.sh — launch 5-fold cross-validation training (default folds 0-4), supports resume
#
# Usage:
#   bash code/01_model/run_cv_folds.sh [start_fold] [end_fold] [--no-resume]
#
# Examples:
#   bash code/01_model/run_cv_folds.sh                # run all folds 0-4 (all, auto-resume)
#   bash code/01_model/run_cv_folds.sh 1 4            # run only folds 1-4
#   bash code/01_model/run_cv_folds.sh 0 4 --no-resume # force retraining all 5 folds from scratch
#
# Dependencies:
#   code/01_model/generate_cv_splits.py has been run, producing cv_splits.json
#
set -euo pipefail

# ── configuration ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/1_0_run.py"
CONFIG="${PKG_ROOT}/configs/pancreas_age_config_npy.json"
# Results root: PANCREASAGE_OUT (default: <package>/results), same convention as code/common/paths.py
OUT_ROOT="${PANCREASAGE_OUT:-${PKG_ROOT}/results}"
CV_SPLITS="${OUT_ROOT}/cv/cv_splits.json"

START_FOLD=${1:-0}
END_FOLD=${2:-4}
NO_RESUME=""
# Check whether the --no-resume argument is present
for arg in "$@"; do
    if [ "$arg" = "--no-resume" ]; then
        NO_RESUME="--no-resume"
    fi
done

# ── GPU settings ──
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# ── NCCL ──
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^lo,docker,virbr"}
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

echo "╔══════════════════════════════════════════════╗"
echo "║     5-Fold CV Training (folds ${START_FOLD}-${END_FOLD})       ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  GPUs       : $CUDA_VISIBLE_DEVICES ($NPROC)"
echo "║  CV splits  : $CV_SPLITS"
echo "║  Script     : $RUN_SCRIPT"
echo "║  Config     : $CONFIG"
[ -n "$NO_RESUME" ] && echo "║  Resume     : OFF (--no-resume)"
echo "╚══════════════════════════════════════════════╝"

# Check whether cv_splits.json exists
if [ ! -f "$CV_SPLITS" ]; then
    echo "[ERROR] cv_splits.json does not exist; please run first: python code/01_model/generate_cv_splits.py"
    exit 1
fi

for FOLD in $(seq "$START_FOLD" "$END_FOLD"); do
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Fold $FOLD / 4"
    echo "══════════════════════════════════════════════"

    torchrun \
        --nproc_per_node="$NPROC" \
        --master_port=$((29500 + FOLD * 10)) \
        "$RUN_SCRIPT" \
        --config "$CONFIG" \
        --cv_splits "$CV_SPLITS" \
        --cv_fold "$FOLD" \
        $NO_RESUME

    echo "[Fold $FOLD] complete."
done

echo ""
echo "══════════════════════════════════════════════"
echo "  All CV folds training complete!"
echo "  Next step: python ${SCRIPT_DIR}/1_1_infer_all_cv.py"
echo "══════════════════════════════════════════════"
