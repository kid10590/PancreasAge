#!/bin/bash
# ============================================================
# run_regenie_full_phase2.sh — wait for full-sample REGENIE to finish, then run the Phase 2 downstream
#
# Phase 2 (full-sample REGENIE):
#   3_3 Manhattan/QQ + significant loci
#   3_5 LD clumping → regenie_lead_loci.csv
#   3_4 REGENIE vs PLINK cross-validation (full)
#   4_2 generate MR exposure files (mr_regenie/)
#
# MR analysis (4_3_run_mr_analysis.R) requires OpenGWAS network access; run it manually.
#
# Usage: bash code/03_genetics/run_regenie_full_phase2.sh &
# ============================================================
# ── paths (override via env: PANCREASAGE_DATA / PANCREASAGE_OUT / PLINK2 / REGENIE) ──
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${PANCREASAGE_DATA:-${PKG_ROOT}/data}"
OUT_ROOT="${PANCREASAGE_OUT:-${PKG_ROOT}/results}"
set -u
RES="${RES:-${OUT_ROOT}}"
LOG="$RES/logs_regenie/phase2_full.log"
mkdir -p "$RES/logs_regenie" "$RES/mr_regenie"
cd "${PKG_ROOT}" || exit 1

echo "[$(date)] Phase2 watcher started" > "$LOG"
while :; do
    n=$(ls "$RES/gwas_regenie_results"/gwas_chr*_catb_delta_with_t1_bc_cole.regenie.gz 2>/dev/null | wc -l)
    done_flag=$(grep -c "REGENIE GWAS complete" "$RES/gwas_regenie_results/run_full.log" 2>/dev/null)
    done_flag=${done_flag:-0}
    echo "[$(date)] chr files=$n / 22, run_full done=$done_flag" >> "$LOG"
    if [ "$done_flag" -ge 1 ] || [ "$n" -ge 22 ]; then
        break
    fi
    sleep 300
done

echo "[$(date)] full REGENIE done, starting Phase 2" >> "$LOG"
python code/03_genetics/3_3_make_plot_regenie.py --mode full >> "$LOG" 2>&1
python code/03_genetics/3_5_make_lead_loci.py --method regenie \
    --gwas-dir "$RES/gwas_regenie_results" \
    --sig-csv "$RES/gwas_regenie_results/significant_snps.csv" \
    --keep "$RES/prs_analysis/keep_all.id" \
    --out "$RES/gwas_regenie_results/regenie_lead_loci.csv" >> "$LOG" 2>&1
python code/03_genetics/3_4_compare_regenie_plink.py --mode full >> "$LOG" 2>&1
python code/03_genetics/4_2_prepare_mr_exposure.py --method regenie \
    --gwas-dir "$RES/gwas_regenie_results" \
    --out "$RES/mr_regenie/apag_exposure_for_mr.csv.gz" >> "$LOG" 2>&1
echo "[$(date)] Phase 2 done" >> "$LOG"
