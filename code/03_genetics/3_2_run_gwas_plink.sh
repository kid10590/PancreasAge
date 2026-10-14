#!/bin/bash
# ============================================================
# 3_2_run_gwas_plink.sh — PLINK2 linear GWAS (dual mode)
#
# Usage (two equivalent forms):
#   bash code/03_genetics/3_2_run_gwas_plink.sh --mode full   (or --full)
#   bash code/03_genetics/3_2_run_gwas_plink.sh --mode discovery (or --discovery / --prs)
#
# Output:
#   full → <RES_ROOT>/gwas_plink_results/
#   discovery → <RES_ROOT>/gwas_plink_discovery/
# ============================================================

# ── paths (override via env: PANCREASAGE_DATA / PANCREASAGE_OUT / PLINK2 / REGENIE) ──
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${PANCREASAGE_DATA:-${PKG_ROOT}/data}"
OUT_ROOT="${PANCREASAGE_OUT:-${PKG_ROOT}/results}"
PLINK2="${PLINK2:-plink2}"

if [[ "$PLINK2" == */* ]]; then
    [ -x "$PLINK2" ] || { echo "Error: plink2 not found: $PLINK2"; exit 1; }
elif ! command -v "$PLINK2" >/dev/null 2>&1; then
    echo "Error: plink2 not found in PATH (use \$PLINK2 to specify a full path)"; exit 1
fi

# ---- mode parsing (accepts --mode full / --full / full / --mode discovery / --discovery) ----
MODE="full"
for ARG in "$@"; do
    case "$ARG" in
        --mode) ;;
        full|--full) MODE="full" ;;
        discovery|--discovery|prs|--prs) MODE="discovery" ;;
    esac
done
if [ "$MODE" != "full" ] && [ "$MODE" != "discovery" ]; then
    echo "Error: unknown mode '$MODE' (options: full / discovery)"
    exit 1
fi

PGEN_DIR="${PGEN_DIR:-${DATA_ROOT}/genotypes/pgen}"
RES_ROOT="${RES_ROOT:-${OUT_ROOT}}"
PHENO_FILE="$RES_ROOT/gwas_inputs/phenotypes.txt"
COVAR_FILE="$RES_ROOT/gwas_inputs/covariates.cov"
KEEP_FILE="$RES_ROOT/prs_analysis/keep_discovery.id"

if [ "$MODE" == "full" ]; then
    OUT_DIR="$RES_ROOT/gwas_plink_results"
    echo "Full-sample GWAS (n=53673, for MR)"
else
    OUT_DIR="$RES_ROOT/gwas_plink_discovery"
    echo "Discovery-only GWAS (n=$(wc -l < $KEEP_FILE), for PRS)"
fi

mkdir -p $OUT_DIR

# File suffix to check
SUFFIX="catb_delta_with_t1_bc_cole.glm.linear"

for CHR in {1..22}; do
    OUT_FILE="${OUT_DIR}/gwas_chr${CHR}.${SUFFIX}"
    
    # ===== resume check =====
    if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
        echo "========================================="
        echo "✅ Chromosome ${CHR} already complete, skipping (${OUT_FILE})"
        echo "========================================="
        continue
    fi
    
    echo "========================================="
    echo "▶️ Starting GWAS for Chromosome ${CHR}..."
    echo "========================================="
    
    # Assemble arguments: add --keep for discovery mode
    KEEP_ARGS=""
    if [ "$MODE" == "discovery" ]; then
        KEEP_ARGS="--keep ${KEEP_FILE}"
    fi
    
    $PLINK2 \
      --pfile ${PGEN_DIR}/ukb22828_c${CHR}_b0_v3 \
      ${KEEP_ARGS} \
      --pheno ${PHENO_FILE} \
      --pheno-name catb_delta_with_t1_bc_cole \
      --covar ${COVAR_FILE} \
      --covar-name Sex,Age,Age2,Array,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 \
      --covar-variance-standardize \
      --mac 100 --maf 0.01 --hwe 1e-15 keep-fewhet --geno 0.1 --mind 0.1 \
      --glm hide-covar \
      --threads 16 \
      --memory 32000 \
      --no-input-missing-phenotype \
      --vif 1000 \
      --out ${OUT_DIR}/gwas_chr${CHR}
    
    # Check whether output was generated successfully
    if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
        echo "✅ Chromosome ${CHR} complete"
    else
        echo "❌ Chromosome ${CHR} failed, please check the log"
    fi
done

echo "All 22 chromosomes processed!"
