#!/bin/bash
# ============================================================
# 4_0_run_prs.sh — APAG PRS construction (based on discovery-only PLINK GWAS)
#
# Workflow:
#   1. Merge the discovery GWAS (gwas_plink_discovery) across 22 chromosomes
#   2. Extract P<5e-8 significant SNPs
#   3. PLINK2 LD clumping (r²=0.1, 250kb) → independent lead SNPs
#   4. Build two score files:
#        - Full APAG PRS (all lead SNPs)
#        - APAG PRS excluding the CTRB1/2 region (chr16:74734872-75734872)
#   5. PLINK2 --export A exports lead SNP dosages chromosome by chromosome
#   6. Allele check (flip to 2-dosage if not matching GWAS A1) + beta-weighted sum → PRS
#      (fix: the old version mistakenly used the unweighted .sscore column NAMED_ALLELE_DOSAGE_SUM, making the PRS-delta correlation ≈0/negative)
#
# Usage:
#   bash code/03_genetics/4_0_run_prs.sh                                   # PLINK discovery (default)
#   bash code/03_genetics/4_0_run_prs.sh --method regenie \
#       --gwas-dir .../gwas_regenie_discovery --prs-dir .../prs_analysis_regenie
# ============================================================
# ── paths (override via env: PANCREASAGE_DATA / PANCREASAGE_OUT / PLINK2 / REGENIE) ──
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${PANCREASAGE_DATA:-${PKG_ROOT}/data}"
OUT_ROOT="${PANCREASAGE_OUT:-${PKG_ROOT}/results}"
set -e

PLINK2="${PLINK2:-plink2}"
THREADS=16

# ---- mode parsing (--method plink|regenie, --gwas-dir, --prs-dir) ----
METHOD="plink"
ARG_GWAS_DIR=""
ARG_PRS_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --gwas-dir) ARG_GWAS_DIR="$2"; shift 2 ;;
        --prs-dir) ARG_PRS_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [ "$METHOD" != "plink" ] && [ "$METHOD" != "regenie" ]; then
    echo "Error: unknown --method '$METHOD' (options: plink / regenie)"
    exit 1
fi

# ---- path configuration ----
PGEN_DIR="${PGEN_DIR:-${DATA_ROOT}/genotypes/pgen}"
RES_ROOT="${RES_ROOT:-${OUT_ROOT}}"
MAIN_PRS_DIR="$RES_ROOT/prs_analysis"
if [ "$METHOD" == "regenie" ]; then
    GWAS_DIR="${ARG_GWAS_DIR:-$RES_ROOT/gwas_regenie_discovery}"
    PRS_DIR="${ARG_PRS_DIR:-$RES_ROOT/prs_analysis_regenie}"
else
    GWAS_DIR="${ARG_GWAS_DIR:-$RES_ROOT/gwas_plink_discovery}"
    PRS_DIR="${ARG_PRS_DIR:-$MAIN_PRS_DIR}"
fi
OUT_DIR="$PRS_DIR/prs_build"
PHENO_NAME="catb_delta_with_t1_bc_cole"
# Full-sample keep list + analysis cohort both come from the main prs_analysis (method-independent)
KEEP_ALL="$MAIN_PRS_DIR/keep_all.id"
COHORT="$MAIN_PRS_DIR/analysis_cohort.csv"
# CTRB1/2 region (rs72802342 chr16:75234872 ±500kb) — fixed before analysis
CTRB12_CHR=16
CTRB12_START=74734872
CTRB12_END=75734872

mkdir -p "$OUT_DIR"
echo "Method: $METHOD | GWAS: $GWAS_DIR | PRS output: $PRS_DIR"
echo "Keep all samples: $(wc -l < $KEEP_ALL)"

# ═══════════════════════════════════════
# Step 1: merge discovery GWAS + extract significant SNPs
# ═══════════════════════════════════════
SIG_FILE="$OUT_DIR/discovery_sig_snps.csv"
SCORE_FULL="$OUT_DIR/apag_prs_full.score"
SCORE_NOCTRB="$OUT_DIR/apag_prs_without_ctrb12.score"

if [ ! -f "$SIG_FILE" ]; then
    echo "========================================="
    echo " Step 1/5: merge discovery GWAS + P<5e-8"
    echo "========================================="
    python3 -c "
import os, glob
import pandas as pd

d = '$GWAS_DIR'
method = '$METHOD'
pheno = '$PHENO_NAME'
if method == 'regenie':
    files = sorted(glob.glob(os.path.join(d, 'gwas_chr*_' + pheno + '.regenie.gz')))
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=r'\s+', comment='#', low_memory=False)
        if 'TEST' in df.columns:
            df = df[df['TEST'].astype(str) == 'ADD']
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['CHROM'] = pd.to_numeric(df['CHROM'], errors='coerce')
    df['POS'] = pd.to_numeric(df['GENPOS'], errors='coerce')
    df['BETA'] = pd.to_numeric(df['BETA'], errors='coerce')
    df['P'] = 10.0 ** (-pd.to_numeric(df['LOG10P'], errors='coerce'))
    df['A1'] = df['ALLELE1'].astype(str).str.upper()
else:
    files = sorted(glob.glob(os.path.join(d, 'gwas_chr*.' + pheno + '.glm.linear*')))
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep='\t', low_memory=False)
        df.columns = [c.strip().lstrip('#') for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['CHROM'] = pd.to_numeric(df['CHROM'], errors='coerce')
    df['POS'] = pd.to_numeric(df['POS'], errors='coerce')
    df['P'] = pd.to_numeric(df['P'], errors='coerce')
    df['BETA'] = pd.to_numeric(df['BETA'], errors='coerce')
    df['A1'] = df['A1'].astype(str).str.upper()

print(f'[{method}] Merged {len(files)} files, {len(df):,} variants')

sig = df[df['P'] < 5e-8].copy()
print(f'P<5e-8: {len(sig):,}')
sig[['CHROM', 'POS', 'ID', 'A1', 'BETA', 'P']].to_csv('$SIG_FILE', index=False)
print(f'  → $SIG_FILE')
"
fi

# ═══════════════════════════════════════
# Step 2: LD clumping (r²=0.1, 250kb)
# ═══════════════════════════════════════
CLUMPED="$OUT_DIR/clumped_lead_snps.csv"

if [ ! -f "$CLUMPED" ]; then
    echo ""
    echo "========================================="
    echo " Step 2/5: LD Clumping (r²=0.1, 250kb, P<5e-8)"
    echo "========================================="

    > "$OUT_DIR/clumped_ids.txt"
    for CHR in $(seq 1 22); do
        CLUMP_IN="$OUT_DIR/clump_chr${CHR}.txt"
        python3 -c "
import pandas as pd
df = pd.read_csv('$SIG_FILE')
c = df[df['CHROM'] == $CHR]
if len(c):
    c[['ID', 'P']].to_csv('$CLUMP_IN', sep='\t', index=False)
    print(f'  chr{$CHR}: {len(c)} sig SNPs')
" 2>/dev/null || true
        if [ -s "$CLUMP_IN" ]; then
            $PLINK2 \
                --rm-dup exclude-all \
                --pfile "${PGEN_DIR}/ukb22828_c${CHR}_b0_v3" \
                --keep "$KEEP_ALL" \
                --clump "$CLUMP_IN" \
                --clump-p1 5e-8 \
                --clump-r2 0.1 \
                --clump-kb 250 \
                --clump-snp-field ID \
                --clump-field P \
                --out "${OUT_DIR}/clump_chr${CHR}" \
                --threads $THREADS \
                --silent
            if [ -f "${OUT_DIR}/clump_chr${CHR}.clumps" ]; then
                awk 'NR>1 && $3!="null" {print $3}' "${OUT_DIR}/clump_chr${CHR}.clumps" >> "$OUT_DIR/clumped_ids.txt"
            fi
        fi
    done
    sort -u "$OUT_DIR/clumped_ids.txt" -o "$OUT_DIR/clumped_ids.txt"
    N=$(wc -l < "$OUT_DIR/clumped_ids.txt")
    echo "  Clumped lead SNPs: $N"

    # Keep lead SNP beta/position (used for score + CTRB1/2 exclusion)
    python3 -c "
import pandas as pd
sig = pd.read_csv('$SIG_FILE')
ids = set(pd.read_csv('$OUT_DIR/clumped_ids.txt', header=None)[0])
lead = sig[sig['ID'].isin(ids)].copy()
lead.to_csv('$CLUMPED', index=False)
print(f'  → $CLUMPED ({len(lead)} lead SNPs)')
"
fi

# ═══════════════════════════════════════
# Step 3: build score files (full version + excluding CTRB1/2)
# ═══════════════════════════════════════
echo ""
echo "========================================="
echo " Step 3/5: build score files"
echo "========================================="
python3 -c "
import pandas as pd
lead = pd.read_csv('$CLUMPED')

# Full version: lead table with positions + ID A1 BETA score
lead_full = lead.copy()
lead_full.to_csv('$OUT_DIR/lead_full.csv', index=False)
lead_full[['ID', 'A1', 'BETA']].to_csv('$SCORE_FULL', sep='\t', index=False, header=False)
print(f'  Full version: {len(lead_full)} SNPs → $SCORE_FULL')

# Exclude the CTRB1/2 region
mask = ~((lead['CHROM'] == $CTRB12_CHR) & (lead['POS'] >= $CTRB12_START) & (lead['POS'] <= $CTRB12_END))
noctrb = lead[mask].copy()
noctrb.to_csv('$OUT_DIR/lead_without_ctrb12.csv', index=False)
noctrb[['ID', 'A1', 'BETA']].to_csv('$SCORE_NOCTRB', sep='\t', index=False, header=False)
print(f'  Excluding CTRB1/2 (chr{$CTRB12_CHR}:{$CTRB12_START}-{$CTRB12_END}): {len(lead_full)-len(noctrb)} SNPs removed, {len(noctrb)} SNPs remain → $SCORE_NOCTRB')
"

# ═══════════════════════════════════════
# Step 4: export lead SNP dosages chromosome by chromosome (--export A, two versions)
# ═══════════════════════════════════════
echo ""
echo "========================================="
echo " Step 4/5: PLINK2 --export A export dosages"
echo "========================================="
for TAG in full without_ctrb12; do
    LEAD_TAG="$OUT_DIR/lead_${TAG}.csv"
    for CHR in $(seq 1 22); do
        RAW_OUT="$OUT_DIR/export_${TAG}_chr${CHR}"
        if [ -f "${RAW_OUT}.raw" ]; then
            continue
        fi
        SNPLIST="$OUT_DIR/export_${TAG}_chr${CHR}.snplist"
        python3 -c "
import pandas as pd
lead = pd.read_csv('$LEAD_TAG')
c = lead[lead['CHROM'] == $CHR]
if len(c):
    c['ID'].to_csv('$SNPLIST', index=False, header=False)
" 2>/dev/null || true
        if [ ! -s "$SNPLIST" ]; then
            continue
        fi
        $PLINK2 \
            --rm-dup exclude-all \
            --pfile "${PGEN_DIR}/ukb22828_c${CHR}_b0_v3" \
            --keep "$KEEP_ALL" \
            --extract "$SNPLIST" \
            --export A \
            --out "$RAW_OUT" \
            --threads $THREADS \
            --silent
    done
done

# ═══════════════════════════════════════
# Step 5: allele check + beta-weighted sum → final PRS
# (fix: the old version mistakenly used the unweighted NAMED_ALLELE_DOSAGE_SUM column of .sscore,
#  and the allele counted by --export A may not be the GWAS effect allele, so a check/flip is needed)
# ═══════════════════════════════════════
echo ""
echo "========================================="
echo " Step 5/5: allele check + beta weighting → final PRS"
echo "========================================="
python3 -c "
import os, sys
import pandas as pd
import numpy as np

OUT_DIR = '$OUT_DIR'
PRS_DIR = '$PRS_DIR'
COHORT = '$COHORT'

def load_raw_allele_checked(raw_file, lead):
    df = pd.read_csv(raw_file, sep=r'\s+')
    beta_map = dict(zip(lead['ID'].astype(str), lead['BETA'].astype(float)))
    a1_map = dict(zip(lead['ID'].astype(str), lead['A1'].astype(str).str.upper()))
    snp_cols = [c for c in df.columns
                if c not in ('FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE')
                and not c.endswith('_HET')]
    score = np.zeros(len(df))
    for col in snp_cols:
        sid, _, counted = col.partition('_')
        counted = counted.upper()
        a1 = a1_map.get(sid)
        dos = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        if a1 is not None and counted and counted != a1:
            dos = 2.0 - dos
            print(f'    [flip] {sid}: counted allele {counted} -> effect allele {a1}')
        score += dos * beta_map.get(sid, 0.0)
    out = df[['FID', 'IID']].copy()
    out['score'] = score
    return out

def build_tag(tag, label):
    lead = pd.read_csv(os.path.join(OUT_DIR, f'lead_{tag}.csv'))
    total = None
    n_chr_used = 0
    for chr_num in range(1, 23):
        raw = os.path.join(OUT_DIR, f'export_{tag}_chr{chr_num}.raw')
        if not os.path.exists(raw):
            continue
        d = load_raw_allele_checked(raw, lead)
        if total is None:
            total = d
        else:
            assert (total['IID'].values == d['IID'].values).all(), f'chr{chr_num} sample order mismatch'
            total['score'] += d['score'].values
        n_chr_used += 1
    if total is None:
        print(f'  [{label}] no raw files')
        return
    # sanity check: correlation with analysis_cohort delta must be positive
    try:
        coh = pd.read_csv(COHORT, low_memory=False, usecols=['eid', 'delta'])
        coh['eid'] = coh['eid'].astype(str)
        total['IID'] = total['IID'].astype(str)
        m = total.merge(coh, left_on='IID', right_on='eid')
        r = np.corrcoef(m['score'], m['delta'])[0, 1]
        print(f'  [{label}] sanity check: corr(PRS, delta) = {r:.4f}')
        if r <= 0:
            print('  [error] PRS-delta correlation is not positive; allele coding still has a problem!')
            sys.exit(1)
    except FileNotFoundError:
        print('  [warning] analysis_cohort.csv not found, skipping sanity check')
    prs = total[['FID', 'IID', 'score']].rename(columns={'score': label})
    prs.to_csv(os.path.join(PRS_DIR, f'{label}_raw.csv'), index=False)
    print(f'  [{label}] {len(prs)} samples, {n_chr_used} chromosomes -> {PRS_DIR}/{label}_raw.csv')

build_tag('full', 'APAG_PRS')
build_tag('without_ctrb12', 'APAG_PRS_without_CTRB12')
"

echo ""
echo "========================================="
echo " APAG PRS construction complete!"
echo " Output: $PRS_DIR/APAG_PRS_raw.csv, APAG_PRS_without_CTRB12_raw.csv"
echo "========================================="
