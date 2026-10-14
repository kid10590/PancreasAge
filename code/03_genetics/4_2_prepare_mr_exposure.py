#!/usr/bin/env python
"""
4_1_prepare_mr_exposure.py — Format PLINK GWAS as TwoSampleMR exposure data

Columns required by TwoSampleMR:
  SNP, effect_allele, other_allele, beta, se, pval, eaf, samplesize, chr, pos

Usage:
  python code/03_genetics/4_2_prepare_mr_exposure.py
  python code/03_genetics/4_2_prepare_mr_exposure.py --gwas-dir /path/to/gwas --out /path/to/exposure.csv
"""
import os, sys, glob, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

GWAS_DIR = os.path.join(OUT_ROOT, "gwas_plink_results")
OUT_DIR = os.path.join(OUT_ROOT, "mr")
PHENO_NAME = "catb_delta_with_t1_bc_cole"


def _load_plink(gwas_dir):
    files = sorted(glob.glob(os.path.join(gwas_dir, f"gwas_chr*.{PHENO_NAME}.glm.linear*")))
    if not files:
        raise FileNotFoundError(f"GLM files not found: {gwas_dir}/gwas_chr*.{PHENO_NAME}.glm.linear*")
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep='\t', low_memory=False)
        df.columns = [c.strip().lstrip('#') for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # In PLINK2 glm output, A1 is the effect allele and REF/ALT are the reference/alternative alleles
    a1 = df['A1'].astype(str).str.upper()
    ref = df['REF'].astype(str).str.upper()
    alt = df['ALT'].astype(str).str.upper()
    other = ref.where(ref != a1, alt)
    return pd.DataFrame({
        'SNP': df['ID'].astype(str),
        'chr': pd.to_numeric(df['CHROM'], errors='coerce').astype('Int64'),
        'pos': pd.to_numeric(df['POS'], errors='coerce').astype('Int64'),
        'effect_allele': a1,
        'other_allele': other,
        'beta': pd.to_numeric(df['BETA'], errors='coerce'),
        'se': pd.to_numeric(df['SE'], errors='coerce'),
        'pval': pd.to_numeric(df['P'], errors='coerce'),
        'eaf': pd.to_numeric(df.get('A1_FREQ', np.nan), errors='coerce'),
        'samplesize': pd.to_numeric(df.get('OBS_CT', 21996), errors='coerce').astype('Int64'),
    }), len(files)


def _load_regenie(gwas_dir):
    files = sorted(glob.glob(os.path.join(gwas_dir, f"gwas_chr*_{PHENO_NAME}.regenie.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(gwas_dir, "gwas_chr*.regenie.gz")))
    if not files:
        raise FileNotFoundError(f"REGENIE files not found: {gwas_dir}/gwas_chr*_{PHENO_NAME}.regenie.gz")
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=r'\s+', comment='#', low_memory=False)
        if 'TEST' in df.columns:
            df = df[df['TEST'].astype(str) == 'ADD']
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # REGENIE: ALLELE1 is the effect allele, ALLELE0 is the other allele, P = 10^(-LOG10P)
    a1 = df['ALLELE1'].astype(str).str.upper()
    other = df['ALLELE0'].astype(str).str.upper()
    log10p = pd.to_numeric(df['LOG10P'], errors='coerce')
    return pd.DataFrame({
        'SNP': df['ID'].astype(str),
        'chr': pd.to_numeric(df['CHROM'], errors='coerce').astype('Int64'),
        'pos': pd.to_numeric(df['GENPOS'], errors='coerce').astype('Int64'),
        'effect_allele': a1,
        'other_allele': other,
        'beta': pd.to_numeric(df['BETA'], errors='coerce'),
        'se': pd.to_numeric(df['SE'], errors='coerce'),
        'pval': 10.0 ** (-log10p),
        'eaf': pd.to_numeric(df.get('A1FREQ', np.nan), errors='coerce'),
        'samplesize': pd.to_numeric(df.get('N', 21996), errors='coerce').astype('Int64'),
    }), len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["plink", "regenie"], default="plink",
                        help="GWAS source: plink (default) / regenie")
    parser.add_argument("--gwas-dir", type=str, default=GWAS_DIR)
    parser.add_argument("--out", type=str, default=os.path.join(OUT_DIR, "apag_exposure_for_mr.csv.gz"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[MR] Method: {args.method}")
    print(f"[MR] GWAS directory: {args.gwas_dir}")
    print(f"[MR] Output: {args.out}")

    if args.method == "regenie":
        mr_df, n_files = _load_regenie(args.gwas_dir)
    else:
        mr_df, n_files = _load_plink(args.gwas_dir)
    print(f"  Merged {n_files} files, {len(mr_df):,} total variants")

    # Clean up
    mr_df = mr_df.dropna(subset=['beta', 'se', 'pval'])
    mr_df = mr_df[mr_df['se'] > 0]
    mr_df = mr_df[mr_df['pval'] > 0]

    # Remove the MHC region (chr6:25-34Mb, to avoid complex LD interfering with MR)
    mhc_mask = (mr_df['chr'] == 6) & (mr_df['pos'].between(25_000_000, 34_000_000))
    n_mhc = mhc_mask.sum()
    mr_df = mr_df[~mhc_mask]
    print(f"  Removed MHC region variants: {n_mhc:,}")

    mr_df.to_csv(args.out, index=False, compression='gzip')
    print(f"  Final {len(mr_df):,} variants → {args.out}")

    # ── Summary statistics ──
    sig = mr_df[mr_df['pval'] < 5e-8]
    print(f"\n  Genome-wide significant (P<5e-8): {len(sig)} variants")
    print(f"  chr distribution: {sig['chr'].value_counts().sort_index().to_dict()}")
    print(f"  Sample size: {mr_df['samplesize'].iloc[0]}")

    # ── Save significant locus list ──
    sig_out = os.path.join(os.path.dirname(args.out), "apag_sig_snps_for_mr.csv")
    sig.to_csv(sig_out, index=False)
    print(f"  Significant loci → {sig_out}")


if __name__ == "__main__":
    main()
