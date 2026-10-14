#!/usr/bin/env python
"""
Mediation analysis: CTRB1/2 → APAG (pancreatic age acceleration) → diabetes

Tests whether genetic variants in the CTRB1/2 region increase diabetes risk by making the pancreas age.

Methods (two in parallel):
  Method 1: Baron & Kenny stepwise approach + Sobel test (traditional)
  Method 2: Bootstrap percentile CI (Preacher & Hayes 2008, modern standard)
    - Bootstrap automatically classifies full/partial/no mediation
    - Does not rely on distributional assumptions, higher statistical power (MacKinnon et al. 2002, 2004)

Inputs:
  - Diabetes matched data: ${ANALYSIS_ROOT}/Diabetes/matched_data.csv
  - GWAS significant loci: ${GWAS_DIR}/significant_snps.csv
  - PLINK pgen: ${PGEN_DIR} (per chromosome)
  - Covariates + PCs: ${INPUT_DIR}/phenotype_covariates_merged.csv

Output directory: ${OUT_DIR}/Mediation/
  - ctrb_snp_genotypes.csv        # extracted CTRB1/2 SNP genotypes
  - merged_mediation_data.csv     # merged analysis data
  - mediation_results.txt          # full mediation analysis results
  - mediation_bootstrap.csv       # Bootstrap indirect effect distribution
  - mediation_summary.csv         # summary of path effects
  - forest_mediation.png          # path-effect forest plot

Usage:
  python code/03_genetics/2_6_run_mediation_snp.py
  python code/03_genetics/2_6_run_mediation_snp.py --lead-snp rs72802342
  python code/03_genetics/2_6_run_mediation_snp.py --sensitivity  # run all sensitivity analyses
"""

import os
import sys
import re
import argparse
import subprocess
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════
# Path configuration
# ═══════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT, PGEN_DIR, PLINK2  # noqa: E402
GWAS_DIR = os.path.join(OUT_ROOT, "gwas_plink_results")
INPUT_DIR = os.path.join(OUT_ROOT, "gwas_inputs")
ANALYSIS_ROOT = os.path.join(OUT_ROOT, "Disease_Phenotype_Analysis")
MATCHED_CSV = os.path.join(ANALYSIS_ROOT, "Diabetes", "matched_data.csv")
PHENO_COVAR_CSV = os.path.join(INPUT_DIR, "phenotype_covariates_merged.csv")
SIGNIFICANT_SNPS = os.path.join(GWAS_DIR, "significant_snps.csv")
OUT_DIR = os.path.join(ANALYSIS_ROOT, "Mediation")

# Significant SNPs in the CTRB1/2 region (chr16) and ABO region (chr9)
# Automatically extracted from GWAS results; ranges below are for reference
CTRB_REGION = {"chr": 16, "start": 75_200_000, "end": 75_400_000}
ABO_REGION = {"chr": 9, "start": 136_100_000, "end": 136_160_000}

# Known T2D loci (for sensitivity analysis — check whether CTRB1/2 is a known T2D locus)
KNOWN_T2D_LOCI = {
    "CTRB1/2": {"chr": 16, "start": 75_200_000, "end": 75_350_000},
}


def setup_output_dir():
    """Create the output directory."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")


# ═══════════════════════════════════════
# Step 0: Extract CTRB1/2 region SNP genotypes from PLINK
# ═══════════════════════════════════════

def get_ctrb_lead_snps(sig_csv: str = SIGNIFICANT_SNPS,
                       region: dict = None) -> pd.DataFrame:
    """Extract lead SNPs in the CTRB1/2 region from the GWAS significant loci.

    Returns DataFrame with columns: CHROM, POS, ID, ALLELE1, BETA, P
    """
    if region is None:
        region = CTRB_REGION

    sig = pd.read_csv(sig_csv)
    sig.columns = [c.strip().upper().replace('#', '') for c in sig.columns]

    # Standardize column names
    col_map = {}
    for c in sig.columns:
        if c in ('CHROM', 'CHR'):
            col_map[c] = 'CHROM'
        elif c in ('POS', 'GENPOS', 'BP'):
            col_map[c] = 'POS'
    sig = sig.rename(columns=col_map)

    ctrb = sig[(sig['CHROM'] == region['chr']) &
               (sig['POS'] >= region['start']) &
               (sig['POS'] <= region['end'])].copy()

    # Deduplicate, keeping the most significant
    ctrb = ctrb.sort_values('P').drop_duplicates(subset=['CHROM', 'POS'])
    ctrb = ctrb.sort_values('P')  # sort by significance

    print(f"\nCTRB1/2 region significant SNPs: {len(ctrb)}")
    for _, r in ctrb.head(10).iterrows():
        print(f"  {r.get('ID', '?')}  chr{r['CHROM']}:{r['POS']}  "
              f"P={r['P']:.2e}  BETA={r.get('BETA', 'NA'):.4f}")
    return ctrb


def check_plink2():
    """Verify that plink2 is available."""
    if not os.path.exists(PLINK2):
        raise FileNotFoundError(f"plink2 not found: {PLINK2}")
    try:
        subprocess.run([PLINK2, "--version"], capture_output=True, timeout=10)
    except Exception:
        raise RuntimeError(f"plink2 cannot be run: {PLINK2}")


def clump_independent_snps(snp_df: pd.DataFrame, chr_num: int,
                           p_threshold: float = 5e-8,
                           r2_threshold: float = 0.001,
                           window_kb: int = 10000,
                           keep_file: str = None) -> pd.DataFrame:
    """Standard LD clumping: identify independent lead SNPs from significant SNPs.

    Uses PLINK2 --clump with UKB's own pgen as the LD reference panel.
    Returns a DataFrame of independent lead SNPs sorted by P.
    """
    print("\n  LD clumping: identifying independent significant SNPs...")

    clump_dir = os.path.join(OUT_DIR, "clump")
    os.makedirs(clump_dir, exist_ok=True)

    # Prepare clump input: two columns SNP and P
    clump_in = os.path.join(clump_dir, f"chr{chr_num}_sig_snps_for_clump.txt")
    snp_df_clump = snp_df[['ID', 'P']].dropna().copy()
    snp_df_clump.to_csv(clump_in, sep='\t', index=False)
    print(f"    Input SNPs: {len(snp_df_clump)}")

    pgen = os.path.join(PGEN_DIR, f"ukb22828_c{chr_num}_b0_v3")
    clump_out = os.path.join(clump_dir, f"chr{chr_num}_clump")

    cmd = [
        PLINK2,
        "--pfile", pgen,
        "--clump", clump_in,
        "--clump-p1", str(p_threshold),
        "--clump-p2", str(p_threshold),
        "--clump-r2", str(r2_threshold),
        "--clump-kb", str(window_kb),
        "--out", clump_out,
        "--threads", "4",
        "--memory", "8000",
    ]
    if keep_file and os.path.exists(keep_file):
        cmd += ["--keep", keep_file]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARNING] clumping failed, falling back to all loci: {result.stderr[-300:]}")
        return snp_df

    clump_file = f"{clump_out}.clumps"
    if not os.path.exists(clump_file):
        print(f"    [WARNING] no .clumps file generated, falling back to all loci")
        return snp_df

    clumps = pd.read_csv(clump_file, sep=r'\s+')
    # .clumps columns: #CHROM POS ID P TOTAL NSIG S05 S01 S001 S0001 SP2
    independent_ids = clumps['ID'].tolist()
    print(f"    Independent signals: {len(independent_ids)} (r²<{r2_threshold}, {window_kb}kb)")

    # Keep independent SNPs, sorted by P
    indep_df = snp_df[snp_df['ID'].isin(independent_ids)].sort_values('P').copy()
    for _, r in indep_df.iterrows():
        print(f"      {r['ID']}  chr{r['CHROM']}:{r['POS']}  P={r['P']:.2e}")

    return indep_df


def load_genotype_dosage(raw_file: str, allele_map: dict = None) -> pd.DataFrame:
    """Load the .raw file output by PLINK --export A.

    Fix: the counted allele (last column-name segment) from --export A may not be the
    GWAS effect allele (A1). When allele_map = {SNP ID: effect allele} is passed and
    the counted allele != effect allele, the dosage is flipped to 2 - dosage so the
    direction matches the GWAS beta.

    Returns DataFrame: IID, dosage (0/1/2 or continuous), columns named by SNP ID.
    """
    df = pd.read_csv(raw_file, sep=r'\s+')

    # PLINK .raw format: FID IID PAT MAT SEX PHENOTYPE SNP1_allele1 SNP1_HET ...
    snp_cols = [c for c in df.columns if c not in
                ('FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE')]

    # Find dosage columns (those without the _HET suffix)
    dosage_cols = [c for c in snp_cols if not c.endswith('_HET')]
    if len(dosage_cols) == 0 and snp_cols:
        dosage_cols = [snp_cols[0]]

    result = df[['IID']].copy()
    result['IID'] = result['IID'].astype(str)

    for col in dosage_cols:
        # PLINK --export A column name format: SNPID_allele (e.g. rs72802342_C)
        # Take the part before the last _ as the SNP ID; the last token is the counted allele
        if '_' in col:
            counted_allele = col.rsplit('_', 1)[1].upper()
            snp_name = col.rsplit('_', 1)[0]
        else:
            counted_allele, snp_name = None, col
        dos = pd.to_numeric(df[col], errors='coerce')
        if allele_map and counted_allele and snp_name in allele_map:
            eff = str(allele_map[snp_name]).upper()
            if eff and counted_allele != eff:
                dos = 2.0 - dos
                print(f"  [flip] {snp_name}: counted allele {counted_allele} -> effect allele {eff}")
        result[snp_name] = dos

    return result


def extract_snp_list_genotypes(snp_ids: list, chr_num: int,
                               keep_file: str = None,
                               batch_name: str = "ctrb_indep",
                               allele_map: dict = None) -> pd.DataFrame:
    """Extract genotypes of multiple SNPs in a single plink2 call (efficient batch extraction).

    Returns DataFrame: IID + one column per SNP (direction corrected to effect-allele dosage via allele_map)
    """
    out_prefix = os.path.join(OUT_DIR, f"snp_batch_{batch_name}")
    raw_file = f"{out_prefix}.raw"

    if os.path.exists(raw_file):
        print(f"  [skip] {raw_file} already exists")
        return load_genotype_dosage(raw_file, allele_map)

    # Write SNP list
    snp_list_file = os.path.join(OUT_DIR, f"_extract_{batch_name}.txt")
    with open(snp_list_file, 'w') as f:
        for sid in snp_ids:
            f.write(f"{sid}\n")

    pgen = os.path.join(PGEN_DIR, f"ukb22828_c{chr_num}_b0_v3")
    keep_arg = []
    if keep_file and os.path.exists(keep_file):
        keep_arg = ["--keep", keep_file]

    cmd = [
        PLINK2,
        "--pfile", pgen,
        "--extract", snp_list_file,
        "--export", "A",
        "--out", out_prefix,
        "--threads", "4",
        "--memory", "8000",
    ] + keep_arg

    print(f"  Batch extracting {len(snp_ids)} chr{chr_num} SNPs...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"plink2 batch extraction failed")

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"genotype file not generated: {raw_file}")

    print(f"  -> {raw_file}")
    return load_genotype_dosage(raw_file, allele_map)


def extract_ctrb_genotypes(lead_snps: pd.DataFrame,
                           keep_file: str = None) -> pd.DataFrame:
    """Extract genotypes of the independent significant SNPs in the CTRB1/2 region (standard pipeline).

    1. LD clumping to identify independent signals
    2. Batch-extract genotypes of all independent SNPs (a single plink2 call)

    Returns DataFrame: IID + one column per independent SNP
    """
    print("\n" + "=" * 60)
    print(" Step 0: Extract CTRB1/2 SNP genotypes (standard LD clumping pipeline)")
    print("=" * 60)

    check_plink2()

    chr_num = CTRB_REGION['chr']

    # Step 0a: LD clumping — identify independent lead SNPs
    independent_snps = clump_independent_snps(
        lead_snps, chr_num, keep_file=keep_file
    )

    if len(independent_snps) == 0:
        raise RuntimeError("No independent significant SNP in the CTRB1/2 region")

    # Step 0b: Batch extract genotypes of all independent SNPs
    snp_ids = independent_snps['ID'].tolist()
    # Build allele map (GWAS effect allele A1) for dosage direction correction
    a1_col = 'A1' if 'A1' in independent_snps.columns else 'ALLELE1'
    allele_map = dict(zip(independent_snps['ID'].astype(str),
                          independent_snps[a1_col].astype(str))) if a1_col in independent_snps.columns else None
    geno_df = extract_snp_list_genotypes(snp_ids, chr_num, keep_file, allele_map=allele_map)

    # Save
    geno_out = os.path.join(OUT_DIR, "ctrb_snp_genotypes.csv")
    geno_df.to_csv(geno_out, index=False)
    print(f"\nExtracted {len(snp_ids)} independent SNPs, n samples: {len(geno_df)}")
    print(f"  -> {geno_out}")

    # Also save the independent SNP list for later use
    indep_out = os.path.join(OUT_DIR, "ctrb_independent_snps.csv")
    independent_snps.to_csv(indep_out, index=False)
    print(f"  -> {indep_out}")

    return geno_df


# ═══════════════════════════════════════
# Step 1: Merge data
# ═══════════════════════════════════════

def merge_mediation_data(geno_df: pd.DataFrame,
                         lead_snp: str) -> pd.DataFrame:
    """Merge genotype, phenotype and covariate data.

    Returns an analysis DataFrame with columns:
      IID, delta, is_case, snp_dosage, age_c, age_c_sq, sex_code, PC1-10
    """
    print("\n" + "=" * 60)
    print(" Step 1: Merge mediation analysis data")
    print("=" * 60)

    # 1. Load diabetes matched data
    matched = pd.read_csv(MATCHED_CSV, low_memory=False)
    matched['eid'] = matched['eid'].astype(str)
    print(f"  Diabetes matched samples: {len(matched)}")

    # 2. Load covariates (PCs)
    covar = pd.read_csv(PHENO_COVAR_CSV, low_memory=False)
    covar['eid'] = covar['eid'].astype(str)
    pc_cols = [f'PC{i}' for i in range(1, 11)]
    pc_cols = [c for c in pc_cols if c in covar.columns]
    covar_cols = ['eid'] + pc_cols
    if 'Array' in covar.columns:
        covar_cols.append('Array')
    if 'Array_bin' in covar.columns:
        covar_cols.append('Array_bin')
    covar = covar[covar_cols]

    # 3. Prepare key columns from matched
    matched_cols = ['eid', 'delta', 'is_case', 'age_use', 'sex_code']
    if 'age_use' not in matched.columns and 'age_gt' in matched.columns:
        matched['age_use'] = matched['age_gt']
    df = matched[matched_cols].copy()

    # 4. Merge genotypes
    geno_df['IID'] = geno_df['IID'].astype(str)
    # Find the column corresponding to the lead SNP
    snp_cols_in_geno = [c for c in geno_df.columns if c != 'IID']
    if lead_snp not in geno_df.columns and len(snp_cols_in_geno) > 0:
        # Use the first available SNP
        actual_snp = snp_cols_in_geno[0]
        print(f"  Note: lead SNP {lead_snp} was not extracted; using {actual_snp}")
        lead_snp = actual_snp

    geno_sub = geno_df[['IID', lead_snp]].copy()
    geno_sub = geno_sub.rename(columns={lead_snp: 'snp_dosage'})

    df = df.merge(geno_sub, left_on='eid', right_on='IID', how='inner')
    print(f"  After merging genotypes: {len(df)} samples")

    # 5. Merge covariates
    df = df.merge(covar, left_on='eid', right_on='eid', how='left')
    print(f"  After merging covariates: {len(df)} samples")

    # 6. Center age
    age_mean = df['age_use'].mean()
    df['age_c'] = df['age_use'] - age_mean
    df['age_c_sq'] = df['age_c'] ** 2
    print(f"  Age centered: mean={age_mean:.2f}")

    # 7. Drop missing values
    required = ['delta', 'is_case', 'snp_dosage', 'age_c', 'age_c_sq', 'sex_code']
    df_clean = df.dropna(subset=required).copy()

    # Fill missing PC values with 0 (a few samples may lack PCs)
    for pc in pc_cols:
        if pc in df_clean.columns:
            df_clean[pc] = df_clean[pc].fillna(0)

    print(f"  Final analysis samples: {len(df_clean)} "
          f"(case={df_clean['is_case'].sum()}, "
          f"ctrl={(df_clean['is_case']==0).sum()})")

    # Save
    merged_out = os.path.join(OUT_DIR, "merged_mediation_data.csv")
    df_clean.to_csv(merged_out, index=False)
    print(f"  -> {merged_out}")
    return df_clean


# ═══════════════════════════════════════
# Step 2: Baron & Kenny three-step mediation analysis
# ═══════════════════════════════════════

def baron_kenny_mediation(df: pd.DataFrame,
                          pc_cols: list = None) -> dict:
    """Baron & Kenny three-step mediation analysis.

    Models:
      Path a (X→M): APAG(delta) ~ SNP + age_c + age_c_sq + sex + PCs
      Path c (X→Y total): Diabetes ~ SNP + age_c + age_c_sq + sex + PCs
      Path c'(X→Y direct, M→Y): Diabetes ~ SNP + APAG + age_c + age_c_sq + sex + PCs

    Returns a dict with each path's coefficient, standard error and P value.
    """
    print("\n" + "=" * 60)
    print(" Step 2: Baron & Kenny three-step mediation analysis")
    print("=" * 60)

    if pc_cols is None:
        pc_cols = [c for c in df.columns if re.match(r'^PC\d+$', c)]

    covar_list = ['age_c', 'age_c_sq', 'sex_code'] + pc_cols
    # Ensure covariates exist
    covar_list = [c for c in covar_list if c in df.columns]

    results = {}

    # ---- Path a: SNP → APAG (linear regression) ----
    print("\n  Path a: SNP → APAG (linear regression)")
    formula_a = f"delta ~ snp_dosage + {' + '.join(covar_list)}"
    model_a = sm.OLS.from_formula(formula_a, data=df).fit()
    a_coef = model_a.params['snp_dosage']
    a_se = model_a.bse['snp_dosage']
    a_p = model_a.pvalues['snp_dosage']
    print(f"    Coefficient a = {a_coef:.4f}  SE = {a_se:.4f}  P = {a_p:.2e}")
    results['path_a'] = {'coef': a_coef, 'se': a_se, 'p': a_p, 'model': model_a}

    # ---- Path c: SNP → diabetes (logistic regression, total effect) ----
    print("\n  Path c: SNP → diabetes (logistic regression, total effect)")
    formula_c = f"is_case ~ snp_dosage + {' + '.join(covar_list)}"
    model_c = sm.Logit.from_formula(formula_c, data=df).fit(disp=0)
    c_coef = model_c.params['snp_dosage']
    c_se = model_c.bse['snp_dosage']
    c_p = model_c.pvalues['snp_dosage']
    c_or = np.exp(c_coef)
    print(f"    Coefficient c = {c_coef:.4f}  SE = {c_se:.4f}  P = {c_p:.2e}  OR = {c_or:.4f}")
    results['path_c'] = {'coef': c_coef, 'se': c_se, 'p': c_p, 'or': c_or, 'model': model_c}

    # ---- Path c' + b: SNP + APAG → diabetes (logistic regression, direct effect) ----
    print("\n  Path c'+b: SNP + APAG → diabetes (logistic regression)")
    formula_cprime = f"is_case ~ snp_dosage + delta + {' + '.join(covar_list)}"
    model_cprime = sm.Logit.from_formula(formula_cprime, data=df).fit(disp=0)
    cprime_coef = model_cprime.params['snp_dosage']
    cprime_se = model_cprime.bse['snp_dosage']
    cprime_p = model_cprime.pvalues['snp_dosage']
    cprime_or = np.exp(cprime_coef)

    b_coef = model_cprime.params['delta']
    b_se = model_cprime.bse['delta']
    b_p = model_cprime.pvalues['delta']
    b_or = np.exp(b_coef)

    print(f"    Direct effect c' = {cprime_coef:.4f}  SE = {cprime_se:.4f}  P = {cprime_p:.2e}  OR = {cprime_or:.4f}")
    print(f"    Path b = {b_coef:.4f}  SE = {b_se:.4f}  P = {b_p:.2e}  OR = {b_or:.4f}")
    results['path_cprime'] = {'coef': cprime_coef, 'se': cprime_se, 'p': cprime_p,
                              'or': cprime_or, 'model': model_cprime}
    results['path_b'] = {'coef': b_coef, 'se': b_se, 'p': b_p, 'or': b_or}

    # ---- Indirect effect (product of coefficients, point estimate) ----
    # Note: a is continuous→continuous, b is continuous→binary (logit); a*b is on the logit scale
    # This is standard practice, though cross-scale product interpretation requires care
    indirect = a_coef * b_coef
    # Sobel SE: sqrt(a²*se_b² + b²*se_a²)
    sobel_se = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    print(f"\n  Indirect effect a×b = {indirect:.4f}  Sobel SE = {sobel_se:.4f}  "
          f"Z = {sobel_z:.2f}  P = {sobel_p:.2e}")

    # Proportion mediated
    total = c_coef
    if abs(total) > 1e-10:
        prop_mediated = indirect / total
        print(f"  Proportion mediated = {prop_mediated*100:.1f}%")
    else:
        prop_mediated = np.nan
        print(f"  Proportion mediated: cannot compute (total effect ~ 0)")

    results['indirect'] = {'coef': indirect, 'sobel_se': sobel_se,
                           'sobel_z': sobel_z, 'sobel_p': sobel_p,
                           'prop_mediated': prop_mediated}

    results['df'] = df
    results['covar_list'] = covar_list
    return results


# ═══════════════════════════════════════
# Step 3: Bootstrap mediation test (modern standard, Preacher & Hayes 2008)
# In parallel with Baron & Kenny; both presented equally
# ═══════════════════════════════════════

def bootstrap_mediation(df: pd.DataFrame,
                        covar_list: list,
                        n_bootstrap: int = 1000,
                        seed: int = 42) -> pd.DataFrame:
    """Bootstrap mediation test — Preacher & Hayes (2008) method.

    Each bootstrap resample estimates all paths:
      - a:   SNP → APAG (linear)
      - b:   APAG → Diabetes, controlling for SNP (logit)
      - c:   SNP → Diabetes, total effect (logit)
      - c':  SNP → Diabetes, direct effect, controlling for APAG (logit)
      - ab:  indirect effect = a × b

    Returns DataFrame: bootstrap distribution + mediation classification.
    """
    print("\n" + "=" * 60)
    print(f" Step 3: Bootstrap mediation test ({n_bootstrap} iterations)")
    print("  (Preacher & Hayes 2008, parallel with Baron-Kenny)")
    print("=" * 60)

    np.random.seed(seed)
    n = len(df)
    covar_str = ' + '.join(covar_list)

    records = []

    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_df = df.iloc[idx]

        try:
            # Path a: SNP → APAG
            m_a = sm.OLS.from_formula(
                f"delta ~ snp_dosage + {covar_str}", data=boot_df
            ).fit()
            a_hat = m_a.params['snp_dosage']

            # Path c: SNP → Diabetes (total effect)
            m_c = sm.Logit.from_formula(
                f"is_case ~ snp_dosage + {covar_str}", data=boot_df
            ).fit(disp=0)
            c_hat = m_c.params['snp_dosage']

            # Path c' + b: SNP + APAG → Diabetes
            m_cp = sm.Logit.from_formula(
                f"is_case ~ snp_dosage + delta + {covar_str}", data=boot_df
            ).fit(disp=0)
            cprime_hat = m_cp.params['snp_dosage']
            b_hat = m_cp.params['delta']

            ab_hat = a_hat * b_hat

            records.append({
                'a_coef': a_hat,
                'b_coef': b_hat,
                'c_coef': c_hat,
                'cprime_coef': cprime_hat,
                'indirect_effect': ab_hat,
            })

        except Exception:
            continue

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_bootstrap}...")

    boot_results = pd.DataFrame(records)

    # Percentile Bootstrap CI for each path
    paths = {
        'indirect_effect': 'ab (indirect effect)',
        'cprime_coef': "c' (direct effect)",
        'c_coef': 'c (total effect)',
        'a_coef': 'a (SNP→APAG)',
        'b_coef': 'b (APAG→DM)',
    }

    ci_info = {}
    print("\n  Bootstrap percentile confidence intervals (95%):")
    print(f"  {'Path':<28s} {'Estimate':>10s} {'95% CI':>30s}  {'Conclusion'}")
    print(f"  {'-'*28} {'-'*10} {'-'*30}  {'-'*16}")

    for col, label in paths.items():
        if col not in boot_results.columns:
            continue
        vals = boot_results[col].dropna()
        if len(vals) == 0:
            continue
        ci_95 = np.percentile(vals, [2.5, 97.5])
        point = np.median(vals)
        sig = "significant ✗" if (ci_95[0] <= 0 <= ci_95[1]) else "significant ✓"
        ci_info[col] = {'ci_95': ci_95, 'median': point, 'significant': sig}

        print(f"  {label:<28s} {point:>10.4f}  [{ci_95[0]:>10.4f}, {ci_95[1]:>10.4f}]  {sig}")

    # ---- Mediation type classification (based on Bootstrap 95% CI) ----
    ab_ci = ci_info.get('indirect_effect', {}).get('ci_95', [0, 0])
    cp_ci = ci_info.get('cprime_coef', {}).get('ci_95', [0, 0])

    ab_sig = not (ab_ci[0] <= 0 <= ab_ci[1])
    cp_sig = not (cp_ci[0] <= 0 <= cp_ci[1])

    if not ab_sig:
        mediation_type = "No mediation effect"
        mediation_desc = "Bootstrap ab 95% CI includes 0; cannot reject the null hypothesis of no mediation effect."
    elif cp_sig:
        mediation_type = "Partial mediation"
        mediation_desc = (
            f"ab 95% CI excludes 0 (mediation supported), and c' 95% CI also excludes 0 (direct effect still significant)."
            f"\n    CTRB1/2 affects diabetes partly through APAG and partly directly, bypassing APAG."
        )
    else:
        mediation_type = "Full mediation"
        mediation_desc = (
            f"ab 95% CI excludes 0 (mediation supported), while c' 95% CI includes 0 (direct effect not significant)."
            f"\n    The effect of CTRB1/2 on diabetes operates almost entirely through APAG."
        )

    print(f"\n  ═══ Bootstrap mediation classification ═══")
    print(f"  Conclusion: {mediation_type}")
    print(f"  {mediation_desc}")

    # Save
    boot_out = os.path.join(OUT_DIR, "mediation_bootstrap.csv")
    boot_results.to_csv(boot_out, index=False)
    print(f"\n  -> {boot_out}")

    # Attach metadata
    boot_results.attrs['ci_info'] = ci_info
    boot_results.attrs['mediation_type'] = mediation_type
    boot_results.attrs['mediation_desc'] = mediation_desc
    boot_results.attrs['ab_sig'] = ab_sig
    boot_results.attrs['cp_sig'] = cp_sig

    return boot_results


# ═══════════════════════════════════════
# Step 4: Sensitivity analysis
# ═══════════════════════════════════════

def sensitivity_multi_snp(df: pd.DataFrame,
                          geno_df: pd.DataFrame,
                          lead_snps: pd.DataFrame,
                          covar_list: list) -> dict:
    """Sensitivity 1: use a weighted genetic score of all independent SNPs instead of a single lead SNP.

    Score = Σ(allele_dosage × BETA_sign), summed with GWAS effect-direction weights and then standardized.
    Covers all LD-clumped independent signals, not limited to a single lead SNP.
    """
    print("\n" + "-" * 40)
    print(" Sensitivity 1: multi-SNP genetic score (CTRB1/2 region)")
    print("-" * 40)

    # Find all available SNP columns
    snp_cols = [c for c in geno_df.columns if c != 'IID']
    available = df['eid'].isin(geno_df['IID'].astype(str)) 

    # Build genetic score
    score = pd.Series(0.0, index=df.index) 
    n_used = 0

    for _, snp_row in lead_snps.iterrows(): 
        snp_id = snp_row['ID']
        if snp_id not in geno_df.columns:
            continue
        beta_sign = 1 if snp_row.get('BETA', 0) >= 0 else -1

        snp_series = df['eid'].map(
            dict(zip(geno_df['IID'].astype(str), geno_df[snp_id]))
        )
        score += snp_series.fillna(0) * beta_sign
        n_used += 1

    if n_used < 2:
        print("  Fewer than 2 available SNPs; skipping multi-SNP analysis")
        return {}

    df_temp = df.copy()
    df_temp['grs'] = score

    # Standardize
    df_temp['grs'] = (df_temp['grs'] - df_temp['grs'].mean()) / df_temp['grs'].std()

    covar_str = ' + '.join(covar_list)

    # Path a
    ma = sm.OLS.from_formula(f"delta ~ grs + {covar_str}", data=df_temp).fit()
    # Path c
    mc = sm.Logit.from_formula(f"is_case ~ grs + {covar_str}", data=df_temp).fit(disp=0)
    # Path c'
    mcp = sm.Logit.from_formula(f"is_case ~ grs + delta + {covar_str}", data=df_temp).fit(disp=0)

    a_grs = ma.params['grs']
    b_grs = mcp.params['delta']
    indirect_grs = a_grs * b_grs

    print(f"  GRS ({n_used} SNPs): a={a_grs:.4f}, b={b_grs:.4f}, a×b={indirect_grs:.4f}")
    print(f"  GRS→DM total effect OR: {np.exp(mc.params['grs']):.4f}, P={mc.pvalues['grs']:.2e}")

    return {
        'n_snps': n_used,
        'a_grs': a_grs, 'b_grs': b_grs,
        'indirect_grs': indirect_grs,
        'c_or': np.exp(mc.params['grs']),
        'c_p': mc.pvalues['grs'],
    }


def sensitivity_abo_control(df: pd.DataFrame,
                            lead_snps: pd.DataFrame,
                            covar_list: list,
                            lead_snp: str) -> dict:
    """Sensitivity 2: ABO-region control mediation analysis (same standard pipeline as CTRB1/2).

    1. Extract all significant SNPs in the ABO region
    2. LD clumping to identify independent signals
    3. Extract genotypes of all independent SNPs
    4. Build a GRS for multiple SNPs, or use dosage for a single SNP
    5. Run the full Baron-Kenny mediation analysis
    """
    print("\n" + "-" * 40)
    print(" Sensitivity 2: ABO-region control analysis (standard LD clumping + mediation)")
    print("-" * 40)

    # Step 2a: Extract all significant SNPs in the ABO region
    abo_snps_raw = get_ctrb_lead_snps(SIGNIFICANT_SNPS, ABO_REGION)
    if len(abo_snps_raw) == 0:
        print("  No significant SNP in the ABO region; skipping")
        return {}

    # Step 2b: LD clumping to identify independent signals (same standard as CTRB1/2)
    chr_num = ABO_REGION['chr']
    keep_file = os.path.join(OUT_DIR, "_keep_fid_iid.txt")
    keep_arg = keep_file if os.path.exists(keep_file) else None

    abo_indep = clump_independent_snps(abo_snps_raw, chr_num, keep_file=keep_arg)

    if len(abo_indep) == 0:
        print("  No independent SNP after LD clumping in the ABO region; skipping")
        return {}

    # Step 2c: Extract genotypes of all independent SNPs
    snp_ids = abo_indep['ID'].tolist()
    a1_col = 'A1' if 'A1' in abo_indep.columns else 'ALLELE1'
    allele_map = dict(zip(abo_indep['ID'].astype(str),
                          abo_indep[a1_col].astype(str))) if a1_col in abo_indep.columns else None
    try:
        abo_geno = extract_snp_list_genotypes(
            snp_ids, chr_num, batch_name="abo_indep", allele_map=allele_map
        )
    except Exception as e:
        print(f"  Could not extract ABO SNPs: {e}")
        return {}

    # Step 2d: Build the ABO genetic instrument
    df_temp = df.copy()
    n_used = 0

    if len(snp_ids) == 1:
        # Single SNP: use dosage directly
        abo_col = snp_ids[0] if snp_ids[0] in abo_geno.columns else abo_geno.columns[1]
        abo_map = dict(zip(abo_geno['IID'].astype(str), abo_geno[abo_col]))
        df_temp['abo_dosage'] = df_temp['eid'].map(abo_map)
        df_temp = df_temp.dropna(subset=['abo_dosage'])
        iv_col = 'abo_dosage'
        n_used = 1
        print(f"  ABO independent signals: {n_used}, using lead SNP {snp_ids[0]}")
    else:
        # Multiple SNPs: build a GRS
        score = pd.Series(0.0, index=df_temp.index)
        for _, snp_row in abo_indep.iterrows():
            sid = snp_row['ID']
            if sid not in abo_geno.columns:
                continue
            beta_sign = 1 if snp_row.get('BETA', 0) >= 0 else -1
            snp_series = df_temp['eid'].map(
                dict(zip(abo_geno['IID'].astype(str), abo_geno[sid]))
            )
            score += snp_series.fillna(0) * beta_sign
            n_used += 1
        df_temp['abo_dosage'] = score
        df_temp['abo_dosage'] = (df_temp['abo_dosage'] - df_temp['abo_dosage'].mean()) / df_temp['abo_dosage'].std()
        iv_col = 'abo_dosage'
        print(f"  ABO independent signals: {n_used}, using GRS")

    covar_str = ' + '.join(covar_list)

    # Step 2e: Baron-Kenny three-step mediation (same specification as the main analysis)
    ma_abo = sm.OLS.from_formula(
        f"delta ~ {iv_col} + {covar_str}", data=df_temp
    ).fit()
    mc_abo = sm.Logit.from_formula(
        f"is_case ~ {iv_col} + {covar_str}", data=df_temp
    ).fit(disp=0)
    mcp_abo = sm.Logit.from_formula(
        f"is_case ~ {iv_col} + delta + {covar_str}", data=df_temp
    ).fit(disp=0)

    a_abo = ma_abo.params[iv_col]
    a_p = ma_abo.pvalues[iv_col]
    b_abo = mcp_abo.params['delta']
    b_p = mcp_abo.pvalues['delta']
    c_abo = mc_abo.params[iv_col]
    c_p = mc_abo.pvalues[iv_col]
    cp_abo = mcp_abo.params[iv_col]
    cp_p = mcp_abo.pvalues[iv_col]
    indirect_abo = a_abo * b_abo

    # Sobel test
    sobel_se_abo = np.sqrt(a_abo**2 * mcp_abo.bse['delta']**2 +
                           b_abo**2 * ma_abo.bse[iv_col]**2)
    sobel_z_abo = indirect_abo / sobel_se_abo if sobel_se_abo > 0 else 0
    sobel_p_abo = 2 * (1 - stats.norm.cdf(abs(sobel_z_abo)))

    print(f"  ═══ ABO region mediation analysis ({n_used} independent SNPs) ═══")
    print(f"  Path a (ABO→APAG):     {a_abo:>10.4f}  P={a_p:.2e}")
    print(f"  Path b (APAG→DM):      {b_abo:>10.4f}  P={b_p:.2e}")
    print(f"  Path c (ABO→DM, total effect): {c_abo:>10.4f}  P={c_p:.2e}  OR={np.exp(c_abo):.4f}")
    print(f"  Path c'(ABO→DM, direct):  {cp_abo:>10.4f}  P={cp_p:.2e}  OR={np.exp(cp_abo):.4f}")
    print(f"  ab (indirect effect):          {indirect_abo:>10.4f}  Sobel Z={sobel_z_abo:.2f}  P={sobel_p_abo:.2e}")

    # Step 2f: Bootstrap mediation test (directly checks whether ab crosses 0; does not rely on Baron-Kenny steps)
    print(f"\n  ABO Bootstrap mediation test (5000 iterations)...")
    np.random.seed(42)
    n_sample = len(df_temp)
    ab_boot_abo = []
    for _ in range(5000):
        idx = np.random.choice(n_sample, size=n_sample, replace=True)
        bdf = df_temp.iloc[idx]
        try:
            ba = sm.OLS.from_formula(f"delta ~ {iv_col} + {covar_str}", data=bdf).fit().params[iv_col]
            bb = sm.Logit.from_formula(f"is_case ~ {iv_col} + delta + {covar_str}", data=bdf).fit(disp=0).params['delta']
            ab_boot_abo.append(ba * bb)
        except Exception:
            continue

    ab_boot_abo = np.array(ab_boot_abo)
    abo_boot_ci = np.percentile(ab_boot_abo, [2.5, 97.5])
    abo_boot_p = 2 * min(np.mean(ab_boot_abo <= 0), np.mean(ab_boot_abo >= 0))
    abo_boot_median = np.median(ab_boot_abo)

    abo_ab_sig = not (abo_boot_ci[0] <= 0 <= abo_boot_ci[1])
    print(f"  ABO Bootstrap ab median: {abo_boot_median:.4f}")
    print(f"  ABO Bootstrap ab 95% CI: [{abo_boot_ci[0]:.4f}, {abo_boot_ci[1]:.4f}]")
    print(f"  ABO Bootstrap P: {abo_boot_p:.4f}")
    print(f"  ABO ab 95% CI {'does not ' if not abo_ab_sig else ''}cross 0 → "
          f"mediation effect {'supported ✓' if abo_ab_sig else 'not supported ✗'}")

    print(f"  Overall conclusion: ABO→DM total effect {'significant' if c_p < 0.05 else 'not significant'}"
          f"{', but ' if c_p >= 0.05 else ', '}Bootstrap ab {'significant' if abo_ab_sig else 'not significant'}")

    return {
        'n_snps': n_used,
        'abo_snp': snp_ids[0],
        'a_abo': a_abo, 'a_p': a_p,
        'b_abo': b_abo, 'b_p': b_p,
        'c_abo': c_abo, 'c_p': c_p, 'c_or': np.exp(c_abo),
        'cp_abo': cp_abo, 'cp_p': cp_p, 'cp_or': np.exp(cp_abo),
        'indirect_abo': indirect_abo,
        'sobel_z_abo': sobel_z_abo, 'sobel_p_abo': sobel_p_abo,
        'abo_boot_ci': abo_boot_ci, 'abo_boot_p': abo_boot_p,
        'abo_boot_median': abo_boot_median, 'abo_ab_sig': abo_ab_sig,
    }


# ═══════════════════════════════════════
# Step 5: Plotting
# ═══════════════════════════════════════

def plot_mediation_results(bk_results: dict,
                           boot_results: pd.DataFrame = None):
    """Plot mediation analysis results — both methods shown in parallel."""
    print("\n" + "=" * 60)
    print(" Step 5: Generate figures (Baron-Kenny + Bootstrap)")
    print("=" * 60)

    ci_info = boot_results.attrs.get('ci_info', {}) if boot_results is not None else {}
    mediation_type = boot_results.attrs.get('mediation_type', '') if boot_results is not None else ''

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # --- Panel 1: Baron-Kenny path-effect plot ---
    ax1 = axes[0]
    effects = {
        'Path a\n(SNP→APAG)': (bk_results['path_a']['coef'],
                               bk_results['path_a']['se'],
                               bk_results['path_a']['p']),
        'Path b\n(APAG→Diabetes)': (bk_results['path_b']['coef'],
                                     bk_results['path_b']['se'],
                                     bk_results['path_b']['p']),
        'Path c\n(SNP→DM, total)': (bk_results['path_c']['coef'],
                                     bk_results['path_c']['se'],
                                     bk_results['path_c']['p']),
        "Path c'\n(SNP→DM, direct)": (bk_results['path_cprime']['coef'],
                                       bk_results['path_cprime']['se'],
                                       bk_results['path_cprime']['p']),
        'Indirect a×b\n(Mediation)': (bk_results['indirect']['coef'],
                                       bk_results['indirect']['sobel_se'],
                                       bk_results['indirect']['sobel_p']),
    }

    y_pos = list(range(len(effects)))
    coefs = [v[0] for v in effects.values()]
    ses = [v[1] for v in effects.values()]
    ps = [v[2] for v in effects.values()]
    colors = ['#2196F3' if p < 0.05 else '#90A4AE' for p in ps]

    ax1.barh(y_pos, coefs, xerr=ses, color=colors, alpha=0.8, capsize=3, height=0.6)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(list(effects.keys()), fontsize=7)
    ax1.set_xlabel('Effect size')
    ax1.set_title('Baron & Kenny \n(Sobel test)', fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    for i, p in enumerate(ps):
        p_text = f"P={p:.2e}" if p < 0.001 else f"P={p:.3f}"
        ax1.text(coefs[i] + ses[i] + abs(max(coefs, key=abs))*0.03, i,
                 p_text, va='center', fontsize=6.5, color='#37474F')

    # --- Panel 2: Bootstrap indirect-effect distribution + classification ---
    ax2 = axes[1]
    if boot_results is not None and len(boot_results) > 0:
        ie = boot_results['indirect_effect'].dropna()
        ax2.hist(ie, bins=50, color='#4CAF50', alpha=0.7, edgecolor='white')
        ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Null (0)')
        ax2.axvline(ie.median(), color='#1B5E20', linestyle='-', linewidth=2,
                    label=f'Median: {ie.median():.4f}')
        ci = np.percentile(ie, [2.5, 97.5])
        ax2.axvline(ci[0], color='#FF9800', linestyle=':', linewidth=1.5)
        ax2.axvline(ci[1], color='#FF9800', linestyle=':', linewidth=1.5,
                    label=f'95% CI\n[{ci[0]:.4f},\n {ci[1]:.4f}]')
        ax2.legend(fontsize=6.5, loc='upper right')
        ax2.set_xlabel('Indirect effect (a×b)')
        ax2.set_ylabel('Frequency')

        # Classification title
        class_colors = {
            'Full mediation': '#1565C0', 'Partial mediation': '#E65100', 'No mediation effect': '#B71C1C'
        }
        ccolor = class_colors.get(mediation_type, '#333')
        ax2.set_title(f'Bootstrap \n→ {mediation_type}',
                      fontsize=10, fontweight='bold', color=ccolor)
    else:
        ax2.text(0.5, 0.5, 'No bootstrap data', transform=ax2.transAxes,
                 ha='center', va='center')
        ax2.set_title('Method 2: Bootstrap distribution')

    # --- Panel 3: Combined mediation path diagram ---
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    a_ = bk_results['path_a']
    b_ = bk_results['path_b']
    c_ = bk_results['path_c']
    cp_ = bk_results['path_cprime']
    ind_ = bk_results['indirect']

    def draw_box(ax, x, y, text, color='#E3F2FD', width=2.5, height=1.2):
        rect = plt.Rectangle((x - width/2, y - height/2), width, height,
                             facecolor=color, edgecolor='#1565C0', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    draw_box(ax3, 2, 7.5, 'CTRB1/2\n(rs72802342)', '#FFF3E0')
    draw_box(ax3, 5, 5, 'Pancreas Age Gap\n(APAG, δ)', '#E8F5E9')
    draw_box(ax3, 8, 7.5, 'Diabetes\n(E10-E14)', '#FCE4EC')

    # c (total effect)
    ax3.annotate('', xy=(4.7, 7.5), xytext=(3.3, 7.5),
                 arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    ax3.text(4, 7.9, f"c={c_['coef']:.3f}, P={c_['p']:.2e}",
             ha='center', fontsize=6.5, color='#555')

    # a
    ax3.annotate('', xy=(3.5, 5.6), xytext=(2.8, 6.9),
                 arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2))
    ax3.text(2.2, 6.2, f"a={a_['coef']:.3f}**",
             ha='center', fontsize=7, color='#2196F3', fontweight='bold')

    # b
    ax3.annotate('', xy=(6.5, 6.9), xytext=(5.5, 5.6),
                 arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
    ax3.text(6.0, 6.2, f"b={b_['coef']:.3f}**",
             ha='center', fontsize=7, color='#4CAF50', fontweight='bold')

    # c'
    ax3.annotate('', xy=(6.8, 7.5), xytext=(5.3, 7.5),
                 arrowprops=dict(arrowstyle='->', color='#999', lw=1.5,
                                 linestyle='dashed'))
    ax3.text(6.0, 7.9, f"c'={cp_['coef']:.3f}", ha='center', fontsize=6.5, color='#999')

    # Results summary box
    # Get Bootstrap CI
    ab_boot_ci = ci_info.get('indirect_effect', {}).get('ci_95', [np.nan, np.nan])
    summary_text = (
        f"Baron-Kenny: ab={ind_['coef']:.4f}, Sobel P={ind_['sobel_p']:.2e}\n"
        f"Bootstrap: ab 95%CI [{ab_boot_ci[0]:.4f}, {ab_boot_ci[1]:.4f}]\n"
        f"Proportion mediated: {ind_['prop_mediated']*100:.1f}%\n"
        f"Classification: {mediation_type}"
    )
    ax3.text(5, 2.5, summary_text, ha='center', fontsize=7.5,
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.85),
             family='monospace')

    ax3.set_title('CTRB1/2 → APAG → Diabetes', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_out = os.path.join(OUT_DIR, "mediation_analysis.pdf")
    fig.savefig(fig_out, bbox_inches='tight',format='pdf')
    plt.close()
    print(f"  -> {fig_out}")


# ═══════════════════════════════════════
# Main function
# ═══════════════════════════════════════

def main(lead_snp: str = "rs72802342",
         chr_num: int = 16,
         n_bootstrap: int = 5000,
         run_all_sensitivity: bool = False,
         keep_file: str = None):
    """Run the full mediation analysis."""

    print("=" * 60)
    print("  CTRB1/2 → APAG → diabetes mediation analysis")
    print(f"  Lead SNP: {lead_snp}")
    print(f"  Bootstrap: {n_bootstrap} iterations")
    print("=" * 60)

    setup_output_dir()

    # Step 0: Extract genotypes
    ctrb_snps = get_ctrb_lead_snps()
    if keep_file is None:
        keep_file = os.path.join(INPUT_DIR, "keep_test_ukb_test_ill.id")
        # PLINK2 requires two columns FID IID
        keep_fid = os.path.join(OUT_DIR, "_keep_fid_iid.txt")
        if not os.path.exists(keep_fid):
            subprocess.run(
                f"awk '{{print $1, $1}}' {keep_file} > {keep_fid}",
                shell=True, check=True
            )
        keep_file = keep_fid

    geno_df = extract_ctrb_genotypes(ctrb_snps, keep_file)

    # Step 1: Merge data
    df = merge_mediation_data(geno_df, lead_snp)

    # Step 2: Baron & Kenny
    pc_cols = [c for c in df.columns if re.match(r'^PC\d+$', c)]
    bk = baron_kenny_mediation(df, pc_cols)

    # Step 3: Bootstrap
    boot_df = bootstrap_mediation(df, bk['covar_list'], n_bootstrap)

    # Step 4: Sensitivity analysis
    sens_results = {}
    if run_all_sensitivity:
        sens_results['multi_snp'] = sensitivity_multi_snp(
            df, geno_df, ctrb_snps, bk['covar_list']
        )
        sens_results['abo_control'] = sensitivity_abo_control(
            df, ctrb_snps, bk['covar_list'], lead_snp
        )

    # Step 5: Plotting
    plot_mediation_results(bk, boot_df)

    # Step 6: Write results summary
    write_summary(bk, boot_df, sens_results, lead_snp)

    print("\n" + "=" * 60)
    print(" Mediation analysis complete!")
    print(f" All outputs: {OUT_DIR}")
    print("=" * 60)

    return bk, boot_df, sens_results


def write_summary(bk: dict, boot_df: pd.DataFrame,
                  sens: dict, lead_snp: str):
    """Write the full results summary — Baron-Kenny and Bootstrap presented in parallel."""
    summary_path = os.path.join(OUT_DIR, "mediation_results.txt")

    a = bk['path_a']
    b = bk['path_b']
    c = bk['path_c']
    cp = bk['path_cprime']
    ind = bk['indirect']

    ci_info = boot_df.attrs.get('ci_info', {})
    mediation_type = boot_df.attrs.get('mediation_type', 'Unclassified')
    mediation_desc = boot_df.attrs.get('mediation_desc', '')

    # Baron-Kenny point estimate of ab
    ab_point = ind['coef']

    # Bootstrap CI of ab
    ab_boot_ci = ci_info.get('indirect_effect', {}).get('ci_95', [np.nan, np.nan])
    cp_boot_ci = ci_info.get('cprime_coef', {}).get('ci_95', [np.nan, np.nan])

    lines = [
        "=" * 70,
        f"  Mediation analysis results: CTRB1/2 → APAG → diabetes",
        f"  Lead SNP: {lead_snp}",
        f"  Date: {pd.Timestamp.now().isoformat()}",
        f"  N samples: {len(bk['df'])} (case={int(bk['df']['is_case'].sum())}, "
        f"ctrl={int((bk['df']['is_case']==0).sum())})",
        "=" * 70,
        "",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│  Method 1: Baron & Kenny stepwise approach + Sobel test (traditional) │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "",
        "  Path coefficients (point estimates):",
        f"    Path a  (SNP → APAG):            {a['coef']:>10.4f}  SE={a['se']:.4f}  P={a['p']:.2e}",
        f"    Path b  (APAG → Diabetes):        {b['coef']:>10.4f}  SE={b['se']:.4f}  P={b['p']:.2e}  OR={b['or']:.4f}",
        f"    Path c  (SNP → Diabetes, total effect):  {c['coef']:>10.4f}  SE={c['se']:.4f}  P={c['p']:.2e}  OR={c['or']:.4f}",
        f"    Path c' (SNP → Diabetes, direct effect):{cp['coef']:>10.4f}  SE={cp['se']:.4f}  P={cp['p']:.2e}  OR={cp['or']:.4f}",
        "",
        "  Indirect effect (Sobel test):",
        f"    ab = a × b = {ab_point:.4f}",
        f"    Sobel SE = {ind['sobel_se']:.4f}",
        f"    Sobel Z  = {ind['sobel_z']:.2f}",
        f"    Sobel P  = {ind['sobel_p']:.2e}",
        f"    Proportion mediated = {ind['prop_mediated']*100:.1f}%",
        "",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│  Method 2: Bootstrap percentile CI (modern standard, Preacher & Hayes 2008) │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "",
        "  Bootstrap 95% CI for each path (median estimates):",
    ]

    path_labels = {
        'a_coef': 'a (SNP→APAG)',
        'b_coef': 'b (APAG→DM)',
        'c_coef': 'c (total effect)',
        'cprime_coef': "c' (direct effect)",
        'indirect_effect': 'ab (indirect effect)',
    }
    for col, label in path_labels.items():
        info = ci_info.get(col, {})
        cilo, cihi = info.get('ci_95', [np.nan, np.nan])
        med = info.get('median', np.nan)
        sig = info.get('significant', 'N/A')
        lines.append(f"    {label:<22s} {med:>10.4f}  [{cilo:>10.4f}, {cihi:>10.4f}]  {sig}")

    lines += [
        "",
        f"  ═══ Bootstrap mediation classification ═══",
        f"  Decision rule: ab 95% CI excludes 0 → mediation supported;",
        f"           then check c' 95% CI: excludes 0 → partial mediation / includes 0 → full mediation",
        f"",
        f"  Conclusion: {mediation_type}",
        f"  {mediation_desc}",
        "",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│  Two-method comparison                                                        │",
        "└─────────────────────────────────────────────────────────────────────┘",
        f"  Baron-Kenny ab point estimate:   {ab_point:.4f}  (Sobel P={ind['sobel_p']:.2e})",
        f"  Bootstrap ab median:     {ci_info.get('indirect_effect', {}).get('median', np.nan):.4f}",
        f"  Bootstrap ab 95% CI:     [{ab_boot_ci[0]:.4f}, {ab_boot_ci[1]:.4f}]",
        f"  Two methods agree:          {'Yes ✓' if (ind['sobel_p'] < 0.05) == boot_df.attrs.get('ab_sig', False) else 'Note discrepancy!'}",
        "",
        f"{'─'*70}",
        f"  Reference: Preacher & Hayes (2008); Hayes, Preacher & Myers (2011)",
        f"        MacKinnon et al. (2002, 2004); Briggs (2006)",
        f"{'─'*70}",
        "",
        f"--- Path a (SNP→APAG) full regression (Baron-Kenny) ---",
        a['model'].summary().as_text(),
        "",
        f"--- Path c (SNP→Diabetes, total effect) full regression (Baron-Kenny) ---",
        c['model'].summary().as_text(),
        "",
        f"--- Path c'+b (SNP+APAG→Diabetes) full regression (Baron-Kenny) ---",
        cp['model'].summary().as_text(),
    ]

    if sens:
        lines.append("\n--- Sensitivity analysis ---")
        if 'multi_snp' in sens and sens['multi_snp']:
            s = sens['multi_snp']
            lines.append(f"Multi-SNP GRS ({s['n_snps']} SNPs): a={s['a_grs']:.4f}, "
                         f"b={s['b_grs']:.4f}, a×b={s['indirect_grs']:.4f}, "
                         f"GRS→DM OR={s['c_or']:.4f}, P={s['c_p']:.2e}")
        if 'abo_control' in sens and sens['abo_control']:
            s = sens['abo_control']
            lines.append(f"\nABO control after LD clumping ({s['n_snps']} independent SNPs, lead: {s['abo_snp']}):")
            lines.append(f"  --- Baron-Kenny ---")
            lines.append(f"  a={s['a_abo']:.4f} (P={s['a_p']:.2e}), "
                         f"b={s['b_abo']:.4f} (P={s['b_p']:.2e})")
            lines.append(f"  c (total effect)={s['c_abo']:.4f} OR={s['c_or']:.4f} (P={s['c_p']:.2e}), "
                         f"c' (direct)={s['cp_abo']:.4f} OR={s['cp_or']:.4f} (P={s['cp_p']:.2e})")
            lines.append(f"  ab (indirect)={s['indirect_abo']:.4f}, "
                         f"Sobel Z={s['sobel_z_abo']:.2f} P={s['sobel_p_abo']:.2e}")
            lines.append(f"  --- Bootstrap (direct test of ab CI) ---")
            lines.append(f"  ab median: {s['abo_boot_median']:.4f}")
            lines.append(f"  ab 95% CI: [{s['abo_boot_ci'][0]:.4f}, {s['abo_boot_ci'][1]:.4f}]")
            lines.append(f"  ab 95% CI {'does not ' if not s['abo_ab_sig'] else ''}cross 0 → "
                         f"mediation effect {'supported ✓' if s['abo_ab_sig'] else 'not supported ✗'}")
            lines.append(f"  Overall: ABO→DM total effect {'significant' if s['c_p'] < 0.05 else 'not significant'} | "
                         f"Bootstrap ab {'significant' if s['abo_ab_sig'] else 'not significant'}")

    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  -> {summary_path}")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTRB1/2 → APAG → diabetes mediation analysis"
    )
    parser.add_argument("--lead-snp", type=str, default="rs72802342",
                        help="Lead SNP ID in the CTRB1/2 region")
    parser.add_argument("--chr", type=int, default=16,
                        help="Chromosome of the CTRB1/2 SNP")
    parser.add_argument("--bootstrap", type=int, default=5000,
                        help="Number of bootstrap iterations (default 5000)")
    parser.add_argument("--sensitivity", action="store_true",default=True,
                        help="Run all sensitivity analyses")
    parser.add_argument("--sig-csv", type=str, default=None,
                        help="Significant SNP table, default gwas_plink_results/significant_snps.csv")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory, default Disease_Phenotype_Analysis/Mediation")
    args = parser.parse_args()

    if args.sig_csv:
        SIGNIFICANT_SNPS = args.sig_csv
    if args.out_dir:
        OUT_DIR = args.out_dir

    main(
        lead_snp=args.lead_snp,
        chr_num=args.chr,
        n_bootstrap=args.bootstrap,
        run_all_sensitivity=args.sensitivity,
    )
