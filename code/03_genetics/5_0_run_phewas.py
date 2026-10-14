#!/usr/bin/env python
"""
PheWAS (phenome-wide association study): association analysis of pancreas age-delta PRS with ICD-10 phenotypes

Methods:
  - Extract disease phenotypes from ICD-10 codes (three-character ICD-10 categories, e.g. I10, E11, etc.)
  - Run logistic regression for phenotypes with >=20 cases
  - Covariates: Age, Age², Sex, PC1-10
  - Multiple-comparison correction: Bonferroni (P < 0.05 / N_phenotypes)
  - Output: Manhattan plot (colored by ICD-10 chapter)

Reference: cardiovascular_ageing/PheWAS/
"""
import os, sys, re, argparse
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ICD-10 chapter definitions (based on WHO ICD-10 classification)
ICD10_CHAPTERS = {
    'I: Infections': (('A', 0, 99), ('B', 0, 99)),
    'II: Neoplasms': (('C', 0, 97), ('D', 0, 48)),
    'III: Blood': (('D', 50, 89),),
    'IV: Endocrine': (('E', 0, 89),),
    'V: Mental': (('F', 0, 99),),
    'VI: Nervous': (('G', 0, 99),),
    'VII: Eye': (('H', 0, 59),),
    'VIII: Ear': (('H', 60, 95),),
    'IX: Circulatory': (('I', 0, 99),),
    'X: Respiratory': (('J', 0, 99),),
    'XI: Digestive': (('K', 0, 93),),
    'XII: Skin': (('L', 0, 99),),
    'XIII: Musculoskeletal': (('M', 0, 99),),
    'XIV: Genitourinary': (('N', 0, 99),),
    'XV: Pregnancy': (('O', 0, 99),),
    'XVI: Perinatal': (('P', 0, 96),),
    'XVII: Congenital': (('Q', 0, 99),),
    'XVIII: Symptoms': (('R', 0, 99),),
    'XIX: Injury': (('S', 0, 99), ('T', 0, 98)),
    'XX: External': (('V', 0, 99), ('W', 0, 99), ('X', 0, 99), ('Y', 0, 98)),
    'XXI: Health factors': (('Z', 0, 99),),
}


def get_icd10_chapter(code: str) -> str:
    """Return the chapter name for an ICD-10 code"""
    code_clean = code.strip().upper()
    m = re.match(r'^([A-Z])(\d+)', code_clean)
    if not m:
        return None
    letter, num_str = m.group(1), m.group(2)
    num = int(num_str) if num_str.isdigit() else 0

    for chapter, ranges in ICD10_CHAPTERS.items():
        for r in ranges:
            if letter == r[0] and r[1] <= num <= r[2]:
                return chapter
    return None


def extract_icd10_codes(icd_field) -> List[str]:
    """Extract a list of three-character codes from an ICD-10 field"""
    if pd.isna(icd_field) or not isinstance(icd_field, str):
        return []
    codes = []
    for part in icd_field.split('|'):
        part = part.strip()
        m = re.match(r'^([A-Za-z]\d{2})', part)
        if m:
            codes.append(m.group(1).upper())
    return codes


def make_phenotype_matrix(df: pd.DataFrame, icd_col: str, min_cases: int = 20) -> Tuple[pd.DataFrame, dict]:
    """Build a binary phenotype matrix from an ICD-10 column"""
    all_codes = set()
    eid_codes: Dict[str, set] = {}

    for _, row in df.iterrows():
        eid = str(row['eid'])
        codes = extract_icd10_codes(row.get(icd_col, ''))
        eid_codes[eid] = set(codes)
        all_codes.update(codes)

    # Filter: at least min_cases cases
    code_counts = {}
    for code in all_codes:
        code_counts[code] = sum(1 for codes in eid_codes.values() if code in codes)

    valid_codes = {c for c, n in code_counts.items() if n >= min_cases}
    valid_codes_sorted = sorted(valid_codes)

    # Get the chapter mapping
    code_chapter = {}
    for c in valid_codes_sorted:
        ch = get_icd10_chapter(c)
        code_chapter[c] = ch if ch else 'Other'

    # Build the phenotype matrix
    pheno_rows = []
    for eid in df['eid'].astype(str):
        row_dict = {'eid': eid}
        codes = eid_codes.get(eid, set())
        for c in valid_codes_sorted:
            row_dict[c] = 1 if c in codes else 0
        pheno_rows.append(row_dict)

    pheno_df = pd.DataFrame(pheno_rows)
    return pheno_df, code_chapter


def run_phewas(pheno_df: pd.DataFrame, prs_df: pd.DataFrame,
               covar_df: pd.DataFrame, code_chapter: dict,
               prs_col: str = 'APAG_PRS') -> pd.DataFrame:
    """Run PheWAS: logistic regression for each phenotype"""
    # Merge data
    merged = pheno_df.merge(prs_df, left_on='eid', right_on='IID', how='inner')
    merged = merged.merge(covar_df, left_on='eid', right_on='IID', how='inner')

    covar_cols = ['age', 'age2', 'sex_code', 'Array_bin'] + [f'PC{i}' for i in range(1, 11)]
    covar_cols = [c for c in covar_cols if c in merged.columns]

    pheno_codes = [c for c in pheno_df.columns if c != 'eid']

    results = []
    for code in pheno_codes:
        if code not in merged.columns:
            continue

        # Extract case/control
        y = merged[code].values
        n_cases = int(y.sum())
        n_controls = int((1 - y).sum())

        if n_cases < 20:
            continue

        # Build the design matrix
        prs_vals = merged[prs_col].fillna(merged[prs_col].mean()).values
        prs_std = (prs_vals - np.mean(prs_vals)) / np.std(prs_vals)

        X_data = {col: merged[col].values for col in covar_cols if col in merged.columns}
        X_data['prs'] = prs_std

        X_df = pd.DataFrame(X_data)
        X_df = X_df.dropna()
        valid_idx = X_df.index.intersection(np.where(~np.isnan(y))[0])

        if len(valid_idx) < 20:
            continue

        X = sm.add_constant(X_df.loc[valid_idx].values)
        y_valid = y[valid_idx]

        try:
            model = sm.Logit(y_valid, X).fit(disp=False)
            # Separation/non-convergence guard: drop SE≈0 (e.g. N48) or non-converged degenerate solutions
            converged = getattr(model, 'mle_retvals', None) or {}
            if not converged.get('converged', True):
                continue
            se = model.bse[-1]
            if not np.isfinite(se) or se < 1e-6:
                continue
            beta = model.params[-1]
            pval = model.pvalues[-1]
            or_val = np.exp(beta)

            results.append({
                'phenotype': code,
                'chapter': code_chapter.get(code, 'Other'),
                'n_cases': n_cases,
                'n_total': len(valid_idx),
                'beta': beta,
                'SE': se,
                'OR': or_val,
                'P': pval,
            })
        except Exception as e:
            continue

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        # Bonferroni correction
        results_df['P_bonferroni'] = results_df['P'] * len(results_df)
        results_df['P_bonferroni'] = results_df['P_bonferroni'].clip(upper=1.0)
        # FDR (Benjamini-Hochberg) correction
        results_df['P_fdr'] = multipletests(results_df['P'], method='fdr_bh')[1]
        results_df['-log10P'] = -np.log10(results_df['P'])
        results_df = results_df.sort_values('P')

    return results_df


def plot_phewas(results_df: pd.DataFrame, out_path: str, title: str = 'PheWAS'):
    """Draw the PheWAS Manhattan plot (colored by ICD-10 chapter)"""
    if len(results_df) == 0:
        print("No results to plot")
        return

    # Assign x coordinates to each chapter
    chapters_ordered = []
    for ch in ICD10_CHAPTERS.keys():
        if ch in results_df['chapter'].values:
            chapters_ordered.append(ch)
    # Add others
    others = results_df[~results_df['chapter'].isin(chapters_ordered)]
    if len(others) > 0:
        chapters_ordered.append('Other')

    # Assign x positions to phenotypes
    results_df = results_df.copy()
    results_df['x'] = 0
    x_positions = []
    chapter_midpoints = []
    chapter_labels = []

    current_x = 0
    for ch in chapters_ordered:
        ch_data = results_df[results_df['chapter'] == ch].sort_values('phenotype')
        if len(ch_data) == 0:
            continue
        for idx in ch_data.index:
            results_df.at[idx, 'x'] = current_x
            current_x += 1
        mid = (results_df.loc[ch_data.index[0], 'x'] + results_df.loc[ch_data.index[-1], 'x']) / 2
        chapter_midpoints.append(mid)
        # Short label
        short_label = ch.split(':')[0] if ':' in ch else ch
        chapter_labels.append(short_label)

    # Coloring
    chapter_colors = plt.cm.tab20(np.linspace(0, 1, len(chapters_ordered)))
    color_map = dict(zip(chapters_ordered, chapter_colors))

    fig, ax = plt.subplots(figsize=(20, 8))

    for ch in chapters_ordered:
        ch_data = results_df[results_df['chapter'] == ch]
        if len(ch_data) == 0:
            continue
        ax.scatter(ch_data['x'], ch_data['-log10P'],
                   c=[color_map[ch]], label=ch.split(':')[1].strip() if ':' in ch else ch,
                   s=8, alpha=0.7)

    # Bonferroni threshold line
    bonf_threshold = 0.05 / len(results_df)
    ax.axhline(-np.log10(bonf_threshold), color='red', linestyle='--', lw=1,
               label=f'Bonferroni (P={bonf_threshold:.2e})')

    # Annotate significant points
    sig = results_df[results_df['P_bonferroni'] < 0.05]
    for _, row in sig.iterrows():
        ax.annotate(row['phenotype'], xy=(row['x'], row['-log10P']),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7, rotation=45,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))

    ax.set_xticks(chapter_midpoints)
    ax.set_xticklabels(chapter_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('ICD-10 Chapter')
    ax.set_ylabel('-log10(P)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=6, ncol=2, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'  → {out_path}')


def main():
    ap = argparse.ArgumentParser(description='PheWAS: pancreas age PRS association with ICD-10 phenotypes')
    ap.add_argument('--prs', required=True, help='PRS score CSV')
    ap.add_argument('--pheno', required=True, help='Phenotype + covariate merged CSV')
    ap.add_argument('--prs-col', default='APAG_PRS', help='PRS column name to use (default: APAG_PRS)')
    ap.add_argument('--icd-col', default='icd-10', help='ICD-10 column name')
    ap.add_argument('--min-cases', type=int, default=20, help='Minimum number of cases (default: 20)')
    ap.add_argument('--keep', default=None, help='List of samples to keep (two columns FID IID, e.g. the independent test set)')
    ap.add_argument('--out-dir', default=None, help='Output directory')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.prs)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    pheno_df = pd.read_csv(args.pheno, low_memory=False)
    prs_df = pd.read_csv(args.prs)

    # Fix: keep only independent samples (e.g. the test 30% not used to build the PRS), avoiding winner's curse
    if args.keep and os.path.exists(args.keep):
        keep = pd.read_csv(args.keep, sep=r'\s+', header=None)
        keep_iids = set(keep.iloc[:, 1].astype(str))
        n0 = len(prs_df)
        prs_df['IID'] = prs_df['IID'].astype(str)
        prs_df = prs_df[prs_df['IID'].isin(keep_iids)]
        pheno_df['eid'] = pheno_df['eid'].astype(str)
        pheno_df = pheno_df[pheno_df['eid'].isin(keep_iids)]
        print(f"  --keep: samples {n0} -> {len(prs_df)} (independent subset)")
    else:
        print("  [NOTE] --keep not provided; using all samples (PRS and GWAS samples overlap, winner's curse risk)")

    # Prepare covariates
    pheno_df['eid'] = pheno_df['eid'].astype(str)
    pheno_df['age'] = pd.to_numeric(pheno_df['age_gt'], errors='coerce')
    pheno_df['age2'] = pheno_df['age'] ** 2

    # Sex
    s = pheno_df['sex'].astype(str).str.strip().str.upper()
    pheno_df['sex_code'] = s.map({'MALE': 1, 'M': 1, 'FEMALE': 0, 'F': 0})
    pheno_df['sex_code'] = pd.to_numeric(pheno_df['sex_code'], errors='coerce')

    # PCs and Array
    for i in range(1, 11):
        col = f'PC{i}'
        if col in pheno_df.columns:
            pheno_df[col] = pd.to_numeric(pheno_df[col], errors='coerce')
        else:
            pheno_df[col] = 0
    if 'Array' in pheno_df.columns:
        pheno_df['Array_bin'] = pheno_df['Array'].fillna("").astype(str).str.contains('UKBILEVE', case=False).astype(int)
    else:
        pheno_df['Array_bin'] = 0

    # Prepare covar_df
    covar_cols = ['eid', 'age', 'age2', 'sex_code', 'Array_bin'] + [f'PC{i}' for i in range(1, 11)]
    covar_df = pheno_df[[c for c in covar_cols if c in pheno_df.columns]].copy()
    covar_df['IID'] = covar_df['eid']

    # PRS IID normalization
    prs_df['IID'] = prs_df['IID'].astype(str)

    # Check PRS column
    if args.prs_col not in prs_df.columns:
        print(f"Warning: PRS column '{args.prs_col}' not found")
        print(f"Available columns: {list(prs_df.columns)}")
        # Try to auto-detect
        prs_candidates = [c for c in prs_df.columns if c not in ('FID', 'IID', 'ALLELE_CT', 'NAMED_ALLELE_DOSAGE_SUM')]
        if prs_candidates:
            args.prs_col = prs_candidates[0]
            print(f"Using: {args.prs_col}")

    # Extract the ICD-10 phenotype matrix
    print("Extracting ICD-10 phenotypes...")
    pheno_matrix, code_chapter = make_phenotype_matrix(pheno_df, args.icd_col, args.min_cases)
    print(f"  Number of phenotypes (>={args.min_cases} cases): {len(pheno_matrix.columns) - 1}")

    # Run PheWAS
    print(f"Running PheWAS (PRS column: {args.prs_col})...")
    results = run_phewas(pheno_matrix, prs_df, covar_df, code_chapter, args.prs_col)

    # Save
    result_csv = os.path.join(out_dir, 'phewas_results.csv')
    results.to_csv(result_csv, index=False)
    print(f"  Results: {result_csv}")
    print(f"  Total phenotypes: {len(results)}")
    sig_count = (results['P_bonferroni'] < 0.05).sum()
    fdr_count = (results['P_fdr'] < 0.05).sum()
    print(f"  Bonferroni significant: {sig_count} | FDR significant: {fdr_count}")

    # Plot
    plot_path = os.path.join(out_dir, 'phewas_manhattan.png')
    plot_phewas(results, plot_path,
                title=f'Pancreas Age-delta PRS PheWAS (n={len(pheno_matrix)})')

    print("\n=== Top 10 associations ===")
    print(results.head(10)[['phenotype', 'chapter', 'n_cases', 'OR', 'P', 'P_bonferroni']].to_string(index=False))


if __name__ == '__main__':
    main()
