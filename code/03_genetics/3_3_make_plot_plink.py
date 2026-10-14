#!/usr/bin/env python
"""PLINK2 GWAS visualization: Manhattan plot + QQ plot + significant locus extraction"""
import os, sys, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

GWAS_DIR = os.path.join(OUT_ROOT, "gwas_plink_discovery")
OUT_DIR = GWAS_DIR
P_THRESHOLD = 5e-8


def load_plink_results(d):
    files = sorted(glob.glob(os.path.join(d, "gwas_chr*.catb_delta_with_t1_bc_cole.glm.linear*")))
    if not files:
        raise FileNotFoundError(f"No PLINK results in {d}")
    
    dfs = []
    for f in files:
        # First read the column names (first line starts with #)
        with open(f, 'r') as handle:   
            header_line = handle.readline().strip()
            if header_line.startswith('#'):
                # Remove the # prefix and split by tab
                header = header_line.lstrip('#').split('\t')
            else:
                # If there is no header line, let pandas handle it automatically
                df_temp = pd.read_csv(f, sep='\t', low_memory=False)
                dfs.append(df_temp)
                continue
        
        # Read data, skipping lines starting with # (to guard against other comment lines)
        df_temp = pd.read_csv(f, sep='\t', comment='#', header=None, low_memory=False)
        
        # Check whether the column count matches
        if len(df_temp.columns) == len(header):
            df_temp.columns = header
        else:
            # If the column count does not match, use the first N column names
            df_temp.columns = header[:len(df_temp.columns)]
        
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Print column names for debugging
    print(f"  File columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['#CHROM', 'POS', 'P']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Try to auto-detect column names
        col_map = {}
        for col in df.columns:
            col_upper = col.upper().strip()
            if col_upper in ['#CHROM', 'CHR', 'CHROM', 'CHROMOSOME']:
                col_map['chrom'] = col
            elif col_upper in ['POS', 'POSITION', 'BP']:
                col_map['pos'] = col
            elif col_upper in ['P', 'PVAL', 'P_VALUE']:
                col_map['pval'] = col
        
        if 'chrom' in col_map and 'pos' in col_map and 'pval' in col_map:
            df = df.rename(columns={
                col_map['chrom']: '#CHROM',
                col_map['pos']: 'POS',
                col_map['pval']: 'P'
            })
        else:
            print(f"  Available columns: {list(df.columns)}")
            raise KeyError(f"Required columns not found: {missing}")
    
    # Convert to numeric
    for c in ['#CHROM', 'POS', 'P']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=['#CHROM', 'POS', 'P'])
    print(f"  {len(files)} files, {len(df)} variants")
    return df


def manhattan(df, out, thr=5e-8, max_background=150_000, leads=None):
    """Standard Manhattan plot: cumulative genomic positions; keep all strong signals, randomly downsample the background.
    leads: DataFrame with an 'ID' column. When provided, only these LD-clumped
    lead loci are labelled, matching Fig4a's annotation."""
    df = df[(df['P'] > 0) & (df['#CHROM'].between(1, 22))].copy()
    if df.empty:
        print("  ⚠️ No valid loci; skipping Manhattan plot")
        return
    strong = df[df['P'] < 1e-3]
    background = df[df['P'] >= 1e-3]
    if len(background) > max_background:
        background = background.sample(max_background, random_state=42)
    pd_ = pd.concat([strong, background], ignore_index=True).sort_values(['#CHROM', 'POS'])
    chrom_max = df.groupby('#CHROM')['POS'].max().sort_index()
    offsets = chrom_max.cumsum().shift(fill_value=0)
    pd_['genomic_pos'] = pd_['POS'] + pd_['#CHROM'].map(offsets)
    pd_['-log10p'] = -np.log10(pd_['P'])

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['#0F4D92', '#A9C4E4']
    xp, xl = [], []
    for n, (chrom, grp) in enumerate(pd_.groupby('#CHROM')):
        ax.scatter(grp['genomic_pos'], grp['-log10p'], color=colors[n % 2], s=3, alpha=0.7, linewidths=0)
        xp.append(offsets.loc[chrom] + chrom_max.loc[chrom] / 2)
        xl.append(str(int(chrom)))
    ax.set_xticks(xp)
    ax.set_xticklabels(xl)
    ax.axhline(-np.log10(thr), color='red', linestyle='--', lw=1)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_ylim(bottom=0)
    
    sig = df[df['P'] < thr].copy()
    if len(sig) > 0:
        if leads is not None and len(leads):
            label_rows = sig[sig['ID'].isin(leads['ID'].astype(str))].drop_duplicates('ID')
            if len(label_rows) == 0:
                label_rows = sig.loc[sig.groupby('#CHROM')['P'].idxmin()]
        else:
            label_rows = sig.loc[sig.groupby('#CHROM')['P'].idxmin()]
        for _, r in label_rows.iterrows():
            sid = str(r.get('ID', f"chr{int(r['#CHROM'])}:{int(r['POS'])}"))
            if not sid.startswith('rs'):
                sid = f"chr{int(r['#CHROM'])}:{r['POS']/1e6:.2f}Mb"
            xpos = r['POS'] + offsets.loc[r['#CHROM']]
            ax.annotate(sid, xy=(xpos, -np.log10(r['P'])), xytext=(0, 7),
                        textcoords="offset points", ha='center', fontsize=8, rotation=30,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  -> {out}")


def qq_plot(df, out):
    pv = df['P'].dropna().values
    pv = pv[pv > 0]
    if len(pv) < 10:
        print("  ⚠️ Too few valid P values; skipping QQ plot")
        return
    
    obs = -np.log10(np.sort(pv))
    exp = -np.log10(np.linspace(1 / len(pv), 1, len(pv)))
    
    try:
        lam = np.median(chi2.isf(pv, 1)) / 0.4549
    except:
        lam = np.nan
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(exp, obs, s=3, alpha=0.5)
    max_val = max(exp.max(), obs.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    ax.set_title(f"QQ plot (λ = {lam:.3f})" if not np.isnan(lam) else "QQ plot")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  -> {out}  lambda={lam:.4f}" if not np.isnan(lam) else f"  -> {out}")


def main():
    parser = argparse.ArgumentParser(description="PLINK GWAS Manhattan/QQ plot")
    parser.add_argument("--mode", choices=["discovery", "full"], default="full")
    parser.add_argument("--gwas-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    gwas_dir = args.gwas_dir or os.path.join(
        OUT_ROOT,
        "gwas_plink_discovery" if args.mode == "discovery" else "gwas_plink_results",
    )
    out_dir = args.out_dir or gwas_dir
    print(f"PLINK GWAS results directory: {gwas_dir}")
    df = load_plink_results(gwas_dir)
    
    # Consistent with Fig4a: label the LD-clumped independent significant loci of the all-sample GWAS.
    lead_path = os.path.join(out_dir, "full_lead_loci.csv")
    leads = None
    if os.path.exists(lead_path):
        leads = pd.read_csv(lead_path)
        if 'ID' not in leads.columns and 'lead' in leads.columns:
            leads = leads.rename(columns={'lead': 'ID'})
        print(f"  → Labeling {len(leads)} independent significant loci: {lead_path}")
    
    # Significant loci
    sig = df[df['P'] < P_THRESHOLD].sort_values('P')
    sig_path = os.path.join(out_dir, "significant_snps.csv")
    sig.to_csv(sig_path, index=False)
    print(f"  Significant loci (P<{P_THRESHOLD}): {len(sig)} → significant_snps.csv")
    
    if len(sig) > 0:
        sig[['#CHROM', 'POS', 'ID', 'P']].to_csv(
            os.path.join(out_dir, "significant_snps_summary.csv"), index=False
        )
    
    manhattan(df, os.path.join(out_dir, "manhattan.png"), leads=leads)
    qq_plot(df, os.path.join(out_dir, "qq.png"))
    
    print("Done.")


if __name__ == "__main__":
    main()
