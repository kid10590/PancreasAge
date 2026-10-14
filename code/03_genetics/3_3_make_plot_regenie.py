#!/usr/bin/env python
"""REGENIE GWAS visualization: Manhattan plot + QQ plot + significant locus extraction
REGENIE output columns: CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P
"""
import os, sys, glob, gzip, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

RES_ROOT = OUT_ROOT
DEFAULT_DIR = os.path.join(RES_ROOT, "gwas_regenie_discovery")
P_THRESHOLD = 5e-8

def load_regenie_results(d):
    """Merge the REGENIE step2 output for all chromosomes"""
    files = sorted(glob.glob(os.path.join(d, "gwas_chr*.regenie*")))
    if not files: raise FileNotFoundError(f"No REGENIE results in {d}")
    dfs = []
    for f in files:
        try:
            opener = gzip.open if f.endswith('.gz') else open
            df = pd.read_csv(opener(f, 'rt'), sep='\s+', low_memory=False, comment='#')
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    # REGENIE: CHROM, GENPOS, ID, ALLELE0, ALLELE1, A1FREQ, INFO, N, TEST, BETA, SE, CHISQ, LOG10P
    df['P'] = 10 ** (-pd.to_numeric(df['LOG10P'], errors='coerce'))
    df['CHROM'] = pd.to_numeric(df['CHROM'], errors='coerce')
    df['GENPOS'] = pd.to_numeric(df['GENPOS'], errors='coerce')
    df = df.dropna(subset=['CHROM', 'GENPOS', 'P'])
    print(f"  {len(files)} files, {len(df)} variants")
    return df.rename(columns={'CHROM': '#CHROM', 'GENPOS': 'POS'})


def manhattan(df, out, thr=5e-8):
    pd_ = df[df['P']<0.1].sort_values(['#CHROM','POS']).copy()
    pd_['-log10p'] = -np.log10(pd_['P']); pd_['ind'] = range(len(pd_))
    fig, ax = plt.subplots(figsize=(16,6))
    colors = ['#1f77b4','#aec7e8']; xl, xp = [], []
    for n,(_,g) in enumerate(pd_.groupby('#CHROM')):
        ax.scatter(g['ind'], g['-log10p'], color=colors[n%2], s=4, alpha=0.7)
        xp.append(g['ind'].median()); xl.append(str(int(n+1)))
    ax.set_xticks(xp); ax.set_xticklabels(xl)
    ax.axhline(-np.log10(thr), color='red', linestyle='--', lw=1)
    ax.set_xlabel("Chromosome"); ax.set_ylabel("-log10(P)")
    ax.set_title("REGENIE GWAS: Pancreas Age-delta")
    sig = pd_[pd_['P']<thr]
    if len(sig)>0:
        for _,r in sig.loc[sig.groupby('#CHROM')['P'].idxmin()].iterrows():
            sid = str(r.get('ID', f"chr{int(r['#CHROM'])}:{int(r['POS'])}"))
            ax.annotate(sid, xy=(r['ind'],r['-log10p']), xytext=(0,7), textcoords="offset points",
                        ha='center', fontsize=8, rotation=30,
                        bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="gray",alpha=0.7))
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print(f"  -> {out}")

def qq_plot(df, out):
    pv = df['P'].dropna().values; pv = pv[pv>0]
    obs = -np.log10(np.sort(pv)); exp = -np.log10(np.linspace(1/len(pv),1,len(pv)))
    lam = np.median(chi2.isf(pv,1))/0.456
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(exp, obs, s=3, alpha=0.5); ax.plot([0,max(exp)],[0,max(exp)],'r--',lw=1)
    ax.set_xlabel("Expected -log10(P)"); ax.set_ylabel("Observed -log10(P)")
    ax.set_title(f"QQ (lambda={lam:.3f})")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print(f"  -> {out}  lambda={lam:.4f}")

def main():
    ap = argparse.ArgumentParser(description="REGENIE GWAS visualization")
    ap.add_argument("--dir", default=None, help="REGENIE results directory (default: discovery)")
    ap.add_argument("--mode", choices=["full", "discovery"], default=None,
                    help="full → gwas_regenie_results; discovery → gwas_regenie_discovery")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same as input)")
    args = ap.parse_args()

    if args.dir:
        gwas_dir = args.dir
    elif args.mode == "full":
        gwas_dir = os.path.join(RES_ROOT, "gwas_regenie_results")
    else:
        gwas_dir = DEFAULT_DIR
    out_dir = args.out_dir or gwas_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"REGENIE GWAS: {gwas_dir}")
    df = load_regenie_results(gwas_dir)
    sig = df[df['P']<P_THRESHOLD].sort_values('P')
    sig.to_csv(os.path.join(out_dir,"significant_snps.csv"), index=False)
    print(f"Significant (P<{P_THRESHOLD}): {len(sig)} -> significant_snps.csv")
    manhattan(df, os.path.join(out_dir,"manhattan.png"))
    qq_plot(df, os.path.join(out_dir,"qq.png"))
    print("Done.")

if __name__=="__main__": main()
