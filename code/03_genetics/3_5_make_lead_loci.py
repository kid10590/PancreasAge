#!/usr/bin/env python
"""
3_5_make_lead_loci.py — LD clumping of significant SNPs to extract independent lead loci

Run LD clumping on GWAS significant loci (default P<5e-8) with PLINK2 --clump
(r²=0.1, 250kb, using UKB's own pgen as the LD reference panel), outputting a table in
the same format as gwas_plink_results/full_lead_loci.csv:
    CHR, lead, BP, P

Works for significant_snps.csv from both PLINK2 and REGENIE.

Usage:
  # REGENIE discovery (default)
  python code/03_genetics/3_5_make_lead_loci.py --method regenie \
      --sig-csv .../gwas_regenie_discovery/significant_snps.csv \
      --keep .../prs_analysis/keep_discovery.id \
      --out .../gwas_regenie_discovery/regenie_lead_loci.csv

  # PLINK full (reproduce full_lead_loci.csv)
  python code/03_genetics/3_5_make_lead_loci.py --method plink \
      --sig-csv .../gwas_plink_results/significant_snps.csv \
      --keep .../prs_analysis/keep_all.id \
      --out .../gwas_plink_results/full_lead_loci.csv
"""
import os
import sys
import argparse
import subprocess

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT, PGEN_DIR, PLINK2  # noqa: E402
RES_ROOT = OUT_ROOT

DEFAULT_GWAS_DIR = {
    "regenie": os.path.join(RES_ROOT, "gwas_regenie_discovery"),
    "plink": os.path.join(RES_ROOT, "gwas_plink_discovery"),
}


def _pick(df, names, what):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Column {what} not found in the significant SNP table (candidates: {names}); actual columns: {list(df.columns)}")


def load_sig(sig_csv, p_threshold):
    df = pd.read_csv(sig_csv, low_memory=False)
    df.columns = [c.strip().lstrip('#') for c in df.columns]
    chrom = _pick(df, ["CHROM", "CHR"], "chromosome")
    pos = _pick(df, ["POS", "GENPOS", "BP"], "position")
    sid = _pick(df, ["ID", "SNP"], "SNP ID")
    pcol = _pick(df, ["P", "PVAL", "P_VALUE"], "P")
    df = df.rename(columns={chrom: "CHROM", pos: "POS", sid: "ID", pcol: "P"})
    df = df[["CHROM", "POS", "ID", "P"]].copy()
    df["CHROM"] = pd.to_numeric(df["CHROM"], errors="coerce")
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["CHROM", "POS", "ID", "P"])
    df = df[df["P"] < p_threshold]
    return df


def clump_chromosome(df_chr, chrom, workdir, pgen_dir, keep, r2, kb, threads, p_threshold):
    prefix = os.path.join(workdir, f"clump_chr{chrom}")
    clump_in = f"{prefix}.txt"
    df_chr[["ID", "P"]].to_csv(clump_in, sep="\t", index=False)

    cmd = [PLINK2, "--rm-dup", "exclude-all",
           "--pfile", os.path.join(pgen_dir, f"ukb22828_c{chrom}_b0_v3")]
    if keep:
        cmd += ["--keep", keep]
    cmd += ["--clump", clump_in,
            "--clump-p1", str(p_threshold),
            "--clump-r2", str(r2),
            "--clump-kb", str(kb),
            "--clump-snp-field", "ID",
            "--clump-field", "P",
            "--out", prefix,
            "--threads", str(threads), "--silent"]
    subprocess.run(cmd, check=True)

    clumps = f"{prefix}.clumps"
    leads = []
    if os.path.exists(clumps):
        with open(clumps) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3 and parts[2] != "null":
                    leads.append(parts[2])
    return leads


def main():
    ap = argparse.ArgumentParser(description="Significant SNP LD clumping → independent lead loci")
    ap.add_argument("--method", choices=["regenie", "plink"], default="regenie",
                    help="GWAS source (determines default directory), default regenie")
    ap.add_argument("--sig-csv", default=None,
                    help="Significant SNP table; default <gwas-dir>/significant_snps.csv")
    ap.add_argument("--gwas-dir", default=None, help="GWAS results directory")
    ap.add_argument("--pgen-dir", default=PGEN_DIR, help="pgen directory")
    ap.add_argument("--keep", default=None, help="Sample keep file (LD panel samples)")
    ap.add_argument("--p-threshold", type=float, default=5e-8, help="Significance threshold")
    ap.add_argument("--r2", type=float, default=0.1, help="clump r²")
    ap.add_argument("--kb", type=int, default=250, help="clump window (kb)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None, help="Output lead loci csv")
    ap.add_argument("--workdir", default=None, help="Directory for clump intermediate files")
    args = ap.parse_args()

    gwas_dir = args.gwas_dir or DEFAULT_GWAS_DIR[args.method]
    sig_csv = args.sig_csv or os.path.join(gwas_dir, "significant_snps.csv")
    out = args.out or os.path.join(gwas_dir, f"{args.method}_lead_loci.csv")
    workdir = args.workdir or os.path.join(os.path.dirname(out), f"clump_{args.method}")
    os.makedirs(workdir, exist_ok=True)

    print("=" * 60)
    print(f"   3_5 LD clumping ({args.method})")
    print(f"   Significant SNP table: {sig_csv}")
    print(f"   Output: {out}")
    print(f"   Intermediate dir: {workdir}")
    print("=" * 60)

    df = load_sig(sig_csv, args.p_threshold)
    print(f"  P<{args.p_threshold:g} significant SNPs: {len(df)}")
    if len(df) == 0:
        print("  No significant SNPs; writing an empty table")
        pd.DataFrame(columns=["CHR", "lead", "BP", "P"]).to_csv(out, index=False)
        return

    lead_ids = []
    for chrom, sub in df.groupby("CHROM"):
        chrom = int(chrom)
        ids = clump_chromosome(sub, chrom, workdir, args.pgen_dir, args.keep,
                               args.r2, args.kb, args.threads, args.p_threshold)
        print(f"  chr{chrom}: {len(sub)} sig SNPs → {len(ids)} lead")
        lead_ids.extend(ids)

    lead_ids = set(lead_ids)
    lead = df[df["ID"].astype(str).isin(lead_ids)].copy()
    lead = lead.drop_duplicates(subset=["ID"])
    lead = lead.rename(columns={"CHROM": "CHR", "ID": "lead", "POS": "BP"})
    lead = lead[["CHR", "lead", "BP", "P"]].sort_values(["CHR", "P"])
    lead["CHR"] = lead["CHR"].astype(int)
    lead["BP"] = lead["BP"].astype("Int64")
    lead.to_csv(out, index=False)

    print(f"\n  Independent lead loci: {len(lead)} → {out}")
    print(lead.to_string(index=False))


if __name__ == "__main__":
    main()
