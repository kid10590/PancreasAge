#!/usr/bin/env python3
"""
4_0_ttest_pancreatic_age.py
============================
Two-sample t test: compare the pancreatic age gap (Δ) between healthy controls (ukb_test) and the diseased population (ukb_test_ill).

Usage:
    python code/02_downstream/2_9_ttest_delta.py
    python code/02_downstream/2_9_ttest_delta.py --input /path/to/merged_test.csv

Outputs:
    - ttest_results.csv       # statistics results table
    - ttest_results.txt       # detailed statistical report
    - delta_comparison.png    # violin plot of Δ distribution
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

# ======================== Config ========================
DEFAULT_CSV = os.path.join(OUT_ROOT, "Disease_Phenotype_Analysis", "merged_phenotype_test_bc_self.csv")
DEFAULT_OUTDIR = os.path.join(OUT_ROOT, "TTest_Analysis")


# ======================== Plotting ========================
def plot_violin(healthy_delta: np.ndarray, ill_delta: np.ndarray,
                p_val: float, cohens_d: float, outdir: str):
    """Draw violin plot of Δ distribution for healthy and disease groups"""
    fig, ax = plt.subplots(figsize=(6, 5))

    positions = [0, 1]
    parts = ax.violinplot([healthy_delta, ill_delta], positions=positions,
                          widths=0.7, showmeans=True, showextrema=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_alpha(0.7)

    ax.axhline(0, linestyle="dotted", color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Healthy\n(ukb_test)", "Disease\n(ukb_test_ill)"], fontsize=11)
    ax.set_ylabel("Pancreatic Age Δ (years)", fontsize=12)

    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ax.set_title(f"Healthy vs Disease\n"
                 f"p = {p_val:.2e}{sig},  Cohen's d = {cohens_d:.3f}", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(outdir, "delta_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


# ======================== Main ========================
def main():
    parser = argparse.ArgumentParser(description="Two-sample t test: between-group comparison of pancreatic age gap")
    parser.add_argument("--input", default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTDIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1. Load data ==========
    print("[Step 1] Loading data...")
    df = pd.read_csv(args.input, low_memory=False)
    df["delta"] = df["age_pred_bc"] - df["age_gt"]

    valid_splits = {"ukb_test", "ukb_test_ill"}
    df = df[df["split"].isin(valid_splits)].copy()

    healthy = df[df["split"] == "ukb_test"]["delta"].dropna().values
    ill = df[df["split"] == "ukb_test_ill"]["delta"].dropna().values

    print(f"   Healthy (ukb_test):      n = {len(healthy)}, mean Δ = {np.mean(healthy):.3f}")
    print(f"   Disease (ukb_test_ill):   n = {len(ill)},    mean Δ = {np.mean(ill):.3f}")

    # ========== 2. Welch's t test ==========
    print("[Step 2] Running Welch's two-sample t test...")
    t_stat, p_val = stats.ttest_ind(healthy, ill, equal_var=False)
    cohens_d = (np.mean(healthy) - np.mean(ill)) / np.sqrt(
        (np.var(healthy, ddof=1) + np.var(ill, ddof=1)) / 2
    )
    delta_diff = np.mean(healthy) - np.mean(ill)

    print(f"   t = {t_stat:.4f}")
    print(f"   p = {p_val:.4e}")
    print(f"   Cohen's d = {cohens_d:.3f}")
    print(f"   Δ mean difference (Healthy − Disease) = {delta_diff:.3f} years")

    # ========== 3. Save results ==========
    print("[Step 3] Saving results...")

    result_df = pd.DataFrame([{
        "Group_A": "Healthy (ukb_test)",
        "Group_B": "Disease (ukb_test_ill)",
        "n_A": len(healthy),
        "n_B": len(ill),
        "Mean_Δ_A": round(np.mean(healthy), 4),
        "Mean_Δ_B": round(np.mean(ill), 4),
        "Std_A": round(np.std(healthy, ddof=1), 4),
        "Std_B": round(np.std(ill, ddof=1), 4),
        "Δ_Diff (A−B)": round(delta_diff, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": p_val,
        "Cohens_d": round(cohens_d, 4),
    }])

    csv_path = os.path.join(args.output_dir, "ttest_results.csv")
    result_df.to_csv(csv_path, index=False, float_format="%.6e",encoding="utf-8-sig")
    print(result_df.to_string(index=False))
    print(f"\nTable saved: {csv_path}")

    # Detailed text report
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    report_path = os.path.join(args.output_dir, "ttest_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Two-sample t test: between-group comparison of pancreatic age gap (Δ)\n")
        f.write("  Method: Welch's t-test (does not assume equal variance)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Healthy controls (ukb_test)\n")
        f.write(f"    Sample size:          {len(healthy)}\n")
        f.write(f"    Δ mean ± SD:          {np.mean(healthy):.3f} ± {np.std(healthy, ddof=1):.3f} years\n\n")
        f.write(f"  Diseased population (ukb_test_ill)\n")
        f.write(f"    Sample size:          {len(ill)}\n")
        f.write(f"    Δ mean ± SD:          {np.mean(ill):.3f} ± {np.std(ill, ddof=1):.3f} years\n\n")
        f.write(f"  Between-group difference (Healthy − Disease): {delta_diff:.3f} years\n")
        f.write(f"  Cohen's d:                    {cohens_d:.3f}\n")
        f.write(f"  t = {t_stat:.4f},  p = {p_val:.4e}  {sig}\n")
    print(f"Report saved: {report_path}")

    # ========== 4. Plotting ==========
    print("[Step 4] Generating violin plot...")
    plot_violin(healthy, ill, p_val, cohens_d, args.output_dir)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

