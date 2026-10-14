#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
6_0 Healthy baseline population -> future disease incidence Cox regression (pancreatic age delta and disease risk)

Design:
  - Population: healthy cohort in merged_phenotype.csv (Mars_45_87) with is_healthy==1 and MRI npy
  - Time origin: MRI examination date (exam_date)
  - Event: first diagnosis date of the target ICD disease > exam_date
  - Censoring: administrative censor date = max diagnosis date in the whole dataset 2022-10-31 (endpoint covered by diagnosis records)
         follow-up time = (min(diagnosis, censor) - exam_date)/365.25 years
         include only subjects with exam_date < censor date (follow-up time > 0)
  - Per-disease handling: those already diagnosed before examination (onset<=exam) are treated as prevalent, and excluded from that disease's analysis
  - Exposure: delta = corrected predicted pancreatic age - actual age (age_pred_bc - age_use)
  - Model A: delta univariable;  Model B: delta + age + sex + BMI + waist + mean systolic BP + mean diastolic BP

Output (cohort_tables/cox_analysis/):
  survival_data_<disease>.csv        per-disease survival analysis dataset
  cox_results.csv                    HR(95%CI) and p for all disease/model/covariate
  cox_summary.txt                    follow-up time and event count summary
  km_<disease>.{png,svg,pdf,tiff}    Kaplan-Meier curves grouped by delta tertile
  forest_plot.{png,svg,pdf,tiff}     per-disease HR forest plot for delta (adjusted)
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

DEFAULT_PHENO = os.path.join(OUT_ROOT, "Disease_Phenotype_Analysis", "merged_phenotype.csv")
DEFAULT_OUT = os.path.join(OUT_ROOT, "cohort_tables", "cox_analysis")

# Administrative censor date = max diagnosis record date in the whole dataset
ADMIN_CENSOR = pd.Timestamp("2022-10-31")

# Target disease ICD-10 prefixes -> display name (metabolic/digestive related)
TARGETS = [
    ("diabetes", "Diabetes (E10–E14)", ["E10", "E11", "E12", "E13", "E14"]),
    ("pancreatic_cancer", "Pancreatic cancer (C25)", ["C25"]),
    ("pancreatitis", "Pancreatitis (K85/K86)", ["K85", "K86"]),
    ("obesity", "Obesity (E66)", ["E66"]),
    ("hypertension", "Hypertension (I10/I15)", ["I10", "I15"]),
    ("dyslipidemia", "Dyslipidemia (E78)", ["E78"]),
    ("biliary", "Biliary disease (K80–K83)", ["K80", "K81", "K82", "K83"]),
    ("fatty_liver", "Fatty liver (K76.0)", ["K76.0"]),
    ("gout", "Gout (M10)", ["M10"]),
]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _rcparams():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def save_pub(fig, filename):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=600, bbox_inches="tight")


def _parse_codes(s):
    if pd.isna(s):
        return []
    out = []
    for it in str(s).split("|"):
        m = re.match(r"([A-Z]\d{2}(?:\.\d)?)", it.strip())
        if m:
            out.append(m.group(1))
    return out


def _build_disease_dataset(pheno, key, label, prefixes):
    """Per-disease survival dataset: exclude prevalent, event=first diagnosis>exam, censoring=administrative censoring."""
    df = pheno.copy()
    onset = df[["p41280_a%d" % i for i in range(240) if ("p41280_a%d" % i) in df.columns]].apply(pd.to_datetime, errors="coerce").values
    first_onset, event = [], []
    for r, codes in enumerate(df["_codes"]):
        fo = None
        for j, c in enumerate(codes):
            if any(c.startswith(p) for p in prefixes):
                d = onset[r, j]
                if not pd.isna(d) and (fo is None or d < fo):
                    fo = d
        first_onset.append(fo)
        event.append(0 if fo is None else 1)
    df["first_onset"] = first_onset
    df["_event"] = event

    df["time"] = ((df["first_onset"] - df["exam"]).dt.days / 365.25).where(df["_event"] == 1,
                                                                           (ADMIN_CENSOR - df["exam"]).dt.days / 365.25)

    # Exclude: already diagnosed before/on examination day (prevalent), or follow-up time <= 0
    at_risk = df[(df["_event"] == 0) | (df["first_onset"] > df["exam"])].copy()
    at_risk = at_risk[(at_risk["time"] > 0) & at_risk["exam"].notna()].copy()
    at_risk["event"] = at_risk["_event"].astype(int)
    at_risk["delta"] = at_risk["delta"].astype(float)
    at_risk = at_risk.reset_index(drop=True)
    return at_risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheno", default=DEFAULT_PHENO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    _rcparams()

    # --- Load data & define healthy analysis population ---
    pheno = pd.read_csv(args.pheno, low_memory=False)
    pheno = pheno[pheno["is_healthy"] == 1].copy()
    pheno["exam"] = pd.to_datetime(pheno["_exam_date"], errors="coerce")
    pheno = pheno[pheno["exam"] < ADMIN_CENSOR].copy()
    pheno["_codes"] = pheno["icd-10"].apply(_parse_codes)
    # Covariates
    for c in ["delta", "age_use", "bmi", "Waist circumference", "sbp_0", "sbp_1", "dbp_0", "dbp_1"]:
        pheno[c] = pd.to_numeric(pheno[c], errors="coerce")
    pheno["sex_male"] = (pheno["sex"] == "Male").astype(int)
    pheno["sbp_mean"] = pheno[["sbp_0", "sbp_1"]].mean(axis=1, skipna=True)
    pheno["dbp_mean"] = pheno[["dbp_0", "dbp_1"]].mean(axis=1, skipna=True)
    print(f"healthy at-risk population (is_healthy==1, exam<{ADMIN_CENSOR.date()}): {len(pheno)}")

    # delta tertiles (common cutoffs for KM comparison)
    q1, q2 = pheno["delta"].quantile([1 / 3, 2 / 3])
    pheno["delta_grp"] = pd.cut(pheno["delta"], [-np.inf, q1, q2, np.inf], labels=["Low", "Mid", "High"])

    cox_rows, summaries = [], []

    # --- Per-disease Cox analysis ---
    for key, label, prefixes in TARGETS:
        d = _build_disease_dataset(pheno, key, label, prefixes)
        covs = ["delta"]
        d["age"] = d["age_use"]
        d["waist"] = d["Waist circumference"]
        d["sbp"] = d["sbp_mean"]
        d["dbp"] = d["dbp_mean"]
        base_cols = ["time", "event", "delta", "age", "sex_male", "bmi", "waist", "sbp", "dbp", "delta_grp"]
        model_df = d[base_cols].dropna(subset=["time", "event", "delta"]).copy()
        summaries.append(dict(disease=label, n_at_risk=len(d), n_events=int(d["event"].sum()),
                              person_years=round(float(d["time"].sum()), 1),
                              median_followup_y=round(float(d["time"].median()), 2)))

        d.to_csv(os.path.join(args.out, f"survival_data_{key}.csv"), index=False)

        for model_name, formula in [
            ("A_delta", "delta"),
            ("B_adjusted", "delta + age + sex_male + bmi + waist + sbp + dbp"),
        ]:
            md = model_df.dropna(subset=[x.split(" ")[0] for x in formula.split(" + ")])
            if len(md) == 0 or md["event"].sum() == 0:
                continue
            py = md["time"].sum()
            cph = CoxPHFitter()
            cph.fit(md[["time", "event"] + [x.split(" ")[0] for x in formula.split(" + ")]],
                    duration_col="time", event_col="event", formula=formula)
            for cov in [x.split(" ")[0] for x in formula.split(" + ")]:
                row = cph.summary.loc[cov]
                cox_rows.append(dict(
                    disease=label, disease_key=key, model=model_name,
                    n_at_risk=len(md), n_events=int(md["event"].sum()), person_years=round(py, 1),
                    covariate=cov,
                    hr=row["exp(coef)"], hr_lower=row["exp(coef) lower 95%"], hr_upper=row["exp(coef) upper 95%"],
                    p=row["p"],
                ))
            print(f"{label:32s} [{model_name}] events={md['event'].sum():5d} n={len(md):6d} "
                  f"delta HR={cph.summary.loc['delta','exp(coef)']:.3f} p={cph.summary.loc['delta','p']:.3g}")

    cox_df = pd.DataFrame(cox_rows)
    cox_df.to_csv(os.path.join(args.out, "cox_results.csv"), index=False)

    summ = pd.DataFrame(summaries)
    summ.to_csv(os.path.join(args.out, "cox_summary.csv"), index=False)
    with open(os.path.join(args.out, "cox_summary.txt"), "w") as f:
        f.write(f"Healthy at-risk population: {len(pheno)} (is_healthy==1, exam < {ADMIN_CENSOR.date()})\n")
        f.write(f"Administrative censor date: {ADMIN_CENSOR.date()} (max diagnosis record date)\n\n")
        f.write(summ.to_string(index=False))

    print(f"\nresults -> {args.out}")
    print(summ.to_string(index=False))

    # --- KM curves (delta tertiles) ---
    for key, label, _ in TARGETS:
        d = pd.read_csv(os.path.join(args.out, f"survival_data_{key}.csv"))
        d = d.dropna(subset=["time", "event", "delta_grp"])
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        for gi, g in enumerate(["Low", "Mid", "High"]):
            sub = d[d["delta_grp"] == g]
            kmf = KaplanMeierFitter()
            kmf.fit(sub["time"], sub["event"], label=g)
            kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=1.4, color=PALETTE[gi])
        ax.set_xlabel("Years since MRI examination")
        ax.set_ylabel("Survival probability (event-free)")
        ax.set_title(label, fontsize=8)
        ax.legend(title="Pancreatic-age\ndelta (tertile)", loc="lower left", fontsize=6.5, title_fontsize=6.5)
        fig.tight_layout()
        save_pub(fig, os.path.join(args.out, f"km_{key}"))
        plt.close(fig)

    # --- Forest plot: delta HR from the adjusted model (log axis symmetric around 1, covering all points) ---
    adj = cox_df[(cox_df["covariate"] == "delta") & (cox_df["model"] == "B_adjusted")].copy()
    adj = adj.iloc[::-1].reset_index(drop=True)
    half = np.log(max(adj["hr_upper"].max(), 1.0 / adj["hr_lower"].min()))
    x_lo, x_hi = np.exp(-half * 1.06), np.exp(half * 1.06)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ypos = np.arange(len(adj))
    for i, r in adj.iterrows():
        ax.errorbar(r["hr"], i, xerr=[[r["hr"] - r["hr_lower"]], [r["hr_upper"] - r["hr"]]],
                    fmt="s", ms=4, color=PALETTE[0], capsize=2, linewidth=1.0, ecolor=PALETTE[0])
    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{d} (n={int(n)})" for d, n in zip(adj["disease"], adj["n_events"])], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio for delta (per 1 year, adjusted)", fontsize=7)
    ax.set_xlim(x_lo, x_hi)
    ax.set_title("Pancreatic-age delta vs. future disease risk", fontsize=8)
    for i, r in adj.iterrows():
        ax.annotate(f"{r['hr']:.2f} [{r['hr_lower']:.2f}, {r['hr_upper']:.2f}]", xy=(r["hr"], i),
                    xytext=(6, 0), textcoords="offset points", fontsize=6, va="center")
    fig.tight_layout()
    save_pub(fig, os.path.join(args.out, "forest_plot"))
    plt.close(fig)

    print("figures saved: km_<disease>.{svg,pdf,tiff}, forest_plot.{svg,pdf,tiff}")


if __name__ == "__main__":
    main()