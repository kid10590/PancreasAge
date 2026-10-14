#!/usr/bin/env python3
"""
2_8_dose_response_diabetes.py — Dose-response analysis of pancreatic age gap and prediabetes/type 2 diabetes risk

Research question: Is pancreatic aging (pancreatic age gap) dose-dependently related to glucose metabolic status?

Design:
  Dependent variable: ogtt_2h_glucose (3-category ordinal: 0=normal, 1=prediabetes, 2=T2D)
  Independent variable: pancreatic age gap delta = age_pred_bc - age_gt, 4 categories:
            - Category 1: delta < 0 (no aging, reference group)
            - Categories 2-4: delta > 0 split into tertiles (low/medium/high accelerated aging)
  Model:  Ordinal logistic regression (proportional odds)
           Model 1 (M1): age + sex + bmi
           Model 2 (M2): M1 + fasting_insulin + CHOL + TG + HDL + ALT + AST + CREA
           Model 3 (M3): M2 + mean_pancreatic_fat + anterior_SC_fat (exploratory)
                       Fat metrics are compressed (head/body/tail subregions removed to avoid collinearity/overfitting)

Missing data handling (revised 2026-09-11):
  - sex coding: 1=male, 2=female, 0=missing (27/210 missing in this dataset)
  - Main analysis (scheme=cc): complete-case
      · Exclude sex=0 (27 subjects), and exclude any row with a missing covariate
      · M1 n=183, M2/M3 n=175
  - Sensitivity A (scheme=meanimpute): exclude sex=0, mean-impute continuous covariates (n=183)
  - Sensitivity B (scheme=sexmissing): keep sex=0 missing subjects, use sex_f + sex_missing
      dummy variables (missing-indicator), complete-case for continuous covariates (n=201)
  - Output n_model / OGTT three-class counts / missing count per covariate / VIF / EPV for each model
  - All schemes: exposure and dependent variable are not imputed

Usage:
  python code/02_downstream/2_8_dose_response_diabetes.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import DATA_ROOT, OUT_ROOT, INFER_DIR  # noqa: E402
PRED_CSV = os.path.join(INFER_DIR, "predictions_ext_part3.csv")
META_CSV = os.path.join(DATA_ROOT, "obesity_cohort", "metadata_norm.csv")
OUT_DIR  = os.path.join(OUT_ROOT, "Dose_Response_Diabetes")

# ── Variable mapping (table column → model variable name) ──
# sex recoded as sex_f (female=1) + sex_missing (sex=0 missing)
COVAR_BASE = ["age", "sex_f", "bmi"]
COVAR_BASE_SM = ["age", "sex_f", "sex_missing", "bmi"]   # missing-indicator version
COVAR_M2 = ["fasting_insulin", "total_cholesterol", "triglyceride",
            "hdl_cholesterol", "alanine_aminotransferase",
            "aspartate_aminotransferase", "creatinine"]
# M3 fat metrics compressed: only whole-pancreas mean fat + anterior abdominal subcutaneous fat (head/body/tail subregions removed)
COVAR_M3 = ["mean_pancreatic_fat_percent", "anterior_subcutaneous_fat_thickness"]

SCHEMES = {
    "cc": "Main analysis (complete-case)",
    "meanimpute": "Sensitivity A (mean imputation for continuous covariates)",
    "sexmissing": "Sensitivity B (sex missing-indicator)",
}

# Display names for output variables
VAR_LABEL = {
    "age": "Age", "sex_f": "Sex (female)", "sex_missing": "Sex missing", "bmi": "BMI",
    "fasting_insulin": "Fasting insulin",
    "total_cholesterol": "Total cholesterol", "triglyceride": "TG",
    "hdl_cholesterol": "HDL-C", "alanine_aminotransferase": "ALT",
    "aspartate_aminotransferase": "AST", "creatinine": "Creatinine",
    "mean_pancreatic_fat_percent": "Mean pancreatic fat (%)",
    "anterior_subcutaneous_fat_thickness": "Anterior SC fat thickness",
    "pancreatic_head_fat_percent": "Pancreatic head fat (%)",
    "pancreatic_body_fat_percent": "Pancreatic body fat (%)",
    "pancreatic_tail_fat_percent": "Pancreatic tail fat (%)",
}


def model_covars(scheme, level):
    base = COVAR_BASE_SM if scheme == "sexmissing" else COVAR_BASE
    if level == "M1":
        return list(base)
    if level == "M2":
        return list(base) + COVAR_M2
    if level == "M3":
        return list(base) + COVAR_M2 + COVAR_M3
    raise ValueError(level)


# ═══════════════════════════════════════
# Step 1: Data merge and cleaning
# ═══════════════════════════════════════
def load_and_merge():
    print("=" * 60)
    print(" Step 1: Data merge")
    print("=" * 60)

    pred = pd.read_csv(PRED_CSV, low_memory=False)
    meta = pd.read_csv(META_CSV, low_memory=False)

    pred["eid"] = pred["eid"].astype(str).str.strip()
    meta["eid"] = meta["eid"].astype(str).str.strip()

    df = pred.merge(meta, on="eid", how="inner")
    print(f"  Predictions: {len(pred)}, clinical: {len(meta)}, merged: {len(df)}")

    # Pancreatic age gap
    df["delta"] = df["age_pred_bc"] - df["age_gt"]
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")

    # Dependent variable
    df["ogtt"] = pd.to_numeric(df["ogtt_2h_glucose"], errors="coerce")
    df = df[df["ogtt"].isin([0, 1, 2])].copy()

    # sex recoding: 1=male, 2=female, 0=missing
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    df["sex_f"] = (df["sex"] == 2).astype(int)
    df["sex_missing"] = (df["sex"] == 0).astype(int)

    # Coerce covariates to numeric
    for c in COVAR_BASE + COVAR_BASE_SM + COVAR_M2 + COVAR_M3:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  Analysis sample: {len(df)}  (ogtt: 0={int((df['ogtt']==0).sum())}, "
          f"1={int((df['ogtt']==1).sum())}, 2={int((df['ogtt']==2).sum())})")
    print(f"  sex missing (sex=0): {int(df['sex_missing'].sum())}")
    return df


# ═══════════════════════════════════════
# Step 1b: Missingness report
# ═══════════════════════════════════════
def missingness_report(df):
    print("\n" + "=" * 60)
    print(" Step 1b: Covariate missingness report")
    print("=" * 60)
    cols = COVAR_BASE + COVAR_M2 + COVAR_M3
    rows = []
    for c in cols:
        if c in df.columns:
            nm = int(df[c].isna().sum())
            rows.append({"variable": c, "n_missing": nm,
                         "pct": round(100 * nm / len(df), 2)})
    out = pd.DataFrame(rows).sort_values("n_missing", ascending=False)
    out.insert(2, "n_total", len(df))
    out.to_csv(os.path.join(OUT_DIR, "missingness_report.csv"), index=False)
    print(out.to_string(index=False))
    print(f"  → missingness_report.csv")
    return out


# ═══════════════════════════════════════
# Step 2: Pancreatic age gap 4-category
# ═══════════════════════════════════════
def build_delta_categories(df):
    print("\n" + "=" * 60)
    print(" Step 2: Pancreatic age gap 4-category")
    print("=" * 60)

    d = df["delta"].dropna()
    n_neg = int((d < 0).sum())
    n_pos = int((d >= 0).sum())
    print(f"  delta<0 (no aging): {n_neg}")
    print(f"  delta>=0: {n_pos} → tertiles")

    pos = d[d >= 0]
    if len(pos) >= 3:
        q1, q2 = pos.quantile([1 / 3, 2 / 3])
        q1, q2 = float(q1), float(q2)
    else:
        q1, q2 = 0.0, 1.0

    print(f"  Cutoffs: {q1:.3f}, {q2:.3f}")

    def _cat(x):
        if pd.isna(x):
            return np.nan
        if x < 0:
            return 0
        if x <= q1:
            return 1
        if x <= q2:
            return 2
        return 3

    df["delta_cat"] = df["delta"].apply(_cat)
    counts = df["delta_cat"].value_counts().sort_index()
    for k in range(4):
        print(f"  Category {k}: n={counts.get(k, 0)}")

    medians = df.groupby("delta_cat")["delta"].median()
    df["delta_cat_median"] = df["delta_cat"].map(medians)
    df["delta_cat_label"] = df["delta_cat"].map({
        0: "No aging\n(<0)", 1: "Low\n", 2: "Medium\n", 3: "High\n"})
    return df, {"q1": q1, "q2": q2, "cutoffs": [q1, q2]}


# ═══════════════════════════════════════
# Step 2b: Prepare analysis data by scheme
# ═══════════════════════════════════════
def prepare_scheme(df, scheme):
    """Return analysis data for the given scheme:
      - cc        : exclude sex=0; complete-case for continuous covariates (dropna at fit time)
      - meanimpute: exclude sex=0; mean-impute continuous covariates
      - sexmissing: keep sex=0 (sex_f + sex_missing dummies); complete-case for continuous covariates
    """
    d = df.copy()
    if scheme in ("cc", "meanimpute"):
        n0 = len(d)
        d = d[d["sex"] != 0].copy()
        print(f"  [{scheme}] Excluding sex=0 missing: {n0} -> {len(d)}")
    if scheme == "meanimpute":
        cont = ["age", "bmi"] + COVAR_M2 + COVAR_M3
        for c in cont:
            if c in d.columns:
                before = int(d[c].isna().sum())
                d[c] = d[c].fillna(d[c].mean())
                if before:
                    print(f"    [mean imputation] {c}: {before} values")
    return d


# ═══════════════════════════════════════
# Step 3: Ordinal logistic regression (supports 3 exposure definitions)
# ═══════════════════════════════════════
def run_ordinal_model(d, covars, name, label, exposure_type="cat4"):
    """
    Ordinal logistic regression (proportional odds).

    exposure_type:
      - "cat4": pancreatic age gap 4-category (delta<0 reference + delta>0 tertiles)
      - "cont": pancreatic age gap as continuous variable (OR per 1-year increase in pancreatic age gap)
      - "bin":  pancreatic age gap binary (delta>=0 aging vs delta<0 no aging)
    """
    # complete-case (applies to schemes cc/sexmissing; meanimpute already filled)
    d = d.copy()
    d["ogtt"] = pd.to_numeric(d["ogtt"], errors="coerce")

    if exposure_type == "cat4":
        d["delta_cat"] = d["delta_cat"].astype(int)
        for k in range(1, 4):
            d[f"delta_cat_{k}"] = (d["delta_cat"] == k).astype(int)
        exog_cols = [f"delta_cat_{k}" for k in range(1, 4)] + list(covars)
        exposure_coefs = [f"delta_cat_{k}" for k in range(1, 4)]
    elif exposure_type == "cont":
        d["delta"] = d["delta"].astype(float)
        exog_cols = ["delta"] + list(covars)
        exposure_coefs = ["delta"]
    elif exposure_type == "bin":
        d["delta_bin"] = (d["delta"] >= 0).astype(int)
        exog_cols = ["delta_bin"] + list(covars)
        exposure_coefs = ["delta_bin"]
    else:
        raise ValueError(f"Unknown exposure_type: {exposure_type}")

    d = d.dropna(subset=["ogtt", "delta"] + list(covars))
    if len(d) < 10:
        print(f"  [{name}] {label}: insufficient sample, skipping")
        return None

    d["sex"] = d["sex"].astype(int)
    d["ogtt"] = d["ogtt"].astype(int)

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    try:
        mod = OrderedModel(d["ogtt"], d[exog_cols], distr="logit")
        res = mod.fit(method="bfgs", disp=False, maxiter=500)
    except Exception as e:
        print(f"    [{name}] Failed: {e}")
        return None

    rows = []
    for coef_name in exposure_coefs:
        if coef_name not in res.params.index:
            continue
        b = res.params[coef_name]
        se = res.bse[coef_name]
        p = res.pvalues[coef_name]
        or_val = np.exp(b)
        or_low = np.exp(b - 1.96 * se)
        or_high = np.exp(b + 1.96 * se)
        if exposure_type == "cat4":
            cat_label = int(coef_name.split("_")[-1])
            n_cat = int((d["delta_cat"] == cat_label).sum())
        elif exposure_type == "bin":
            cat_label = "aging"
            n_cat = int((d["delta_bin"] == 1).sum())
        else:
            cat_label = "per1yr"
            n_cat = len(d)
        rows.append({
            "model": name, "category": cat_label, "n": n_cat,
            "OR": or_val, "OR_low95": or_low, "OR_high95": or_high,
            "beta": b, "se": se, "P": p,
        })

    p_trend, or_trend = np.nan, np.nan
    if exposure_type == "cat4":
        try:
            trend_terms = ["delta_cat"] + list(covars)
            d_trend = d.copy()
            mod_t = OrderedModel(d_trend["ogtt"], d_trend[trend_terms], distr="logit")
            res_t = mod_t.fit(method="bfgs", disp=False, maxiter=200)
            p_trend = res_t.pvalues["delta_cat"]
            or_trend = np.exp(res_t.params["delta_cat"])
        except Exception:
            pass
    else:
        p_trend = rows[0]["P"] if rows else np.nan
        or_trend = rows[0]["OR"] if rows else np.nan

    ogtt_counts = d["ogtt"].value_counts().sort_index().to_dict()
    result = {"rows": rows, "p_trend": p_trend, "or_trend": or_trend,
              "n": len(d), "model": name, "label": label,
              "exposure_type": exposure_type,
              "cat_medians": d.groupby("delta_cat")["delta"].median().to_dict()
                              if "delta_cat" in d.columns else {},
              "ogtt_counts": ogtt_counts,
              "n_sex_missing": int(d["sex_missing"].sum()) if "sex_missing" in d.columns else 0}
    print(f"  [{name}] {label}: n={len(d)}  "
          f"ogtt 0/1/2 = {ogtt_counts.get(0,0)}/{ogtt_counts.get(1,0)}/{ogtt_counts.get(2,0)}")
    return result


# ═══════════════════════════════════════
# Step 3b: VIF and EPV
# ═══════════════════════════════════════
def compute_vif(d, covars, name):
    d = d.dropna(subset=["ogtt", "delta_cat"] + list(covars)).copy()
    if len(d) < 20:
        return pd.DataFrame()
    for k in range(1, 4):
        d[f"delta_cat_{k}"] = (d["delta_cat"] == k).astype(int)
    cols = [f"delta_cat_{k}" for k in range(1, 4)] + list(covars)
    X = sm.add_constant(d[cols], has_constant="add")
    rows = []
    for i, c in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.nan
        rows.append({"model": name, "variable": c, "VIF": round(float(v), 3)})
    return pd.DataFrame(rows)


def compute_epv(d, covars, name):
    d = d.dropna(subset=["ogtt", "delta_cat"] + list(covars)).copy()
    if len(d) < 10:
        return None
    n_params = 3 + len(covars) + 2   # 3 exposure dummies + covariates + 2 intercepts
    events = int(d["ogtt"].value_counts().min())
    return {"model": name, "n": len(d), "n_params": n_params,
            "min_outcome_n": events, "EPV": round(events / n_params, 2)}


# ═══════════════════════════════════════
# Step 4: Results summary + plotting
# ═══════════════════════════════════════
def _draw_forest(ax, all_results, etag):
    """Forest plot (panel a): OR ± 95%CI for each model × exposure definition."""
    y = 0
    labels = []
    for res in all_results:
        if res is None:
            continue
        for r in res["rows"]:
            c = "#2E86C1" if r["P"] < 0.05 else "#90A4AE"
            ax.errorbar(r["OR"], y, xerr=[[r["OR"] - r["OR_low95"]],
                                          [r["OR_high95"] - r["OR"]]],
                        fmt="o", color=c, markersize=7, capsize=3, zorder=3)
            ax.text(r["OR_high95"] + 0.02, y,
                    f"OR={r['OR']:.2f} [{r['OR_low95']:.2f}-{r['OR_high95']:.2f}] P={r['P']:.3f}",
                    va="center", fontsize=7)
            etype_str = etag.get(res.get("exposure_type", "cat4"), "?")
            labels.append(f"{res['label']} [{etype_str}] {r['category']} (n={r['n']})")
            y += 1
        if res.get("exposure_type") == "cat4":
            ax.text(0.02, y - 0.4, f"P-trend = {res.get('p_trend', np.nan):.3f}",
                    fontsize=7, style="italic", color="#333")
        y += 1.2
    ax.axvline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return len(labels)


def _draw_trend(ax, all_results):
    """Dose-response line (panel b): OR curves + CI bands for each cat4 model."""
    drew_any = False
    for res in all_results:
        if res is None or res.get("exposure_type") != "cat4" or not res["rows"]:
            continue
        if not res.get("cat_medians"):
            continue
        ors = [r["OR"] for r in res["rows"]]
        lows = [r["OR_low95"] for r in res["rows"]]
        highs = [r["OR_high95"] for r in res["rows"]]
        ax.plot(range(4), [1] + ors, marker="o", label=res["label"], linewidth=1.5)
        ax.fill_between(range(4), [1] + lows, [1] + highs, alpha=0.15)
        drew_any = True
    if drew_any:
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["No aging", "Low", "Medium", "High"])
        ax.set_ylabel("OR for worse glucose status")
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_results(all_results, out_dir, tag=""):
    os.makedirs(out_dir, exist_ok=True)

    etag = {"cat4": "4-cat", "cont": "continuous", "bin": "binary"}

    # ── Standalone forest plot (kept) ──
    fig, ax = plt.subplots(figsize=(9, 5 + 1.0 * len(all_results)))
    _draw_forest(ax, all_results, etag)
    ax.set_title("Dose-Response: Pancreatic Age Gap vs Glucose Status")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"forest_plot{tag}.pdf"))
    plt.close()

    # ── Standalone dose-response plot (kept) ──
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    _draw_trend(ax2, all_results)
    ax2.set_title("Dose-Response Trend (4-category)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"dose_response_plot{tag}.pdf"))
    plt.close(fig2)

    # ── Combined figure: (a) forest plot | (b) dose-response trend ──
    n_rows = sum(len(res["rows"]) if res else 0 for res in all_results)
    fig3, (ax_f, ax_t) = plt.subplots(
        1, 2, figsize=(12, 4 + 0.45 * n_rows),
        gridspec_kw={"width_ratios": [1.4, 1]})
    _draw_forest(ax_f, all_results, etag)
    ax_f.set_title("(a) Forest plot")
    _draw_trend(ax_t, all_results)
    ax_t.set_title("(b) Dose–response trend")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, f"dose_response_combined{tag}.pdf"))
    fig3.savefig(os.path.join(out_dir, f"dose_response_combined{tag}.png"), dpi=300)
    plt.close(fig3)
    print(f"  -> dose_response_combined{tag}.pdf / .png (a=forest plot, b=trend)")


# ═══════════════════════════════════════
# Step 5: Proportional odds assumption diagnostics (based on main analysis CC)
# ═══════════════════════════════════════
def proportional_odds_diagnostic(d, covars, name):
    d = d.dropna(subset=["ogtt", "delta_cat"] + list(covars)).copy()
    if len(d) < 20:
        return pd.DataFrame()

    d["delta_cat"] = d["delta_cat"].astype(int)
    exposure_cols = []
    for k in range(1, 4):
        col = f"delta_cat_{k}"
        d[col] = (d["delta_cat"] == k).astype(int)
        exposure_cols.append(col)
    model_cols = exposure_cols + list(covars)
    x = sm.add_constant(d[model_cols], has_constant="add")

    out = []
    fits = {}
    for cut_name, y in {
        "ogtt_ge1_vs_0": (d["ogtt"] >= 1).astype(int),
        "ogtt_eq2_vs_le1": (d["ogtt"] == 2).astype(int),
    }.items():
        try:
            fits[cut_name] = sm.Logit(y, x).fit(disp=0)
        except Exception as exc:
            print(f"  [PO diagnostic skipped] {name} {cut_name}: {exc}")
            return pd.DataFrame()

    for col in exposure_cols:
        m1, m2 = fits["ogtt_ge1_vs_0"], fits["ogtt_eq2_vs_le1"]
        b1, se1 = m1.params[col], m1.bse[col]
        b2, se2 = m2.params[col], m2.bse[col]
        z = (b1 - b2) / np.sqrt(se1 ** 2 + se2 ** 2)
        p_diff = 2 * stats.norm.sf(abs(z))
        out.append({
            "model": name, "exposure": col,
            "beta_ge1_vs_0": b1, "se_ge1_vs_0": se1,
            "beta_eq2_vs_le1": b2, "se_eq2_vs_le1": se2,
            "beta_difference": b1 - b2, "approx_p_difference": p_diff,
            "po_supported_approx": int(p_diff >= 0.05), "n": len(d),
        })
    return pd.DataFrame(out)


# ═══════════════════════════════════════
# Main function
# ═══════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_and_merge()
    df, cat_info = build_delta_categories(df)
    with open(os.path.join(OUT_DIR, "delta_categories.json"), "w") as f:
        json.dump(cat_info, f, indent=2)

    missingness_report(df)

    models = [
        ("M1", "Model 1 (basic)"),
        ("M2", "Model 2 (metabolic)"),
        ("M3", "Model 3 (final, exploratory)"),
    ]
    exposure_types = [
        ("cat4", "4-category"),
        ("cont", "continuous"),
        ("bin", "binary"),
    ]

    all_rows = []
    per_scheme_results = {}

    for scheme in ["cc", "meanimpute", "sexmissing"]:
        print("\n" + "=" * 60)
        print(f" Analysis scheme: [{scheme}] {SCHEMES[scheme]}")
        print("=" * 60)
        d = prepare_scheme(df, scheme)
        results = []
        for name, label in models:
            covars = model_covars(scheme, name)
            for etype, etag in exposure_types:
                res = run_ordinal_model(d, covars, name, label, exposure_type=etype)
                if res is not None:
                    results.append(res)
                    for r in res["rows"]:
                        all_rows.append({**r, "scheme": scheme,
                                         "model_label": label,
                                         "exposure_type": etype,
                                         "p_trend": res["p_trend"],
                                         "n_model": res["n"],
                                         "ogtt0": res["ogtt_counts"].get(0, 0),
                                         "ogtt1": res["ogtt_counts"].get(1, 0),
                                         "ogtt2": res["ogtt_counts"].get(2, 0),
                                         "n_sex_missing": res["n_sex_missing"]})
        per_scheme_results[scheme] = results
        plot_results(results, OUT_DIR, tag=f"_{scheme}")

    if all_rows:
        df_res = pd.DataFrame(all_rows)
        df_res.to_csv(os.path.join(OUT_DIR, "dose_response_results.csv"), index=False)
        print(f"\n[Saved] dose_response_results.csv ({len(df_res)} rows, including scheme)")

    # ── Sensitivity comparison table: cat4 High (cat3) OR and p_trend ──
    print("\n" + "=" * 60)
    print(" Sensitivity comparison (cat4, High category OR and P-trend)")
    print("=" * 60)
    comp_rows = []
    for scheme in ["cc", "meanimpute", "sexmissing"]:
        for name, _ in models:
            for res in per_scheme_results[scheme]:
                if res.get("exposure_type") != "cat4" or res["model"] != name:
                    continue
                cat3 = next((r for r in res["rows"] if r["category"] == 3), None)
                comp_rows.append({
                    "scheme": scheme, "model": name,
                    "n_model": res["n"],
                    "ogtt0": res["ogtt_counts"].get(0, 0),
                    "ogtt1": res["ogtt_counts"].get(1, 0),
                    "ogtt2": res["ogtt_counts"].get(2, 0),
                    "n_sex_missing": res["n_sex_missing"],
                    "High_OR": cat3["OR"] if cat3 else np.nan,
                    "High_OR_low95": cat3["OR_low95"] if cat3 else np.nan,
                    "High_OR_high95": cat3["OR_high95"] if cat3 else np.nan,
                    "High_P": cat3["P"] if cat3 else np.nan,
                    "p_trend": res["p_trend"],
                })
                break
    if comp_rows:
        comp = pd.DataFrame(comp_rows)
        comp.to_csv(os.path.join(OUT_DIR, "dose_response_sensitivity_comparison.csv"), index=False)
        print(comp.to_string(index=False))
        print(f"[Saved] dose_response_sensitivity_comparison.csv")

    # ── VIF and EPV (based on main analysis CC) ──
    d_cc = prepare_scheme(df, "cc")
    vif_rows, epv_rows = [], []
    for name, _ in models:
        covars = model_covars("cc", name)
        v = compute_vif(d_cc, covars, name)
        if not v.empty:
            vif_rows.append(v)
        e = compute_epv(d_cc, covars, name)
        if e:
            epv_rows.append(e)
    if vif_rows:
        vif_df = pd.concat(vif_rows, ignore_index=True)
        vif_df.to_csv(os.path.join(OUT_DIR, "vif_report.csv"), index=False)
        print(f"\n[Saved] vif_report.csv (based on main analysis CC)")
        print(vif_df[vif_df["VIF"] >= 5].to_string(index=False))
    if epv_rows:
        epv_df = pd.DataFrame(epv_rows)
        epv_df.to_csv(os.path.join(OUT_DIR, "epv_report.csv"), index=False)
        print(f"[Saved] epv_report.csv")
        print(epv_df.to_string(index=False))

    # ── Proportional odds diagnostics (main analysis CC) ──
    po_rows = []
    for name, _ in models:
        covars = model_covars("cc", name)
        po = proportional_odds_diagnostic(d_cc, covars, name)
        if not po.empty:
            po_rows.append(po)
    if po_rows:
        po_df = pd.concat(po_rows, ignore_index=True)
        po_df.to_csv(os.path.join(OUT_DIR, "po_diagnostic.csv"), index=False)
        unsupported = int((po_df["po_supported_approx"] == 0).sum())
        print(f"[Saved] po_diagnostic.csv ({len(po_df)} rows, approx. non-parallel terms={unsupported})")

    # ── Sensitivity analysis: BMI>=30 (main analysis CC) ──
    print("\n" + "=" * 60)
    print(" Sensitivity analysis: BMI >= 30 (scheme cc)")
    print("=" * 60)
    df_ob = d_cc[pd.to_numeric(d_cc["bmi"], errors="coerce") >= 30].copy()
    print(f"  BMI>=30 sample: {len(df_ob)}")
    ob_rows = []
    ob_results = []
    for name, label in models:
        covars = model_covars("cc", name)
        for etype, etag in exposure_types:
            res = run_ordinal_model(df_ob, covars, name + "_ob", label + " (BMI>=30)",
                                    exposure_type=etype)
            if res is not None:
                ob_results.append(res)
                for r in res["rows"]:
                    ob_rows.append({**r, "model_label": label,
                                    "exposure_type": etype,
                                    "p_trend": res["p_trend"],
                                    "n_model": res["n"],
                                    "ogtt0": res["ogtt_counts"].get(0, 0),
                                    "ogtt1": res["ogtt_counts"].get(1, 0),
                                    "ogtt2": res["ogtt_counts"].get(2, 0),
                                    "n_sex_missing": res["n_sex_missing"]})
    if ob_rows:
        df_ob_res = pd.DataFrame(ob_rows)
        df_ob_res.to_csv(os.path.join(OUT_DIR, "sensitivity_bmi30_results.csv"), index=False)
        print(f"[Saved] sensitivity_bmi30_results.csv")
    plot_results(ob_results, OUT_DIR, tag="_bmi30")

    # ── Save analysis data ──
    df.to_csv(os.path.join(OUT_DIR, "merged_analysis_data.csv"), index=False, encoding="utf-8-sig")
    print(f"[Saved] merged_analysis_data.csv")

    print(f"\nDone! All outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()