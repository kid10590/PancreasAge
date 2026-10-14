import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from paths import OUT_ROOT  # noqa: E402

DEFAULT_CSV = os.path.join(OUT_ROOT, "Disease_Phenotype_Analysis", "merged_phenotype.csv")
DEFAULT_OUTPUT_ROOT = os.path.join(OUT_ROOT, "Disease_Phenotype_Analysis")

# -----------------------------
# Core analysis pipeline
# -----------------------------


def fmt_p(p: float) -> str:
    """Format p-value as scientific notation with two decimals."""
    if pd.isna(p):
        return "nan"
    return f"{p:.2e}"


def load_base_data(csv_path: str, split_json: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["_exam_date"] = df["date"]
    df["eid"] = df["eid"].astype(str)

    # Filter using the split column already present in the CSV
    valid_splits = {"ukb_test", "ukb_test_ill"}
    if "split" in df.columns:
        df = df[df["split"].isin(valid_splits)].copy()
        df["split_group"] = df["split"]
        print(f"[Data] Filtered split in {valid_splits}: {len(df)} rows")
    else:
        print("Warning: no split column, using full dataset")

    df["age_use"] = df["age_gt"]
    # bc_col = "age_pred_bc_self" if "age_pred_bc_self" in df.columns else "age_pred_bc"
    bc_col = "age_pred_bc"
    df["delta"] = df[bc_col] - df["age_gt"]

    # ponytail: robust sex_code — compatible with 0/1 numeric, "Male"/"Female" strings, NaN
    # print(df["sex"].dtype)
    if df["sex"].dtype == "O" or df["sex"].dtype == "str":
        df["sex_code"] = df["sex"].map({"Male": 1, "Female": 0, "M": 1, "F": 0})
        # If the map yields all NaN, fall back to numeric conversion (strings "1", "0", etc.)
        if df["sex_code"].isna().all() and df["sex"].notna().any():
            df["sex_code"] = pd.to_numeric(df["sex"], errors="coerce")
    else:
        df["sex_code"] = pd.to_numeric(df["sex"], errors="coerce")
    # Clip sex_code to 0/1
    df["sex_code"] = df["sex_code"].clip(0, 1)
    return df


class GenericPhenotypeAnalysis:
    def __init__(self, disease_name: str, df: pd.DataFrame, output_root: str, adjust_bmi: bool = True):
        self.disease_name = disease_name
        self.df = df
        self.output_path = os.path.join(output_root, disease_name)
        # Obesity is defined by a BMI threshold, so it cannot be matched/adjusted by BMI (otherwise no common support region).
        self.adjust_bmi = adjust_bmi
        os.makedirs(self.output_path, exist_ok=True)
        self.cohort: pd.DataFrame | None = None
        self.matched_df: pd.DataFrame | None = None
        self.main_model = None

    def define_cohort(self, logic_func: Callable[[pd.Series], float], control_mask: pd.Series | None = None) -> bool:
        self.df["phenotype_status"] = self.df.apply(logic_func, axis=1)
        self.df["is_case"] = self.df["phenotype_status"]
        if control_mask is not None:
            control_mask = control_mask.reindex(self.df.index).fillna(False).astype(bool)
            keep = (self.df["is_case"] == 1) | ((self.df["is_case"] == 0) & control_mask)
            cohort_source = self.df[keep].copy()
        else:
            cohort_source = self.df
        required = ["is_case", "age_use", "sex_code", "delta"]
        if self.adjust_bmi:
            required.append("bmi")
        self.cohort = cohort_source.dropna(subset=required).copy()
        n_case = int((self.cohort["is_case"] == 1).sum())
        n_ctrl = int((self.cohort["is_case"] == 0).sum())
        print(f"[Cohort] {self.disease_name}: {n_case} cases / {n_ctrl} controls")
        self.cohort.to_csv(os.path.join(self.output_path, "organized_data.csv"), index=False)
        return n_case >= 5 and n_ctrl >= 5

    def run_propensity_matching(self, caliper: float = 0.2):
        data = self.cohort.copy()
        data["intercept"] = 1.0
        psm_cols = ["is_case", "age_use", "sex_code"] + (["bmi"] if self.adjust_bmi else [])
        for col in psm_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=psm_cols)
        print(f"[PSM] Valid samples: {len(data)} (original cohort: {len(self.cohort)})")

        case_df = data[data["is_case"] == 1].copy()
        ctrl_df = data[data["is_case"] == 0].copy()
        n_case, n_ctrl = len(case_df), len(ctrl_df)
        if n_case < 5 or n_ctrl < 5:
            print(f"[PSM] Insufficient cases or controls (case={n_case}, ctrl={n_ctrl}), skipping")
            self.matched_df = None
            return

        # ── 1. Fit propensity score (logistic regression) ──
        psm_design = ["intercept", "age_use", "sex_code"] + (["bmi"] if self.adjust_bmi else [])
        model = sm.Logit(data["is_case"], data[psm_design]).fit(disp=0)
        
        # ✅ Fix: correctly compute logit(PS)
        case_df["ps"] = model.predict(case_df[psm_design])
        ctrl_df["ps"] = model.predict(ctrl_df[psm_design])
        # Clip extreme probabilities to avoid log(0) or log(∞)
        eps = 1e-8
        case_df["lp"] = np.log(np.clip(case_df["ps"], eps, 1 - eps) / (1 - np.clip(case_df["ps"], eps, 1 - eps)))
        ctrl_df["lp"] = np.log(np.clip(ctrl_df["ps"], eps, 1 - eps) / (1 - np.clip(ctrl_df["ps"], eps, 1 - eps)))

        # ── 2. KDTree matching ──
        if n_case <= n_ctrl:
            query_df, pool_df = case_df, ctrl_df
        else:
            query_df, pool_df = ctrl_df, case_df

        all_lp = np.concatenate([query_df["lp"].values, pool_df["lp"].values])
        radius = caliper * float(np.std(all_lp))

        k_max = min(200, len(pool_df))
        nn = NearestNeighbors(n_neighbors=k_max, algorithm="kd_tree")
        nn.fit(pool_df[["lp"]].values)
        distances, indices = nn.kneighbors(query_df[["lp"]].values)

        # Sort query by best distance (match harder-to-match samples first, i.e. those with larger nearest-neighbor distance first)
        best_d = distances[:, 0]
        query_order = np.argsort(-best_d)

        matched_query_idx = []
        matched_pool_idx = []
        used_pool = set()
        n_skip_cal, n_skip_used = 0, 0

        for qi in query_order:
            found = False
            for k in range(k_max):
                pi = indices[qi, k]
                d = distances[qi, k]
                if d > max(radius, 1e-8):
                    break
                if pi in used_pool:
                    continue
                matched_query_idx.append(qi)
                matched_pool_idx.append(pi)
                used_pool.add(pi)
                found = True
                break
            if not found:
                if distances[qi, 0] > max(radius, 1e-8):
                    n_skip_cal += 1
                else:
                    n_skip_used += 1

        matched_query = [query_df.iloc[qi] for qi in matched_query_idx]
        matched_pool = [pool_df.iloc[pi] for pi in matched_pool_idx]

        self.matched_df = pd.concat([pd.DataFrame(matched_query), pd.DataFrame(matched_pool)], axis=0)
        self.matched_df.to_csv(os.path.join(self.output_path, "matched_data.csv"), index=False)
        print(f"[PSM] matched: {len(matched_query)} pairs "
            f"(caliper={caliper}×σ={radius:.4f}, "
            f"skipped_caliper={n_skip_cal}, skipped_used={n_skip_used}, "
            f"query={len(query_df)}, pool={len(pool_df)})")

    def run_regression(self):
        # Center age (remove collinearity between age and age²)
        age_mean = float(self.matched_df["age_use"].mean())
        self.matched_df["age_c"] = self.matched_df["age_use"] - age_mean
        self.matched_df["age_c_sq"] = self.matched_df["age_c"] ** 2

        bmi_term = " + bmi" if self.adjust_bmi else ""
        self.main_model = smf.ols(f"delta ~ is_case + age_c + age_c_sq + sex_code{bmi_term}", data=self.matched_df).fit()

        # Female subset
        female_df = self.matched_df[self.matched_df["sex_code"] == 0].copy()
        self.female_model = smf.ols(f"delta ~ is_case + age_c + age_c_sq{bmi_term}", data=female_df).fit()

        # Male subset
        male_df = self.matched_df[self.matched_df["sex_code"] == 1].copy()
        self.male_model = smf.ols(f"delta ~ is_case + age_c + age_c_sq{bmi_term}", data=male_df).fit()

        with open(os.path.join(self.output_path, "stats_summary.txt"), "w") as f:
            f.write(self.main_model.summary().as_text())
            f.write("\n\nExact P-values:\n")
            for k, v in self.main_model.pvalues.items():
                f.write(f"{k}: {v}\n")
            f.write("\n\nFemale model:\n")
            f.write(self.female_model.summary().as_text())
            f.write("\n\nExact P-values:\n")
            for k, v in self.female_model.pvalues.items():
                f.write(f"{k}: {v}\n")
            f.write("\n\nMale model:\n")
            f.write(self.male_model.summary().as_text())
            f.write("\n\nExact P-values:\n")
            for k, v in self.male_model.pvalues.items():
                f.write(f"{k}: {v}\n")

        print(self.main_model.summary())
        print("\n--- Female subset ---")
        print(self.female_model.summary())
        print("\n--- Male subset ---")
        print(self.male_model.summary())

    def _fmt_p(self, p: float) -> str:
        """Format p-value: <0.001 for very small, else 3 significant digits."""
        if pd.isna(p):
            return "NA"
        if p < 0.001:
            return "<0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        elif p < 0.05:
            return f"{p:.3f}"
        else:
            return f"{p:.3f}"

    def _save_fig(self, basename: str):
        """Save figure as PDF (primary) + PNG (fallback)."""
        pdf_path = os.path.join(self.output_path, f"{basename}.pdf")
        png_path = os.path.join(self.output_path, f"{basename}.png")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [plot] {pdf_path}")

    def plot_mean_ci(self):
        summary = self.matched_df.groupby("is_case").agg(mean_delta=("delta", "mean"), n=("delta", "count"))
        se = self.matched_df.groupby("is_case")["delta"].apply(lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        ci = 1.96 * se.values
        labels = ["Healthy Controls", self.disease_name]
        positions = np.arange(len(summary))

        fig, ax = plt.subplots(figsize=(5, 4.5))
        colors = ["#2E86C1", "#E74C3C"]
        for i, (pos, color) in enumerate(zip(positions, colors)):
            ax.errorbar(pos, summary["mean_delta"].iloc[i],
                        yerr=ci[i], fmt="o", color=color, ecolor="gray",
                        capsize=6, markersize=10, markeredgewidth=1.5,
                        markeredgecolor="white", zorder=3)

        ax.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Pancreatic Age Gap (years)", fontsize=11)
        ax.set_title(f"Mean Age Gap by {self.disease_name} Status", fontsize=12, fontweight="bold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._save_fig("mean_ci_plot")

    def plot_forest(self):
        params = self.main_model.params
        conf = self.main_model.conf_int()
        pvals = self.main_model.pvalues

        # Term name pretty-printing map
        term_map = {
            "is_case": self.disease_name,
            "age_c": "Age",
            "age_c_sq": "Age²",
            "sex_code": "Sex",
            "bmi": "BMI",
            "Intercept": "Intercept",
        }

        forest_df = pd.DataFrame({
            "term": [term_map.get(t, t) for t in params.index],
            "coef": params.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
            "pval": pvals.values,
        })
        forest_df = forest_df[forest_df["term"] != "Intercept"].iloc[::-1]

        # Dynamic height
        row_h = 0.9
        fig, ax = plt.subplots(figsize=(11, len(forest_df) * row_h + 2.2))

        y_pos = np.arange(len(forest_df))
        ax.errorbar(forest_df["coef"], y_pos,
                    xerr=[forest_df["coef"] - forest_df["ci_lower"],
                          forest_df["ci_upper"] - forest_df["coef"]],
                    fmt="o", markersize=10, color="#2E86C1", ecolor="#5D6D7E",
                    elinewidth=2.5, capsize=6, markeredgewidth=1.5,
                    markeredgecolor="white", zorder=3)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

        # X-axis range + text space
        x_min = min(forest_df["ci_lower"].min(), 0)
        x_max = forest_df["ci_upper"].max()
        x_range = max(x_max - x_min, 0.2)
        xlim_left  = x_min - x_range * 0.10
        xlim_right = x_max + x_range * 0.75
        ax.set_xlim(xlim_left, xlim_right)

        text_x_coef = x_max + x_range * 0.08
        text_x_pval = x_max + x_range * 0.50

        # Header
        header_y = len(forest_df) - 0.2
        ax.text(text_x_coef, header_y, "β [95% CI]", fontweight="bold",
                ha="left", va="center", fontsize=11)
        ax.text(text_x_pval, header_y, "P", fontweight="bold",
                ha="left", va="center", fontsize=11)

        # Data rows
        for y, row in zip(y_pos, forest_df.itertuples()):
            coef_str = f"{row.coef:.2f} [{row.ci_lower:.2f}, {row.ci_upper:.2f}]"
            p_str = self._fmt_p(row.pval)
            # Bold if significant
            fw = "bold" if row.pval < 0.05 else "normal"
            ax.text(text_x_coef, y, coef_str, ha="left", va="center",
                    fontsize=10, color="#222222", fontfamily="monospace", fontweight=fw)
            ax.text(text_x_pval, y, p_str, ha="left", va="center",
                    fontsize=10, color="#222222", fontfamily="monospace", fontweight=fw)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(forest_df["term"], fontsize=11)
        ax.set_xlabel("β Coefficient (years)", fontsize=11)
        ax.set_title(f"Association of {self.disease_name} with Pancreatic Age Gap",
                     fontsize=13, fontweight="bold", pad=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.8, len(forest_df) + 0.3)
        ax.tick_params(left=False)

        fig.tight_layout()
        self._save_fig("forest_plot")

    def plot_violin(self):
        if self.matched_df is None or self.matched_df.empty:
            return
        ctrl = self.matched_df.loc[self.matched_df["is_case"] == 0, "delta"].dropna()
        case = self.matched_df.loc[self.matched_df["is_case"] == 1, "delta"].dropna()
        if ctrl.empty and case.empty:
            return

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        data = [ctrl, case]
        positions = [0, 1]
        parts = ax.violinplot(data, positions=positions, widths=0.7,
                               showmeans=True, showextrema=True, showmedians=True)
        colors = ["#2E86C1", "#E74C3C"]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.60)
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(1.2)
        for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if partname in parts:
                parts[partname].set_color("#333333")
                parts[partname].set_linewidth(1.0)

        ax.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Healthy Controls", self.disease_name], fontsize=11)
        ax.set_ylabel("Pancreatic Age Gap (years)", fontsize=11)
        ax.set_title(f"Age Gap Distribution by {self.disease_name} Status",
                     fontsize=12, fontweight="bold", pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        self._save_fig("delta_violin_plot")

    def run_all(self, logic_func: Callable[[pd.Series], float], control_mask: pd.Series | None = None):
        if not self.define_cohort(logic_func, control_mask=control_mask):
            print(f"[Skip] {self.disease_name}: insufficient samples")
            return
        self.run_propensity_matching()
        if self.matched_df is None or self.matched_df.empty:
            print(f"[Skip] {self.disease_name}: PSM failed, skipping subsequent analysis")
            return
        self.run_regression()
        self.plot_mean_ci()
        self.plot_violin()
        self.plot_forest()


# -----------------------------
# Phenotype-specific logic
# -----------------------------

code_regex = re.compile(r"^[A-Za-z]\d{2}(?:\.[A-Za-z0-9]+)?")


def normalize_code(token: str | None) -> str | None:
    if not isinstance(token, str):
        return None
    m = code_regex.match(token.strip().upper())
    return m.group(0) if m else None


# Obesity (E66 / BMI >= 28)
OB_RANGES = [("E", 66, 66)]


def matches_range(code: str | None, ranges: list[tuple[str, int, int]]) -> bool:
    if not code:
        return False
    m = re.match(r"^([A-Z])(\d{2})(?:\..*)?$", code)
    if not m:
        return False
    letter, num = m.group(1), int(m.group(2))
    return any(letter == l and s <= num <= e for l, s, e in ranges)


def check_icd_history(row: pd.Series, ranges: list[tuple[str, int, int]]) -> bool:
    icd_field = row.get("icd-10", "")
    if not isinstance(icd_field, str) or not icd_field.strip():
        return False
    exam_dt = row.get("_exam_date", pd.NaT)
    if pd.isna(exam_dt):
        exam_dt = pd.to_datetime(row.get("date", pd.NaT), errors="coerce")
    if pd.isna(exam_dt):
        return False
    for idx, part in enumerate(icd_field.split("|")):
        code = normalize_code(part)
        if code and matches_range(code, ranges):
            onset_dt = pd.to_datetime(row.get(f"p41280_a{idx}", pd.NA), errors="coerce")
            if pd.notna(onset_dt) and onset_dt < exam_dt:
                return True
    return False


def logic_obesity(row: pd.Series) -> float:
    bmi = row.get("bmi", np.nan)
    icd_flag = check_icd_history(row, OB_RANGES)
    if (not pd.isna(bmi) and bmi >= 28) or icd_flag:
        return 1
    if pd.isna(bmi) and not icd_flag:
        return np.nan
    return 0


# Hypertension (BP >=140/90 or I10-I15 before exam)
HTN_RANGES = [("I", 10, 15)]


def logic_hypertension(row: pd.Series) -> float:
    sbp_mean = np.nanmean([row.get("sbp_0", np.nan), row.get("sbp_1", np.nan)])
    dbp_mean = np.nanmean([row.get("dbp_0", np.nan), row.get("dbp_1", np.nan)])
    has_high_bp = (not np.isnan(sbp_mean) and sbp_mean >= 140) or (not np.isnan(dbp_mean) and dbp_mean >= 90)
    icd_flag = check_icd_history(row, HTN_RANGES)
    if has_high_bp or icd_flag:
        return 1
    if np.isnan(sbp_mean) and np.isnan(dbp_mean) and not icd_flag:
        return np.nan
    return 0


# Diabetes (E10-E14)
DM_RANGES = [("E", 10, 14)]


def flag_diabetes(df: pd.DataFrame, ranges: list[tuple[str, int, int]] = DM_RANGES) -> pd.DataFrame:
    out = {"diabetes_any_pre": []}
    exam_dates = pd.to_datetime(df["_exam_date"], errors="coerce")
    for _, row in df.iterrows():
        icd_field = row.get("icd-10", "")
        exam_dt = row.get("_exam_date", pd.NaT)
        stats_pre = 0
        if isinstance(icd_field, str) and icd_field.strip() and not pd.isna(exam_dt):
            for idx, part in enumerate(icd_field.split("|")):
                code = normalize_code(part)
                if code and matches_range(code, ranges):
                    onset_raw = row.get(f"p41280_a{idx}", pd.NA)
                    onset_dt = pd.to_datetime(onset_raw, errors="coerce")
                    if pd.notna(onset_dt) and onset_dt < exam_dt:
                        stats_pre += 1
        out["diabetes_any_pre"].append(1 if stats_pre > 0 else 0)
    df = df.copy()
    df["diabetes_any_pre"] = out["diabetes_any_pre"]
    return df


def logic_diabetes(row: pd.Series) -> float:
    val = row.get("diabetes_any_pre", pd.NA)
    if pd.isna(val):
        return np.nan
    return 1 if val == 1 else 0


def flag_diabetes_e11(df: pd.DataFrame) -> pd.DataFrame:
    """Diabetes sensitivity definition: only E11 (type 2 diabetes) diagnosed before imaging examination."""
    return flag_diabetes(df, ranges=[("E", 11, 11)])


# Metabolic Syndrome

def check_waist(row: pd.Series) -> int:
    waist = row.get("Waist circumference", np.nan)
    sex = row.get("sex_code", np.nan)
    if pd.isna(waist) or pd.isna(sex):
        return 0
    if sex == 0:
        return 1 if waist >= 88 else 0
    if sex == 1:
        return 1 if waist >= 102 else 0
    return 0


def check_bp_component(row: pd.Series) -> int:
    sbp_mean = np.nanmean([row.get("sbp_0", np.nan), row.get("sbp_1", np.nan)])
    dbp_mean = np.nanmean([row.get("dbp_0", np.nan), row.get("dbp_1", np.nan)])
    measure_high = (not np.isnan(sbp_mean) and sbp_mean >= 130) or (not np.isnan(dbp_mean) and dbp_mean >= 85)
    icd_high = check_icd_history(row, HTN_RANGES)
    return 1 if (measure_high or icd_high) else 0


def check_diabetes_component(row: pd.Series) -> int:
    return 1 if check_icd_history(row, DM_RANGES) else 0


def check_lipid_component(row: pd.Series) -> int:
    icd_field = row.get("icd-10", "")
    if not isinstance(icd_field, str) or not icd_field.strip():
        return 0
    exam_dt = row.get("_exam_date", pd.NaT)
    if pd.isna(exam_dt):
        return 0
    for idx, part in enumerate(icd_field.split("|")):
        code = normalize_code(part)
        if code and code.startswith("E78"):
            onset_dt = pd.to_datetime(row.get(f"p41280_a{idx}", pd.NA), errors="coerce")
            if pd.notna(onset_dt) and onset_dt < exam_dt:
                return 1
    return 0


def flag_mets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def has_e8881(row: pd.Series) -> int:
        icd_field = row.get("icd-10", "")
        if not isinstance(icd_field, str) or not icd_field.strip():
            return 0
        exam_dt = row.get("_exam_date", pd.NaT)
        if pd.isna(exam_dt):
            return 0
        for idx, part in enumerate(icd_field.split("|")):
            code = normalize_code(part)
            if code == "E88.81":
                onset_dt = pd.to_datetime(row.get(f"p41280_a{idx}", pd.NA), errors="coerce")
                if pd.notna(onset_dt) and onset_dt < exam_dt:
                    return 1
        return 0

    df["flag_direct_E8881"] = df.apply(has_e8881, axis=1)
    df["flag_waist"] = df.apply(check_waist, axis=1)
    df["flag_bp"] = df.apply(check_bp_component, axis=1)
    df["flag_diabetes"] = df.apply(check_diabetes_component, axis=1)
    df["flag_lipid"] = df.apply(check_lipid_component, axis=1)
    df["mets_score"] = df[["flag_waist", "flag_bp", "flag_diabetes", "flag_lipid"]].sum(axis=1)
    df["is_mets"] = df.apply(lambda r: 1 if r["flag_direct_E8881"] == 1 or r["mets_score"] >= 3 else 0, axis=1)
    return df


def logic_mets(row: pd.Series) -> float:
    return row.get("is_mets", np.nan)


# -----------------------------
# Registry and CLI
# -----------------------------


@dataclass
class PhenotypeConfig:
    name: str
    label: str
    logic: Callable[[pd.Series], float]
    flagger: Callable[[pd.DataFrame], pd.DataFrame]


identity_flagger = lambda df: df

PHENOTYPE_REGISTRY: Dict[str, PhenotypeConfig] = {
    "obesity": PhenotypeConfig("obesity", "Obesity", logic_obesity, identity_flagger),
    "hypertension": PhenotypeConfig("hypertension", "Hypertension", logic_hypertension, identity_flagger),
    "diabetes": PhenotypeConfig("diabetes", "Diabetes", logic_diabetes, flag_diabetes),
    "metabolic_syndrome": PhenotypeConfig("metabolic_syndrome", "Metabolic Syndrome", logic_mets, flag_mets),
}


def main():
    parser = argparse.ArgumentParser(description="Unified phenotype analysis with forest plots")
    parser.add_argument("--phenotype", default=None,
                        choices=list(PHENOTYPE_REGISTRY.keys()),
                        help="Single phenotype name (default: --all runs all)")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Run all phenotype analyses (default)")
    parser.add_argument("--no-sensitivities", action="store_true",
                        help="Do not run E11-only and mutually-exclusive control sensitivity analyses")
    parser.add_argument("--input", default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    args = parser.parse_args()

    base_df = load_base_data(args.input)
    print(f"[Data] loaded {len(base_df):,} rows, "
          f"test={int((base_df['split']=='ukb_test').sum()):,}, "
          f"ill={int((base_df['split']=='ukb_test_ill').sum()):,}")

    phenotypes_to_run = list(PHENOTYPE_REGISTRY.keys()) if args.all else [args.phenotype]

    # Precompute the four target disease statuses for the "mutually exclusive control" sensitivity: controls exclude the other three diseases.
    status_by_key = {}
    for key, cfg in PHENOTYPE_REGISTRY.items():
        flagged = cfg.flagger(base_df.copy())
        status_by_key[key] = flagged.apply(cfg.logic, axis=1)

    for pheno_key in phenotypes_to_run:
        print(f"\n{'='*60}")
        print(f"  {PHENOTYPE_REGISTRY[pheno_key].label}")
        print(f"{'='*60}")
        config = PHENOTYPE_REGISTRY[pheno_key]
        flagged_df = config.flagger(base_df.copy())

        adjust_bmi = pheno_key != "obesity"
        analysis = GenericPhenotypeAnalysis(config.label, flagged_df, args.output_root, adjust_bmi=adjust_bmi)
        analysis.run_all(config.logic)

        if not args.no_sensitivities:
            other_case = pd.Series(False, index=base_df.index)
            for other_key, other_status in status_by_key.items():
                if other_key != pheno_key:
                    other_case |= other_status.eq(1)
            mutex_analysis = GenericPhenotypeAnalysis(
                f"{config.label}_mutex", flagged_df.copy(), args.output_root, adjust_bmi=adjust_bmi
            )
            # main keeps "non-case means control"; this sensitivity retains only controls free of the other three diseases.
            mutex_analysis.run_all(config.logic, control_mask=~other_case)

    if not args.no_sensitivities and "diabetes" in phenotypes_to_run:
        print(f"\n{'='*60}")
        print("  Diabetes E11-only sensitivity")
        print(f"{'='*60}")
        e11_df = flag_diabetes_e11(base_df.copy())
        e11_analysis = GenericPhenotypeAnalysis("Diabetes_E11", e11_df, args.output_root)
        e11_analysis.run_all(logic_diabetes)

    print(f"\n{'='*60}")
    print(f"  All phenotype analyses complete! Results → {args.output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
