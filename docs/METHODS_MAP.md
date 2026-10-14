# Manuscript → code map

Maps each manuscript figure/table to the script(s) in this package that produce it.
Omitted items (no code in this package) are listed at the bottom.

| Manuscript item | Content | Script(s) |
|---|---|---|
| Fig. 1 (a–c) | Model development / calibration (per-fold MAE, predicted vs chronological, Bland–Altman) | `04_figures/make_sci_figures.py::fig1_model_performance`; `04_figures/figures_nature_code/fig1_model_performance.py` |
| Fig. 1 (d) | Bias-corrected MAE across cohorts | same as above (external MAE panel) |
| Table 1 | Baseline characteristics of the four cohorts | `02_downstream/5_2_ukb_cohort_tables.py`, `5_4_fuyi_part3_table.py`, `5_4b_fuyi_part1_table.py` |
| Fig. 3 | APAG in prevalent diabetes / hypertension / obesity (PSM + linear regression) | `02_downstream/2_5_phenotype_analysis_runner.py`; figure `04_figures/make_sci_figures.py::fig2_disease_association` |
| Supp. Fig. S1c | Healthy vs diseased APAG Δ (t-test) | `02_downstream/2_9_ttest_delta.py` |
| Fig. 4 / Table 2 | Incident T2D: Cox model, Kaplan–Meier, forest plot | `02_downstream/6_0_healthy_cox_analysis.py` |
| Fig. 5, Fig. 7 | APAG / pancreatic fat across OGTT groups | *not included* (see below) |
| Fig. 6 / Table 3 / Supp. Table S10 | Obesity cohort ordinal + fat models | *not included* (see below) |
| Supp. Fig. S6 | APAG dose–response vs OGTT status | `02_downstream/2_8_dose_response_diabetes.py`; figure `04_figures/make_sci_figures.py::fig8_dose_response` |
| Fig. 8 | GWAS of APAG (Manhattan, QQ, regional, lead loci) | `03_genetics/3_1_prepare_plink_data.py`, `3_2_run_gwas_regenie.sh`, `3_3_make_plot_regenie.py`, `3_5_make_lead_loci.py`; figure `04_figures/figures_nature_code/fig3_gwas.py` |
| Supp. Fig. S6 (validation) | REGENIE vs PLINK concordance | `03_genetics/3_2_run_gwas_plink.sh`, `3_4_compare_regenie_plink.py`, `3_6_compare_regenie_plink.py` |
| Fig. 9 | CTRB1/2 SNP → APAG → diabetes mediation | `03_genetics/2_6_run_mediation_snp.py`; figure `04_figures/figures_nature_code/fig4_mediation.py` |
| Fig. 10 / Table S5 | Two-sample MR (APAG → T2D) | `03_genetics/4_2_prepare_mr_exposure.py`, `4_3_run_mr_analysis.R`; figure `04_figures/figures_nature_code/fig5_mr_prs.py` |
| Fig. 11 | APAG PRS and independent contribution | `03_genetics/4_0_run_prs.sh`, `4_1_evaluate_prs.py`, `4_4_build_t2d_prs.R`, `4_4_run_t2d_prs.sh`, `4_5_prs_diabetes_assoc.py` |
| Supp. Fig. S7 | PheWAS of APAG PRS | `03_genetics/5_0_run_phewas.py`; figure `04_figures/figures_nature_code/fig6_phewas.py` |
| Methods flowchart | Study design | `04_figures/make_methods_figure.py` |

## Not included in this package

- **Pancreas segmentation (nnU-Net)** — external to this repository; preprocessing
  assumes paired image/mask volumes.
- **Obesity-cohort per-SD ordinal model with pancreatic fat (Fig. 6), the
  APAG–fat linear model, and the per-group fat/ANOVA/Spearman analyses** — no code
  available in this package.
- **Cox landmark analyses and multi-outcome FDR correction** — `6_0` implements the
  main and adjusted Cox models, Kaplan–Meier and forest plot, but not the landmark
  windows or the ten-outcome FDR correction.
