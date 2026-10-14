# Pipeline overview

End-to-end pipeline for the study *"MRI-derived pancreatic biological ageing gap
reveals metabolic risk and a genetic link to type 2 diabetes"*.

Main line: UK Biobank pancreatic MRI → deep-learning **pancreatic age** prediction
→ accelerated pancreatic ageing gap (**APAG** = bias-corrected predicted age −
chronological age) → disease-phenotype association → GWAS → PRS / MR / mediation /
PheWAS.

## Paths

All scripts read their input/output roots from `code/common/paths.py`, which is
driven by environment variables (see the top-level `README.md`):

| variable | default | meaning |
|---|---|---|
| `PANCREASAGE_PROJ` | package root | package location |
| `PANCREASAGE_DATA` | `$PROJ/data` | raw / intermediate inputs |
| `PANCREASAGE_OUT`  | `$PROJ/results` | all outputs |
| `PLINK2`, `REGENIE`, `DCM2NIIX`, `PGEN_DIR`, `OPENGWAS_JWT` | — | external tools (see `environment/external_tools.md`) |

## Script map

### A. Data preparation — `code/00_data_prep/`

| script | function |
|---|---|
| `0_0_select_icd10.py` | Scan UK Biobank ICD-10 records; extract first matching code and onset date; classify health status relative to the MRI date → `is_healthy` label. |
| `0_1_dcm2nii.py` | Batch DICOM → NIfTI conversion for the external (SYSUFAH) cohort via `dcm2niix`. |
| `0_2_preprocess_data.py` | NIfTI preprocessing: mask-centred crop/pad to 160×128×64, in-mask linear normalisation to [−1, 1] (background −1); external set resampled to 1.4×1.4×1.8 mm; save `.npy`; build `metadata.csv` (train/val/test split for healthy UKB). |

> Pancreas masks come from a separate nnU-Net segmentation model that is **not**
> part of this package (see `environment/external_tools.md`).

### B. Model training / inference — `code/01_model/`

| script | function |
|---|---|
| `1_0_generate_cv_splits.py` | Age-stratified (5-year bins) 5-fold cross-validation split → `cv_splits.json`, `metadata_cv.csv`. |
| `1_0_run.py` | DDP training of the Swin3D age regressor (label-distribution learning: KL + L1, bias correction, AMP). Saves `best_val_mae.pt` per fold. Launcher: `run_cv_folds.sh`. |
| `1_1_infer_all_cv.py` | 5-fold inference: per-fold test predictions concatenated into out-of-fold (OOF) healthy results; unhealthy / external cohorts predicted per fold and averaged. → `master_ukb_predictions.csv`, `predictions_ext_*.csv`. |
| `Model/` | `Swin3DBackbone` + `Swin3DAgeRegressor` (MONAI Swin Transformer V2), base regressor, heads. |
| `Dataset/` | `.npy` dataset and GPU augmentation. |
| `utils/` | training utilities, metrics, plotting. |
| `../../configs/pancreas_age_config_npy.json` | model/training configuration (the published run). |

### C. Disease-phenotype analysis — `code/02_downstream/`

| script | function |
|---|---|
| `2_0_merge_results.py` | Merge master predictions × ICD-10 flags × phenotype table; compute `delta` (APAG) → `merged_phenotype.csv`. |
| `2_5_phenotype_analysis_runner.py` | Generic cohort definition (ICD rules) → propensity-score matching (PSM, age/sex caliper; +BMI for diabetes/hypertension) → multivariable linear regression. `--all` runs diabetes, hypertension, obesity, metabolic syndrome. |
| `2_8_dose_response_diabetes.py` | Ordinal logistic dose-response between APAG (4 categories) and OGTT-defined glycaemic status in the external cohort, with nested covariate models, P-trend, VIF/EPV and proportional-odds diagnostics. |
| `2_9_ttest_delta.py` | Healthy vs diseased APAG Δ: Welch t-test + Cohen's d + violin plot. |
| `5_1…5_4b_*.py` | Cohort tables / baseline statistics for the UK Biobank and SYSUFAH cohorts. |
| `6_0_healthy_cox_analysis.py` | Cox proportional-hazards analysis of incident disease (per-SD APAG), Kaplan-Meier curves and forest plot. |

### D. Genetics — `code/03_genetics/`

| script | function |
|---|---|
| `3_1_prepare_plink_data.py` | Build GWAS inputs: APAG + covariates (sex, age, age², array, PC1–10); stratified 7:3 discovery/test split. |
| `3_2_run_gwas_plink.sh` | PLINK2 `--glm` GWAS per chromosome (primary validation analysis). |
| `3_2_run_gwas_regenie.sh`, `run_regenie_full_phase*.sh` | REGENIE two-step GWAS (primary analysis). |
| `3_3_make_plot_plink.py`, `3_3_make_plot_regenie.py` | Manhattan + QQ plots, significant-SNP extraction. |
| `3_4_compare_regenie_plink.py`, `3_6_compare_regenie_plink.py` | REGENIE vs PLINK concordance. |
| `3_5_make_lead_loci.py` | LD clumping → independent lead loci. |
| `4_0_run_prs.sh`, `4_1_evaluate_prs.py` | APAG PRS construction (LD clumping + PLINK2 scoring) and validation / diabetes association. |
| `4_2_prepare_mr_exposure.py`, `4_2_install_mr_packages.R`, `4_3_run_mr_analysis.R` | Two-sample MR (exposure = APAG; outcomes = T2D GWAS). |
| `4_4_build_t2d_prs.R`, `4_4_run_t2d_prs.sh`, `4_5_prs_diabetes_assoc.py` | External T2D PRS and APAG-PRS independent-contribution analysis. |
| `2_6_run_mediation_snp.py`, `2_7_run_mediation_prs.py` | CTRB1/2 SNP → APAG → diabetes, and PRS-based mediation. |
| `5_0_run_phewas.py` | APAG PRS × ICD-10 three-character phenotypes (logistic regression, Bonferroni + FDR). |

### E. Figures — `code/04_figures/`

| script | function |
|---|---|
| `make_sci_figures.py` | Main figures (Fig 1–10), house style, PDF + PNG + editable SVG. |
| `make_methods_figure.py` | Methods flowchart. |
| `recompute_external_metrics.py` | Recompute external-cohort metrics after age filtering. |
| `figures_nature_code/fig1…6` | Nature-style figure variants. |

## Run order

Run from the package root (`PancreasAge_public/`). Set the environment variables first.

```bash
export PANCREASAGE_DATA=/path/to/input_data
export PANCREASAGE_OUT=/path/to/results

# ── A. data preparation ──
python code/00_data_prep/0_0_select_icd10.py
python code/00_data_prep/0_1_dcm2nii.py --source-dir ... --target-dir ...
python code/00_data_prep/0_2_preprocess_data.py

# ── B. model training / inference ──
python code/01_model/1_0_generate_cv_splits.py
bash   code/01_model/run_cv_folds.sh 0 4
python code/01_model/1_1_infer_all_cv.py

# ── C. disease-phenotype analysis ──
python code/02_downstream/2_0_merge_results.py
python code/02_downstream/2_5_phenotype_analysis_runner.py --all
python code/02_downstream/2_9_ttest_delta.py
python code/02_downstream/2_8_dose_response_diabetes.py
python code/02_downstream/6_0_healthy_cox_analysis.py

# ── D. genetics ──
python code/03_genetics/3_1_prepare_plink_data.py
bash   code/03_genetics/3_2_run_gwas_regenie.sh      # or 3_2_run_gwas_plink.sh
python code/03_genetics/3_3_make_plot_regenie.py
python code/03_genetics/3_5_make_lead_loci.py
bash   code/03_genetics/4_0_run_prs.sh
python code/03_genetics/4_1_evaluate_prs.py
Rscript code/03_genetics/4_2_install_mr_packages.R
python code/03_genetics/4_2_prepare_mr_exposure.py
Rscript code/03_genetics/4_3_run_mr_analysis.R --outcome ieu-a-23
Rscript code/03_genetics/4_4_build_t2d_prs.R
bash   code/03_genetics/4_4_run_t2d_prs.sh
python code/03_genetics/4_5_prs_diabetes_assoc.py
python code/03_genetics/2_6_run_mediation_snp.py --lead-snp rs72802342
python code/03_genetics/2_7_run_mediation_prs.py
python code/03_genetics/5_0_run_phewas.py

# ── E. figures ──
python code/04_figures/make_sci_figures.py
```

> Note: the mediation scripts (`2_6`, `2_7`) are numbered in the `2_x` group but
> depend on GWAS (`3_x`) and PRS (`4_0`) outputs, so they run after those.

## Data flow

```
is_healthy (0_0) ──► metadata.csv (0_2) ──► cv_splits.json (1_0_generate)
                                                  │
                                    5-fold models (1_0_run) ──► checkpoints
                                                  │
                        master_ukb_predictions.csv + predictions_ext_*.csv (1_1_infer)
                                                  │
                             merged_phenotype.csv (2_0) ──► phenotype analyses (2_5/2_8/2_9/6_0)
                                                  │
                         GWAS inputs (3_1) ──► GWAS (3_2/3_3) ──► lead loci (3_5)
                                                  │
                          PRS (4_0/4_1/4_4/4_5) · MR (4_2/4_3) · mediation (2_6/2_7) · PheWAS (5_0)
```

## Not included in this package

- Pancreas segmentation model (nnU-Net) training/inference.
- Obesity-cohort per-SD ordinal-logistic model with pancreatic fat (Fig 6) and its
  linear/fat association analyses.
- Cox landmark analyses, multi-outcome FDR correction.
