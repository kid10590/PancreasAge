# Input data schema

This package contains **no data**. Place inputs under `$PANCREASAGE_DATA` and
point scripts there (defaults in `code/common/paths.py`). Below are the table
schemas the scripts expect. Column names must match.

## 1. UK Biobank ICD-10 records — `icd-10-all.csv` → `icd-10-all_with_flags.csv`

Input to `0_0_select_icd10.py`.

| column | type | description |
|---|---|---|
| `eid` | str/int | participant id |
| `icd-10` | str | one or more ICD-10 codes per row, separated by `|` |
| `date` | date | record date (used as onset date) |

Output adds health-status flags relative to the MRI date (used downstream as
`is_healthy`).

## 2. UK Biobank participant table — `age_sex_bmi_etal_participant.csv`

| column | description |
|---|---|
| `eid` | participant id |
| `age` | chronological age at MRI |
| `sex` | 1 = male, 2 = female |
| `bmi` | body-mass index (kg/m²) |
| systolic / diastolic blood pressure, waist circumference | used by phenotype / Cox analyses |

## 3. Processed image dataset — `ukb_healthy_npy/`

Produced by `0_2_preprocess_data.py`.

```
ukb_healthy_npy/
├── images_npy/{eid}.npy     # float32, shape (160,128,64), values in [-1,1]
└── metadata.csv             # columns: eid, age, split  (split ∈ train/val/test)
```

## 4. External healthy cohort — `ext_healthy_npy/`

Same layout as the UKB set (`images_npy/`, `metadata.csv` with
`eid, age, split`), resampled to 1.4×1.4×1.8 mm. Used for external validation.

## 5. Obesity cohort — `obesity_cohort/`

Used by `2_8_dose_response_diabetes.py`, `5_4_*`, `recompute_external_metrics.py`.

```
obesity_cohort/
├── images_npy/{eid}.npy
├── metadata.csv          # eid, age, split
└── metadata_norm.csv     # clinical table, see below
```

`metadata_norm.csv` clinical columns (excerpt):

| column | description |
|---|---|
| `eid` | participant id |
| `sex` | 1 = male, 2 = female, 0 = missing |
| `age` | chronological age |
| `age_pred_raw`, `age_pred_bc` | predicted pancreatic age (uncorrected / bias-corrected) |
| `delta` | APAG (bias-corrected predicted age − chronological age) |
| `delta_cat`, `delta_cat_median`, `delta_cat_label` | APAG grouping (0 = <0; 1–3 = tertiles of ≥0) |
| `bmi` | body-mass index |
| `ogtt_2h_glucose` | OGTT glycaemic status code (0 normal / 1 prediabetes / 2 diabetes-range) |
| `fasting_insulin`, `total_cholesterol`, `triglyceride`, `hdl_cholesterol`, `ldl_cholesterol`, `alanine_aminotransferase`, `aspartate_aminotransferase`, `creatinine`, … | laboratory covariates |
| `mean_pancreatic_fat_percent`, `pancreatic_head_fat_percent`, `pancreatic_body_fat_percent`, `tail_corrected`, `anterior_subcutaneous_fat_thickness` | MRI fat measures (percent / mm) |

## 6. Genotype data (PLINK 2 / REGENIE)

```
genotypes/pgen/ukb22828_c{1..22}_b0_v3.{pgen,pvar,psam}
```

GWAS inputs written by `3_1_prepare_plink_data.py`:

- `gwas_inputs/phenotypes.txt` — `FID IID APAG`
- `gwas_inputs/covariates.cov` — `FID IID Sex Age Age2 Array PC1..PC10`
- `prs_analysis/keep_{discovery,test,all}.id`

## 7. Mendelian randomisation outcomes

`4_3_run_mr_analysis.R` / `4_4_build_t2d_prs.R` fetch outcome GWAS
(`ieu-a-23`, `ebi-a-GCST007515`, `finn-b-T2D`) from OpenGWAS at runtime — requires
`OPENGWAS_JWT` (see `environment/external_tools.md`).
