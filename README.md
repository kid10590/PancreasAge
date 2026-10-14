# PancreasAge

Public code release for the study **"MRI-derived pancreatic biological ageing gap
reveals metabolic risk and a genetic link to type 2 diabetes"**.

We train a 3D Swin Transformer to predict **pancreatic age** from segmentation-masked
abdominal MRI, define the **accelerated pancreatic ageing gap (APAG)** as the
age-bias-corrected predicted age minus chronological age, and characterise its
metabolic associations and genetic architecture.

## Contents

```
PancreasAge_public/
├── code/
│   ├── common/paths.py        # central path configuration (env-var driven)
│   ├── 00_data_prep/          # ICD-10 health screening, DICOM→NIfTI, preprocessing
│   ├── 01_model/              # Swin3D model + 5-fold training / inference
│   ├── 02_downstream/         # phenotype / PSM / dose-response / t-test / Cox / cohort tables
│   ├── 03_genetics/           # GWAS, PRS, MR, mediation, PheWAS
│   └── 04_figures/            # manuscript figures
├── configs/                   # model/training configuration
├── environment/               # conda / pip environment + external tools
├── docs/                      # PIPELINE.md, DATA_SCHEMA.md, METHODS_MAP.md
└── data/                      # (empty) place input data here
```

## Environment

Python 3.11. Install the Python dependencies from `environment/requirements.txt`;
see `environment/external_tools.md` for `plink2`, `regenie`, `dcm2niix`, the R
packages and the OpenGWAS token.

```bash
conda create -n pancreas_age python=3.11 -y
conda activate pancreas_age
pip install -r environment/requirements.txt
```

## Configuration

Paths are read from `code/common/paths.py`, driven by environment variables:

```bash
export PANCREASAGE_DATA=/path/to/input_data    # inputs   (default: <pkg>/data)
export PANCREASAGE_OUT=/path/to/results        # outputs  (default: <pkg>/results)
# optional external tools:
export PLINK2=/path/to/plink2
export REGENIE=/path/to/regenie
export DCM2NIIX=/path/to/dcm2niix
export PGEN_DIR=/path/to/genotypes/pgen
export OPENGWAS_JWT=<your-token>               # for MR / external PRS
```

Input table schemas are documented in `docs/DATA_SCHEMA.md`.

## Usage

See `docs/PIPELINE.md` for the full script map and dependency-ordered run
instructions. Minimal model workflow:

```bash
python code/01_model/1_0_generate_cv_splits.py
bash   code/01_model/run_cv_folds.sh 0 4
python code/01_model/1_1_infer_all_cv.py
```

## Model

- **Backbone**: MONAI Swin Transformer V2 (3D).
- **Output**: probability distribution over 1-year age bins (45–87 y); predicted
  age = expectation.
- **Loss**: KL divergence (distribution) + L1 (point), with optional residual /
  evidential heads.
- **Cross-validation**: participant-level 5-fold (train 64% / val 16% / test 20%);
  each participant is tested in exactly one fold (out-of-fold evaluation).
- **Age-bias correction**: linear correction `pred_bc = pred − (α·age + β)` fitted
  on the training partition of each fold, applied unchanged to held-out / external
  data (Beheshti et al.).

## Notes / scope

The following components of the paper are **not** part of this code release:
pancreas segmentation (nnU-Net), the obesity-cohort per-SD ordinal + pancreatic-fat
model, and the Cox landmark / multi-outcome FDR analyses. See
`docs/METHODS_MAP.md` for details.

## Data availability

- UK Biobank: https://www.ukbiobank.ac.uk/enable-your-research (application required).
- SYSUFAH cohorts: not publicly available due to participant privacy/ethics; requests
  to the corresponding author.

## Code availability

https://github.com/kid10590/PancreasAge

## License

Apache License 2.0 — see `LICENSE`.

## Citation

`<citation to be added>`
