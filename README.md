# PancreasAge

Code accompanying the study **"MRI-derived pancreatic biological ageing gap
reveals metabolic risk and a genetic link to type 2 diabetes"**.

The repository covers the complete analysis pipeline:

- **Cohort construction & imaging** — UK Biobank health screening from ICD-10
  records; abdominal MRI preprocessing (DICOM → NIfTI, segmentation-masked
  cropping and intensity normalisation). Pancreas segmentation used
  [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
- **Pancreatic age model** — a 3D Swin Transformer trained to predict pancreatic
  age from abdominal MRI under 5-fold cross-validation with age-bias correction;
  the **accelerated pancreatic ageing gap (APAG)** is the bias-corrected predicted
  age minus chronological age.
- **External validation** — evaluation of the frozen model in independent hospital
  cohorts.
- **Metabolic & disease associations** — APAG vs prevalent diabetes, hypertension
  and obesity (propensity-score matching + multivariable regression), incident type
  2 diabetes (Cox models, Kaplan–Meier), and OGTT-defined glycaemic status in an
  obesity cohort.
- **Genetic analyses** — genome-wide association (REGENIE / PLINK 2.0), polygenic
  risk scores, two-sample Mendelian randomisation, mediation and phenome-wide
  association.

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

## Data availability

- UK Biobank: https://www.ukbiobank.ac.uk/enable-your-research (application required).
- SYSUFAH cohorts: not publicly available due to participant privacy/ethics; requests
  to the corresponding author.

## License

Apache License 2.0 — see `LICENSE`.

## Citation

`<citation to be added>`
