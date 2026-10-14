# External tools and data access

This package depends on a few third-party tools and external datasets that are
**not** bundled here. Install / obtain them as described below.

## Python environment

The analyses were run in the conda environment referenced by
`/nas_3/LBX/requirement_list/lbx` (Python 3.11). Two environment files are provided:

- `environment_lbx.yml` — conda environment export (reference).
- `requirements_lbx.txt` — `pip freeze` from the working environment (authoritative).

Key packages: `torch`, `torchvision`, `monai`, `numpy`, `pandas`, `scipy`,
`scikit-learn`, `statsmodels`, `lifelines`, `nibabel`, `SimpleITK`, `matplotlib`,
`openpyxl`, `h5py`, `einops`, `einops-exts`, `tqdm`.

```bash
# option A: recreate from the pip freeze
conda create -n pancreas_age python=3.11 -y
conda activate pancreas_age
pip install -r environment/requirements_lbx.txt

# option B: recreate from the conda yml
conda env create -f environment/environment_lbx.yml
```

## Command-line tools

| Tool | Used by | Notes |
|---|---|---|
| `dcm2niix` | `00_data_prep/0_1_dcm2nii.py` | DICOM → NIfTI. Set `$DCM2NIIX` or put it on `PATH`. |
| `plink2` | `03_genetics/3_2_*`, `3_5`, `4_0`, `4_4`, `4_3_run_mr_analysis.R` | Download from https://www.cog-genomics.org/plink/2.0/. Set `$PLINK2`. |
| `regenie` (v4.1.3) | `03_genetics/3_2_run_gwas_regenie.sh`, `run_regenie_full_phase*.sh` | Download from https://rgcgithub.github.io/regenie/. Set `$REGENIE`. |

## R packages (Mendelian randomisation / PRS)

`03_genetics/4_2_install_mr_packages.R` installs the required R packages
(`TwoSampleMR`, `ieugwasr`, `MendelianRandomization`, `MR-PRESSO`, `data.table`, `ggplot2`).

```bash
Rscript 03_genetics/4_2_install_mr_packages.R
```

## OpenGWAS token

`4_3_run_mr_analysis.R` and `4_4_build_t2d_prs.R` query OpenGWAS and require a
personal token (no token is shipped in this repository):

```bash
export OPENGWAS_JWT="<your-token>"   # obtain at https://api.opengwas.io/
```

## Pancreas segmentation

The pancreas masks used as model input were produced with
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet), a self-configuring segmentation
framework. `00_data_prep/0_2_preprocess_data.py` expects paired image/mask volumes
(see `docs/DATA_SCHEMA.md`).

## Data

- **UK Biobank**: available to approved researchers via
  https://www.ukbiobank.ac.uk/enable-your-research (application required).
- **SYSUFAH** (First Affiliated Hospital of Sun Yat-sen University) cohorts:
  individual-level data are not publicly available due to participant privacy and
  ethics restrictions. Requests go to the corresponding author.

Input table schemas are documented in `docs/DATA_SCHEMA.md`.
