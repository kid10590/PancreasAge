"""
Central path configuration for the PancreasAge public package.

All scripts read their input/output roots from here so that no machine-specific
absolute paths are hard-coded. Override with environment variables:

    export PANCREASAGE_PROJ=/path/to/PancreasAge_public   # package root
    export PANCREASAGE_DATA=/path/to/input_data           # raw/intermediate inputs
    export PANCREASAGE_OUT=/path/to/results               # all outputs

Defaults:
    PROJ_ROOT = <package root>            (three levels above this file)
    DATA_ROOT = $PROJ_ROOT/data
    OUT_ROOT  = $PROJ_ROOT/results
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.environ.get("PANCREASAGE_PROJ", os.path.dirname(os.path.dirname(_HERE)))
DATA_ROOT = os.environ.get("PANCREASAGE_DATA", os.path.join(PROJ_ROOT, "data"))
OUT_ROOT = os.environ.get("PANCREASAGE_OUT", os.path.join(PROJ_ROOT, "results"))

# Commonly used sub-roots (kept for convenience / documentation)
UKB_NPY_DIR = os.path.join(DATA_ROOT, "ukb_healthy_npy", "images_npy")
UKB_META_CSV = os.path.join(DATA_ROOT, "ukb_healthy_npy", "metadata.csv")
EXT_NPY_DIR = os.path.join(DATA_ROOT, "ext_healthy_npy", "images_npy")
EXT_META_CSV = os.path.join(DATA_ROOT, "ext_healthy_npy", "metadata.csv")
OBESITY_NPY_DIR = os.path.join(DATA_ROOT, "obesity_cohort", "images_npy")
OBESITY_META_CSV = os.path.join(DATA_ROOT, "obesity_cohort", "metadata.csv")
CV_DIR = os.path.join(OUT_ROOT, "cv")
INFER_DIR = os.path.join(OUT_ROOT, "infer_cv")

# Genetics inputs / third-party tools (see environment/external_tools.md)
PGEN_DIR = os.environ.get("PGEN_DIR", os.path.join(DATA_ROOT, "genotypes", "pgen"))
PLINK2 = os.environ.get("PLINK2", "plink2")
REGENIE = os.environ.get("REGENIE", "regenie")
