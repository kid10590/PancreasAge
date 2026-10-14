# Data

This directory is intentionally empty. Place input data here, or set
`PANCREASAGE_DATA` to point elsewhere.

Expected layout (see `../docs/DATA_SCHEMA.md` for column definitions):

```
data/
├── icd-10-all.csv
├── icd-10-all_with_flags.csv
├── age_sex_bmi_etal_participant.csv
├── ukb_healthy_npy/{images_npy/, metadata.csv}
├── ext_healthy_npy/{images_npy/, metadata.csv}
├── obesity_cohort/{images_npy/, metadata.csv, metadata_norm.csv}
└── genotypes/pgen/ukb22828_c{1..22}_b0_v3.{pgen,pvar,psam}
```

UK Biobank data are available to approved researchers; SYSUFAH individual-level
data are not publicly available (see the top-level `README.md`).
