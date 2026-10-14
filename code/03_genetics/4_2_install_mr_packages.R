#!/usr/bin/env Rscript

# ============================================================
# Install all R packages required for MR analysis
# ============================================================

# Set CRAN mirror (users in China may prefer the Tsinghua mirror)
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# 1. Install base packages (via CRAN)
base_packages <- c(
  "data.table",      # efficient data processing
  "ggplot2",         # plotting
  "remotes",         # install from GitHub
  "devtools"         # development tools
)

install.packages(base_packages)

# 2. Install BiocManager (for Bioconductor packages)
if (!require("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# 3. Install core MR packages
# 3a. TwoSampleMR (main analysis package)
remotes::install_github("MRCIEU/TwoSampleMR")

# 3b. ieugwasr (data extraction)
remotes::install_github("MRCIEU/ieugwasr")

# 3c. MendelianRandomization (supplementary MR methods)
BiocManager::install("MendelianRandomization")

# 3d. MR-PRESSO (outlier detection)
remotes::install_github("rondolab/MR-PRESSO")

# 4. Install other helper packages
install.packages(c(
  "meta",           # meta-analysis
  "forestplot",     # forest plot
  "gridExtra",      # figure arrangement
  "cowplot"         # plot themes
))

# 5. Install LD reference panel packages (for online clumping)
remotes::install_github("MRCIEU/ieugwasr")

cat("\n========================================\n")
cat("All packages installed successfully!\n")
cat("========================================\n")

# Verify installation
cat("\nVerifying installed packages:\n")
packages_to_check <- c("TwoSampleMR", "ieugwasr", "MendelianRandomization", 
                       "MRPRESSO", "data.table", "ggplot2")

for (pkg in packages_to_check) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("  [OK] %s\n", pkg))
  } else {
    cat(sprintf("  [FAIL] %s\n", pkg))
  }
}
