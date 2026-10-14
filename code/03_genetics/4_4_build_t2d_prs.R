###############################################################################
# 4_3_build_t2d_prs.R — build a known T2D PRS (weights from external published GWAS)
#
# Data sources (external, none include UKB):
#   - DIAMANTE  (ieu-a-23)    European population, excludes UKB
#   - FinnGen   (finn-b-T2D)  independent Finnish population, no UKB overlap
#
# Workflow:
#   1. ieugwasr::extract_instruments(P<5e-8) extracts T2D significant SNPs
#   2. LD clumping (r²=0.1, 250kb) → independent lead SNPs
#   3. Output score file: SNP effect_allele beta (for PLINK2 --score)
#
# Output: $PRS_DIR/t2d_score_<id>.txt
# Next: PLINK2 --score scoring on the full UKB sample (see the bash command commented at the end of the script)
#
# Usage:
#   Rscript code/03_genetics/4_4_build_t2d_prs.R
###############################################################################

library(data.table)
library(ieugwasr)
library(TwoSampleMR)

# ---- paths ----
# Results root: override with the PANCREASAGE_OUT env var.
pkg_root <- Sys.getenv("PANCREASAGE_OUT", unset = file.path(getwd(), "results"))
PRS_DIR <- file.path(pkg_root, "prs_analysis")
dir.create(PRS_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- OpenGWAS authentication ----
# Provide your own token via the OPENGWAS_JWT environment variable, e.g.
#   export OPENGWAS_JWT="<your-token>"   (obtain at https://api.opengwas.io/)
if (Sys.getenv("OPENGWAS_JWT") == "") {
  warning("OPENGWAS_JWT is not set; OpenGWAS queries will fail. See environment/external_tools.md.")
}
tryCatch({ Sys.setenv(OPENSSL_CONF = "/dev/null") }, error = function(e) NULL)

options(timeout = 600)

OUTCOME_IDS <- c("ieu-a-23", "finn-b-T2D")

for (oid in OUTCOME_IDS) {
  cat("\n========================================\n")
  cat(sprintf(" Processing %s ...\n", oid))
  cat("========================================\n")

  # 1. Extract P<5e-8 significant SNPs
  inst <- tryCatch({
    extract_instruments(outcomes = oid, p1 = 5e-8, clump = FALSE)
  }, error = function(e) {
    cat(sprintf("  [failed] extract_instruments: %s\n", e$message))
    NULL
  })
  if (is.null(inst) || nrow(inst) == 0) {
    cat("  No significant SNPs, skipping\n")
    next
  }
  cat(sprintf("  P<5e-8: %d SNPs\n", nrow(inst)))

  # 2. LD clumping (r²=0.1, 250kb)
  inst <- tryCatch({
    clump_data(inst, clump_kb = 250, clump_r2 = 0.1,
               pop = "EUR", bfile = NULL)
  }, error = function(e) {
    cat(sprintf("  [warning] online clump failed: %s, trying without the pop argument\n", e$message))
    tryCatch({
      clump_data(inst, clump_kb = 250, clump_r2 = 0.1)
    }, error = function(e2) {
      cat(sprintf("  [warning] second clump failed: %s, using all significant SNPs\n", e2$message))
      inst
    })
  })
  cat(sprintf("  Clumped lead SNPs: %d\n", nrow(inst)))

  if (nrow(inst) == 0) next

  # 3. Output score file (SNP, effect_allele, beta)
  score <- data.frame(
    SNP = inst$SNP,
    effect_allele = inst$effect_allele.exposure,
    beta = inst$beta.exposure,
    stringsAsFactors = FALSE
  )
  score <- score[complete.cases(score), ]
  score <- unique(score)
  out_file <- file.path(PRS_DIR, sprintf("t2d_score_%s.txt", oid))
  write.table(score, out_file, sep = "\t", row.names = FALSE,
              col.names = FALSE, quote = FALSE)
  cat(sprintf("  → %s  (%d SNPs)\n", out_file, nrow(score)))

  # Back up full information
  inst_info <- inst[, c("SNP", "chr.exposure", "pos.exposure",
                        "effect_allele.exposure", "other_allele.exposure",
                        "beta.exposure", "se.exposure", "pval.exposure",
                        "eaf.exposure")]
  write.csv(inst_info, file.path(PRS_DIR, sprintf("t2d_instruments_%s.csv", oid)),
            row.names = FALSE)
}

cat("\n✓ T2D PRS score files generated!\n")
cat("Next step (bash): score the full UKB sample for each score file:\n")
cat("  PLINK2 --pfile ukb22828_c{chr}_b0_v3 --keep keep_all.id --score t2d_score_<id>.txt 1 2 3 --out ...\n")
cat("  then merge chromosomes → standardize per 1 SD\n")
