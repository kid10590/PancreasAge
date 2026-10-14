###############################################################################
# 4_2_run_mr_analysis.R — two-sample Mendelian randomization (TwoSampleMR)
#
# Exposure: APAG (pancreas aging acceleration) GWAS — from PLINK results
# Outcome: type 2 diabetes — one of several optional GWAS datasets
#
# Usage:
#   Rscript code/03_genetics/4_2_run_mr_analysis.R
#   Rscript code/03_genetics/4_2_run_mr_analysis.R --outcome ieu-a-23
#   Rscript code/03_genetics/4_2_run_mr_analysis.R --outcome ebi-a-GCST007515
#   Rscript code/03_genetics/4_2_run_mr_analysis.R --outcome finn-b-T2D
###############################################################################

# ═══════════════════════════════════════
# 0. Environment setup
# ═══════════════════════════════════════
library(data.table)
library(TwoSampleMR)
library(ieugwasr)
# library(MendelianRandomization)
library(ggplot2)

# ---- parse command-line arguments ----
args <- commandArgs(trailingOnly = TRUE)
outcome_id <- "ieu-a-23"  # default
exposure_file_arg <- NULL
out_root_arg <- NULL
if (length(args) >= 1) {
  # Supports the --outcome xxx / --exposure xxx / --out-root xxx format
  for (i in seq_along(args)) {
    if (args[i] == "--outcome" && i < length(args)) {
      outcome_id <- args[i + 1]
    }
    if (args[i] == "--exposure" && i < length(args)) {
      exposure_file_arg <- args[i + 1]
    }
    if (args[i] == "--out-root" && i < length(args)) {
      out_root_arg <- args[i + 1]
    }
  }
}

# ---- set OpenGWAS authentication ----
# Provide your own token via the OPENGWAS_JWT environment variable, e.g.
#   export OPENGWAS_JWT="<your-token>"   (obtain at https://api.opengwas.io/)
if (Sys.getenv("OPENGWAS_JWT") == "") {
  warning("OPENGWAS_JWT is not set; OpenGWAS queries will fail. See environment/external_tools.md.")
}

# ---- path configuration ----
# Results root: override with --out-root or the PANCREASAGE_OUT env var.
pkg_root <- Sys.getenv("PANCREASAGE_OUT", unset = "")
default_mr_root <- if (nzchar(pkg_root)) file.path(pkg_root, "mr") else file.path(getwd(), "results", "mr")
MR_ROOT <- if (is.null(out_root_arg)) default_mr_root else out_root_arg
EXPOSURE_FILE <- if (is.null(exposure_file_arg)) file.path(MR_ROOT, "apag_exposure_for_mr.csv.gz") else exposure_file_arg
EXPOSURE_SIG_FILE <- file.path(dirname(EXPOSURE_FILE), "apag_sig_snps_for_mr.csv")
OUT_DIR <- file.path(MR_ROOT, outcome_id)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- outcome dataset info ----
outcome_info <- list(
  "ieu-a-23" = list(
    name = "DIAMANTE (2017)",
    cases = "12,171",
    controls = "56,862",
    desc = "European population, excludes UKB"
  ),
  "ebi-a-GCST007515" = list(
    name = "Mahajan 2018",
    cases = "74,124",
    controls = "824,006",
    desc = "Large-scale meta-analysis, includes some UKB"
  ),
  "finn-b-T2D" = list(
    name = "FinnGen T2D",
    cases = "~50,000",
    controls = "~180,000",
    desc = "Independent Finnish population, no UKB overlap"
  )
)

cat("\n")
cat("========================================\n")
cat("MR analysis configuration\n")
cat("========================================\n")
cat(sprintf("  Exposure: APAG (pancreas aging acceleration)\n"))
cat(sprintf("  Outcome: %s (%s)\n", 
    ifelse(outcome_id %in% names(outcome_info), outcome_info[[outcome_id]]$name, outcome_id),
    ifelse(outcome_id %in% names(outcome_info), outcome_info[[outcome_id]]$desc, "custom")))
cat(sprintf("  Outcome sample size: cases=%s, controls=%s\n",
    ifelse(outcome_id %in% names(outcome_info), outcome_info[[outcome_id]]$cases, "unknown"),
    ifelse(outcome_id %in% names(outcome_info), outcome_info[[outcome_id]]$controls, "unknown")))
cat("========================================\n\n")

# ═══════════════════════════════════════
# 1. Load and prepare exposure data
# ═══════════════════════════════════════
cat("\n========== Step 1: load exposure GWAS ==========\n")
exposure_raw <- fread(EXPOSURE_FILE)
exposure_raw <- as.data.frame(exposure_raw)
cat(sprintf("  Exposure variants: %s\n", format(nrow(exposure_raw), big.mark = ",")))

# Format as TwoSampleMR exposure format
exposure <- format_data(
  exposure_raw,
  type = "exposure",
  snp_col = "SNP",
  beta_col = "beta",
  se_col = "se",
  effect_allele_col = "effect_allele",
  other_allele_col = "other_allele",
  eaf_col = "eaf",
  pval_col = "pval",
  samplesize_col = "samplesize",
  chr_col = "chr",
  pos_col = "pos"
)
cat(sprintf("  After formatting: %s variants\n", nrow(exposure)))

# ═══════════════════════════════════════
# 2. LD Clumping: independent instrumental variables (local PLINK2, r²<0.001, 10000kb)
# ═══════════════════════════════════════
cat("\n========== Step 2: LD Clumping (PLINK2 r²<0.001) ==========\n")

exposure_sig <- exposure[exposure$pval.exposure < 5e-8, ]
cat(sprintf("  Significant SNPs (P<5e-8): %d\n", nrow(exposure_sig)))
if (nrow(exposure_sig) < 3) stop("significant SNPs <3")

PLINK2 <- Sys.getenv("PLINK2", unset = "plink2")
PGEN_DIR <- Sys.getenv("PGEN_DIR", unset = file.path(Sys.getenv("PANCREASAGE_DATA", unset = "data"), "genotypes", "pgen"))
clump_tmp <- file.path(OUT_DIR, "mr_clump")

# Get the chr position columns
sig_chr <- merge(exposure_sig[, c("SNP", "pval.exposure")],
                 exposure_raw[, c("SNP", "chr")], by = "SNP", all.x = TRUE)
names(sig_chr)[names(sig_chr) == "pval.exposure"] <- "P"

clumped_ids <- c()
for (chr_num in 1:22) {
  chr_snps <- subset(sig_chr, chr == chr_num)
  if (nrow(chr_snps) == 0) next
  chr_in <- file.path(OUT_DIR, sprintf("mr_clump_chr%d.txt", chr_num))
  # PLINK2 --clump: header=SNP P, --clump-id-field SNP, --clump-p-field P
  write.table(chr_snps[, c("SNP", "P")], chr_in,
              sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
  pgen <- file.path(PGEN_DIR, sprintf("ukb22828_c%d_b0_v3", chr_num))
  if (!file.exists(paste0(pgen, ".pgen"))) next

  cmd <- sprintf('%s --pfile %s --clump %s --clump-p1 5e-8 --clump-r2 0.001 --clump-kb 10000 --clump-id-field SNP --clump-p-field P --out %s_chr%d --threads 4',
                 PLINK2, pgen, chr_in, clump_tmp, chr_num)
  ret <- system(cmd, ignore.stdout = FALSE, ignore.stderr = FALSE)
  cf <- sprintf("%s_chr%d.clumps", clump_tmp, chr_num)
  if (file.exists(cf)) {
    cr <- fread(cf)
    if (nrow(cr) > 0) {
      clumped_ids <- c(clumped_ids, cr$ID[cr$ID != "null"])
    }
  }
}

clumped_ids <- unique(clumped_ids)
cat(sprintf("  PLINK2 clumped (r²<0.001): %d\n", length(clumped_ids)))

# If too few, relax to r²=0.01
if (length(clumped_ids) < 3) {
  cat("  Relaxing to r²=0.01...\n")
  clumped_ids <- c()
  for (chr_num in 1:22) {
    chr_snps <- subset(sig_chr, chr == chr_num)
    if (nrow(chr_snps) == 0) next
    chr_in <- file.path(OUT_DIR, sprintf("mr_clump_chr%d.txt", chr_num))
    pgen <- file.path(PGEN_DIR, sprintf("ukb22828_c%d_b0_v3", chr_num))
    if (!file.exists(paste0(pgen, ".pgen"))) next
    cmd <- sprintf('%s --pfile %s --clump %s --clump-p1 5e-8 --clump-r2 0.01 --clump-kb 10000 --clump-id-field SNP --clump-p-field P --out %s_chr%d_r --threads 4',
                   PLINK2, pgen, chr_in, clump_tmp, chr_num)
    system(cmd, ignore.stdout = FALSE, ignore.stderr = FALSE)
    cf <- sprintf("%s_chr%d_r.clumps", clump_tmp, chr_num)
    if (file.exists(cf)) {
      cr <- fread(cf)
      if (nrow(cr) > 0) clumped_ids <- c(clumped_ids, cr$ID[cr$ID != "null"])
    }
  }
  clumped_ids <- unique(clumped_ids)
  cat(sprintf("  After relaxing: %d\n", length(clumped_ids)))
}

exposure_clumped <- exposure_sig[exposure_sig$SNP %in% clumped_ids, ]
cat(sprintf("  Final independent instrumental variables: %d\n", nrow(exposure_clumped)))

# Compute F-statistic
exposure_clumped$F_stat <- (exposure_clumped$beta.exposure / exposure_clumped$se.exposure)^2
n_weak <- sum(exposure_clumped$F_stat < 10)
cat(sprintf("  Weak instruments (F<10): %d / %d\n", n_weak, nrow(exposure_clumped)))
cat(sprintf("  Mean F = %.1f, median F = %.1f\n",
            mean(exposure_clumped$F_stat), median(exposure_clumped$F_stat)))

write.csv(exposure_clumped[, c("SNP", "F_stat")],
          file.path(OUT_DIR, paste0("mr_f_statistics_", outcome_id, ".csv")), row.names = FALSE)

# ═══════════════════════════════════════
# 3. Extract outcome GWAS (diabetes)
# ═══════════════════════════════════════
cat("\n========== Step 3: extract outcome GWAS ==========\n")

# Network timeout settings
options(timeout = 600)
tryCatch({ Sys.setenv(OPENSSL_CONF = "/dev/null") }, error = function(e) NULL)

cat(sprintf("  Outcome GWAS ID: %s\n", outcome_id))

# Try online extraction
outcome_data <- tryCatch({
  extract_outcome_data(
    snps = exposure_clumped$SNP,
    outcomes = outcome_id,
    proxies = TRUE,
    rsq = 0.8,
    maf_threshold = 0.01
  )
}, error = function(e) {
  cat(sprintf("  [warning] online extraction failed: %s\n", e$message))
  cat("  Please download the outcome GWAS summary statistics manually and place them locally.\n")
  cat(sprintf("  Download URL: https://gwas.mrcieu.ac.uk/datasets/%s/\n", outcome_id))
  return(NULL)
})

if (is.null(outcome_data) || nrow(outcome_data) == 0) {
  stop("Unable to obtain outcome GWAS data. Check the network or use local files.")
}
cat(sprintf("  Outcome SNPs extracted: %d\n", nrow(outcome_data)))

# ═══════════════════════════════════════
# 4. Harmonise data
# ═══════════════════════════════════════
cat("\n========== Step 4: Harmonise ==========\n")
dat <- harmonise_data(
  exposure_dat = exposure_clumped,
  outcome_dat = outcome_data
)
cat(sprintf("  Harmonised SNPs: %d\n", nrow(dat)))

# Remove palindromic SNPs
dat <- dat[!(dat$ambiguous == TRUE & dat$palindromic == TRUE), ]
cat(sprintf("  After removing palindromic: %d SNPs\n", nrow(dat)))

# ponytail: if harmonised <5, local clumping already tried r²=0.01, no room left
if (nrow(dat) < 5) {
  cat(sprintf("  [warning] only %d harmonised SNPs, MR results may be unstable\n", nrow(dat)))
}

write.csv(dat, file.path(OUT_DIR, paste0("mr_harmonised_data_", outcome_id, ".csv")), row.names = FALSE)

# ═══════════════════════════════════════
# 5. Main MR analysis
# ═══════════════════════════════════════
cat("\n========== Step 5: main MR analysis ==========\n")
mr_results <- mr(dat, method_list = c(
  "mr_ivw",
  "mr_egger_regression",
  "mr_weighted_median",
  "mr_weighted_mode",
  "mr_ivw_mre"
))

mr_results$or <- exp(mr_results$b)
mr_results$or_lci95 <- exp(mr_results$b - 1.96 * mr_results$se)
mr_results$or_uci95 <- exp(mr_results$b + 1.96 * mr_results$se)

cat("\n  ═══ MR causal effect estimates (outcome: ", outcome_id, ") ═══\n", sep="")
print(mr_results[, c("method", "nsnp", "b", "se", "pval", "or", "or_lci95", "or_uci95")], row.names = FALSE)
write.csv(mr_results, file.path(OUT_DIR, paste0("mr_main_results_", outcome_id, ".csv")), row.names = FALSE)

# ═══════════════════════════════════════
# 6. Sensitivity analysis
# ═══════════════════════════════════════
cat("\n========== Step 6: sensitivity analysis ==========\n")

# 6a. Heterogeneity test
cat("\n--- 6a. Heterogeneity test (Cochran's Q) ---\n")
het <- mr_heterogeneity(dat)
print(het)
write.csv(het, file.path(OUT_DIR, paste0("mr_heterogeneity_", outcome_id, ".csv")), row.names = FALSE)

# 6b. Pleiotropy test
cat("\n--- 6b. Pleiotropy test (MR-Egger intercept) ---\n")
pleio <- mr_pleiotropy_test(dat)
cat(sprintf("  Egger intercept = %.4f, SE = %.4f, P = %.4f\n",
            pleio$egger_intercept, pleio$se, pleio$pval))
cat(sprintf("  → %s\n",
            ifelse(pleio$pval < 0.05,
                   "Significant directional horizontal pleiotropy detected! (interpret IVW results with caution)",
                   "No significant directional pleiotropy (IVW results are reliable)")))
write.csv(pleio, file.path(OUT_DIR, paste0("mr_pleiotropy_", outcome_id, ".csv")), row.names = FALSE)

# 6c. Leave-one-out analysis
cat("\n--- 6c. Leave-one-out analysis ---\n")
loo <- mr_leaveoneout(dat)
write.csv(loo, file.path(OUT_DIR, paste0("mr_leaveoneout_", outcome_id, ".csv")), row.names = FALSE)

# 6d. MR-PRESSO
cat("\n--- 6d. MR-PRESSO ---\n")
tryCatch({
  library(MRPRESSO)
  presso <- mr_presso(
    BetaOutcome = "beta.outcome", 
    BetaExposure = "beta.exposure",
    SdOutcome = "se.outcome", 
    SdExposure = "se.exposure",
    OUTLIERtest = TRUE, 
    DISTORTIONtest = TRUE,
    data = dat, 
    NbDistribution = 5000,  # increase iterations to improve precision
    SignifThreshold = 0.05
  )
  cat(sprintf("  MR-PRESSO Global RSS P = %.4f\n", presso$`MR-PRESSO results`$`Global Test`$Pvalue))
  if (!is.na(presso$`MR-PRESSO results`$`Distortion Test`$Pvalue)) {
    cat(sprintf("  Distortion P = %.4f\n", presso$`MR-PRESSO results`$`Distortion Test`$Pvalue))
  }
  outlier_snps <- presso$`MR-PRESSO results`$`Outlier Test`$SNP
  if (length(outlier_snps) > 0) {
    cat(sprintf("  Detected %d outlier SNPs: %s\n",
                length(outlier_snps), paste(outlier_snps, collapse = ", ")))
  }
}, error = function(e) {
  cat(sprintf("  [skip] MR-PRESSO failed: %s\n", e$message))
})

# 6e. Steiger directionality test
cat("\n--- 6e. Steiger directionality test ---\n")
steiger <- tryCatch({
  # ponytail: precompute r.exposure/r.outcome to avoid internal crash in directionality_test
  dat$r.exposure <- get_r_from_pn(dat$pval.exposure, dat$samplesize.exposure)
  dat$r.outcome  <- get_r_from_pn(dat$pval.outcome,  dat$samplesize.outcome)
  directionality_test(dat)
}, error = function(e) {
  cat(sprintf("  [skip] Steiger failed: %s\n", e$message))
  return(NULL)
})

if (!is.null(steiger)) {
  cat(sprintf("  Steiger: correct causal direction = %s, P = %.4f\n",
              steiger$correct_causal_direction, steiger$steiger_pval))
  write.csv(steiger, file.path(OUT_DIR, paste0("mr_steiger_", outcome_id, ".csv")), row.names = FALSE)
}

# ═══════════════════════════════════════
# 7. Reverse MR (diabetes → APAG)
# ═══════════════════════════════════════
cat("\n========== Step 7: reverse MR (diabetes → APAG) ==========\n")

# Fix: reverse instrumental variables should be extracted from genome-wide significant loci of the outcome (diabetes) GWAS (extract_instruments);
# the old version selected from outcome_data (which only contains SNPs queried for the exposure), which is methodologically invalid.
outcome_sig <- tryCatch({
  extract_instruments(outcomes = outcome_id, p1 = 5e-8, clump = FALSE)
}, error = function(e) {
  cat(sprintf("  [warning] extract_instruments failed: %s\n", e$message))
  NULL
})

if (!is.null(outcome_sig) && nrow(outcome_sig) > 0) {
  outcome_sig <- tryCatch({
    clump_data(outcome_sig, clump_kb = 10000, clump_r2 = 0.001, pop = "EUR")
  }, error = function(e) {
    cat(sprintf("  [warning] reverse clumping failed: %s, using all significant SNPs\n", e$message))
    outcome_sig
  })
  cat(sprintf("  Diabetes GWAS independent significant SNPs (reverse instrumental variables): %d\n", nrow(outcome_sig)))
} else {
  cat("  [skip] unable to obtain outcome GWAS instrumental variables\n")
}

if (!is.null(outcome_sig) && nrow(outcome_sig) > 3) {
  reverse_exp <- outcome_sig

  if (nrow(reverse_exp) >= 3) {
    reverse_snps <- reverse_exp$SNP
    apag_full <- fread(EXPOSURE_FILE)

    reverse_out <- apag_full[apag_full$SNP %in% reverse_snps, ]
    reverse_out <- as.data.frame(reverse_out)   # fread returns a data.table; format_data requires a data.frame
    reverse_out <- format_data(
      reverse_out,
      type = "outcome",
      snp_col = "SNP",
      beta_col = "beta",
      se_col = "se",
      effect_allele_col = "effect_allele",
      other_allele_col = "other_allele",
      eaf_col = "eaf",
      pval_col = "pval",
      samplesize_col = "samplesize"
    )

    reverse_dat <- harmonise_data(exposure_dat = reverse_exp, outcome_dat = reverse_out)
    reverse_dat <- reverse_dat[!(reverse_dat$ambiguous == TRUE & reverse_dat$palindromic == TRUE), ]
    cat(sprintf("  Reverse MR harmonised: %d SNPs\n", nrow(reverse_dat)))

    if (nrow(reverse_dat) >= 3) {
      reverse_mr <- mr(reverse_dat, method_list = c("mr_ivw", "mr_egger_regression", "mr_weighted_median"))
      reverse_mr$or <- exp(reverse_mr$b)
      cat("\n  ═══ Reverse MR (diabetes→APAG) ═══\n")
      print(reverse_mr[, c("method", "nsnp", "b", "se", "pval", "or")])
      write.csv(reverse_mr, file.path(OUT_DIR, paste0("mr_reverse_results_", outcome_id, ".csv")), row.names = FALSE)

      ivw_rev <- reverse_mr[reverse_mr$method == "Inverse variance weighted", ]
      if (nrow(ivw_rev) > 0) {
        if (ivw_rev$pval < 0.05) {
          cat("\n  ⚠️ Reverse MR significant! Bidirectional causation or shared genetic architecture may exist.\n")
        } else {
          cat("\n  ✅ Reverse MR not significant; supports a causal direction of APAG→diabetes.\n")
        }
      }
    }
  }
} else {
  cat("  Diabetes GWAS significant SNPs <3, skipping reverse MR\n")
}

# ═══════════════════════════════════════
# 8. CTRB1/2-specific analysis
# ═══════════════════════════════════════
cat("\n========== Step 8: CTRB1/2 pleiotropy-specific analysis ==========\n")

if ("chr.exposure" %in% names(dat)) {
  dat$is_ctrb <- (dat$chr.exposure == 16 &
                  dat$pos.exposure >= 75200000 &
                  dat$pos.exposure <= 75400000)
} else if (file.exists(EXPOSURE_SIG_FILE)) {
  sig_snps <- fread(EXPOSURE_SIG_FILE)
  ctrb_ids <- sig_snps$SNP[sig_snps$chr == 16 &
                           sig_snps$pos >= 75200000 &
                           sig_snps$pos <= 75400000]
  dat$is_ctrb <- dat$SNP %in% ctrb_ids
} else {
  dat$is_ctrb <- grepl("^rs72802342$|^rs72802358$|^rs889512$", dat$SNP)
}

n_ctrb <- sum(dat$is_ctrb, na.rm = TRUE)
cat(sprintf("  CTRB1/2 region SNPs among instrumental variables: %d / %d\n", n_ctrb, nrow(dat)))

if (n_ctrb > 0) {
  cat("\n--- 8a. All IVs (including CTRB1/2) ---\n")
  mr_all <- mr(dat, method_list = c("mr_ivw", "mr_egger_regression", "mr_weighted_median"))
  print(mr_all[, c("method", "nsnp", "b", "se", "pval")])

  cat("\n--- 8b. After excluding CTRB1/2 ---\n")
  dat_noctrb <- dat[!dat$is_ctrb, ]
  if (nrow(dat_noctrb) >= 3) {
    mr_noctrb <- mr(dat_noctrb, method_list = c("mr_ivw", "mr_egger_regression", "mr_weighted_median"))
    print(mr_noctrb[, c("method", "nsnp", "b", "se", "pval")])
    write.csv(mr_noctrb, file.path(OUT_DIR, paste0("mr_no_ctrb_results_", outcome_id, ".csv")), row.names = FALSE)

    cat("\n  ═══ CTRB1/2 before vs. after exclusion ═══\n")
    ivw_all <- mr_all[mr_all$method == "Inverse variance weighted", ]
    ivw_noctrb <- mr_noctrb[mr_noctrb$method == "Inverse variance weighted", ]
    if (nrow(ivw_all) > 0 && nrow(ivw_noctrb) > 0) {
      cat(sprintf("  IVW (including CTRB1/2): b=%.4f, P=%.4f\n", ivw_all$b, ivw_all$pval))
      cat(sprintf("  IVW (excluding CTRB1/2): b=%.4f, P=%.4f\n", ivw_noctrb$b, ivw_noctrb$pval))
      cat("  → If the effect weakens substantially after excluding CTRB1/2, the original MR result may be driven by this locus.\n")
    }
  }
}

# ═══════════════════════════════════════
# 9. Generate figures
# ═══════════════════════════════════════
cat("\n========== Step 9: generate figures ==========\n")

p_scatter <- mr_scatter_plot(mr_results, dat)
ggsave(file.path(OUT_DIR, paste0("mr_scatter_", outcome_id, ".pdf")), p_scatter[[1]], width = 8, height = 8)

p_forest <- mr_forest_plot(mr_singlesnp(dat))
ggsave(file.path(OUT_DIR, paste0("mr_forest_", outcome_id, ".pdf")), p_forest[[1]], width = 10, height = 6)

p_loo <- mr_leaveoneout_plot(loo)
ggsave(file.path(OUT_DIR, paste0("mr_leaveoneout_", outcome_id, ".pdf")), p_loo[[1]], width = 8, height = 8)

p_funnel <- mr_funnel_plot(mr_singlesnp(dat))
ggsave(file.path(OUT_DIR, paste0("mr_funnel_", outcome_id, ".pdf")), p_funnel[[1]], width = 8, height = 8)

# ═══════════════════════════════════════
# 10. Summary report
# ═══════════════════════════════════════
cat("\n========== MR analysis complete ==========\n")
cat(sprintf("  Outcome: %s\n", outcome_id))
cat(sprintf("  Result directory: %s\n", OUT_DIR))
cat("  Output files:\n")
cat(sprintf("    mr_main_results_%s.csv       — main MR results\n", outcome_id))
cat(sprintf("    mr_heterogeneity_%s.csv      — heterogeneity test\n", outcome_id))
cat(sprintf("    mr_pleiotropy_%s.csv         — pleiotropy test\n", outcome_id))
cat(sprintf("    mr_leaveoneout_%s.csv        — leave-one-out\n", outcome_id))
cat(sprintf("    mr_steiger_%s.csv            — directionality test\n", outcome_id))
cat(sprintf("    mr_scatter_%s.pdf            — scatter plot\n", outcome_id))
cat(sprintf("    mr_forest_%s.pdf             — forest plot\n", outcome_id))
cat(sprintf("    mr_leaveoneout_%s.pdf        — leave-one-out plot\n", outcome_id))
cat(sprintf("    mr_funnel_%s.pdf             — funnel plot\n", outcome_id))
cat("========================================\n")
