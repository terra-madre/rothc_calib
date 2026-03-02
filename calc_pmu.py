"""
calc_pmu.py
-----------
Calculates the Pooled Measurement Uncertainty (PMU) for the calibration dataset.

PMU formula:
    PMU = sqrt( sum_j( sigma_j^2 * (n_j - 1) ) / sum_j( n_j - 1 ) )

where sigma_j is the standard error and n_j the number of replicates
of the j-th observation (see OPTIMIZATION_CHAT_INSTRUCTIONS.md §6).

Steps:
  1. Read inputs/raw/cases_uncertainty.csv
  2. Compute mean delta CV (ignoring missing values)
  3. Impute missing 'delta sd (t C/ha/y)' as:  mean_CV * |Delta SOC (t C/ha/year)|
  4. Impute missing 'delta se (t C/ha/y)' as:  sd / sqrt(n)   (where n is available)
  5. Compute PMU using SE as sigma_j, over all cases that have both se and n >= 2
"""

import math
import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_IN  = ROOT / "inputs" / "raw" / "cases_uncertainty.csv"
CSV_OUT = ROOT / "outputs" / "cases_uncertainty_filled.csv"

# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_IN, dtype=str)

# Rename columns to shorter aliases for convenience
col_delta_soc = "delta SOC (t C/ha/year)"
col_n         = "n"
col_cv        = "delta CV"
col_sd        = "delta sd (t C/ha/y)"
col_se        = "delta se  (t C/ha/y)"   # two spaces — matches header exactly

# Convert numeric columns; coerce non-numeric (empty / text) → NaN
for col in [col_delta_soc, col_n, col_cv, col_sd, col_se]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Loaded {len(df)} cases from {CSV_IN.name}")

# ---------------------------------------------------------------------------
# Step 2 — Mean delta CV (ignoring missing)
# ---------------------------------------------------------------------------
mean_cv = df[col_cv].mean()          # pandas .mean() skips NaN by default
n_cv    = df[col_cv].notna().sum()
print(f"\nStep 2 — Mean delta CV: {mean_cv:.4f}  (from {n_cv} cases with CV reported)")

# ---------------------------------------------------------------------------
# Step 3 — Impute missing delta sd (t C/ha/y)
# ---------------------------------------------------------------------------
missing_sd = df[col_sd].isna()

df.loc[missing_sd, col_sd] = mean_cv * df.loc[missing_sd, col_delta_soc].abs()

n_imputed = missing_sd.sum()
print(f"\nStep 3 — Imputed {n_imputed} missing 'delta sd (t C/ha/y)' values")
print(f"         (formula: mean_CV × |Delta SOC| = {mean_cv:.4f} × |Delta SOC|)")

# ---------------------------------------------------------------------------
# Step 4 — Impute missing delta se (t C/ha/y) where n is available
# ---------------------------------------------------------------------------
can_impute_se = df[col_se].isna() & df[col_n].notna() & (df[col_n] >= 1)
df.loc[can_impute_se, col_se] = (
    df.loc[can_impute_se, col_sd] / np.sqrt(df.loc[can_impute_se, col_n])
)

n_imputed_se = can_impute_se.sum()
print(f"\nStep 4 — Imputed {n_imputed_se} missing 'delta se (t C/ha/y)' values")
print(f"         (formula: sd / sqrt(n))")

# ---------------------------------------------------------------------------
# Step 5 — Pooled Measurement Uncertainty (PMU)
#
# sigma_j = standard error (delta se)
# Only cases with:
#   • a valid (possibly imputed) delta se (t C/ha/y)
#   • n >= 2 so that (n – 1) > 0
# contribute to the sum.
# ---------------------------------------------------------------------------
pmu_mask = df[col_se].notna() & df[col_n].notna() & (df[col_n] >= 2)
pmu_df   = df[pmu_mask].copy()

sigma = pmu_df[col_se].values
n_j   = pmu_df[col_n].values
w     = n_j - 1       # degrees-of-freedom weights

numerator   = np.sum(sigma**2 * w)
denominator = np.sum(w)
pmu         = math.sqrt(numerator / denominator)

print(f"\nStep 5 — PMU calculation")
print(f"         Cases used     : {len(pmu_df)} (out of {len(df)} total)")
print(f"         Excluded cases : {len(df) - len(pmu_df)}  "
      f"(n missing or n < 2: cannot contribute)")
print(f"         Numerator      : sum(sigma^2 * (n-1)) = {numerator:.6f}")
print(f"         Denominator    : sum(n-1)             = {denominator:.0f}")
print(f"\n  *** PMU = {pmu:.4f} t C/ha/y ***\n")

# ---------------------------------------------------------------------------
# Save filled table for reference
# ---------------------------------------------------------------------------
df.to_csv(CSV_OUT, index=False)
print(f"Filled uncertainty table saved to {CSV_OUT.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Breakdown by case (informational)
# ---------------------------------------------------------------------------
print("\nPer-case contributions to PMU:")
print(f"{'Case':>6}  {'SE (sigma)':>10}  {'n':>4}  {'n-1':>4}  {'sigma^2*(n-1)':>14}  {'imputed_se':>10}")
print("-" * 64)
for _, row in pmu_df.iterrows():
    s   = row[col_se]
    nv  = int(row[col_n])
    wv  = nv - 1
    contrib = s**2 * wv
    was_imputed = can_impute_se.loc[row.name]
    print(f"{int(row['case']):>6}  {s:>10.4f}  {nv:>4}  {wv:>4}  {contrib:>14.6f}  {'yes' if was_imputed else 'no':>10}")
