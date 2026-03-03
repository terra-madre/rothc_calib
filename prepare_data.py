"""
prepare_data.py
---------------
Data-preparation pipeline for RothC calibration.

Run this once (or whenever raw inputs change) to produce the processed
files that the optimisation scripts consume via precompute_data().

Steps
  1. Preprocess cases    — fetch SoilGrids, ERA5 climate, NUTS3   (slow, API calls)
  2. Fetch stat. yields  — EUROSTAT statistical yields by NUTS     (slow, API calls)
  3. Compute derivatives — initial pools + plant-cover schedule    (fast, deterministic)
  4. Outlier filtering   — Modified Z-score → no_outliers subset   (fast)

Outputs
  inputs/processed/
    cases_info.csv  cases_treatments.csv  initial_pools.csv  plant_cover.csv
  inputs/processed/no_outliers/
    cases_info.csv  cases_treatments.csv  initial_pools.csv  plant_cover.csv

Usage (from project root):
    cd git_code && conda run -n terra-plus python prepare_data.py && cd ..
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import preprocess as step1
import calc_initial_pools as step3
import calc_plant_cover as step4

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(do_preprocess_cases=False, do_get_st_yields=False,
        do_compute_derived=True, do_remove_outliers=True,
        soil_depth=30):
    """Execute the data-preparation pipeline.

    Args:
        do_preprocess_cases: Fetch/enrich case data from APIs (slow).
        do_get_st_yields: Fetch EUROSTAT statistical yields (slow).
        do_compute_derived: Compute initial_pools.csv + plant_cover.csv.
        do_remove_outliers: Create the no_outliers/ subset.
        soil_depth: Soil depth in cm.
    """
    input_dir = REPO_ROOT / "inputs"
    loc_data_dir = input_dir / "loc_data"
    loc_data_dir.mkdir(parents=True, exist_ok=True)
    proc_data_dir = input_dir / "processed"
    proc_data_dir.mkdir(parents=True, exist_ok=True)
    fixed_data_dir = input_dir / "fixed_values"
    fixed_data_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: preprocess cases ──────────────────────────────────────────────
    if do_preprocess_cases:
        print("Step 1: Preprocessing cases (fetching external data)…")
        cases_info_raw_df = pd.read_csv(input_dir / "raw" / "cases_info.csv")
        cases_treatments_df = pd.read_csv(input_dir / "raw" / "cases_treatments.csv")

        cases_info_df, cases_treatments_df = step1.prepare_variables(
            cases_info_raw_df, cases_treatments_df,
        )
        cases_info_df, climate_df = step1.prepare_cases_df(
            cases_info_df,
            input_dir=input_dir,
            loc_data_dir=loc_data_dir,
            soil_depth_cm=soil_depth,
            nuts_version="2024",
        )

        cases_info_df.to_csv(proc_data_dir / "cases_info.csv", index=False)
        cases_treatments_df.to_csv(proc_data_dir / "cases_treatments.csv", index=False)
        print(f"  Saved {len(cases_info_df)} cases → {proc_data_dir.relative_to(REPO_ROOT)}/")
    else:
        cases_info_df = pd.read_csv(proc_data_dir / "cases_info.csv")
        cases_treatments_df = pd.read_csv(proc_data_dir / "cases_treatments.csv")
        print(f"Step 1: Loaded {len(cases_info_df)} cases from "
              f"{proc_data_dir.relative_to(REPO_ROOT)}/")

    # ── Step 2: statistical yields ────────────────────────────────────────────
    if do_get_st_yields:
        print("Step 2: Fetching statistical yields…")
        st_yields_all = step1.get_st_yields(
            cases_info_df=cases_info_df,
            fixed_data_dir=fixed_data_dir,
            loc_data_dir=loc_data_dir,
        )
        print(f"  Saved {len(st_yields_all)} rows → {loc_data_dir.relative_to(REPO_ROOT)}/")
    else:
        print("Step 2: Statistical yields — skipped (already on disk)")

    # ── Step 3: derived files (initial pools + plant cover) ───────────────────
    if do_compute_derived:
        print("Step 3: Computing initial pools and plant cover…")
        initial_pools_df = step3.get_rothc_pools(cases_info_df, type="transient")
        initial_pools_df.to_csv(proc_data_dir / "initial_pools.csv", index=False)

        plant_cover_df = step4.plant_cover(cases_treatments_df)
        plant_cover_df.to_csv(proc_data_dir / "plant_cover.csv", index=False)

        print(f"  initial_pools.csv : {len(initial_pools_df)} rows")
        print(f"  plant_cover.csv   : {len(plant_cover_df)} rows")
    else:
        print("Step 3: Derived files — skipped")

    # ── Step 4: outlier filtering ─────────────────────────────────────────────
    if do_remove_outliers:
        print("Step 4: Outlier filtering (Modified Z-score, threshold=2.5)…")
        no_outlier_dir = proc_data_dir / "no_outliers"
        no_outlier_dir.mkdir(parents=True, exist_ok=True)

        pc_path = proc_data_dir / "plant_cover.csv"
        plant_cover_df = pd.read_csv(pc_path) if pc_path.exists() else None

        filtered_ci, filtered_ct, filtered_pc, removed = step1.remove_outlier_cases(
            cases_info_df.copy(),
            cases_treatments_df.copy(),
            plant_cover_df=plant_cover_df.copy() if plant_cover_df is not None else None,
        )

        filtered_ci.to_csv(no_outlier_dir / "cases_info.csv", index=False)
        filtered_ct.to_csv(no_outlier_dir / "cases_treatments.csv", index=False)
        if filtered_pc is not None:
            filtered_pc.to_csv(no_outlier_dir / "plant_cover.csv", index=False)

        filtered_pools = step3.get_rothc_pools(filtered_ci, type="transient")
        filtered_pools.to_csv(no_outlier_dir / "initial_pools.csv", index=False)

        print(f"  Removed {len(removed)} cases: {sorted(removed)}")
        print(f"  Remaining: {len(filtered_ci)} cases")
        print(f"  Saved to {no_outlier_dir.relative_to(REPO_ROOT)}/")
    else:
        print("Step 4: Outlier filtering — skipped")

    print("\nData preparation complete.")


if __name__ == "__main__":
    run()
