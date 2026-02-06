# This is the main scrpt for a calibration of the RothC model

# Import necessary modules
from pathlib import Path
import pandas as pd
import step1_preprocess as step1
import step2_calc_inputs as step2

# configuration and paths
repo_root = Path(__file__).resolve().parents[1]
input_dir = repo_root / "inputs"
cases_info_raw_file = input_dir / "raw" / "cases_info.csv"
cases_treatments_raw_file = input_dir / "raw" / "cases_treatments.csv"
loc_data_dir = input_dir / "loc_data"
loc_data_dir.mkdir(parents=True, exist_ok=True)
cases_info_proc_file = input_dir / "processed" / "cases_info.csv"

do_preprocess_cases = True  # Whether to run the preprocessing step (fetching/enriching data for cases)
soil_depth = 30  # Soil depth in cm


# Step 1: Preprocess cases (fetch/enrich data)

if do_preprocess_cases:

    # Read raw data from files
    cases_info_df = pd.read_csv(cases_info_raw_file)

    # Step 1: fetch/prepare auxiliary datasets and enrich cases
    cases_info_df = step1.prepare_cases_df(
        cases_info_df,
        input_dir=input_dir,
        loc_data_dir=loc_data_dir,
        soil_depth_cm=soil_depth,
        nuts_version='2024'
    )

    # Save processed cases table
    cases_info_df.to_csv(cases_info_proc_file, index=False)

else:
    # Just load the preprocessed cases info
    cases_info_df = pd.read_csv(cases_info_proc_file)


# Step 2: Calculate soil C inputs and initial SOC stocks for each case

cases_treatments_df = pd.read_csv(cases_treatments_raw_file)

cases_initial_pools_df = step2.get_rothc_pools(cases_info_df, type="transient")

cases_inputs_df = step2.calc_c_inputs(cases_treatments_df)