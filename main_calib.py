# This is the main scrpt for a calibration of the RothC model

# Import necessary modules
from pathlib import Path
import pandas as pd
import step1_preprocess as step1
import step2_calc_inputs as step2

# configuration and paths
repo_root = Path(__file__).resolve().parents[1]
input_dir = repo_root / "inputs"
loc_data_dir = input_dir / "loc_data"
loc_data_dir.mkdir(parents=True, exist_ok=True)
proc_data_dir = input_dir / "processed"
proc_data_dir.mkdir(parents=True, exist_ok=True)
fixed_data_dir = input_dir / "fixed_values"

do_preprocess_cases = False  # Whether to run the preprocessing step (fetching/enriching data for cases)
do_get_st_yields = True  # Whether to calculate st_yields (can be time-consuming)
soil_depth = 30  # Soil depth in cm

cases_info_raw_df = pd.read_csv(input_dir / "raw" / "cases_info.csv")
cases_treatments_df = pd.read_csv(input_dir / "raw" / "cases_treatments.csv")

# Load fixed parameter files
ps_herbaceous = pd.read_csv(fixed_data_dir / "ps_herbaceous.csv")
ps_management = pd.read_csv(fixed_data_dir / "ps_management.csv")
ps_general = pd.read_csv(fixed_data_dir / "ps_general.csv")
ps_trees = pd.read_csv(fixed_data_dir / "ps_trees.csv")
ps_amendments = pd.read_csv(fixed_data_dir / "ps_amendments.csv")

# Step 1: Preprocess cases (fetch/enrich data)
if do_preprocess_cases:

    # Step 1: fetch/prepare auxiliary datasets and enrich cases
    cases_info_df = step1.prepare_cases_df(
        cases_info_raw_df,
        input_dir=input_dir,
        loc_data_dir=loc_data_dir,
        proc_data_dir=proc_data_dir,
        soil_depth_cm=soil_depth,
        nuts_version='2024'
    )

else:
    # Just load the preprocessed data
    cases_info_df = pd.read_csv(proc_data_dir / "cases_info.csv")

if do_get_st_yields:
    st_yields_all = step1.get_st_yields(
        cases_info_df=cases_info_df,
        fixed_data_dir=fixed_data_dir, 
        loc_data_dir=loc_data_dir
    )
else:
    st_yields_all = pd.read_csv(loc_data_dir / "st_yields_selected.csv")


# Step 2: Calculate soil C inputs and initial SOC stocks for each case

carbon_inputs_df = step2.calc_c_inputs(
    cases_treatments_df=cases_treatments_df,
    cases_info_df=cases_info_df,
    fixed_data_dir=fixed_data_dir,
    st_yields_all=st_yields_all,
    ps_herbaceous=ps_herbaceous,
    ps_management=ps_management,
    ps_general=ps_general,
    ps_trees=ps_trees,
    ps_amendments=ps_amendments
)

initial_pools_df = step2.get_rothc_pools(cases_info_df, type="transient")

# Display results
print("\n=== Carbon Inputs Summary ===")
print(carbon_inputs_df.head())
print(f"\nTotal rows: {len(carbon_inputs_df)}")
print(f"\nC input statistics (t/ha):")
print(carbon_inputs_df['c_input_t_ha'].describe())
print(f"\nDPM/RPM ratio statistics:")
print(carbon_inputs_df['dpm_rpm_ratio'].describe())

print("\n=== Initial Pools Summary ===")
print(initial_pools_df.head())

print("\nProcessing complete!")
