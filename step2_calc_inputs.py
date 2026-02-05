# Import necessary libraries
from pathlib import Path
from git_code.rothc import *
import pandas as pd
from git_code.io_utils import *

# Configurations and paths
soil_depth = 30  # Soil depth in cm
cases_df = pd.read_csv("./inputs/data_clean.csv")
location = cases_df.iloc[0]['lonlat']
case_id = cases_df.iloc[0]['case']
loc_data_dir = Path("./inputs/loc_data/") / location
output_dir = Path("./outputs/") / case_id
output_dir.mkdir(parents=True, exist_ok=True)

# Read soil girds and climate data
soil_grids_file = loc_data_dir / 'soil_data' / 'soil_data_soilgrids.csv'
farm_soil_grids = pd.read_csv(soil_grids_file)
climate_file = loc_data_dir / 'climate_data' / 'climate_monthly_mean.csv'
farm_climate = pd.read_csv(climate_file)


rothc_carbon_baseline = prepare_rothc_carbon(marged_yearsel, year, "baseline", initial_pools=soilc_baseline,
                                                output_dir=output_dir, farm_id=farm_id)


cases_inputs_df = pd.read_csv("./inputs/data_inputs.csv")

pools_baseline = []
pools_project = []

# Loop through each plot and run RothC for baseline and project scenarios
for _, case in cases_inputs_df.iterrows():

    # Clay percentage is the same for both scenarios
    clay = case['clay_perc']
    # Get tillage decomposition modifiers for this plot
    tillage_data = processed_data['tillage']
    plot_tillage = tillage_data[tillage_data['plot_name'] == plot_name].iloc[0]
    tillage_modifier_bl = plot_tillage['decomp_modifier_bl']
    tillage_modifier_pr = plot_tillage['decomp_modifier_pr']

    # Prepare common monthly data
    monthly = rothc_climate.copy()
    monthly['t_FYM_Inp'] = 0.0
    monthly['t_PC'] = 1
    
    # --- Run RothC for baseline scenario --- #
    # Get baseline carbon inputs for this plot and add to monthly data, then run RothC
    plot_carbon_baseline = rothc_carbon_baseline[rothc_carbon_baseline['plot_name'] == plot_name].iloc[0]
    c_inp_total = plot_carbon_baseline['t_C_Inp']
    dpm_rpm = plot_carbon_baseline['t_DPM_RPM']
    monthly['t_C_Inp'] = c_inp_total / 12  # Split evenly over 12 months
    monthly['t_DPM_RPM'] = dpm_rpm
    rothc_results = rothc_transient(year, clay, soil_depth, monthly, plot_carbon_baseline, tillage_modifier=tillage_modifier_bl)
    rothc_results['plot_name'] = plot_name
    pools_baseline.append(rothc_results)

    # --- Run RothC for project scenario --- #
    # Get project carbon inputs for this plot and add to monthly data, then run RothC
    plot_carbon_project = rothc_carbon_project[rothc_carbon_project['plot_name'] == plot_name].iloc[0]
    c_inp_total = plot_carbon_project['t_C_Inp']
    dpm_rpm = plot_carbon_project['t_DPM_RPM']
    monthly['t_C_Inp'] = c_inp_total / 12  # Split evenly over 12 months
    monthly['t_DPM_RPM'] = dpm_rpm
    rothc_results = rothc_transient(year, clay, soil_depth, monthly, plot_carbon_project, tillage_modifier=tillage_modifier_pr)
    rothc_results['plot_name'] = plot_name
    pools_project.append(rothc_results)

# Convert to DataFrame add year column and reorder columns
pools_baseline = pd.DataFrame(pools_baseline)
pools_project = pd.DataFrame(pools_project)
pools_baseline['year'] = year  # Initial soil carbon is for the year before project start
pools_project['year'] = year
cols_order = ['plot_name', 'year', 'DPM', 'RPM', 'BIO', 'HUM', 'IOM', 'SOC']
pools_baseline = pools_baseline[cols_order]
pools_project = pools_project[cols_order]

soilc_baseline = pd.concat([soilc_baseline, pools_baseline], ignore_index=True)
soilc_project = pd.concat([soilc_project, pools_project], ignore_index=True)

# Save the RothC input datasets as separate CSV files
for input_type in ["climate", "soil", "carbon_baseline", "carbon_project"]:
    df = locals()[f"rothc_{input_type}"]
    output_file = soilc_dir_path / f'rothc_{input_type}_{year}.csv'
    df.to_csv(output_file, index=False)
    