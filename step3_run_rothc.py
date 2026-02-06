
pools_baseline = []
pools_project = []
# Loop through each plot and run RothC for baseline and project scenarios
for _, case in cases_inputs_df.iterrows():

    clay = case['rothc_clay_pct']
    
    # Prepare monthly data
    monthly = rothc_climate.copy()
    monthly['t_FYM_Inp'] = 0.0
    monthly['t_PC'] = 1

    # --- Run RothC --- #
    # Get carbon inputs for this case and add to monthly data, then run RothC
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