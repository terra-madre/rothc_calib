import pandas as pd
from rothc import rothc

def run_rothc(
        cases_treatments_df,
        cases_info_df,
        climate_df,
        carbon_inputs_df,
        initial_pools_df,
        plant_cover_df,
        soil_depth_cm=30
    ):
    """
    Run RothC model for all cases, looping over a single year of averaged climate data.
    
    Returns:
        tuple: (yearly_results_df, monthly_results_df) with subcase and group columns
    """
    
    # Initialize lists to accumulate results across all cases
    all_years_list = []
    all_months_list = []
    
    # Loop through each subcase and run RothC for each
    for _, subcase in cases_treatments_df.iterrows():

        # Get case information and parameters
        subcase_id = subcase['subcase']
        case_id = subcase['case']
        group = subcase['group']  # control or treatment
        case_info = cases_info_df[cases_info_df['case'] == case_id].iloc[0]
        case_climate = climate_df[climate_df['case'] == case_id].reset_index(drop=True)
        case_c_inputs = carbon_inputs_df[carbon_inputs_df['subcase'] == subcase_id].iloc[0]
        case_plant_cover = plant_cover_df[plant_cover_df['subcase'] == subcase_id].reset_index(drop=True)

        # Get start year and duration
        start_year = 1
        duration_years = int(case_info['duration_years'])
        nsteps = duration_years * 12  # Convert years to months

        # Prepare monthly climate data (single year, 12 months)
        # Ensure we have exactly 12 months of climate data
        if len(case_climate) != 12:
            raise ValueError(f"Expected 12 months of climate data for case {case_id}, got {len(case_climate)}")
        
        monthly = case_climate.copy()
        monthly['t_FYM_Inp'] = 0.0  # FYM inputs already in C_Inp column
        monthly['t_PC'] = 1.0  # Assuming full plant cover; adjust as needed

        # Initialize soil carbon pools
        case_pools = initial_pools_df[initial_pools_df['case'] == case_id].iloc[0]
        DPM = [case_pools['DPM']]
        RPM = [case_pools['RPM']]
        BIO = [case_pools['BIO']]
        HUM = [case_pools['HUM']]
        IOM = [case_pools['IOM']]
        SOC = [DPM[0] + RPM[0] + BIO[0] + HUM[0] + IOM[0]]

        # Set initial soil water content (deficit) and clay content
        SWC = [0.0]
        clay = case_info['rothc_clay_pct']
        c_inputs = case_c_inputs['c_input_t_ha'] / 12  # Convert annual C inputs to monthly
        dpm_rpm = case_c_inputs['dpm_rpm_ratio']

        # Store initial state (year 0, January)
        all_years_list.append([subcase_id, case_id, group, 0, 1, DPM[0], RPM[0], BIO[0], HUM[0], IOM[0], SOC[0]])

        # Run RothC for each month
        for i in range(nsteps):
            # Calculate current year and month (1-indexed months: 1=Jan, 12=Dec)
            current_month = (i % 12) + 1
            current_year = start_year + (i // 12)
            
            # Get climate data for this month (cycling through 12-month climate)
            monthly_idx = i % 12
            TEMP = monthly.iloc[monthly_idx]['t_tmp']
            RAIN = monthly.iloc[monthly_idx]['t_rain']
            PEVAP = monthly.iloc[monthly_idx]['t_evap']
            PC = case_plant_cover.iloc[monthly_idx]['t_PC']  # Plant cover for this month
            FYM_Inp = 0.0  # Amendment inputs already calculated in C_Inp column
            
            DPM_RPM = dpm_rpm
            C_Inp = c_inputs
            depth = soil_depth_cm
            
            # Run RothC for this month
            rothc(12, DPM, RPM, BIO, HUM, IOM, SOC, clay, depth, TEMP, RAIN, PEVAP, 
                  PC, DPM_RPM, C_Inp, FYM_Inp, SWC, RM_TILL=1.0)
            
            # Store monthly results
            all_months_list.append([subcase_id, case_id, group, current_year, current_month, 
                                   DPM[0], RPM[0], BIO[0], HUM[0], IOM[0], SOC[0]])
            
            # Store yearly results (at the end of December)
            if current_month == 12:
                all_years_list.append([subcase_id, case_id, group, current_year, 12, 
                                      DPM[0], RPM[0], BIO[0], HUM[0], IOM[0], SOC[0]])

    # Create output dataframes
    year_columns = ["subcase", "case", "group", "year", "month", "DPM_t_C_ha", "RPM_t_C_ha", 
                   "BIO_t_C_ha", "HUM_t_C_ha", "IOM_t_C_ha", "SOC_t_C_ha"]
    month_columns = ["subcase", "case", "group", "year", "month", "DPM_t_C_ha", "RPM_t_C_ha", 
                    "BIO_t_C_ha", "HUM_t_C_ha", "IOM_t_C_ha", "SOC_t_C_ha"]
    
    output_years = pd.DataFrame(all_years_list, columns=year_columns)
    output_months = pd.DataFrame(all_months_list, columns=month_columns)

    return output_years, output_months
