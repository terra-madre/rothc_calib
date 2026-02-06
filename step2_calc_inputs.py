# Import necessary libraries
import pandas as pd
from io_utils import *



def get_rothc_pools(cases_info_df, type, initial_pools=None):
    """Prepare carbon inputs and initial pool values for RothC model.
    
    Args:
        soil_data (pd.DataFrame): DataFrame with carbon input data per plot
        type (str): "spinup" or "transient" - determines how pools are initialized
        initial_pools (pd.DataFrame, optional): DataFrame with initial pool values (columns: plot_name, DPM, RPM, BIO, HUM, IOM, SOC)
    
    Returns:
        pd.DataFrame: Carbon inputs and pool values ready for RothC

    """

    rothc_pools = cases_info_df[['case', 'rothc_clay_pct', 'rothc_soc30_t_ha']].copy()
    rothc_pools.rename(columns={'rothc_soc30_t_ha': 'SOC'}, inplace=True)
    clay_pct = rothc_pools['rothc_clay_pct'].values
    rothc_pools.drop(columns=['rothc_clay_pct'], inplace=True)

    pool_cols = ['DPM', 'RPM', 'BIO', 'HUM', 'IOM', 'SOC']

    # If period is "spinup", pools start at zero (will be calculated)
    if type == "spinup":
        for col in pool_cols:
            rothc_pools[col] = 0.0
        return rothc_pools
        
    required_cols = ['case'] + pool_cols
    
    # Use initial_pools if provided
    if initial_pools is not None:
        init_pools = initial_pools.copy()
        rothc_pools = rothc_pools.merge(
            init_pools[required_cols],
            on='case',
            how='left'
        )
        return rothc_pools
    else:
        # If no initial_pools provided, use soc to calculate pool sizes
        # Pool distribution: Calculated using pedotransfer functions (Weihermüller et al., 2013):
        # RPM = (0.1847 × SOC + 0.1555) × (clay + 1.2750)^(-0.1158)
        # HUM = (0.7148 × SOC + 0.5069) × (clay + 0.3421)^(0.0184)
        # BIO = (0.0140 × SOC + 0.0075) × (clay + 8.8473)^(0.0567)
        # IOM = 0.049 × SOC^1.139
        # DPM = 0 (at equilibrium)
        rothc_pools['RPM'] = (0.1847 * rothc_pools['SOC'] + 0.1555) * (clay_pct + 1.2750)**(-0.1158)
        rothc_pools['HUM'] = (0.7148 * rothc_pools['SOC'] + 0.5069) * (clay_pct + 0.3421)**(0.0184)
        rothc_pools['BIO'] = (0.0140 * rothc_pools['SOC'] + 0.0075) * (clay_pct + 8.8473)**(0.0567)
        rothc_pools['IOM'] = 0.049 * rothc_pools['SOC']**1.139
        rothc_pools['DPM'] = 0.0
        return rothc_pools

# Here we calculate the soil C inputs for each case in cases_inputs_df
def calc_c_inputs(cases_inputs_df):
    """Calculate carbon inputs for RothC model based on treatments and case info.
    
    Args:
        cases_inputs_df (pd.DataFrame): DataFrame with treatment info per case (columns: case, year, treatment, etc.)
    
    Returns:
        pd.DataFrame: DataFrame with calculated carbon inputs per case and year
    """
    

    pass