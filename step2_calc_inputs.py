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
def calc_c_herb(
    cases_treatments_df,
    cases_info_df,
    fixed_data_dir,
    st_yields_all,
    ps_herbaceous,
    ps_management,
    ps_general
):
    """Calculate carbon inputs for RothC model based on treatments and case info.
    
    Args:
        cases_treatments_df (pd.DataFrame): DataFrame with treatment info per case
        cases_info_df (pd.DataFrame): DataFrame with case information (nuts3, map_mm)
        fixed_data_dir (Path): Directory containing fixed input data files
        st_yields_all (pd.DataFrame): DataFrame with selected yields by nuts3
        ps_herbaceous (pd.DataFrame): DataFrame with herbaceous plant parameters
        ps_management (pd.DataFrame): DataFrame with management parameters
        ps_general (pd.DataFrame): DataFrame with general parameters
    
    Returns:
        pd.DataFrame: DataFrame with calculated carbon inputs (case, group, c_input_herbaceous_t_ha)
    """
    
    # Merge with cases_info to get nuts3 and map_mm
    df = cases_treatments_df.merge(
        cases_info_df[['case', 'nuts3', 'map_mm']],
        on='case',
        how='left'
    )
    
    # Get map_to_prod coefficient from ps_general
    map_to_prod = ps_general[ps_general['name'] == 'map_to_prod']['value'].values[0]
    
    # Initialize result column
    df['c_input_herbaceous_t_ha'] = 0.0
    
    # Process each row
    for idx, row in df.iterrows():
        total_c_input = 0.0
        
        # Get case info
        nuts3 = row['nuts3']
        map_mm = row['map_mm']
        irrigation = row['irrigation']
        
        # --- Process main crops (crop1, crop2, crop3) ---
        crop_cols = ['crop1_name', 'crop2_name', 'crop3_name']
        crop_perc_cover = row['crop_perc_cover'] / 100 if pd.notna(row['crop_perc_cover']) else 0
        crops_residues = row['crops_residues']
        
        for crop_col in crop_cols:
            crop_name = row[crop_col]
            if pd.isna(crop_name) or crop_name == 'NA':
                continue
                
            # Get plant parameters
            crop_params = ps_herbaceous[ps_herbaceous['group_cover'] == crop_name]
            if crop_params.empty:
                print(f"Warning: No parameters found for crop '{crop_name}'")
                continue
            crop_params = crop_params.iloc[0]
            
            # Get yield
            yield_data = st_yields_all[(st_yields_all['nuts3'] == nuts3) & 
                                       (st_yields_all['group_name'] == crop_name)]
            if yield_data.empty:
                print(f"Warning: No yield data for crop '{crop_name}' in nuts3 '{nuts3}'")
                continue
            
            # Select yield based on irrigation
            if irrigation:
                yield_tdm_ha = yield_data['yield_dry_irrigated_t_ha'].values[0]
            else:
                yield_tdm_ha = yield_data['yield_dry_rainfed_t_ha'].values[0]
            
            if pd.isna(yield_tdm_ha):
                continue
            
            # Get management parameters
            mgmt_params = ps_management[ps_management['management'] == crops_residues]
            if mgmt_params.empty:
                print(f"Warning: No management parameters for '{crops_residues}'")
                continue
            frac_remaining = mgmt_params['frac_remaining'].values[0]
            
            # Calculate biomass and inputs
            residues_tdm_ha = yield_tdm_ha * crop_params['residue_yield_ratio (kg/kg)']
            agb_t_ha = yield_tdm_ha + residues_tdm_ha
            bgb_t_ha = agb_t_ha * crop_params['r_s_ratio (kg/kg)']
            
            agb_input_t_ha = (agb_t_ha - yield_tdm_ha) * frac_remaining
            bgb_input_t_ha = bgb_t_ha * crop_params['turnover_bg (y-1)']
            
            c_input_t_ha = (agb_input_t_ha + bgb_input_t_ha) * crop_params['c_frac (kgC/kgDM)'] * crop_perc_cover
            total_c_input += c_input_t_ha
        
        # --- Process covercrop ---
        covercrop_name = row['covercrop']
        if pd.notna(covercrop_name) and covercrop_name != 'NA':
            # Get plant parameters
            cc_params = ps_herbaceous[ps_herbaceous['group_cover'] == covercrop_name]
            if not cc_params.empty:
                cc_params = cc_params.iloc[0]
                
                # Get management parameters
                covercrop_residues = row['covercrop_residues']
                mgmt_params = ps_management[ps_management['management'] == covercrop_residues]
                if not mgmt_params.empty:
                    frac_remaining = mgmt_params['frac_remaining'].values[0]
                    
                    # Calculate biomass using precipitation productivity
                    agb_t_ha = map_mm * map_to_prod
                    bgb_t_ha = agb_t_ha * cc_params['r_s_ratio (kg/kg)']
                    
                    agb_input_t_ha = agb_t_ha * frac_remaining
                    bgb_input_t_ha = bgb_t_ha * cc_params['turnover_bg (y-1)']
                    
                    c_input_t_ha = (agb_input_t_ha + bgb_input_t_ha) * cc_params['c_frac (kgC/kgDM)']
                    total_c_input += c_input_t_ha
        
        # --- Process grassland ---
        grass_perc_cover = row['grass_perc_cover'] / 100 if pd.notna(row['grass_perc_cover']) else 0
        if grass_perc_cover > 0:
            # Get parameters for "grassland - permanent grasses or shrubs"
            grass_params = ps_herbaceous[ps_herbaceous['group_cover'] == 'grassland - permanent grasses or shrubs']
            if not grass_params.empty:
                grass_params = grass_params.iloc[0]
                
                # Get management parameters for natural grassland with continuous grazing
                grass_mgmt = 'natural grasses/shrubs with continuous grazing'
                mgmt_params = ps_management[ps_management['management'] == grass_mgmt]
                if not mgmt_params.empty:
                    frac_remaining = mgmt_params['frac_remaining'].values[0]
                    prod_modifier = mgmt_params['prod_modifier'].values[0]
                    
                    # Calculate biomass using precipitation productivity with modifier
                    agb_t_ha = map_mm * map_to_prod * prod_modifier
                    bgb_t_ha = agb_t_ha * grass_params['r_s_ratio (kg/kg)']
                    
                    agb_input_t_ha = agb_t_ha * frac_remaining
                    bgb_input_t_ha = bgb_t_ha * grass_params['turnover_bg (y-1)']
                    
                    c_input_t_ha = (agb_input_t_ha + bgb_input_t_ha) * grass_params['c_frac (kgC/kgDM)'] * grass_perc_cover
                    total_c_input += c_input_t_ha
        
        # Store result
        df.at[idx, 'c_input_herbaceous_t_ha'] = round(total_c_input, 2)
    
    # Return result with only needed columns
    result = df[['case', 'group', 'c_input_herbaceous_t_ha']].copy()
    return result


def calc_c_tree(
    cases_treatments_df,
    ps_trees,
    ps_management
):
    """Calculate carbon inputs from tree crops based on treatments.
    
    Args:
        cases_treatments_df (pd.DataFrame): DataFrame with treatment info per case
        ps_trees (pd.DataFrame): DataFrame with tree species parameters
        ps_management (pd.DataFrame): DataFrame with management parameters
    
    Returns:
        pd.DataFrame: DataFrame with calculated carbon inputs (case, group, c_input_tree_t_ha)
    """
    
    df = cases_treatments_df.copy()
    df['c_input_tree_t_ha'] = 0.0
    
    # Process each row
    for idx, row in df.iterrows():
        tree_name = row['tree_name']
        
        if pd.isna(tree_name) or tree_name == 'NA':
            continue
        
        # Get tree parameters
        tree_params = ps_trees[ps_trees['species'] == tree_name]
        if tree_params.empty:
            print(f"Warning: No parameters found for tree '{tree_name}'")
            continue
        tree_params = tree_params.iloc[0]
        
        # Get AGB from input data (already provided in t/ha)
        agb_t_ha = row['tree_agb_t_ha']
        if pd.isna(agb_t_ha) or agb_t_ha == 0:
            continue
        
        # Calculate BGB
        bgb_t_ha = agb_t_ha * tree_params['r_s_ratio (kg/kg)']
        
        # Calculate carbon content
        c_agb_t_ha = agb_t_ha * tree_params['dry_c']
        c_bgb_t_ha = bgb_t_ha * tree_params['dry_c']
        
        # Calculate litter production
        c_leaf_litter_t_ha = c_agb_t_ha * tree_params['turnover_ag (y-1)']
        c_root_litter_t_ha = c_bgb_t_ha * tree_params['fine_tot_r_ratio (kg/kg)'] * tree_params['turnover_bg (y-1)']
        
        # Calculate pruning carbon
        c_pruning_total_t_ha = c_agb_t_ha * tree_params['pruning_ag (kg/kg)']
        
        # Get management parameters for pruning
        tree_prunings = row['tree_prunings']
        if pd.notna(tree_prunings) and tree_prunings != 'NA':
            mgmt_params = ps_management[ps_management['management'] == tree_prunings]
            if not mgmt_params.empty:
                frac_remaining = mgmt_params['frac_remaining'].values[0]
                c_pruning_remaining_t_ha = c_pruning_total_t_ha * frac_remaining
            else:
                c_pruning_remaining_t_ha = 0.0
        else:
            c_pruning_remaining_t_ha = 0.0
        
        # Total C input = litter + prunings
        total_c_input = c_leaf_litter_t_ha + c_root_litter_t_ha + c_pruning_remaining_t_ha
        df.at[idx, 'c_input_tree_t_ha'] = round(total_c_input, 2)
    
    # Return result with only needed columns
    result = df[['case', 'group', 'c_input_tree_t_ha']].copy()
    return result


def calc_c_amend(
    cases_treatments_df,
    ps_amendments
):
    """Calculate carbon inputs from amendments based on treatments.
    
    Args:
        cases_treatments_df (pd.DataFrame): DataFrame with treatment info per case
        ps_amendments (pd.DataFrame): DataFrame with amendment parameters
    
    Returns:
        pd.DataFrame: DataFrame with calculated carbon inputs (case, group, c_input_amend_t_ha)
    """
    
    df = cases_treatments_df.copy()
    df['c_input_amend_t_ha'] = 0.0
    
    # Process each row
    for idx, row in df.iterrows():
        amend_name = row['amend_name']
        
        if pd.isna(amend_name) or amend_name == 'NA':
            continue
        
        # Get amendment parameters
        amend_params = ps_amendments[ps_amendments['sub_type'] == amend_name]
        if amend_params.empty:
            print(f"Warning: No parameters found for amendment '{amend_name}'")
            continue
        amend_params = amend_params.iloc[0]
        
        # Get fresh amendment amount from input data
        amend_fresh_t_ha = row['amend_fresh_t_ha']
        if pd.isna(amend_fresh_t_ha) or amend_fresh_t_ha == 0:
            continue
        
        # Calculate dry matter amount
        amend_dry_t_ha = amend_fresh_t_ha * amend_params['dry_frac (kgDM/kgFM)']
        
        # Calculate carbon input
        c_input_t_ha = amend_dry_t_ha * amend_params['c_frac (kgC/kgDM)']
        
        df.at[idx, 'c_input_amend_t_ha'] = round(c_input_t_ha, 2)
    
    # Return result with only needed columns
    result = df[['case', 'group', 'c_input_amend_t_ha']].copy()
    return result


def calc_c_inputs(
    cases_treatments_df,
    cases_info_df,
    fixed_data_dir,
    st_yields_all,
    ps_herbaceous,
    ps_management,
    ps_general,
    ps_trees,
    ps_amendments
):
    """Calculate total carbon inputs for RothC model with weighted DPM/RPM ratio.
    
    This is a wrapper function that:
    1. Calls calc_c_herb, calc_c_tree, and calc_c_amend
    2. Sums all C inputs
    3. Calculates weighted DPM/RPM ratio based on input sources
    
    Args:
        cases_treatments_df (pd.DataFrame): DataFrame with treatment info per case
        cases_info_df (pd.DataFrame): DataFrame with case information (nuts3, map_mm)
        fixed_data_dir (Path): Directory containing fixed input data files
        st_yields_all (pd.DataFrame): DataFrame with selected yields by nuts3
        ps_herbaceous (pd.DataFrame): DataFrame with herbaceous plant parameters
        ps_management (pd.DataFrame): DataFrame with management parameters
        ps_general (pd.DataFrame): DataFrame with general parameters
        ps_trees (pd.DataFrame): DataFrame with tree species parameters
        ps_amendments (pd.DataFrame): DataFrame with amendment parameters
    
    Returns:
        pd.DataFrame: DataFrame with calculated carbon inputs (case, group, c_input_t_ha, dpm_rpm_ratio)
    """
    
    # Calculate C inputs from each source
    c_herb = calc_c_herb(
        cases_treatments_df=cases_treatments_df,
        cases_info_df=cases_info_df,
        fixed_data_dir=fixed_data_dir,
        st_yields_all=st_yields_all,
        ps_herbaceous=ps_herbaceous,
        ps_management=ps_management,
        ps_general=ps_general
    )
    
    c_tree = calc_c_tree(
        cases_treatments_df=cases_treatments_df,
        ps_trees=ps_trees,
        ps_management=ps_management
    )
    
    c_amend = calc_c_amend(
        cases_treatments_df=cases_treatments_df,
        ps_amendments=ps_amendments
    )
    
    # Merge all results
    result = c_herb.merge(c_tree[['case', 'group', 'c_input_tree_t_ha']], on=['case', 'group'], how='left')
    result = result.merge(c_amend[['case', 'group', 'c_input_amend_t_ha']], on=['case', 'group'], how='left')
    
    # Fill NaN with 0
    result['c_input_tree_t_ha'] = result['c_input_tree_t_ha'].fillna(0)
    result['c_input_amend_t_ha'] = result['c_input_amend_t_ha'].fillna(0)
    
    # Get DPM/RPM ratios from ps_general
    dr_ratio_annuals = ps_general[ps_general['name'] == 'dr_ratio_annuals']['value'].values[0]
    dr_ratio_treegrass = ps_general[ps_general['name'] == 'dr_ratio_treegrass']['value'].values[0]
    dr_ratio_amend = ps_general[ps_general['name'] == 'dr_ratio_amend']['value'].values[0]
    
    # Calculate weighted DPM/RPM ratio
    # DPM/RPM = (C_herb * dr_annuals + C_tree * dr_treegrass + C_amend * dr_amend) / C_total
    result['c_input_t_ha'] = (
        result['c_input_herbaceous_t_ha'] + 
        result['c_input_tree_t_ha'] + 
        result['c_input_amend_t_ha']
    )
    
    result['dpm_rpm_ratio'] = (
        (result['c_input_herbaceous_t_ha'] * dr_ratio_annuals +
         result['c_input_tree_t_ha'] * dr_ratio_treegrass +
         result['c_input_amend_t_ha'] * dr_ratio_amend) /
        result['c_input_t_ha']
    ).where(result['c_input_t_ha'] > 0, 1.44)  # Default to annuals ratio if no inputs
    
    # Round results
    result['c_input_t_ha'] = result['c_input_t_ha'].round(2)
    result['dpm_rpm_ratio'] = result['dpm_rpm_ratio'].round(3)
    
    # Return final result
    result = result[['case', 'group', 'c_input_herbaceous_t_ha', 'c_input_tree_t_ha', 
                     'c_input_amend_t_ha', 'c_input_t_ha', 'dpm_rpm_ratio']].copy()
    
    return result

