# Import necessary libraries
import pandas as pd
from io_utils import *


# Here we calculate the soil C inputs for each case in cases_inputs_df
def calc_c_herb(
    cases_treatments_df,
    cases_info_df,
    st_yields_all,
    ps_herbaceous,
    ps_management,
    ps_general
):
    """Calculate carbon inputs for RothC model based on treatments and case info.
    
    Args:
        cases_treatments_df (pd.DataFrame): DataFrame with treatment info per case
        cases_info_df (pd.DataFrame): DataFrame with case information (nuts3, map_mm)
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
            if pd.isna(crop_name):
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
        if pd.notna(covercrop_name):
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
    result = df[['subcase', 'c_input_herbaceous_t_ha']].copy()
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
        
        if pd.isna(tree_name):
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
        if pd.notna(tree_prunings):
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
    result = df[['subcase', 'c_input_tree_t_ha']].copy()
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
        
        if pd.isna(amend_name):
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
    result = df[['subcase', 'c_input_amend_t_ha']].copy()
    return result


def calc_c_inputs(
    cases_treatments_df,
    cases_info_df,
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
    result = c_herb.merge(c_tree[['subcase', 'c_input_tree_t_ha']], on=['subcase'], how='left')
    result = result.merge(c_amend[['subcase', 'c_input_amend_t_ha']], on=['subcase'], how='left')
    
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
    result = result[['subcase', 'c_input_herbaceous_t_ha', 'c_input_tree_t_ha', 
                     'c_input_amend_t_ha', 'c_input_t_ha', 'dpm_rpm_ratio']].copy()
    
    return result


def plant_cover(cases_treatments_df):
    """Calculate monthly plant cover for each subcase.
    
    Args:
        cases_treatments_df (pd.DataFrame): DataFrame with treatment info per case
    
    Returns:
        pd.DataFrame: DataFrame with columns (subcase, month, t_PC) in long format
    """
    
    results = []
    
    for _, subcase in cases_treatments_df.iterrows():
        # Monthly bare soil status (January to December)
        # Index 0 = January, Index 11 = December
        plant_cover_conventional = [
            1,  # January - Winter cereals established, active growth
            1,  # February - Peak vegetation cover, rainy season
            1,  # March - Spring growth, maximum cover
            1,  # April - Cereals maturing, still covered
            1,  # May - Late growth before harvest begins
            0,  # June - Harvest begins, stubble/bare fields
            0,  # July - Post-harvest, summer fallow
            0,  # August - Peak bare period, hot & dry
            0,  # September - Field preparation, still bare
            0,  # October - Plowing & seeding, minimal cover
            0,  # November - Early germination, insufficient cover
            1   # December - Seedlings established, coverage begins
        ]
        plant_cover_with_cover_crop = [
            1,  # January - Winter cereals established, active growth
            1,  # February - Peak vegetation cover, rainy season
            1,  # March - Spring growth, maximum cover
            1,  # April - Cereals maturing, still covered
            1,  # May - Late growth before harvest begins
            0,  # June - Harvest period, transition to cover crop
            1,  # July - Cover crop established (fast-growing summer species)
            1,  # August - Cover crop providing cover during dry period
            1,  # September - Cover crop still growing with first rains
            0,  # October - Cover crop terminated, field preparation begins
            0,  # November - Plowing & seeding main crop, minimal cover
            1   # December - Winter cereal seedlings established
        ]

        # Determine the soil cover:
        if subcase['grass_percent_cover'] == 100:
            # Full grass cover: array of 1s (fully covered)
            PC = [1] * 12
        elif pd.notna(subcase['covercrop']):
            # Cover crop present: use predefined cover pattern for cover crops. 
            # In the case of tree crops with cover crops we assume cover crop cover over the year.
            PC = plant_cover_with_cover_crop
        elif any(pd.notna(subcase[crop_col]) for crop_col in ['crop1_name', 'crop2_name', 'crop3_name']):
            # Conventional crops: use predefined cover pattern for conventional crops
            PC = plant_cover_conventional
        else:
            # No crops or cover crop: assume bare soil (0) except for January (1)
            PC = [1] + [0] * 11
        
        # Create one row per month for this subcase
        for month in range(1, 13):
            results.append({
                'subcase': subcase['subcase'],
                'month': month,
                't_PC': PC[month - 1]
            })
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    return result_df
