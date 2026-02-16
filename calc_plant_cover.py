import pandas as pd

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
        if subcase['grass_perc_cover'] == 100:
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
