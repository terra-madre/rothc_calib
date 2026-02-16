import pandas as pd

def calc_deltas(rothc_yearly_results):
    """
    Calculate treatment-control differences (deltas) for RothC yearly results.
    
    Returns:
        DataFrame with case and two key metrics:
        1. delta_treatment_control: Treatment SOC - Control SOC at final year
        2. delta_control_baseline: Control SOC final - Control SOC initial (year 0)
    """

    # Separate control and treatment groups
    columns_to_merge = ['subcase', 'case', 'group', 'year', 'month', 'SOC_t_C_ha']
    control_df = rothc_yearly_results[rothc_yearly_results['group'] == 'control'][columns_to_merge].copy()
    treatment_df = rothc_yearly_results[rothc_yearly_results['group'] == 'treatment'][columns_to_merge].copy()

    # For each control subcase, calculate the difference between final SOC and initial SOC (year 0)
    control_baseline = control_df.groupby('subcase')['SOC_t_C_ha'].first().reset_index()
    control_baseline.columns = ['subcase', 'baseline_SOC']
    
    # Keep only the last year for each subcase
    control_final = control_df.groupby('subcase').tail(1).reset_index(drop=True)
    treatment_final = treatment_df.groupby('subcase').tail(1).reset_index(drop=True)
    
    # Merge control final with baseline
    control_final = control_final.merge(control_baseline, on='subcase')
    control_final['delta_control_baseline'] = control_final['SOC_t_C_ha'] - control_final['baseline_SOC']
    
    # Merge control and treatment on case (both subcases belong to same case)
    merged_df = pd.merge(
        control_final[['case', 'year', 'month', 'SOC_t_C_ha', 'delta_control_baseline']], 
        treatment_final[['case', 'SOC_t_C_ha']], 
        on='case', 
        suffixes=('_control', '_treatment')
    )
    
    # Calculate treatment-control difference at final year
    merged_df['delta_treatment_control'] = merged_df['SOC_t_C_ha_treatment'] - merged_df['SOC_t_C_ha_control']
    
    # Calculate annual rates
    merged_df['delta_treatment_control_per_year'] = merged_df['delta_treatment_control'] / merged_df['year']
    merged_df['delta_control_baseline_per_year'] = merged_df['delta_control_baseline'] / merged_df['year']
    
    # Select relevant columns for output
    delta_columns = [
        'case', 'year', 'month', 
        'SOC_t_C_ha_control', 'SOC_t_C_ha_treatment',
        'delta_treatment_control', 'delta_treatment_control_per_year',
        'delta_control_baseline', 'delta_control_baseline_per_year'
    ]
    deltas_df = merged_df[delta_columns].round(2)
    
    return deltas_df    
    