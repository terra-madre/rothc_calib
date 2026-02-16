"""
Compare predictions: Baseline vs Optimized Parameters

Shows the impact of Phase 2 optimization on model predictions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimization import precompute_data, objective, PARAM_CONFIG

def get_predictions(param_values, param_names, data):
    """
    Get detailed predictions for comparison.
    
    Returns:
        DataFrame with case, observed, predicted, and residuals
    """
    rmse, details = objective(param_values, param_names, data, return_details=True)
    
    comparison_df = details['comparison_df'].copy()
    comparison_df = comparison_df.rename(columns={
        'delta_treatment_control_per_year': 'predicted',
        'delta_soc_t_ha_y': 'observed'
    })
    
    return comparison_df[['case', 'observed', 'predicted']].copy()


def main():
    print("\n" + "="*70)
    print("COMPARING BASELINE VS OPTIMIZED PREDICTIONS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    BASE_DIR = Path(__file__).parent.parent
    data = precompute_data(repo_root=BASE_DIR)
    
    # Baseline (default parameters)
    print("\n[1/2] Computing baseline predictions...")
    param_names = ['dr_ratio_annuals', 'map_to_prod']
    baseline_values = [PARAM_CONFIG[p]['default'] for p in param_names]
    baseline_pred = get_predictions(baseline_values, param_names, data)
    baseline_pred = baseline_pred.rename(columns={'predicted': 'pred_baseline'})
    
    # Optimized (from Phase 2 DE results)
    print("[2/2] Computing optimized predictions...")
    optimized_values = [2.48073630677393, 0.003001213338264137, 0.808465157020152]
    optimized_pred = get_predictions(optimized_values, param_names, data)
    optimized_pred = optimized_pred.rename(columns={'predicted': 'pred_optimized'})
    
    # Merge
    comparison = baseline_pred.merge(optimized_pred, on=['case', 'observed'])
    comparison['resid_baseline'] = comparison['observed'] - comparison['pred_baseline']
    comparison['resid_optimized'] = comparison['observed'] - comparison['pred_optimized']
    comparison['improvement'] = abs(comparison['resid_baseline']) - abs(comparison['resid_optimized'])
    
    # Add group info
    cases_info = data['cases_info_df'][['case', 'group_calib', 'duration_years']]
    comparison = comparison.merge(cases_info, on='case')
    
    # Summary statistics
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    
    baseline_rmse = np.sqrt(np.mean(comparison['resid_baseline']**2))
    optimized_rmse = np.sqrt(np.mean(comparison['resid_optimized']**2))
    baseline_mae = np.mean(abs(comparison['resid_baseline']))
    optimized_mae = np.mean(abs(comparison['resid_optimized']))
    
    print(f"\nRMSE:")
    print(f"  Baseline:   {baseline_rmse:.4f}")
    print(f"  Optimized:  {optimized_rmse:.4f}")
    print(f"  Improvement: {baseline_rmse - optimized_rmse:.4f} ({(baseline_rmse - optimized_rmse)/baseline_rmse*100:.1f}%)")
    
    print(f"\nMAE:")
    print(f"  Baseline:   {baseline_mae:.4f}")
    print(f"  Optimized:  {optimized_mae:.4f}")
    print(f"  Improvement: {baseline_mae - optimized_mae:.4f} ({(baseline_mae - optimized_mae)/baseline_mae*100:.1f}%)")
    
    # Per-group performance
    print("\n" + "="*70)
    print("PERFORMANCE BY GROUP")
    print("="*70)
    
    for group in comparison['group_calib'].unique():
        group_data = comparison[comparison['group_calib'] == group]
        n = len(group_data)
        
        group_baseline_rmse = np.sqrt(np.mean(group_data['resid_baseline']**2))
        group_optimized_rmse = np.sqrt(np.mean(group_data['resid_optimized']**2))
        
        improvement_pct = (group_baseline_rmse - group_optimized_rmse) / group_baseline_rmse * 100
        
        print(f"\n{group} (n={n}):")
        print(f"  Baseline RMSE:   {group_baseline_rmse:.4f}")
        print(f"  Optimized RMSE:  {group_optimized_rmse:.4f}")
        print(f"  Improvement:      {improvement_pct:.1f}%")
    
    # Detailed case-by-case comparison
    print("\n" + "="*70)
    print("CASE-BY-CASE COMPARISON")
    print("="*70)
    print()
    
    # Sort by improvement
    comparison_sorted = comparison.sort_values('improvement', ascending=False)
    
    print(comparison_sorted[[
        'case', 'group_calib', 'observed', 'pred_baseline', 
        'pred_optimized', 'resid_baseline', 'resid_optimized', 'improvement'
    ]].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    # Save results
    output_dir = BASE_DIR / "outputs"
    output_path = output_dir / "baseline_vs_optimized_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\n\nDetailed comparison saved to:")
    print(f"  {output_path}")
    
    # Summary of improvements
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    
    better = (comparison['improvement'] > 0).sum()
    worse = (comparison['improvement'] < 0).sum()
    same = (comparison['improvement'] == 0).sum()
    
    print(f"\nCases with better predictions:  {better}/{len(comparison)} ({better/len(comparison)*100:.0f}%)")
    print(f"Cases with worse predictions:   {worse}/{len(comparison)} ({worse/len(comparison)*100:.0f}%)")
    print(f"Cases with same predictions:    {same}/{len(comparison)} ({same/len(comparison)*100:.0f}%)")
    
    avg_improvement = comparison['improvement'].mean()
    print(f"\nAverage reduction in absolute error: {avg_improvement:.3f} t C/ha/y")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
