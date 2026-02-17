"""
Phase 3: Cross-Validation Testing

Implements stratified k-fold cross-validation with held-out test set
to assess parameter generalization and avoid overfitting.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

# Add git_code to path
sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG,
    PARAM_SETS,
    OPTIM_SETTINGS,
    precompute_data,
    objective,
    cross_validate,
    get_baseline_rmse
)


def run_phase3_cv(output_dir='../outputs'):
    """
    Run Phase 3 cross-validation tests with different parameter sets.
    
    Saves results to:
    - phase3_cv_folds.csv: Per-fold results
    - phase3_summary.csv: Aggregated metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PHASE 3: CROSS-VALIDATION")
    print("="*70)
    print()
    
    # Load data
    print("Loading data...")
    data = precompute_data()
    n_cases = len(data['cases_info_df'])
    groups = data['cases_info_df']['group_calib'].value_counts()
    
    print(f"Total cases: {n_cases}")
    print(f"Groups distribution:")
    for group, count in groups.items():
        print(f"  {group}: {count}")
    print()
    
    # Get baseline
    print("Computing baseline metrics (default parameters)...")
    baseline = get_baseline_rmse(data)
    print(f"Baseline RMSE: {baseline['rmse']:.4f}")
    print(f"Baseline MAE: {baseline['mae']:.4f}")
    print(f"Baseline Bias: {baseline['bias']:.4f}")
    print(f"Baseline R²: {baseline['r2']:.4f}")
    print()
    
    # Define parameter sets to test (loaded from config files)
    param_sets = PARAM_SETS
    
    # Storage for results
    all_fold_results = []
    summary_results = []
    
    # Test each parameter set
    for set_name, param_names in param_sets.items():
        print("="*70)
        print(f"Testing: {set_name}")
        print(f"Parameters: {param_names}")
        print("="*70)
        print()
        
        start_time = time.time()
        
        # Run cross-validation
        try:
            cv_df, test_rmse, test_cases = cross_validate(
                param_names=param_names,
                data=data,
                verbose=True
            )
            
            elapsed = time.time() - start_time
            
            # Add set name to fold results
            cv_df['param_set'] = set_name
            all_fold_results.append(cv_df)
            
            # Compute summary statistics
            mean_train_rmse = cv_df['train_rmse'].mean()
            std_train_rmse = cv_df['train_rmse'].std()
            mean_val_rmse = cv_df['val_rmse'].mean()
            std_val_rmse = cv_df['val_rmse'].std()
            
            # Check for overfitting
            overfit_ratio = mean_val_rmse / mean_train_rmse
            overfit_flag = overfit_ratio > 1.1  # >10% worse on validation
            
            # Improvement vs baseline
            improvement_train = baseline['rmse'] - mean_train_rmse
            improvement_val = baseline['rmse'] - mean_val_rmse
            improvement_test = baseline['rmse'] - test_rmse
            pct_improve_train = (improvement_train / baseline['rmse']) * 100
            pct_improve_val = (improvement_val / baseline['rmse']) * 100
            pct_improve_test = (improvement_test / baseline['rmse']) * 100
            
            summary = {
                'param_set': set_name,
                'n_params': len(param_names),
                'baseline_rmse': baseline['rmse'],
                'mean_train_rmse': mean_train_rmse,
                'std_train_rmse': std_train_rmse,
                'mean_val_rmse': mean_val_rmse,
                'std_val_rmse': std_val_rmse,
                'test_rmse': test_rmse,
                'overfit_ratio': overfit_ratio,
                'overfit_flag': overfit_flag,
                'improve_train': improvement_train,
                'improve_val': improvement_val,
                'improve_test': improvement_test,
                'pct_improve_train': pct_improve_train,
                'pct_improve_val': pct_improve_val,
                'pct_improve_test': pct_improve_test,
                'time_s': elapsed,
                'n_folds': len(cv_df)
            }
            
            # Add mean parameter values
            for p in param_names:
                summary[f'mean_{p}'] = cv_df[p].mean()
                summary[f'std_{p}'] = cv_df[p].std()
            
            summary_results.append(summary)
            
            print(f"\n{'='*70}")
            print(f"Summary for {set_name}:")
            print(f"  Train RMSE: {mean_train_rmse:.4f} ± {std_train_rmse:.4f}")
            print(f"  Val RMSE: {mean_val_rmse:.4f} ± {std_val_rmse:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Overfit ratio: {overfit_ratio:.3f} {'⚠️ OVERFIT' if overfit_flag else '✓ OK'}")
            print(f"  Improvement (val): {pct_improve_val:.2f}%")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"ERROR in {set_name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Combine and save results
    if all_fold_results:
        fold_df = pd.concat(all_fold_results, ignore_index=True)
        fold_output = output_dir / 'phase3_cv_folds.csv'
        fold_df.to_csv(fold_output, index=False)
        print(f"Saved fold results to: {fold_output}")
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_output = output_dir / 'phase3_summary.csv'
        summary_df.to_csv(summary_output, index=False)
        print(f"Saved summary to: {summary_output}")
        
        # Print final comparison
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(summary_df[['param_set', 'mean_val_rmse', 'test_rmse', 
                         'pct_improve_val', 'overfit_flag']].to_string(index=False))
        print()
    
    return fold_df, summary_df


def run_phase3_with_differential_evolution(output_dir='../outputs'):
    """
    Run Phase 3 using differential_evolution (best method from Phase 2).
    
    This is more computationally intensive but may find better parameter values.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PHASE 3: CV with Differential Evolution")
    print("="*70)
    print()
    
    # Load data
    print("Loading data...")
    data = precompute_data()
    
    # Get baseline
    baseline = get_baseline_rmse(data)
    print(f"Baseline RMSE: {baseline['rmse']:.4f}\n")
    
    # Test Tier1_Core
    param_names = PARAM_SETS.get('Tier1_Core', ['dr_ratio_annuals', 'map_to_prod'])
    
    print(f"Parameters: {param_names}")
    print("Method: Differential Evolution")
    print()
    
    start_time = time.time()
    
    cv_df, test_rmse, test_cases = cross_validate(
        param_names=param_names,
        data=data,
        method='differential_evolution',
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    cv_output = output_dir / 'phase3_cv_de_folds.csv'
    cv_df.to_csv(cv_output, index=False)
    
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved results to: {cv_output}")
    
    return cv_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3: Cross-Validation")
    parser.add_argument('--method', choices=['lbfgsb', 'de'], default='lbfgsb',
                       help='Optimization method (lbfgsb=fast, de=slower but better)')
    parser.add_argument('--output', default='../outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.method == 'lbfgsb':
        fold_df, summary_df = run_phase3_cv(output_dir=args.output)
    else:
        cv_df = run_phase3_with_differential_evolution(output_dir=args.output)
    
    print("\nPhase 3 complete!")
