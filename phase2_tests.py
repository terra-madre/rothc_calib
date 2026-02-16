"""
Phase 2: Single-fold Optimization Testing

Tests different parameter sets, optimization methods, and benchmarks runtime.
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize, differential_evolution

# Add git_code to path
sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG,
    precompute_data,
    objective,
    run_optimization,
    get_baseline_rmse
)


def benchmark_iteration(param_names, data, n_iterations=5):
    """
    Benchmark average time per objective function call.
    
    Args:
        param_names: List of parameter names
        data: Precomputed data dict
        n_iterations: Number of calls to average over
    
    Returns:
        float: Average time per iteration (seconds)
    """
    x0 = np.array([PARAM_CONFIG[p]['default'] for p in param_names])
    
    times = []
    for i in range(n_iterations):
        start = time.time()
        rmse = objective(x0, param_names, data)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}/{n_iterations}: {elapsed:.3f}s, RMSE={rmse:.4f}")
    
    avg_time = np.mean(times)
    print(f"\nAverage: {avg_time:.3f}s per iteration")
    return avg_time


def test_lbfgsb(param_names, data, maxiter=200):
    """
    Test L-BFGS-B optimization with specified parameters.
    
    Args:
        param_names: List of parameter names to optimize
        data: Precomputed data dict
        maxiter: Maximum iterations
    
    Returns:
        dict: Results with optimized params, RMSE, time, iterations
    """
    print(f"\n{'='*70}")
    print(f"L-BFGS-B Optimization: {param_names}")
    print(f"{'='*70}")
    
    x0 = np.array([PARAM_CONFIG[p]['default'] for p in param_names])
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    
    start_time = time.time()
    result = minimize(
        objective,
        x0,
        args=(param_names, data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': maxiter}
    )
    elapsed = time.time() - start_time
    
    optimized_params = dict(zip(param_names, result.x))
    
    print(f"\n--- Results ---")
    print(f"Status: {result.message}")
    print(f"Iterations: {result.nit}")
    print(f"Function calls: {result.nfev}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per iteration: {elapsed/result.nfev:.3f}s")
    print(f"Final RMSE: {result.fun:.4f}")
    print(f"\nOptimized parameters:")
    for name, value in optimized_params.items():
        default = PARAM_CONFIG[name]['default']
        bounds = PARAM_CONFIG[name]['bounds']
        pct_change = (value - default) / default * 100
        print(f"  {name:25s}: {value:.4f}  (default: {default:.4f}, {pct_change:+.1f}%, bounds: {bounds})")
    
    return {
        'method': 'L-BFGS-B',
        'param_names': param_names,
        'optimized_params': optimized_params,
        'rmse': result.fun,
        'time': elapsed,
        'iterations': result.nit,
        'func_calls': result.nfev,
        'success': result.success,
        'message': result.message
    }


def test_differential_evolution(param_names, data, maxiter=100, popsize=15):
    """
    Test differential evolution (global optimization).
    
    Args:
        param_names: List of parameter names to optimize
        data: Precomputed data dict
        maxiter: Maximum generations
        popsize: Population size multiplier
    
    Returns:
        dict: Results with optimized params, RMSE, time, iterations
    """
    print(f"\n{'='*70}")
    print(f"Differential Evolution: {param_names}")
    print(f"{'='*70}")
    print(f"Population size: {popsize * len(param_names)}")
    print(f"Max generations: {maxiter}")
    
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    
    start_time = time.time()
    
    # Track progress
    iteration_count = [0]
    def callback(xk, convergence):
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            rmse = objective(xk, param_names, data)
            elapsed = time.time() - start_time
            print(f"  Gen {iteration_count[0]}: RMSE={rmse:.4f}, Time={elapsed:.1f}s, Convergence={convergence:.6f}")
        return False
    
    result = differential_evolution(
        objective,
        bounds,
        args=(param_names, data),
        maxiter=maxiter,
        popsize=popsize,
        seed=42,
        callback=callback,
        disp=True,
        workers=1  # Sequential for reproducibility
    )
    elapsed = time.time() - start_time
    
    optimized_params = dict(zip(param_names, result.x))
    
    print(f"\n--- Results ---")
    print(f"Status: {result.message}")
    print(f"Iterations: {result.nit}")
    print(f"Function calls: {result.nfev}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per function call: {elapsed/result.nfev:.3f}s")
    print(f"Final RMSE: {result.fun:.4f}")
    print(f"\nOptimized parameters:")
    for name, value in optimized_params.items():
        default = PARAM_CONFIG[name]['default']
        bounds = PARAM_CONFIG[name]['bounds']
        pct_change = (value - default) / default * 100
        print(f"  {name:25s}: {value:.4f}  (default: {default:.4f}, {pct_change:+.1f}%, bounds: {bounds})")
    
    return {
        'method': 'differential_evolution',
        'param_names': param_names,
        'optimized_params': optimized_params,
        'rmse': result.fun,
        'time': elapsed,
        'iterations': result.nit,
        'func_calls': result.nfev,
        'success': result.success,
        'message': result.message
    }


def run_phase2_tests():
    """
    Run comprehensive Phase 2 tests.
    """
    print("\n" + "="*70)
    print("PHASE 2: SINGLE-FOLD OPTIMIZATION TESTING")
    print("="*70)
    
    # Setup
    BASE_DIR = Path(__file__).parent.parent
    
    print("\n[1/7] Loading and precomputing data...")
    data = precompute_data(repo_root=BASE_DIR)
    n_cases = len(data['cases_info_df'])
    groups = data['cases_info_df']['group_calib'].unique().tolist()
    print(f"  Loaded {n_cases} cases")
    print(f"  Groups: {groups}")
    
    print("\n[2/7] Computing baseline RMSE...")
    baseline_metrics = get_baseline_rmse(data)
    baseline_rmse = baseline_metrics['rmse']
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  Baseline MAE:  {baseline_metrics['mae']:.4f}")
    print(f"  Baseline Bias: {baseline_metrics['bias']:.4f}")
    print(f"  Baseline R²:   {baseline_metrics['r2']:.4f}")
    
    # Parameter sets to test
    param_sets = {
        'Tier1_Core': ['dr_ratio_annuals', 'map_to_prod'],
        'Set_A_Annuals': ['dr_ratio_annuals', 'map_to_prod', 'covercrop_rs_ratio'],
        'Set_B_Trees': ['dr_ratio_treegrass', 'tree_fine_root_ratio'],
        'Tier1_All': ['dr_ratio_annuals', 'dr_ratio_treegrass', 'map_to_prod', 
                      'covercrop_rs_ratio'],
    }
    
    # Benchmark
    print("\n[3/7] Benchmarking iteration time (Tier1_Core)...")
    avg_iter_time = benchmark_iteration(param_sets['Tier1_Core'], data, n_iterations=5)
    
    # Test L-BFGS-B with different parameter sets
    results = []
    
    print("\n[4/7] Testing L-BFGS-B with different parameter sets...")
    for set_name, param_names in param_sets.items():
        result = test_lbfgsb(param_names, data, maxiter=200)
        result['set_name'] = set_name
        results.append(result)
        
        # Brief pause between tests
        time.sleep(1)
    
    # Test differential evolution (only with smaller sets due to runtime)
    print("\n[5/7] Testing Differential Evolution (Tier1_Core only)...")
    de_result = test_differential_evolution(
        param_sets['Tier1_Core'], 
        data, 
        maxiter=50,  # Fewer iterations for DE due to population size
        popsize=10
    )
    de_result['set_name'] = 'Tier1_Core'
    results.append(de_result)
    
    # Summary comparison
    print("\n[6/7] Results Summary")
    print("="*70)
    
    summary_data = []
    for r in results:
        rmse_improvement = baseline_rmse - r['rmse']
        pct_improvement = (rmse_improvement / baseline_rmse) * 100
        
        summary_data.append({
            'Set': r['set_name'],
            'Method': r['method'],
            'N_Params': len(r['param_names']),
            'RMSE': r['rmse'],
            'Improvement': rmse_improvement,
            'Pct_Improve': pct_improvement,
            'Time_s': r['time'],
            'Func_Calls': r['func_calls'],
            'Success': r['success']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('RMSE')
    
    print("\n" + summary_df.to_string(index=False))
    
    # Save results
    print("\n[7/7] Saving results...")
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    summary_df.to_csv(output_dir / "phase2_summary.csv", index=False)
    
    # Save detailed results
    detailed_results = []
    for r in results:
        for param_name, param_value in r['optimized_params'].items():
            detailed_results.append({
                'set': r['set_name'],
                'method': r['method'],
                'parameter': param_name,
                'optimized_value': param_value,
                'default_value': PARAM_CONFIG[param_name]['default'],
                'bounds_min': PARAM_CONFIG[param_name]['bounds'][0],
                'bounds_max': PARAM_CONFIG[param_name]['bounds'][1],
                'pct_change': (param_value - PARAM_CONFIG[param_name]['default']) / 
                             PARAM_CONFIG[param_name]['default'] * 100
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(output_dir / "phase2_detailed_params.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir}/phase2_summary.csv")
    print(f"  - {output_dir}/phase2_detailed_params.csv")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    best_result = summary_df.iloc[0]
    print(f"\nBest Result:")
    print(f"  Set: {best_result['Set']}")
    print(f"  Method: {best_result['Method']}")
    print(f"  RMSE: {best_result['RMSE']:.4f} (baseline: {baseline_rmse:.4f})")
    print(f"  Improvement: {best_result['Improvement']:.4f} ({best_result['Pct_Improve']:.1f}%)")
    print(f"  Time: {best_result['Time_s']:.1f}s")
    
    if best_result['Pct_Improve'] < 1.0:
        print("\n⚠️  WARNING: Minimal improvement detected (<1%)")
        print("  Possible reasons:")
        print("    - Parameters may have weak effect on current dataset")
        print("    - Model structure may be limiting fit quality")
        print("    - Local minimum near default values")
        print("    - Need more diverse cases for parameter identifiability")
        print("\n  Recommended next steps:")
        print("    - Expand dataset to 60-70 cases (Phase 3)")
        print("    - Perform sensitivity analysis to identify impactful parameters")
        print("    - Consider alternative model structures or constraints")
    
    return results, summary_df


if __name__ == "__main__":
    results, summary = run_phase2_tests()
    print("\nPhase 2 testing complete!")
