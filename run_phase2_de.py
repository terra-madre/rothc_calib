"""
Phase 2: Differential Evolution optimization on key parameter sets.

Runs DE with settings from optim_settings.csv on selected param sets.
Saves results to outputs/phase2_de_*.csv
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, PARAM_SETS, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse
)


def run_de(set_name, param_names, data, maxiter, popsize, seed=42):
    """Run DE on a single parameter set."""
    print(f"\n{'='*70}")
    print(f"DE: {set_name} — {param_names}")
    n_params = len(param_names)
    pop_total = popsize * n_params
    print(f"Pop={pop_total}, MaxGen={maxiter}, Seed={seed}")
    print(f"{'='*70}")

    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    
    start = time.time()
    gen_count = [0]

    def callback(xk, convergence):
        gen_count[0] += 1
        if gen_count[0] % 5 == 0:
            rmse = objective(xk, param_names, data)
            elapsed = time.time() - start
            print(f"  Gen {gen_count[0]:3d}: RMSE={rmse:.4f}  Conv={convergence:.6f}  Time={elapsed:.0f}s")
        return False

    result = differential_evolution(
        objective,
        bounds,
        args=(param_names, data),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        callback=callback,
        disp=False,
        tol=0.001,
        workers=1
    )
    elapsed = time.time() - start

    opt_params = dict(zip(param_names, result.x))
    print(f"\n--- {set_name} Results ---")
    print(f"RMSE:       {result.fun:.4f}")
    print(f"Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Func calls: {result.nfev}")
    print(f"Converged:  {result.success} ({result.message})")
    for name, val in opt_params.items():
        default = PARAM_CONFIG[name]['default']
        pct = (val - default) / default * 100
        print(f"  {name:25s}: {val:.4f}  (default: {default:.4f}, {pct:+.1f}%)")

    return {
        'set_name': set_name,
        'rmse': result.fun,
        'time_s': elapsed,
        'func_calls': result.nfev,
        'success': result.success,
        'params': opt_params,
        'message': result.message
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    
    maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    seed = int(OPTIM_SETTINGS.get('seed', 42))
    
    print("Phase 2: Differential Evolution Tests")
    print(f"Settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")
    
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    n_cases = len(data['cases_info_df'])
    print(f"Loaded {n_cases} cases")

    baseline = get_baseline_rmse(data)
    baseline_rmse = baseline['rmse']
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # Run DE on selected sets (ordered by param count for runtime)
    sets_to_test = ['Amendments', 'Annuals', 'Pasture', 'Trees',
                    'Tier1', 'Tier1_Tier2']
    
    all_results = []
    for set_name in sets_to_test:
        if set_name not in PARAM_SETS:
            print(f"WARNING: {set_name} not found in PARAM_SETS, skipping")
            continue
        param_names = PARAM_SETS[set_name]
        result = run_de(set_name, param_names, data, maxiter, popsize, seed)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Set':20s} | {'RMSE':>7s} | {'Improve':>8s} | {'Time':>7s} | {'Calls':>6s}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x['rmse']):
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        print(f"{r['set_name']:20s} | {r['rmse']:7.4f} | {imp:+7.1f}% | {r['time_s']:6.0f}s | {r['func_calls']:6d}")
    
    # Save summary
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    summary_rows = []
    detail_rows = []
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        summary_rows.append({
            'set_name': r['set_name'],
            'method': 'differential_evolution',
            'n_params': len(r['params']),
            'rmse': r['rmse'],
            'baseline_rmse': baseline_rmse,
            'improvement_pct': imp,
            'time_s': r['time_s'],
            'func_calls': r['func_calls'],
            'success': r['success'],
            'maxiter': maxiter,
            'popsize': popsize
        })
        for pname, pval in r['params'].items():
            detail_rows.append({
                'set_name': r['set_name'],
                'parameter': pname,
                'optimized_value': pval,
                'default_value': PARAM_CONFIG[pname]['default'],
                'bound_min': PARAM_CONFIG[pname]['bounds'][0],
                'bound_max': PARAM_CONFIG[pname]['bounds'][1],
                'pct_change': (pval - PARAM_CONFIG[pname]['default']) / PARAM_CONFIG[pname]['default'] * 100
            })
    
    pd.DataFrame(summary_rows).to_csv(output_dir / "phase2_de_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "phase2_de_params.csv", index=False)
    print(f"\nSaved: outputs/phase2_de_summary.csv, outputs/phase2_de_params.csv")
    print("Phase 2 DE complete!")
