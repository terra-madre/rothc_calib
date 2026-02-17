"""L-BFGS-B optimization on Tier1_Tier2 (7 params) with eps=0.01."""
import time
import sys
from optimization import (precompute_data, run_optimization, get_baseline_rmse,
                          PARAM_SETS, PARAM_CONFIG, OPTIM_SETTINGS)

print(f'eps setting: {OPTIM_SETTINGS.get("eps", "not set")}')
print('Loading data...')
sys.stdout.flush()

t0 = time.time()
data = precompute_data()
print(f'Data loaded in {time.time()-t0:.1f}s')
sys.stdout.flush()

baseline = get_baseline_rmse(data)
print(f'Baseline RMSE: {baseline["rmse"]:.4f}, R2: {baseline["r2"]:.4f}')
sys.stdout.flush()

set_name = 'Tier1_Tier2'
param_names = PARAM_SETS[set_name]
print(f'\n=== L-BFGS-B: {set_name} ({len(param_names)} params) ===')
print(f'Parameters: {param_names}')
defaults = {p: PARAM_CONFIG[p]["default"] for p in param_names}
bounds = {p: PARAM_CONFIG[p]["bounds"] for p in param_names}
print(f'Defaults: {defaults}')
print(f'Bounds: {bounds}')
sys.stdout.flush()

t1 = time.time()
result = run_optimization(param_names, data, method='L-BFGS-B', maxiter=200, verbose=True)
elapsed = time.time() - t1

print(f'\n--- Results ---')
print(f'Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)')
print(f'Success: {result["success"]}')
print(f'RMSE: {result["rmse"]:.4f} (baseline: {baseline["rmse"]:.4f})')
print(f'Improvement: {(baseline["rmse"] - result["rmse"])/baseline["rmse"]*100:.1f}%')
print(f'MAE: {result["mae"]:.4f}')
print(f'R2: {result["r2"]:.4f} (baseline: {baseline["r2"]:.4f})')
print(f'Bias: {result["bias"]:.4f}')
print(f'Iterations: {result["iterations"]}')
print(f'\nOptimized params:')
for p, v in result['params'].items():
    d = PARAM_CONFIG[p]['default']
    b = PARAM_CONFIG[p]['bounds']
    at_bound = ' [AT LOWER BOUND]' if abs(v - b[0]) < 0.001 else (' [AT UPPER BOUND]' if abs(v - b[1]) < 0.001 else '')
    print(f'  {p}: {d} -> {v:.4f}{at_bound}')
print(f'\nScipy message: {result["scipy_result"].message}')
sys.stdout.flush()
