"""Quick L-BFGS-B probe on Tier1 to test if gradient method works with eps=0.01."""
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
print(f'Baseline RMSE: {baseline["rmse"]:.4f}')
sys.stdout.flush()

set_name = 'Tier1'
param_names = PARAM_SETS[set_name]
print(f'\n=== L-BFGS-B probe: {set_name} ({len(param_names)} params) ===')
print(f'Parameters: {param_names}')
print(f'Defaults: {[PARAM_CONFIG[p]["default"] for p in param_names]}')
sys.stdout.flush()

t1 = time.time()
result = run_optimization(param_names, data, method='L-BFGS-B', maxiter=100, verbose=True)
elapsed = time.time() - t1

print(f'\n--- Results ---')
print(f'Completed in {elapsed:.1f}s')
print(f'RMSE: {result["rmse"]:.4f} (baseline: {baseline["rmse"]:.4f})')
print(f'Improvement: {(baseline["rmse"] - result["rmse"])/baseline["rmse"]*100:.1f}%')
print(f'Iterations: {result["iterations"]}')
print(f'Params: {result["params"]}')
print(f'R2: {result["r2"]:.4f}')
print(f'Message: {result["scipy_result"].message}')
sys.stdout.flush()
