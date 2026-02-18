"""
Phase 2 re-run: L-BFGS-B with widened bounds on Tier1_Tier2 (8 params).

Changes from previous run:
  - dr_ratio_annuals upper: 2.5 → 3.5
  - plant_cover_modifier upper: 0.8 → 1.0
  - tree_fine_root_ratio lower: 0.1 → 0.05
  - map_to_prod now included in Tier1_Tier2 (8 params total)

Usage:
  cd rothc_calibration
  PYTHONPATH=git_code conda run -n terra-plus python3 git_code/run_lbfgsb_widened.py
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, PARAM_SETS, OPTIM_SETTINGS,
    precompute_data, objective, run_optimization, get_baseline_rmse
)

BASE_DIR = Path(__file__).parent.parent


def main():
    # ---- Setup ----
    print("=" * 70)
    print("L-BFGS-B RE-RUN: Tier1_Tier2 with widened bounds")
    print("=" * 70)

    param_names = PARAM_SETS['Tier1_Tier2']
    print(f"\nParameter set ({len(param_names)} params): {param_names}")
    print("\nBounds (from param_config.csv):")
    for p in param_names:
        cfg = PARAM_CONFIG[p]
        print(f"  {p:25s}: default={cfg['default']}, bounds={cfg['bounds']}")

    print(f"\nOptimization settings:")
    print(f"  method: L-BFGS-B")
    print(f"  eps: {OPTIM_SETTINGS.get('eps', 1e-8)}")
    print(f"  maxiter: 200")

    # ---- Load data ----
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    n_cases = len(data['cases_info_df'])
    groups = data['cases_info_df']['group_calib'].value_counts()
    print(f"Loaded {n_cases} cases")
    print(f"Groups:\n{groups.to_string()}")

    # ---- Baseline + benchmark ----
    print("\nBaseline metrics (default parameters):")
    baseline = get_baseline_rmse(data)
    print(f"  RMSE: {baseline['rmse']:.4f}")
    print(f"  MAE:  {baseline['mae']:.4f}")
    print(f"  Bias: {baseline['bias']:.4f}")
    print(f"  R²:   {baseline['r2']:.4f}")

    # ---- Benchmark: time one objective call to estimate total runtime ----
    n_params = len(param_names)
    x0 = [PARAM_CONFIG[p]['default'] for p in param_names]
    print("\nBenchmarking: timing 3 objective function calls...")
    bench_times = []
    for i in range(3):
        t0 = time.time()
        objective(x0, param_names, data)
        dt = time.time() - t0
        bench_times.append(dt)
        print(f"  Call {i+1}: {dt:.2f}s")
    avg_call = np.mean(bench_times)
    # L-BFGS-B: ~(2*n_params+1) calls per iteration for finite-diff gradient
    calls_per_iter = 2 * n_params + 1
    maxiter = 200
    est_calls = calls_per_iter * maxiter
    est_total = avg_call * est_calls
    print(f"\n  Avg call time: {avg_call:.2f}s")
    print(f"  Calls/iteration (finite-diff, {n_params} params): ~{calls_per_iter}")
    print(f"  Estimated max func calls: ~{est_calls} (maxiter={maxiter})")
    print(f"  Estimated max runtime: {est_total:.0f}s ({est_total/60:.1f} min)")
    print(f"  (Likely converges much earlier than maxiter)")

    # ---- Heartbeat thread: print status every 5 minutes ----
    heartbeat_stop = threading.Event()
    opt_start_time = [None]  # mutable container for closure

    def heartbeat():
        while not heartbeat_stop.wait(300):  # 300s = 5 min
            if opt_start_time[0] is not None:
                elapsed_hb = time.time() - opt_start_time[0]
                print(f"  [heartbeat] Still running... {elapsed_hb:.0f}s ({elapsed_hb/60:.1f} min) elapsed",
                      flush=True)

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    # ---- Optimize ----
    print(f"\n{'='*70}")
    print("Starting L-BFGS-B optimization...")
    start_time = time.time()
    opt_start_time[0] = start_time

    result = run_optimization(
        param_names=param_names,
        data=data,
        method='L-BFGS-B',
        maxiter=maxiter,
        verbose=True
    )

    heartbeat_stop.set()

    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ---- Results ----
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Success: {result['success']}")
    print(f"RMSE:    {result['rmse']:.4f}  (baseline: {baseline['rmse']:.4f})")
    improvement = (baseline['rmse'] - result['rmse']) / baseline['rmse'] * 100
    print(f"Improvement: {improvement:.1f}%")
    print(f"MAE:     {result['mae']:.4f}")
    print(f"Bias:    {result['bias']:.4f}")
    print(f"R²:      {result['r2']:.4f}")
    print(f"Cases:   {result['n_cases']}")

    print(f"\nOptimized parameters:")
    for name, value in result['params'].items():
        cfg = PARAM_CONFIG[name]
        default = cfg['default']
        lo, hi = cfg['bounds']
        pct = (value - default) / default * 100 if default != 0 else 0
        at_bound = ""
        if abs(value - lo) < 1e-4:
            at_bound = " [AT LOWER BOUND]"
        elif abs(value - hi) < 1e-4:
            at_bound = " [AT UPPER BOUND]"
        print(f"  {name:25s}: {value:.4f}  (default: {default}, bounds: [{lo}, {hi}], {pct:+.1f}%){at_bound}")

    # ---- Save log ----
    log_path = BASE_DIR / "outputs" / "lbfgsb_widened.log"
    with open(log_path, 'w') as f:
        f.write(f"L-BFGS-B Tier1_Tier2 (widened bounds)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Params: {param_names}\n")
        f.write(f"Baseline RMSE: {baseline['rmse']:.4f}\n")
        f.write(f"Optimized RMSE: {result['rmse']:.4f}\n")
        f.write(f"Improvement: {improvement:.1f}%\n")
        f.write(f"R²: {result['r2']:.4f}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")
        f.write(f"Optimized parameters:\n")
        for name, value in result['params'].items():
            cfg = PARAM_CONFIG[name]
            f.write(f"  {name}: {value:.6f} (default: {cfg['default']}, bounds: {cfg['bounds']})\n")
        f.write(f"\nScipy result:\n{result['scipy_result']}\n")

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
