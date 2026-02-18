"""
Overnight optimization: DE (warm-started from L-BFGS-B) → L-BFGS-B polish.

Strategy:
  1. DE global search with x0 = L-BFGS-B solution (seeded into population)
  2. L-BFGS-B local polish from DE result (with tighter eps=0.001)

Expected runtime: ~4-6h for DE (popsize=8, 8 params → 64 pop, 30 gen)
                  + ~20min for L-BFGS-B polish

Usage:
  cd rothc_calibration
  PYTHONPATH=git_code mamba run -n terra-plus python3 -u git_code/run_overnight.py 2>&1 | tee outputs/overnight.log
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, PARAM_SETS, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse
)
from scipy.optimize import differential_evolution, minimize

BASE_DIR = Path(__file__).parent.parent
LOG_PATH = BASE_DIR / "outputs" / "overnight.log"


# L-BFGS-B results to warm-start DE
LBFGSB_SOLUTION = {
    'dr_ratio_annuals': 1.568459,
    'dr_ratio_treegrass': 0.705350,
    'dr_ratio_wood': 0.243019,
    'dr_ratio_amend': 1.129834,
    'plant_cover_modifier': 1.000000,
    'tree_fine_root_ratio': 0.204236,
    'grass_rs_ratio': 0.848794,
    'map_to_prod': 0.004000,
}
LBFGSB_RMSE = 1.5830


def log_and_print(msg, f=None):
    """Print to stdout and optionally write to log file."""
    print(msg, flush=True)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


def heartbeat_thread(stop_event, start_time_ref, label=""):
    """Background thread that prints every 5 minutes."""
    while not stop_event.wait(300):
        if start_time_ref[0] is not None:
            elapsed = time.time() - start_time_ref[0]
            print(f"  [heartbeat {label}] Still running... {elapsed:.0f}s ({elapsed/60:.1f} min)",
                  flush=True)


def main():
    param_names = PARAM_SETS['Tier1_Tier2']
    n_params = len(param_names)
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    x0_lbfgsb = np.array([LBFGSB_SOLUTION[p] for p in param_names])
    x0_default = np.array([PARAM_CONFIG[p]['default'] for p in param_names])

    # DE settings
    de_maxiter = 30
    de_popsize = 8  # multiplier → actual pop = 8 * 8 = 64
    de_seed = 42
    actual_pop = de_popsize * n_params

    with open(LOG_PATH, 'w') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        log_and_print(f"{'='*70}", f)
        log_and_print(f"OVERNIGHT OPTIMIZATION: DE → L-BFGS-B", f)
        log_and_print(f"Started: {now}", f)
        log_and_print(f"{'='*70}", f)

        log_and_print(f"\nParameter set ({n_params} params): {param_names}", f)
        log_and_print(f"\nBounds:", f)
        for p in param_names:
            cfg = PARAM_CONFIG[p]
            log_and_print(f"  {p:25s}: bounds={cfg['bounds']}", f)

        log_and_print(f"\nWarm-start (L-BFGS-B solution, RMSE={LBFGSB_RMSE}):", f)
        for p in param_names:
            log_and_print(f"  {p:25s}: {LBFGSB_SOLUTION[p]:.6f}", f)

        # ---- Load data ----
        log_and_print(f"\nLoading data...", f)
        data = precompute_data(repo_root=BASE_DIR)
        n_cases = len(data['cases_info_df'])
        log_and_print(f"Loaded {n_cases} cases", f)

        # ---- Baseline ----
        baseline = get_baseline_rmse(data)
        log_and_print(f"\nBaseline: RMSE={baseline['rmse']:.4f}, R²={baseline['r2']:.4f}", f)

        # ---- Benchmark ----
        log_and_print(f"\nBenchmarking (3 calls)...", f)
        bench_times = []
        for i in range(3):
            t0 = time.time()
            objective(x0_default, param_names, data)
            dt = time.time() - t0
            bench_times.append(dt)
            log_and_print(f"  Call {i+1}: {dt:.2f}s", f)
        avg_call = np.mean(bench_times)

        # DE estimate: (popsize*n_params + 1) calls per generation (init) + popsize*n_params per gen
        de_calls_init = actual_pop
        de_calls_per_gen = actual_pop
        de_total_calls = de_calls_init + de_calls_per_gen * de_maxiter
        de_est_time = avg_call * de_total_calls

        # L-BFGS-B estimate
        lbfgsb_calls = (2 * n_params + 1) * 50  # ~50 iterations max
        lbfgsb_est_time = avg_call * lbfgsb_calls

        total_est = de_est_time + lbfgsb_est_time

        log_and_print(f"\n  Avg call time: {avg_call:.2f}s", f)
        log_and_print(f"\n  --- DE time estimate ---", f)
        log_and_print(f"  Population: {actual_pop} (popsize={de_popsize} × {n_params} params)", f)
        log_and_print(f"  Max generations: {de_maxiter}", f)
        log_and_print(f"  Est. func calls: ~{de_total_calls}", f)
        log_and_print(f"  Est. DE runtime: {de_est_time:.0f}s ({de_est_time/60:.1f} min, {de_est_time/3600:.1f}h)", f)
        log_and_print(f"\n  --- L-BFGS-B polish estimate ---", f)
        log_and_print(f"  Est. func calls: ~{lbfgsb_calls}", f)
        log_and_print(f"  Est. runtime: {lbfgsb_est_time:.0f}s ({lbfgsb_est_time/60:.1f} min)", f)
        log_and_print(f"\n  Total estimated: {total_est:.0f}s ({total_est/60:.1f} min, {total_est/3600:.1f}h)", f)
        log_and_print(f"  (DE often converges earlier)", f)

        # ================================================================
        # PHASE 1: Differential Evolution (global search)
        # ================================================================
        log_and_print(f"\n{'='*70}", f)
        log_and_print(f"PHASE 1: Differential Evolution (warm-started)", f)
        log_and_print(f"{'='*70}", f)

        # Heartbeat
        hb_stop = threading.Event()
        hb_start = [None]
        hb = threading.Thread(target=heartbeat_thread, args=(hb_stop, hb_start, "DE"), daemon=True)
        hb.start()

        de_start = time.time()
        hb_start[0] = de_start

        # Track generations
        gen_count = [0]
        gen_start = [de_start]
        best_rmse = [LBFGSB_RMSE]

        def de_callback(xk, convergence):
            gen_count[0] += 1
            gen_elapsed = time.time() - gen_start[0]
            total_elapsed = time.time() - de_start
            rmse = objective(xk, param_names, data)
            if rmse < best_rmse[0]:
                best_rmse[0] = rmse
            msg = (f"  Gen {gen_count[0]:3d}/{de_maxiter}: "
                   f"RMSE={rmse:.4f} (best={best_rmse[0]:.4f}) "
                   f"conv={convergence:.6f} "
                   f"gen_time={gen_elapsed:.0f}s "
                   f"total={total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
            log_and_print(msg, f)
            gen_start[0] = time.time()
            return False

        de_result = differential_evolution(
            objective,
            bounds,
            args=(param_names, data),
            x0=x0_lbfgsb,             # Warm-start with L-BFGS-B solution
            maxiter=de_maxiter,
            popsize=de_popsize,
            seed=de_seed,
            mutation=(0.5, 1.0),       # dithering for diversity
            recombination=0.7,
            tol=0.001,
            atol=0.001,
            disp=True,
            workers=1,
            polish=False,              # We'll do our own L-BFGS-B polish
            callback=de_callback
        )

        hb_stop.set()
        de_elapsed = time.time() - de_start

        log_and_print(f"\n--- DE Results ---", f)
        log_and_print(f"Converged: {de_result.success} ({de_result.message})", f)
        log_and_print(f"RMSE: {de_result.fun:.4f}", f)
        log_and_print(f"Generations: {gen_count[0]}", f)
        log_and_print(f"Func calls: {de_result.nfev}", f)
        log_and_print(f"Time: {de_elapsed:.0f}s ({de_elapsed/60:.1f} min, {de_elapsed/3600:.1f}h)", f)

        de_params = dict(zip(param_names, de_result.x))
        log_and_print(f"\nDE optimized parameters:", f)
        for name, value in de_params.items():
            cfg = PARAM_CONFIG[name]
            default = cfg['default']
            lo, hi = cfg['bounds']
            pct = (value - default) / default * 100 if default != 0 else 0
            at_bound = ""
            if abs(value - lo) < 1e-4:
                at_bound = " [AT LOWER BOUND]"
            elif abs(value - hi) < 1e-4:
                at_bound = " [AT UPPER BOUND]"
            log_and_print(f"  {name:25s}: {value:.6f}  (default={default}, {pct:+.1f}%){at_bound}", f)

        # ================================================================
        # PHASE 2: L-BFGS-B polish (local refinement)
        # ================================================================
        log_and_print(f"\n{'='*70}", f)
        log_and_print(f"PHASE 2: L-BFGS-B polish (from DE result)", f)
        log_and_print(f"{'='*70}", f)

        hb_stop2 = threading.Event()
        hb_start2 = [None]
        hb2 = threading.Thread(target=heartbeat_thread, args=(hb_stop2, hb_start2, "L-BFGS-B"), daemon=True)
        hb2.start()

        polish_start = time.time()
        hb_start2[0] = polish_start

        iter_count = [0]
        def lbfgsb_callback(xk):
            iter_count[0] += 1
            if iter_count[0] % 5 == 0:
                rmse = objective(xk, param_names, data)
                elapsed = time.time() - polish_start
                msg = f"  L-BFGS-B iter {iter_count[0]}: RMSE={rmse:.4f} ({elapsed:.0f}s)"
                log_and_print(msg, f)

        polish_result = minimize(
            objective,
            de_result.x,
            args=(param_names, data),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'disp': True, 'eps': 0.001},  # Tighter eps
            callback=lbfgsb_callback
        )

        hb_stop2.set()
        polish_elapsed = time.time() - polish_start

        log_and_print(f"\n--- L-BFGS-B Polish Results ---", f)
        log_and_print(f"Converged: {polish_result.success} ({polish_result.message})", f)
        log_and_print(f"RMSE: {polish_result.fun:.4f}", f)
        log_and_print(f"Iterations: {polish_result.nit}", f)
        log_and_print(f"Func calls: {polish_result.nfev}", f)
        log_and_print(f"Time: {polish_elapsed:.0f}s ({polish_elapsed/60:.1f} min)", f)

        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        total_elapsed = time.time() - de_start
        final_rmse, details = objective(polish_result.x, param_names, data, return_details=True)

        log_and_print(f"\n{'='*70}", f)
        log_and_print(f"FINAL SUMMARY", f)
        log_and_print(f"{'='*70}", f)
        log_and_print(f"Baseline RMSE:       {baseline['rmse']:.4f}  R²={baseline['r2']:.4f}", f)
        log_and_print(f"L-BFGS-B (prev):     {LBFGSB_RMSE:.4f}  (from widened-bounds run)", f)
        log_and_print(f"DE result:           {de_result.fun:.4f}", f)
        log_and_print(f"DE + L-BFGS-B final: {final_rmse:.4f}  R²={details['r2']:.4f}", f)
        improvement = (baseline['rmse'] - final_rmse) / baseline['rmse'] * 100
        log_and_print(f"Total improvement:   {improvement:.1f}%", f)
        log_and_print(f"Total time:          {total_elapsed:.0f}s ({total_elapsed/60:.1f} min, {total_elapsed/3600:.1f}h)", f)
        log_and_print(f"MAE:  {details['mae']:.4f}", f)
        log_and_print(f"Bias: {details['bias']:.4f}", f)

        log_and_print(f"\nFinal optimized parameters:", f)
        for name, value in zip(param_names, polish_result.x):
            cfg = PARAM_CONFIG[name]
            default = cfg['default']
            lo, hi = cfg['bounds']
            pct = (value - default) / default * 100 if default != 0 else 0
            at_bound = ""
            if abs(value - lo) < 1e-4:
                at_bound = " [AT LOWER BOUND]"
            elif abs(value - hi) < 1e-4:
                at_bound = " [AT UPPER BOUND]"
            vs_lbfgsb = value - LBFGSB_SOLUTION[name]
            log_and_print(f"  {name:25s}: {value:.6f}  (default={default}, {pct:+.1f}%, vs_prev={vs_lbfgsb:+.4f}){at_bound}", f)

        log_and_print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f)

    print(f"\nFull log: {LOG_PATH}")


if __name__ == "__main__":
    main()
