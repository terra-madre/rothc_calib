"""
Phase 3: Cross-Validation with DE, warm-started from Phase 2 best solution.

Runs stratified 5-fold CV on Tier1_Tier2 (8 params) using DE,
with held-out test set (~20%).

Each fold: DE(30 gen, pop=64, warm-started) → ~2-3h
Total: ~10-15h

Usage:
  cd rothc_calibration
  PYTHONPATH=git_code mamba run -n terra-plus python3 -u git_code/run_phase3_cv.py 2>&1 | tee outputs/phase3_cv.log
"""

import sys
import time
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, PARAM_SETS, OPTIM_SETTINGS,
    precompute_data, objective, cross_validate, get_baseline_rmse
)

BASE_DIR = Path(__file__).parent.parent

# Phase 2 best solution (DE → L-BFGS-B, RMSE=1.4450)
# NOTE: Values at bounds nudged slightly inside for DE compatibility
PHASE2_BEST = np.array([
    3.489367,   # dr_ratio_annuals
    0.305041,   # dr_ratio_treegrass
    0.275310,   # dr_ratio_wood
    1.452270,   # dr_ratio_amend
    0.999,      # plant_cover_modifier (was 1.0, nudged from upper bound 1.0)
    0.050,      # tree_fine_root_ratio (now within bounds 0.03-0.2)
    1.995390,   # grass_rs_ratio
    0.00404,    # map_to_prod (was 0.004, nudged from lower bound)
])


def heartbeat_thread(stop_event, start_time_ref, label=""):
    """Print status every 5 minutes."""
    while not stop_event.wait(300):
        if start_time_ref[0] is not None:
            elapsed = time.time() - start_time_ref[0]
            print(f"  [heartbeat {label}] Still running... "
                  f"{elapsed:.0f}s ({elapsed/60:.1f} min, {elapsed/3600:.1f}h)",
                  flush=True)


def main():
    param_set_name = 'Tier1_Tier2'
    param_names = PARAM_SETS[param_set_name]
    n_params = len(param_names)

    de_maxiter = OPTIM_SETTINGS.get('maxiter', 30)
    de_popsize = OPTIM_SETTINGS.get('popsize', 8)
    actual_pop = de_popsize * n_params
    n_splits = OPTIM_SETTINGS.get('n_splits', 5)

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"{'='*70}")
    print(f"PHASE 3: CROSS-VALIDATION ({param_set_name})")
    print(f"Started: {now}")
    print(f"{'='*70}")
    print(f"\nParams ({n_params}): {param_names}")
    print(f"Method: DE (maxiter={de_maxiter}, popsize={de_popsize}, pop={actual_pop})")
    print(f"CV: {n_splits}-fold, test_size={OPTIM_SETTINGS.get('test_size', 0.2)}")
    print(f"Warm-start from Phase 2: {dict(zip(param_names, PHASE2_BEST))}")

    # ---- Load data ----
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    n_cases = len(data['cases_info_df'])
    groups = data['cases_info_df']['group_calib'].value_counts()
    print(f"Loaded {n_cases} cases")
    print(f"Groups:\n{groups.to_string()}")

    # ---- Baseline ----
    baseline = get_baseline_rmse(data)
    print(f"\nBaseline: RMSE={baseline['rmse']:.4f}, R²={baseline['r2']:.4f}")

    # ---- Benchmark ----
    print("\nBenchmarking (3 calls)...")
    bench_times = []
    x0_default = [PARAM_CONFIG[p]['default'] for p in param_names]
    for i in range(3):
        t0 = time.time()
        objective(x0_default, param_names, data)
        dt = time.time() - t0
        bench_times.append(dt)
        print(f"  Call {i+1}: {dt:.2f}s")
    avg_call = np.mean(bench_times)

    # Estimate: ~80% of full 98 cases per fold (train set)
    scale = 0.8
    calls_per_fold = actual_pop * (1 + de_maxiter)  # init + generations
    est_fold_time = avg_call * scale * calls_per_fold
    est_total = est_fold_time * n_splits

    print(f"\n  Avg call time: {avg_call:.2f}s")
    print(f"  Est. calls/fold: ~{calls_per_fold} (pop={actual_pop}, {de_maxiter} gen)")
    print(f"  Est. time/fold: ~{est_fold_time:.0f}s ({est_fold_time/60:.1f} min, {est_fold_time/3600:.1f}h)")
    print(f"  Est. total ({n_splits} folds): ~{est_total:.0f}s ({est_total/60:.1f} min, {est_total/3600:.1f}h)")
    print(f"  (DE often converges earlier)")

    # ---- Start heartbeat ----
    hb_stop = threading.Event()
    hb_start = [time.time()]
    hb = threading.Thread(target=heartbeat_thread, args=(hb_stop, hb_start, "CV"), daemon=True)
    hb.start()

    # ---- Run CV ----
    print(f"\n{'='*70}")
    print(f"Starting {n_splits}-fold CV...")
    cv_start = time.time()

    cv_df, test_rmse, test_cases = cross_validate(
        param_names=param_names,
        data=data,
        method='differential_evolution',
        x0_warmstart=PHASE2_BEST,
        verbose=True
    )

    hb_stop.set()
    cv_elapsed = time.time() - cv_start

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"PHASE 3 RESULTS: {param_set_name}")
    print(f"{'='*70}")
    print(f"Baseline RMSE:     {baseline['rmse']:.4f}")
    print(f"Phase 2 full-set:  1.4450")
    print(f"CV Train RMSE:     {cv_df['train_rmse'].mean():.4f} ± {cv_df['train_rmse'].std():.4f}")
    print(f"CV Val RMSE:       {cv_df['val_rmse'].mean():.4f} ± {cv_df['val_rmse'].std():.4f}")
    print(f"Held-out Test:     {test_rmse:.4f}")

    overfit_ratio = cv_df['val_rmse'].mean() / cv_df['train_rmse'].mean()
    print(f"Overfit ratio:     {overfit_ratio:.3f} {'! OVERFIT' if overfit_ratio > 1.15 else 'OK'}")
    print(f"Total time:        {cv_elapsed:.0f}s ({cv_elapsed/60:.1f} min, {cv_elapsed/3600:.1f}h)")

    print(f"\nMean calibrated parameters:")
    for p in param_names:
        cfg = PARAM_CONFIG[p]
        mean_val = cv_df[p].mean()
        std_val = cv_df[p].std()
        lo, hi = cfg['bounds']
        at_bound = ""
        if abs(mean_val - lo) < std_val * 0.5 + 1e-4:
            at_bound = " [NEAR LOWER]"
        elif abs(mean_val - hi) < std_val * 0.5 + 1e-4:
            at_bound = " [NEAR UPPER]"
        print(f"  {p:25s}: {mean_val:.4f} ± {std_val:.4f}  (default={cfg['default']}){at_bound}")

    print(f"\nPer-fold details:")
    for _, row in cv_df.iterrows():
        print(f"  Fold {int(row['fold'])}: train={row['train_rmse']:.4f}, val={row['val_rmse']:.4f}, "
              f"time={row['time_s']:.0f}s")

    # ---- Save results ----
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    cv_df.to_csv(output_dir / "phase3_cv_folds.csv", index=False)
    print(f"\nFold results: {output_dir / 'phase3_cv_folds.csv'}")

    # Save summary
    summary = {
        'param_set': param_set_name,
        'n_params': n_params,
        'n_folds': n_splits,
        'method': 'differential_evolution',
        'baseline_rmse': baseline['rmse'],
        'baseline_r2': baseline['r2'],
        'phase2_rmse': 1.4450,
        'cv_train_rmse_mean': cv_df['train_rmse'].mean(),
        'cv_train_rmse_std': cv_df['train_rmse'].std(),
        'cv_val_rmse_mean': cv_df['val_rmse'].mean(),
        'cv_val_rmse_std': cv_df['val_rmse'].std(),
        'test_rmse': test_rmse,
        'overfit_ratio': overfit_ratio,
        'total_time_s': cv_elapsed,
    }
    for p in param_names:
        summary[f'mean_{p}'] = cv_df[p].mean()
        summary[f'std_{p}'] = cv_df[p].std()

    pd.DataFrame([summary]).to_csv(output_dir / "phase3_summary.csv", index=False)
    print(f"Summary: {output_dir / 'phase3_summary.csv'}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
