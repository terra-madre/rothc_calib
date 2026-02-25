"""
Cover Crop Test Optimization.

Targeted single-run DE to address systematic overestimation in cover crop cases.

Key changes vs Phase 2:
  - Optimizes only cover crop cases (covercrop, covercrop_amendment, covercrop_cropresid)
  - Parameters: cc_yield_mod, decomp_mod, plant_cover_modifier
  - plant_cover_modifier upper bound raised to 1.0 (wider than param_config default of 0.8)
  - Warm-starts from Phase 2 'all' optimized values (decomp_mod, plant_cover_modifier)
  - cc_yield_mod warm-starts at its default (1.0) as it is new

Rationale:
  cc_yield_mod directly scales cover crop aboveground (and thus belowground via r_s_ratio)
  biomass, providing a mechanism to correct systematic overestimation without affecting
  non-cover-crop cases. cover_crop_rs_ratio was removed from optimization and fixed at
  0.2 in ps_herbaceous.csv.

Usage (from project root):
    nohup mamba run -n terra-plus python -u git_code/run_covercrop_test.py > outputs/covercrop_test.log 2>&1 &
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse
)

# =============================================================================
# Configuration
# =============================================================================

# Parameters to optimize in this test run
PARAM_NAMES = ['cc_yield_mod', 'decomp_mod', 'plant_cover_modifier']

# Covercrop target groups
TARGET_GROUPS = ['covercrop', 'covercrop_amendment', 'covercrop_cropresid']

# Override plant_cover_modifier upper bound for this run (default is 0.8)
PCM_UPPER_BOUND_OVERRIDE = 1.0

# Checkpoint to warm-start from
WARMSTART_CHECKPOINT = "outputs/phase2_sequential_checkpoints/all.json"

# Output checkpoint directory
CKPT_DIR = "outputs/covercrop_test_checkpoints"

# Set name
SET_NAME = "covercrop_test"

# =============================================================================
# Helpers
# =============================================================================

def get_bounds(param_names, pcm_upper=PCM_UPPER_BOUND_OVERRIDE):
    """Get bounds list for the given params, overriding plant_cover_modifier upper."""
    bounds = []
    for p in param_names:
        lo, hi = PARAM_CONFIG[p]['bounds']
        if p == 'plant_cover_modifier':
            hi = pcm_upper
        bounds.append((lo, hi))
    return bounds


def load_warmstart(checkpoint_path):
    """Load optimized params dict from a checkpoint JSON."""
    p = Path(checkpoint_path)
    if not p.exists():
        print(f"WARNING: Warm-start checkpoint not found at {p}")
        return {}
    with open(p) as f:
        ckpt = json.load(f)
    params = ckpt.get('params', {})
    print(f"Loaded warm-start from: {p.name}")
    print(f"  Available params: {list(params.keys())}")
    return params


def save_result(result_dict, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{SET_NAME}.json"
    with open(ckpt_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"  [checkpoint saved → {ckpt_file}]")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent

    maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    seed    = int(OPTIM_SETTINGS.get('seed', 42))

    print("Cover Crop Test Optimization")
    print(f"Parameters:    {PARAM_NAMES}")
    print(f"Target groups: {TARGET_GROUPS}")
    print(f"plant_cover_modifier upper bound: {PCM_UPPER_BOUND_OVERRIDE}")
    print(f"Settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    groups_present = sorted(data['cases_info_df']['group_calib'].unique().tolist())
    print(f"All groups present: {groups_present}")

    # Get covercrop case subset
    ci = data['cases_info_df']
    case_subset = ci[ci['group_calib'].isin(TARGET_GROUPS)]['case'].tolist()
    if not case_subset:
        print(f"\nERROR: No cases found for groups {TARGET_GROUPS}.")
        print(f"Available group_calib values: {groups_present}")
        sys.exit(1)
    print(f"Covercrop case subset: {len(case_subset)} cases")
    print(f"  Groups: {ci[ci['case'].isin(case_subset)]['group_calib'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    baseline_all = get_baseline_rmse(data)
    baseline_cc  = get_baseline_rmse(data, case_subset=case_subset)
    print(f"\nBaseline RMSE (all cases):        {baseline_all['rmse']:.4f}")
    print(f"Baseline RMSE (covercrop subset): {baseline_cc['rmse']:.4f}")

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------
    warmstart_params = load_warmstart(BASE_DIR / WARMSTART_CHECKPOINT)

    bounds = get_bounds(PARAM_NAMES)
    x0 = np.array([
        warmstart_params.get(p, PARAM_CONFIG[p]['default'])
        for p in PARAM_NAMES
    ])

    # Clip x0 to bounds (plant_cover_modifier may need clipping with new upper)
    for i, (pname, (lo, hi)) in enumerate(zip(PARAM_NAMES, bounds)):
        if not (lo <= x0[i] <= hi):
            clipped = float(np.clip(x0[i], lo, hi))
            print(f"  WARNING: x0[{pname}]={x0[i]:.4f} outside bounds ({lo},{hi}) → clipped to {clipped:.4f}")
            x0[i] = clipped

    print(f"\nx0 (warm-start): {dict(zip(PARAM_NAMES, np.round(x0, 4)))}")
    print(f"Bounds:          {dict(zip(PARAM_NAMES, bounds))}")

    # ------------------------------------------------------------------
    # Run DE
    # ------------------------------------------------------------------
    n = len(PARAM_NAMES)
    pop_total = popsize * n
    print(f"\nStarting DE: {n} params, pop={pop_total}, maxiter={maxiter}")
    print(f"{'='*70}")

    start = time.time()
    gen_count = [0]

    def callback(xk, convergence):
        gen_count[0] += 1
        if gen_count[0] % 5 == 0:
            rmse = objective(xk, PARAM_NAMES, data, case_subset)
            elapsed = time.time() - start
            print(f"  Gen {gen_count[0]:3d}: RMSE={rmse:.4f}  Conv={convergence:.6f}  Time={elapsed:.0f}s")
        return False

    result = differential_evolution(
        objective,
        bounds,
        args=(PARAM_NAMES, data, case_subset),
        x0=x0,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        callback=callback,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.001,
        atol=0.001,
        disp=False,
        workers=1,
        polish=False,
    )
    elapsed = time.time() - start

    opt_params = dict(zip(PARAM_NAMES, result.x))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\n--- {SET_NAME} Results ---")
    print(f"RMSE (covercrop subset): {result.fun:.4f}")
    print(f"Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Func calls: {result.nfev}")
    print(f"Converged:  {result.success} ({result.message})")
    print(f"\nOptimized parameters:")
    for pname, val in opt_params.items():
        default = PARAM_CONFIG[pname]['default']
        pct = (val - default) / abs(default) * 100 if default != 0 else float('nan')
        lo, hi = bounds[PARAM_NAMES.index(pname)]
        at_bound = " ← AT LOWER" if abs(val - lo) < 1e-6 else (" ← AT UPPER" if abs(val - hi) < 1e-6 else "")
        print(f"  {pname:30s}: {val:.4f}  (default: {default:.4f}, {pct:+.1f}%){at_bound}")

    # Full-dataset evaluation with optimized params
    print(f"\nEvaluating optimized params on all cases...")
    final_rmse_all, final_details_all = objective(
        list(opt_params.values()), PARAM_NAMES, data, case_subset=None, return_details=True
    )
    final_rmse_cc, final_details_cc = objective(
        list(opt_params.values()), PARAM_NAMES, data, case_subset=case_subset, return_details=True
    )

    imp_all = (baseline_all['rmse'] - final_rmse_all) / baseline_all['rmse'] * 100
    imp_cc  = (baseline_cc['rmse'] - final_rmse_cc)  / baseline_cc['rmse']  * 100

    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Covercrop subset  → RMSE={final_rmse_cc:.4f}  (baseline={baseline_cc['rmse']:.4f}, {imp_cc:+.1f}%)")
    print(f"                       MAE={final_details_cc['mae']:.4f}, Bias={final_details_cc['bias']:.4f}, R²={final_details_cc['r2']:.4f}")
    print(f"  All cases         → RMSE={final_rmse_all:.4f}  (baseline={baseline_all['rmse']:.4f}, {imp_all:+.1f}%)")
    print(f"                       MAE={final_details_all['mae']:.4f}, Bias={final_details_all['bias']:.4f}, R²={final_details_all['r2']:.4f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    result_dict = {
        'set_name': SET_NAME,
        'params': opt_params,
        'param_names': PARAM_NAMES,
        'target_groups': TARGET_GROUPS,
        'n_cases_cc': len(case_subset),
        'rmse_cc': float(result.fun),
        'rmse_all': float(final_rmse_all),
        'baseline_rmse_cc': baseline_cc['rmse'],
        'baseline_rmse_all': baseline_all['rmse'],
        'improvement_pct_cc': float(imp_cc),
        'improvement_pct_all': float(imp_all),
        'mae_cc': float(final_details_cc['mae']),
        'bias_cc': float(final_details_cc['bias']),
        'r2_cc': float(final_details_cc['r2']),
        'mae_all': float(final_details_all['mae']),
        'bias_all': float(final_details_all['bias']),
        'r2_all': float(final_details_all['r2']),
        'time_s': elapsed,
        'func_calls': result.nfev,
        'success': bool(result.success),
        'message': result.message,
        'maxiter': maxiter,
        'popsize': popsize,
        'pcm_upper_bound': PCM_UPPER_BOUND_OVERRIDE,
        'warmstart_checkpoint': WARMSTART_CHECKPOINT,
    }
    save_result(result_dict, BASE_DIR / CKPT_DIR)

    print(f"\nCover crop test optimization complete!")
