"""
Phase 2 Sequential Group Optimization.

Runs DE on each calibration group in sequence. Each run:
  - Uses only cases belonging to its target group(s)
  - Warm-starts from all previously optimized parameter values
  - Saves a checkpoint JSON on completion

Run order and group/parameter mapping:
  1. amendment        → dr_ratio_fym
  2. cropresid        → residue_frac_remaining
  3. covercrop_all    → dr_ratio_annuals, cc_yield_mod, decomp_mod
                        (groups: covercrop, covercrop_amendment, covercrop_cropresid)
  4. grass_all        → grass_rsr_b, turnover_bg_grass, dr_ratio_treegrass
                        (groups: grass, grass_annuals)
  5. pruning          → dr_ratio_wood
                        (groups: grass_pruning, covercrop_pruning)
  6. all              → all above + plant_cover_modifier  (all cases)

Usage (from project root):
    nohup mamba run -n terra-plus python -u git_code/run_phase2_sequential.py > outputs/phase2_sequential.log 2>&1 &
"""

import sys
import json
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

# =============================================================================
# Sub-run definitions
# Each entry: (set_name, param_names, target_groups)
# target_groups=None → use all cases
# =============================================================================

SUB_RUNS = [
    (
        'amendment',
        PARAM_SETS['amendment'],
        ['amendment'],
    ),
    (
        'cropresid',
        PARAM_SETS['cropresid'],
        ['cropresid'],
    ),
    (
        'covercrop_all',
        PARAM_SETS['covercrop_all'],
        ['covercrop', 'covercrop_amendment', 'covercrop_cropresid'],
    ),
    (
        'grass_all',
        PARAM_SETS['grass_all'],
        ['grass', 'grass_annuals'],
    ),
    (
        'pruning',
        PARAM_SETS['pruning'],
        ['grass_pruning', 'covercrop_pruning'],
    ),
    (
        'all',
        PARAM_SETS['all'],
        None,   # all cases
    ),
]


# =============================================================================
# Helpers
# =============================================================================

def get_case_subset(data, groups):
    """Return list of case IDs belonging to the given group(s)."""
    if groups is None:
        return None
    cases_info = data['cases_info_df']
    return cases_info[cases_info['group_calib'].isin(groups)]['case'].tolist()


def run_de(set_name, param_names, data, case_subset, best_params,
           maxiter, popsize, seed, verbose=True):
    """Run DE on a single sub-run, warm-started from best_params."""
    n = len(param_names)
    pop_total = popsize * n
    subset_label = f"{len(case_subset)} cases" if case_subset is not None else "all cases"

    print(f"\n{'='*70}")
    print(f"Sub-run: {set_name}  [{subset_label}]")
    print(f"Params ({n}): {param_names}")
    print(f"Pop={pop_total}, MaxGen={maxiter}, Seed={seed}")
    print(f"{'='*70}")

    # x0: warm-start from any previously optimized values
    x0 = np.array([best_params.get(p, PARAM_CONFIG[p]['default']) for p in param_names])
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]

    # Validate x0 within bounds (clip if needed)
    for i, (pname, (lo, hi)) in enumerate(zip(param_names, bounds)):
        if not (lo <= x0[i] <= hi):
            clipped = float(np.clip(x0[i], lo, hi))
            print(f"  WARNING: x0[{pname}]={x0[i]:.4f} outside bounds ({lo},{hi}) → clipped to {clipped:.4f}")
            x0[i] = clipped

    print(f"x0 (warm-start): {dict(zip(param_names, np.round(x0, 4)))}")

    start = time.time()
    gen_count = [0]

    def callback(xk, convergence):
        gen_count[0] += 1
        if gen_count[0] % 5 == 0:
            rmse = objective(xk, param_names, data, case_subset)
            elapsed = time.time() - start
            print(f"  Gen {gen_count[0]:3d}: RMSE={rmse:.4f}  Conv={convergence:.6f}  Time={elapsed:.0f}s")
        return False

    result = differential_evolution(
        objective,
        bounds,
        args=(param_names, data, case_subset),
        x0=x0,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        callback=callback,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.001,
        atol=0.001,
        disp=verbose,
        workers=1,
        polish=False,
    )
    elapsed = time.time() - start

    opt_params = dict(zip(param_names, result.x))

    print(f"\n--- {set_name} Results ---")
    print(f"RMSE:       {result.fun:.4f}")
    print(f"Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Func calls: {result.nfev}")
    print(f"Converged:  {result.success} ({result.message})")
    for pname, val in opt_params.items():
        default = PARAM_CONFIG[pname]['default']
        pct = (val - default) / abs(default) * 100 if default != 0 else float('nan')
        bound_lo, bound_hi = PARAM_CONFIG[pname]['bounds']
        at_bound = " ← AT LOWER" if abs(val - bound_lo) < 1e-6 else (" ← AT UPPER" if abs(val - bound_hi) < 1e-6 else "")
        print(f"  {pname:30s}: {val:.4f}  (default: {default:.4f}, {pct:+.1f}%){at_bound}")

    return {
        'set_name': set_name,
        'rmse': float(result.fun),
        'time_s': elapsed,
        'func_calls': result.nfev,
        'success': bool(result.success),
        'message': result.message,
        'params': opt_params,
        'n_cases': len(case_subset) if case_subset is not None else None,
    }


def save_checkpoint(run_result, baseline_rmse, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{run_result['set_name']}.json"
    n_cases = run_result['n_cases']
    imp = (baseline_rmse - run_result['rmse']) / baseline_rmse * 100 if baseline_rmse else None
    payload = {**run_result, 'baseline_rmse': baseline_rmse, 'improvement_pct': imp}
    with open(ckpt_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  [checkpoint saved → {ckpt_file.name}]")


def load_checkpoint(set_name, ckpt_dir):
    ckpt_file = ckpt_dir / f"{set_name}.json"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            return json.load(f)
    return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent

    maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    seed    = int(OPTIM_SETTINGS.get('seed', 42))

    print("Phase 2: Sequential Group-Specific Optimization")
    print(f"Settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")
    print(f"Sub-runs: {[sr[0] for sr in SUB_RUNS]}")

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "phase2_sequential_checkpoints"

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    n_cases = len(data['cases_info_df'])
    groups_present = sorted(data['cases_info_df']['group_calib'].unique().tolist())
    print(f"Loaded {n_cases} cases")
    print(f"Groups: {groups_present}")

    baseline = get_baseline_rmse(data)
    baseline_rmse = baseline['rmse']
    print(f"\nBaseline RMSE (all cases, defaults): {baseline_rmse:.4f}")
    print(f"Baseline MAE:  {baseline['mae']:.4f}")
    print(f"Baseline Bias: {baseline['bias']:.4f}")
    print(f"Baseline R²:   {baseline['r2']:.4f}")

    # -------------------------------------------------------------------
    # Run sub-runs sequentially, accumulating best_params
    # -------------------------------------------------------------------
    best_params = {}   # grows as each sub-run completes
    all_results = []

    for set_name, param_names, target_groups in SUB_RUNS:
        if not param_names:
            print(f"\nWARNING: No params defined for set '{set_name}', skipping.")
            continue

        # Resume: use checkpoint if it exists
        existing = load_checkpoint(set_name, ckpt_dir)
        if existing is not None:
            print(f"\n[RESUME] {set_name}: checkpoint found — RMSE={existing['rmse']:.4f}, skipping run.")
            # Still update best_params from checkpoint
            for pname, pval in existing['params'].items():
                best_params[pname] = pval
            all_results.append(existing)
            continue

        # Get cases for this sub-run
        case_subset = get_case_subset(data, target_groups)
        if case_subset is not None and len(case_subset) == 0:
            print(f"\nWARNING: No cases found for groups {target_groups}, skipping '{set_name}'.")
            continue

        # Run DE
        result = run_de(
            set_name, param_names, data, case_subset, best_params,
            maxiter=maxiter, popsize=popsize, seed=seed
        )

        # Merge optimized params into best_params (warm-start for next run)
        for pname, pval in result['params'].items():
            best_params[pname] = pval

        save_checkpoint(result, baseline_rmse, ckpt_dir)
        all_results.append(result)

        print(f"\nCumulative best_params after '{set_name}':")
        for pname, pval in best_params.items():
            print(f"  {pname:30s}: {pval:.4f}")

    # -------------------------------------------------------------------
    # Final evaluation: best_params on all cases
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (all cases, accumulated best_params)")
    print(f"{'='*70}")
    if best_params:
        final_param_names = list(best_params.keys())
        final_param_values = [best_params[p] for p in final_param_names]
        from optimization import objective
        final_rmse, final_details = objective(
            final_param_values, final_param_names, data, case_subset=None, return_details=True
        )
        imp = (baseline_rmse - final_rmse) / baseline_rmse * 100
        print(f"Final RMSE (all cases): {final_rmse:.4f}  ({imp:+.1f}% vs baseline)")
        print(f"Final MAE:  {final_details['mae']:.4f}")
        print(f"Final Bias: {final_details['bias']:.4f}")
        print(f"Final R²:   {final_details['r2']:.4f}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sub-run':20s} | {'RMSE':>7s} | {'Improve':>8s} | {'Time':>7s}")
    print("-" * 55)
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        t = r['time_s']
        print(f"{r['set_name']:20s} | {r['rmse']:7.4f} | {imp:+7.1f}% | {t:6.0f}s")

    # Save summary CSVs
    summary_rows = []
    detail_rows = []
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        summary_rows.append({
            'set_name': r['set_name'],
            'n_cases': r.get('n_cases'),
            'n_params': len(r['params']),
            'rmse': r['rmse'],
            'baseline_rmse': baseline_rmse,
            'improvement_pct': imp,
            'time_s': r['time_s'],
            'func_calls': r['func_calls'],
            'success': r['success'],
            'maxiter': maxiter,
            'popsize': popsize,
        })
        for pname, pval in r['params'].items():
            cfg = PARAM_CONFIG[pname]
            detail_rows.append({
                'set_name': r['set_name'],
                'parameter': pname,
                'optimized_value': pval,
                'default_value': cfg['default'],
                'bound_min': cfg['bounds'][0],
                'bound_max': cfg['bounds'][1],
                'pct_change': (pval - cfg['default']) / abs(cfg['default']) * 100 if cfg['default'] != 0 else float('nan'),
            })

    pd.DataFrame(summary_rows).to_csv(output_dir / "phase2_sequential_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "phase2_sequential_params.csv", index=False)
    print(f"\nSaved: outputs/phase2_sequential_summary.csv, outputs/phase2_sequential_params.csv")
    print("Phase 2 Sequential complete!")
