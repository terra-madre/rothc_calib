"""
Sequential Group Optimization.

Runs DE on each calibration group in sequence. Each run:
  - Uses only cases belonging to its target group(s)
  - Warm-starts from all previously optimized parameter values
  - Saves a checkpoint JSON on completion

Run order and group/parameter mapping:
  1. amendment        → dr_ratio_fym
  2. cropresid        → residue_frac_remaining
  3. covercrop_all    → dr_ratio_annuals, cc_yield_mod, decomp_mod
  4. grass_all        → grass_rsr_b, turnover_bg_grass, dr_ratio_treegrass
  5. pruning          → dr_ratio_wood
  6. all              → all above + plant_cover_modifier  (all cases)

Usage (from project root):
    nohup mamba run -n terra-plus python -u git_code/run_sequential_groups.py \
        --proc-subdir no_outliers --output-dir outputs/no_outliers \
        > outputs/no_outliers/sequential_groups_no_outliers.log 2>&1 &
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse,
    SUB_RUNS, get_case_subset, run_sequential_subruns,
)

BASE_DIR = Path(__file__).parent.parent


# =============================================================================
# Public API
# =============================================================================

def run_sequential_groups(data, output_dir, maxiter=None, popsize=None, seed=None):
    """Run sequential group-specific optimization.

    Returns:
        (best_params, baseline_rmse, all_results)
    """
    if maxiter is None:
        maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    if popsize is None:
        popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    if seed is None:
        seed = int(OPTIM_SETTINGS.get('seed', 42))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "sequential_groups_checkpoints"

    print("Sequential Groups: Group-Specific Optimization")
    print(f"Settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")
    print(f"Sub-runs: {[sr[0] for sr in SUB_RUNS]}")
    print(f"Loaded {len(data['cases_info_df'])} cases")
    print(f"Groups: {sorted(data['cases_info_df']['group_calib'].unique().tolist())}")

    baseline = get_baseline_rmse(data)
    baseline_rmse = baseline['rmse']
    print(f"\nBaseline RMSE: {baseline_rmse:.4f},  MAE: {baseline['mae']:.4f},  "
          f"Bias: {baseline['bias']:.4f},  R²: {baseline['r2']:.4f}")

    # ── Sequential sub-runs ───────────────────────────────────────────────────
    best_params = {}
    all_results = run_sequential_subruns(
        data, ckpt_dir, best_params,
        case_subset_fn=get_case_subset,
        maxiter=maxiter, popsize=popsize, seed=seed,
        baseline_rmse=baseline_rmse,
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (all cases, accumulated best_params)")
    print(f"{'='*70}")
    if best_params:
        pnames = list(best_params.keys())
        pvals = [best_params[p] for p in pnames]
        final_rmse, final_details = objective(
            pvals, pnames, data, case_subset=None, return_details=True,
        )
        imp = (baseline_rmse - final_rmse) / baseline_rmse * 100
        print(f"Final RMSE: {final_rmse:.4f}  ({imp:+.1f}% vs baseline)")
        print(f"Final MAE:  {final_details['mae']:.4f}")
        print(f"Final Bias: {final_details['bias']:.4f}")
        print(f"Final R²:   {final_details['r2']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sub-run':20s} | {'RMSE':>7s} | {'Improve':>8s} | {'Time':>7s}")
    print("-" * 55)
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        print(f"{r['set_name']:20s} | {r['rmse']:7.4f} | {imp:+7.1f}% | {r['time_s']:6.0f}s")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    summary_rows, detail_rows = [], []
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        summary_rows.append({
            'set_name': r['set_name'], 'n_cases': r.get('n_cases'),
            'n_params': len(r['params']), 'rmse': r['rmse'],
            'baseline_rmse': baseline_rmse, 'improvement_pct': imp,
            'time_s': r['time_s'], 'func_calls': r['func_calls'],
            'success': r['success'], 'maxiter': maxiter, 'popsize': popsize,
        })
        for pname, pval in r['params'].items():
            cfg = PARAM_CONFIG[pname]
            detail_rows.append({
                'set_name': r['set_name'], 'parameter': pname,
                'optimized_value': pval, 'default_value': cfg['default'],
                'bound_min': cfg['bounds'][0], 'bound_max': cfg['bounds'][1],
                'pct_change': ((pval - cfg['default']) / abs(cfg['default']) * 100
                               if cfg['default'] != 0 else float('nan')),
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "sequential_groups_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "sequential_groups_params.csv", index=False)
    print(f"\nSaved: sequential_groups_summary.csv, sequential_groups_params.csv")
    print("Sequential Groups run complete!")

    return best_params, baseline_rmse, all_results


# =============================================================================
# Standalone CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sequential Group Optimization")
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: outputs/)')
    ap.add_argument('--proc-subdir', default=None,
                    help='Subdirectory under inputs/processed/ (e.g. "no_outliers")')
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "outputs"

    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR, proc_subdir=args.proc_subdir)

    run_sequential_groups(data, output_dir)
