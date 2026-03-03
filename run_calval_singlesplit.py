"""
Calibration-Validation run using a stratified 70/30 train/test split.

Strategy:
  1. Stratified 70/30 split by SUB_RUNS super-groups
    2. Run sequential-group optimization on training cases only,
         warm-starting from the sequential_groups 'all' checkpoint
  3. Evaluate calibration (train) and validation (test) performance
  4. Report acceptance criteria:
       C1  Mean bias ≤ PMU
       C2  ≥90% of test obs within 90% PI
       C3  R² > 0 on test set

Usage (from project root):
    nohup mamba run -n terra-plus python -u git_code/run_calval_singlesplit.py \
        --proc-subdir no_outliers --output-dir outputs/no_outliers \
        > outputs/no_outliers/calval_no_outliers.log 2>&1 &
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse,
    SUB_RUNS, run_sequential_subruns,
    load_warmstart_params, get_train_case_subset, build_supergroup_labels,
)

BASE_DIR = Path(__file__).parent.parent


# =============================================================================
# Calval-specific helpers
# =============================================================================

def supergroup_stratified_split(cases_info, super_labels, test_frac, random_state):
    """Stratified train/test split per super-group.

    Uses round() so small groups (e.g. pruning n=5, 30%) get 2 test / 3 train
    instead of 1/4.  Minimum 1 test case per group with n >= 2.
    """
    rng = np.random.default_rng(random_state)
    train_cases, test_cases = [], []
    for sg in sorted(super_labels.unique()):
        sg_cases = cases_info.loc[(super_labels == sg).values, 'case'].values
        n = len(sg_cases)
        n_test = max(1, round(n * test_frac)) if n > 1 else 0
        n_test = min(n_test, n - 1)
        perm = rng.permutation(n)
        test_cases.extend(sg_cases[perm[:n_test]])
        train_cases.extend(sg_cases[perm[n_test:]])
    return sorted(train_cases), sorted(test_cases)


def print_split_summary(cases_info, train_cases, test_cases):
    """Print group-level train/test breakdown."""
    print(f"\n{'Group':30s}  {'Total':>5}  {'Train':>5}  {'Test':>5}")
    print("-" * 55)
    train_set, test_set = set(train_cases), set(test_cases)
    for grp in sorted(cases_info['group_calib'].unique()):
        grp_cases = cases_info[cases_info['group_calib'] == grp]['case'].tolist()
        n_train = sum(c in train_set for c in grp_cases)
        n_test = sum(c in test_set for c in grp_cases)
        print(f"  {grp:28s}  {len(grp_cases):>5}  {n_train:>5}  {n_test:>5}")
    print("-" * 55)
    print(f"  {'TOTAL':28s}  {len(cases_info):>5}  {len(train_cases):>5}  {len(test_cases):>5}")


def compute_acceptance_criteria(cal_rmse, cal_details, val_details):
    """Evaluate three cal-val acceptance criteria.

    C1  Mean bias ≤ PMU  (skipped — replicate data unavailable)
    C2  ≥90% of val obs within PI_90 = pred ± 1.645 × RMSE_cal
    C3  R² > 0 on validation set
    """
    results = []

    # C1
    val_bias = val_details['bias']
    results.append({
        'criterion': 'C1',
        'description': 'Mean bias ≤ PMU',
        'value': f"val bias = {val_bias:+.4f} t C ha⁻¹ yr⁻¹",
        'passed': None,
        'note': 'PMU not computable — replicate data (σ_j, n_j) not in dataset',
    })

    # C2
    half_width = 1.645 * cal_rmse
    comp = val_details['comparison_df'].copy()
    obs = comp['delta_soc_t_ha_y'].values
    pred = comp['delta_treatment_control_per_year'].values
    within = np.abs(obs - pred) <= half_width
    coverage = within.mean()
    results.append({
        'criterion': 'C2',
        'description': '≥90% of val obs within 90% PI',
        'value': f"coverage = {coverage*100:.1f}%  (PI ±{half_width:.3f})",
        'passed': bool(coverage >= 0.90),
        'note': f"PI = pred ± 1.645 × RMSE_cal ({cal_rmse:.4f}); assumes Gaussian residuals",
    })

    # C3
    val_r2 = val_details['r2']
    results.append({
        'criterion': 'C3',
        'description': 'R² > 0 on validation set',
        'value': f"val R² = {val_r2:.4f}",
        'passed': bool(val_r2 > 0),
        'note': '',
    })

    return results


def format_acceptance_report(criteria_results, cal_details, val_details,
                             train_cases, test_cases):
    """Build formatted acceptance-criteria report string."""
    sep = "=" * 70
    lines = [
        sep,
        "CAL-VAL ACCEPTANCE CRITERIA REPORT",
        sep,
        f"Split: {len(train_cases)} train / {len(test_cases)} test  "
        f"(stratified by super-groups, seed=42)",
        "",
        f"Calibration (train):  RMSE={cal_details['rmse']:.4f}  "
        f"MAE={cal_details['mae']:.4f}  Bias={cal_details['bias']:+.4f}  "
        f"R²={cal_details['r2']:.4f}  n={cal_details['n_cases']}",
        f"Validation  (test) :  RMSE={val_details['rmse']:.4f}  "
        f"MAE={val_details['mae']:.4f}  Bias={val_details['bias']:+.4f}  "
        f"R²={val_details['r2']:.4f}  n={val_details['n_cases']}",
        "",
        f"{'Criterion':<6}  {'Description':<35}  {'Result':<45}  {'Pass?':<6}",
        "-" * 100,
    ]
    for r in criteria_results:
        p = "PASS" if r['passed'] is True else "FAIL" if r['passed'] is False else "N/A"
        lines.append(f"  {r['criterion']:<4}  {r['description']:<35}  {r['value']:<45}  {p:<6}")
        if r['note']:
            lines.append(f"        Note: {r['note']}")
    lines.append(sep)
    return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================

def run_calval(data, output_dir, warmstart_path=None,
               maxiter=None, popsize=None, seed=None,
               train_size=0.7, random_state=42):
    """Run calibration-validation with stratified 70/30 split.

    Returns:
        (best_params, criteria_results)
    """
    if maxiter is None:
        maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    if popsize is None:
        popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    if seed is None:
        seed = int(OPTIM_SETTINGS.get('seed', 42))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "calval_checkpoints"
    if warmstart_path is None:
        warmstart_path = output_dir / "sequential_groups_checkpoints" / "all.json"

    print("Cal-Val: Stratified 70/30 Split + Sequential Group Optimization")
    print(f"Train size: {int(train_size*100)}%  |  Random state: {random_state}")
    print(f"Settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")
    print(f"Warm-start: {warmstart_path}")
    print(f"Loaded {len(data['cases_info_df'])} cases")

    cases_info = data['cases_info_df']
    print(f"Groups: {sorted(cases_info['group_calib'].unique().tolist())}")

    baseline = get_baseline_rmse(data)
    baseline_rmse = baseline['rmse']
    print(f"\nBaseline RMSE (defaults): {baseline_rmse:.4f}")

    # ── Super-group labels + stratified split ─────────────────────────────────
    super_labels = build_supergroup_labels(cases_info)
    print("\nSuper-group mapping (for stratification):")
    for sg in sorted(super_labels.unique()):
        members = cases_info[super_labels == sg]['group_calib'].unique().tolist()
        n = (super_labels == sg).sum()
        print(f"  {sg:20s} ({n:2d} cases): {', '.join(sorted(members))}")

    train_cases, test_cases = supergroup_stratified_split(
        cases_info, super_labels,
        test_frac=1.0 - train_size,
        random_state=random_state,
    )
    train_set = set(train_cases)

    print(f"\nDataset split: {len(train_cases)} train, {len(test_cases)} test")
    print_split_summary(cases_info, train_cases, test_cases)

    # ── Warm-start from sequential_groups checkpoint ─────────────────────────
    print("\nLoading warm-start parameters...")
    best_params = load_warmstart_params(warmstart_path)
    if best_params:
        print("  Values:")
        for p, v in best_params.items():
            print(f"    {p:30s}: {v:.4f}")

    # ── Sequential sub-runs on training set ───────────────────────────────────
    all_results = run_sequential_subruns(
        data, ckpt_dir, best_params,
        case_subset_fn=lambda d, groups: get_train_case_subset(d, groups, train_set),
        maxiter=maxiter, popsize=popsize, seed=seed,
        baseline_rmse=baseline_rmse,
    )

    # ── Final evaluation: train and test ──────────────────────────────────────
    if not best_params:
        print("ERROR: no parameters calibrated — exiting.")
        sys.exit(1)

    pnames = list(best_params.keys())
    pvals = [best_params[p] for p in pnames]

    print(f"\n{'='*70}")
    print("FINAL PARAMETER EVALUATION")
    print(f"{'='*70}")

    cal_rmse, cal_details = objective(
        pvals, pnames, data, case_subset=train_cases, return_details=True,
    )
    val_rmse, val_details = objective(
        pvals, pnames, data, case_subset=test_cases, return_details=True,
    )

    print(f"Calibration (train, n={cal_details['n_cases']}):")
    print(f"  RMSE={cal_rmse:.4f}  MAE={cal_details['mae']:.4f}  "
          f"Bias={cal_details['bias']:+.4f}  R²={cal_details['r2']:.4f}")
    print(f"Validation  (test,  n={val_details['n_cases']}):")
    print(f"  RMSE={val_rmse:.4f}  MAE={val_details['mae']:.4f}  "
          f"Bias={val_details['bias']:+.4f}  R²={val_details['r2']:.4f}")

    # ── Acceptance criteria ───────────────────────────────────────────────────
    print("")
    criteria = compute_acceptance_criteria(cal_rmse, cal_details, val_details)
    report = format_acceptance_report(
        criteria, cal_details, val_details, train_cases, test_cases,
    )
    print(report)

    # ── Sub-run summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUB-RUN SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sub-run':20s} | {'RMSE (train subset)':>20s} | {'Improve':>8s} | {'Time':>7s}")
    print("-" * 65)
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        print(f"{r['set_name']:20s} | {r['rmse']:>20.4f} | {imp:+7.1f}% | {r['time_s']:6.0f}s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_rows = []
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        summary_rows.append({
            'set_name': r['set_name'], 'n_cases': r.get('n_cases'),
            'n_params': len(r['params']), 'rmse': r['rmse'],
            'baseline_rmse': baseline_rmse, 'improvement_pct': imp,
            'time_s': r['time_s'], 'func_calls': r['func_calls'],
            'success': r['success'],
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "calval_summary.csv", index=False)
    pd.DataFrame([{'parameter': p, 'value': v} for p, v in best_params.items()]).to_csv(
        output_dir / "calval_params.csv", index=False,
    )
    pd.DataFrame(
        [{'case': c, 'split': 'train'} for c in train_cases]
        + [{'case': c, 'split': 'test'} for c in test_cases]
    ).to_csv(output_dir / "calval_split.csv", index=False)
    (output_dir / "calval_report.txt").write_text(report)

    print(f"\nSaved: calval_summary.csv, calval_params.csv, calval_split.csv, calval_report.txt")
    print("Cal-Val complete!")

    return best_params, criteria


# =============================================================================
# Standalone CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cal-Val: Stratified split + sequential-groups optimization")
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: outputs/)')
    ap.add_argument('--proc-subdir', default=None,
                    help='Subdirectory under inputs/processed/ (e.g. "no_outliers")')
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "outputs"

    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR, proc_subdir=args.proc_subdir)

    run_calval(data, output_dir)
