"""
Stratified K-fold cross-validation using sequential group optimization.

Strategy:
  1. Build super-group labels and split all cases into K stratified folds.
  2. For each fold k:
       a. Train = all cases NOT in fold k.   Val = fold k.
         b. Run the full sequential-group sub-runs on train cases,
             warm-starting from the sequential_groups 'all' checkpoint.
       c. Evaluate calibration and validation metrics.
  3. Report per-fold and mean±std metrics with acceptance criteria.

Usage (from project root):
    nohup mamba run -n terra-plus python -u git_code/run_kfold.py \
        --proc-subdir no_outliers --output-dir outputs/no_outliers \
        > outputs/no_outliers/kfold_no_outliers.log 2>&1 &
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse,
    SUB_RUNS,
    run_sequential_subruns,
    load_warmstart_params, get_train_case_subset, build_supergroup_labels,
)

BASE_DIR = Path(__file__).parent.parent


# =============================================================================
# K-fold-specific helpers
# =============================================================================

def print_fold_split(cases_info, fold, train_cases, val_cases):
    """Print group-level train/val breakdown for one fold."""
    print(f"\n  {'Group':30s}  {'Total':>5}  {'Train':>5}  {'Val':>5}")
    print("  " + "-" * 53)
    train_set, val_set = set(train_cases), set(val_cases)
    for grp in sorted(cases_info['group_calib'].unique()):
        grp_cases = cases_info[cases_info['group_calib'] == grp]['case'].tolist()
        n_train = sum(c in train_set for c in grp_cases)
        n_val = sum(c in val_set for c in grp_cases)
        print(f"  {grp:30s}  {len(grp_cases):>5}  {n_train:>5}  {n_val:>5}")
    print("  " + "-" * 53)
    print(f"  {'TOTAL':30s}  {len(cases_info):>5}  {len(train_cases):>5}  {len(val_cases):>5}")


def compute_fold_acceptance(cal_rmse, val_details):
    """Evaluate per-fold acceptance criteria (C2 + C3)."""
    half_width = 1.645 * cal_rmse
    comp = val_details['comparison_df'].copy()
    obs = comp['delta_soc_t_ha_y'].values
    pred = comp['delta_treatment_control_per_year'].values
    within = np.abs(obs - pred) <= half_width
    coverage = float(within.mean())
    return {
        'pi_half_width': half_width,
        'c2_coverage': coverage,
        'c2_passed': coverage >= 0.90,
        'c3_r2': val_details['r2'],
        'c3_passed': val_details['r2'] > 0,
    }


# =============================================================================
# Public API
# =============================================================================

def run_kfold(data, output_dir, warmstart_path=None,
              maxiter=None, popsize=None, seed=None,
              n_splits=None, random_state=None):
    """Run stratified K-fold cross-validation.

    Returns:
        cv_df  (DataFrame of per-fold metrics)
    """
    if maxiter is None:
        maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    if popsize is None:
        popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    if seed is None:
        seed = int(OPTIM_SETTINGS.get('seed', 42))
    if n_splits is None:
        n_splits = int(OPTIM_SETTINGS.get('n_splits', 5))
    if random_state is None:
        random_state = int(OPTIM_SETTINGS.get('cv_random_state', 42))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kfold_ckpt_root = output_dir / "kfold_checkpoints"
    if warmstart_path is None:
        warmstart_path = output_dir / "sequential_groups_checkpoints" / "all.json"

    print("K-Fold CV: Stratified Phase 2 Sequential Optimization")
    print(f"K={n_splits}  |  Random state={random_state}")
    print(f"DE settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")
    print(f"Warm-start: {warmstart_path}")
    print(f"Loaded {len(data['cases_info_df'])} cases")

    cases_info = data['cases_info_df']
    print(f"Groups: {sorted(cases_info['group_calib'].unique().tolist())}")

    baseline = get_baseline_rmse(data)
    baseline_rmse = baseline['rmse']
    print(f"\nBaseline RMSE (defaults): {baseline_rmse:.4f}")

    # ── Super-group labels ────────────────────────────────────────────────────
    super_labels = build_supergroup_labels(cases_info)
    cases_arr = cases_info['case'].values
    strata_arr = super_labels.values

    print("\nSuper-group mapping:")
    for sg in sorted(np.unique(strata_arr)):
        members = cases_info[super_labels == sg]['group_calib'].unique().tolist()
        n = (super_labels == sg).sum()
        print(f"  {sg:20s} ({n:2d} cases): {', '.join(sorted(members))}")

    # ── Warm-start parameters (shared across folds) ──────────────────────────
    print("\nLoading warm-start parameters...")
    warmstart_params = load_warmstart_params(warmstart_path)
    if warmstart_params:
        print("  Values:")
        for p, v in warmstart_params.items():
            print(f"    {p:30s}: {v:.4f}")

    # ── K-fold loop ───────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []
    param_rows = []
    fold_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cases_arr, strata_arr)):
        fold_num = fold_idx + 1
        train_cases = cases_arr[train_idx].tolist()
        val_cases = cases_arr[val_idx].tolist()
        train_set = set(train_cases)
        ckpt_dir = kfold_ckpt_root / f"fold_{fold_num}"

        print(f"\n{'#'*70}")
        print(f"FOLD {fold_num} / {n_splits}")
        print(f"{'#'*70}")
        print(f"Train: {len(train_cases)} cases   Val: {len(val_cases)} cases")
        print_fold_split(cases_info, fold_num, train_cases, val_cases)

        best_params = dict(warmstart_params)

        # Sequential sub-runs on this fold's training set
        # Use a per-fold seed perturbation so each fold explores differently
        run_sequential_subruns(
            data, ckpt_dir, best_params,
            case_subset_fn=lambda d, groups, _ts=train_set: get_train_case_subset(d, groups, _ts),
            maxiter=maxiter, popsize=popsize, seed=seed + fold_idx,
            baseline_rmse=baseline_rmse,
        )

        # ── Evaluate this fold ────────────────────────────────────────────────
        if not best_params:
            print(f"  ERROR: no parameters calibrated for fold {fold_num} — skipping.")
            continue

        pnames = list(best_params.keys())
        pvals = [best_params[p] for p in pnames]

        print(f"\n  {'='*60}")
        print(f"  FOLD {fold_num} EVALUATION")
        print(f"  {'='*60}")

        cal_rmse, cal_details = objective(
            pvals, pnames, data, case_subset=train_cases, return_details=True,
        )
        val_rmse, val_details = objective(
            pvals, pnames, data, case_subset=val_cases, return_details=True,
        )

        print(f"  Calibration (train, n={cal_details['n_cases']}):")
        print(f"    RMSE={cal_rmse:.4f}  MAE={cal_details['mae']:.4f}  "
              f"Bias={cal_details['bias']:+.4f}  R²={cal_details['r2']:.4f}")
        print(f"  Validation  (val,   n={val_details['n_cases']}):")
        print(f"    RMSE={val_rmse:.4f}  MAE={val_details['mae']:.4f}  "
              f"Bias={val_details['bias']:+.4f}  R²={val_details['r2']:.4f}")

        ac = compute_fold_acceptance(cal_rmse, val_details)
        print(f"  C2: {ac['c2_coverage']*100:.1f}% coverage "
              f"(PI ±{ac['pi_half_width']:.3f})  "
              f"{'PASS' if ac['c2_passed'] else 'FAIL'}")
        print(f"  C3: R²={ac['c3_r2']:.4f}  "
              f"{'PASS' if ac['c3_passed'] else 'FAIL'}")

        fold_elapsed = time.time() - fold_start
        print(f"  Total elapsed: {fold_elapsed:.0f}s ({fold_elapsed/60:.1f} min)")

        fold_rows.append({
            'fold': fold_num,
            'n_train': len(train_cases),
            'n_val': len(val_cases),
            'cal_rmse': cal_rmse, 'val_rmse': val_rmse,
            'cal_mae': cal_details['mae'], 'val_mae': val_details['mae'],
            'cal_bias': cal_details['bias'], 'val_bias': val_details['bias'],
            'cal_r2': cal_details['r2'], 'val_r2': val_details['r2'],
            'c2_coverage': ac['c2_coverage'], 'c2_passed': ac['c2_passed'],
            'c3_passed': ac['c3_passed'], 'time_s': fold_elapsed,
            'train_cases': ','.join(str(c) for c in sorted(train_cases)),
            'val_cases': ','.join(str(c) for c in sorted(val_cases)),
        })
        for pname, pval in best_params.items():
            param_rows.append({
                'fold': fold_num, 'parameter': pname,
                'optimized_value': pval,
                'default_value': PARAM_CONFIG[pname]['default'],
            })

    # ── Aggregate report ──────────────────────────────────────────────────────
    total_time = time.time() - fold_start
    cv_df = pd.DataFrame(fold_rows)

    print(f"\n{'='*70}")
    print("K-FOLD CV AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"K={n_splits}, seed={random_state}, DE: maxiter={maxiter}, popsize={popsize}")
    print(f"Baseline RMSE (defaults, all cases): {baseline_rmse:.4f}")
    print(f"Total run time: {total_time:.0f}s ({total_time/60:.1f} min, {total_time/3600:.1f} h)")
    print("")

    header = (f"{'Fold':>4}  {'Cal RMSE':>9}  {'Val RMSE':>9}  "
              f"{'Cal R²':>7}  {'Val R²':>7}  "
              f"{'Val Bias':>9}  {'C2':>6}  {'C3':>5}")
    print(header)
    print("-" * len(header))
    for _, row in cv_df.iterrows():
        print(f"  {int(row['fold']):2d}  "
              f"  {row['cal_rmse']:7.4f}  "
              f"  {row['val_rmse']:7.4f}  "
              f"  {row['cal_r2']:6.4f}  "
              f"  {row['val_r2']:6.4f}  "
              f"  {row['val_bias']:+8.4f}  "
              f"{'PASS' if row['c2_passed'] else 'FAIL':>6}  "
              f"{'PASS' if row['c3_passed'] else 'FAIL':>5}")
    print("-" * len(header))

    def _stat(col):
        return f"{cv_df[col].mean():.4f} ± {cv_df[col].std():.4f}"

    print(f"  MEAN  {_stat('cal_rmse'):>17}  {_stat('val_rmse'):>17}  "
          f"{_stat('cal_r2'):>15}  {_stat('val_r2'):>15}  "
          f"{cv_df['val_bias'].mean():+8.4f}")
    print("")
    print(f"C2 folds passed: {cv_df['c2_passed'].sum()}/{n_splits}  "
          f"(mean coverage = {cv_df['c2_coverage'].mean()*100:.1f}%)")
    print(f"C3 folds passed: {cv_df['c3_passed'].sum()}/{n_splits}")

    # ── Parameter table ───────────────────────────────────────────────────────
    param_df = pd.DataFrame(param_rows)
    if not param_df.empty:
        print(f"\n{'='*70}")
        print("PARAMETER SUMMARY ACROSS FOLDS")
        print(f"{'='*70}")
        pivot = param_df.pivot(index='parameter', columns='fold', values='optimized_value')
        pivot['mean'] = pivot.mean(axis=1)
        pivot['std'] = pivot.std(axis=1)
        pivot['default'] = param_df.groupby('parameter')['default_value'].first()
        with pd.option_context('display.float_format', '{:.4f}'.format,
                               'display.max_columns', 20):
            print(pivot.to_string())

    # ── Save outputs ──────────────────────────────────────────────────────────
    cv_df_save = cv_df.drop(columns=['train_cases', 'val_cases'], errors='ignore')
    cv_df_save.to_csv(output_dir / "kfold_summary.csv", index=False)
    param_df.to_csv(output_dir / "kfold_fold_params.csv", index=False)

    case_rows = []
    for _, row in cv_df.iterrows():
        fn = int(row['fold'])
        for c in row['train_cases'].split(','):
            case_rows.append({'fold': fn, 'case': int(c), 'split': 'train'})
        for c in row['val_cases'].split(','):
            case_rows.append({'fold': fn, 'case': int(c), 'split': 'val'})
    pd.DataFrame(case_rows).to_csv(output_dir / "kfold_case_splits.csv", index=False)

    # Text report
    sep = "=" * 70
    lines = [
        sep,
        f"K-FOLD CV REPORT  (K={n_splits}, seed={random_state})",
        sep,
        f"Baseline RMSE (defaults, all cases): {baseline_rmse:.4f}",
        f"Mean Cal RMSE:  {cv_df['cal_rmse'].mean():.4f} ± {cv_df['cal_rmse'].std():.4f}",
        f"Mean Val RMSE:  {cv_df['val_rmse'].mean():.4f} ± {cv_df['val_rmse'].std():.4f}",
        f"Mean Val R²:    {cv_df['val_r2'].mean():.4f} ± {cv_df['val_r2'].std():.4f}",
        f"Mean Val Bias:  {cv_df['val_bias'].mean():+.4f} ± {cv_df['val_bias'].std():.4f}",
        f"Mean Val MAE:   {cv_df['val_mae'].mean():.4f} ± {cv_df['val_mae'].std():.4f}",
        "",
        "Acceptance Criteria (per fold):",
        f"  C1  Mean bias ≤ PMU     : N/A (PMU not computable)",
        f"  C2  ≥90% PI coverage    : {cv_df['c2_passed'].sum()}/{n_splits} folds passed "
        f"(mean coverage={cv_df['c2_coverage'].mean()*100:.1f}%)",
        f"  C3  R² > 0              : {cv_df['c3_passed'].sum()}/{n_splits} folds passed",
        "",
        "Overall judgement:",
    ]
    c2_pass = cv_df['c2_passed'].sum() >= (n_splits * 0.6)
    c3_pass = cv_df['c3_passed'].sum() >= (n_splits * 0.6)
    for label, passed in [("C2", c2_pass), ("C3", c3_pass)]:
        lines.append(f"  {label}: {'PASS (majority of folds)' if passed else 'FAIL (majority of folds)'}")
    lines += ["", "Note: C2 PI = pred ± 1.645 × RMSE_cal (per fold). Assumes Gaussian residuals.", sep]
    report_text = "\n".join(lines)
    print("\n" + report_text)
    (output_dir / "kfold_report.txt").write_text(report_text)

    print(f"\nSaved: kfold_summary.csv, kfold_fold_params.csv, kfold_case_splits.csv, kfold_report.txt")
    print("K-fold CV complete!")

    return cv_df


# =============================================================================
# Standalone CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="K-Fold CV: Stratified Phase 2 Sequential Optimization")
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: outputs/)')
    ap.add_argument('--proc-subdir', default=None,
                    help='Subdirectory under inputs/processed/ (e.g. "no_outliers")')
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "outputs"

    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR, proc_subdir=args.proc_subdir)

    run_kfold(data, output_dir)
