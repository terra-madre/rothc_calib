"""
run_calval.py
-------------
Calibration-Validation run using a stratified 70/30 train/test split.

Strategy:
  1. Stratified 70/30 split by SUB_RUNS super-groups (amendment, cropresid,
     covercrop_all, grass_all, pruning) — ensures even small groups like
     pruning (n=5) get a 3/2 train/test split
  2. Run Phase 2 sequential optimization on training cases only,
     warm-starting from the Phase 2 'all' checkpoint
  3. Evaluate calibration (train) and validation (test) performance
  4. Report three acceptance criteria:
       C1  Mean bias ≤ PMU           (skipped — replicate data not available)
       C2  ≥90% of test obs within 90% PI  (PI = pred ± 1.645 × RMSE_cal)
       C3  R² > 0 on test set

Outputs:
  outputs/calval_checkpoints/        — per-sub-run JSONs (resume support)
  outputs/calval_summary.csv         — sub-run RMSE table
  outputs/calval_report.txt          — acceptance criteria report

Usage (from project root):
    nohup mamba run -n terra-plus python -u git_code/run_calval.py > outputs/calval.log 2>&1 &
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
    PARAM_CONFIG, PARAM_SETS, OPTIM_SETTINGS,
    precompute_data, objective, get_baseline_rmse,
)
from run_phase2_sequential import SUB_RUNS, run_de, save_checkpoint, load_checkpoint

# =============================================================================
# Configuration
# =============================================================================

TRAIN_SIZE   = 0.7     # 70% train, 30% test (~49 / ~21 cases)
RANDOM_STATE = 42

# Phase 2 'all' checkpoint to warm-start from
WARMSTART_CKPT = "phase2_sequential_checkpoints/all.json"

BASE_DIR = Path(__file__).parent.parent


# =============================================================================
# Helpers
# =============================================================================

def supergroup_stratified_split(cases_info, super_labels, test_frac, random_state):
    """
    Stratified train/test split per super-group.

    Uses round() instead of int() so that small groups (e.g. pruning n=5)
    get round(n * test_frac) test cases rather than floor(n * test_frac).
    This ensures pruning (n=5, 30%) → 2 test, 3 train instead of 1/4.

    Minimum 1 test case per group with n >= 2. Groups with n == 1 go to train.
    """
    rng = np.random.default_rng(random_state)
    train_cases, test_cases = [], []
    for sg in sorted(super_labels.unique()):
        sg_cases = cases_info.loc[(super_labels == sg).values, 'case'].values
        n        = len(sg_cases)
        n_test   = max(1, round(n * test_frac)) if n > 1 else 0
        n_test   = min(n_test, n - 1)           # keep at least 1 in train
        perm     = rng.permutation(n)
        test_cases .extend(sg_cases[perm[:n_test]])
        train_cases.extend(sg_cases[perm[n_test:]])
    return sorted(train_cases), sorted(test_cases)


def load_warmstart_params(base_dir):
    """Load param dict from the Phase 2 'all' checkpoint."""
    ckpt_path = base_dir / "outputs" / WARMSTART_CKPT
    if not ckpt_path.exists():
        print(f"  WARNING: warm-start checkpoint not found at {ckpt_path}.")
        print("  Starting from PARAM_CONFIG defaults.")
        return {}
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    params = ckpt["params"]
    print(f"  Warm-start loaded from: {ckpt_path.name}")
    return params


def get_train_case_subset(data, groups, train_cases_set):
    """Return cases belonging to groups AND present in the training set."""
    cases_info = data['cases_info_df']
    if groups is None:
        return sorted(train_cases_set)
    group_cases = cases_info[cases_info['group_calib'].isin(groups)]['case'].tolist()
    return [c for c in group_cases if c in train_cases_set]


def print_split_summary(cases_info, train_cases, test_cases):
    """Print group-level train/test breakdown."""
    print(f"\n{'Group':30s}  {'Total':>5}  {'Train':>5}  {'Test':>5}")
    print("-" * 55)
    train_set, test_set = set(train_cases), set(test_cases)
    for grp in sorted(cases_info['group_calib'].unique()):
        grp_cases = cases_info[cases_info['group_calib'] == grp]['case'].tolist()
        n_train = sum(c in train_set for c in grp_cases)
        n_test  = sum(c in test_set  for c in grp_cases)
        print(f"  {grp:28s}  {len(grp_cases):>5}  {n_train:>5}  {n_test:>5}")
    print("-" * 55)
    print(f"  {'TOTAL':28s}  {len(cases_info):>5}  {len(train_cases):>5}  {len(test_cases):>5}")


# =============================================================================
# Acceptance criteria evaluation
# =============================================================================

def compute_acceptance_criteria(cal_rmse, cal_details, val_details):
    """
    Evaluate the three cal-val acceptance criteria.

    C1  Mean bias ≤ PMU              — skipped (replicate data unavailable)
    C2  ≥90% of validation obs within PI_90 = pred ± 1.645 × RMSE_cal
    C3  R² > 0 on validation set

    bias = mean(pred - obs): positive = overprediction, negative = underprediction.

    Returns a list of result dicts.
    """
    results = []

    # ── C1: bias ≤ PMU ───────────────────────────────────────────────────────
    val_bias = val_details['bias']
    results.append({
        'criterion': 'C1',
        'description': 'Mean bias ≤ PMU',
        'value': f"val bias = {val_bias:+.4f} t C ha⁻¹ yr⁻¹",
        'passed': None,   # cannot evaluate
        'note': 'PMU not computable — per-observation replicate data (σ_j, n_j) not in dataset',
    })

    # ── C2: 90% PI coverage ──────────────────────────────────────────────────
    half_width = 1.645 * cal_rmse
    comp   = val_details['comparison_df'].copy()
    obs    = comp['delta_soc_t_ha_y'].values
    pred   = comp['delta_treatment_control_per_year'].values
    within = np.abs(obs - pred) <= half_width
    coverage = within.mean()
    passed_c2 = bool(coverage >= 0.90)
    results.append({
        'criterion': 'C2',
        'description': '≥90% of val obs within 90% PI',
        'value': f"coverage = {coverage*100:.1f}%  (PI half-width = ±{half_width:.3f})",
        'passed': passed_c2,
        'note': f"PI = pred ± 1.645 × RMSE_cal (RMSE_cal = {cal_rmse:.4f}); assumes Gaussian residuals",
    })

    # ── C3: R² > 0 ───────────────────────────────────────────────────────────
    val_r2 = val_details['r2']
    passed_c3 = bool(val_r2 > 0)
    results.append({
        'criterion': 'C3',
        'description': 'R² > 0 on validation set',
        'value': f"val R² = {val_r2:.4f}",
        'passed': passed_c3,
        'note': '',
    })

    return results


def print_acceptance_report(criteria_results, cal_details, val_details, train_cases, test_cases):
    """Print and return formatted report string."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("CAL-VAL ACCEPTANCE CRITERIA REPORT")
    lines.append(sep)
    lines.append(f"Split: {len(train_cases)} train / {len(test_cases)} test  (stratified by SUB_RUNS super-groups, seed=42)")
    lines.append("")
    lines.append(f"Calibration set (train):  RMSE={cal_details['rmse']:.4f}  "
                 f"MAE={cal_details['mae']:.4f}  "
                 f"Bias={cal_details['bias']:+.4f}  "
                 f"R²={cal_details['r2']:.4f}  "
                 f"n={cal_details['n_cases']}")
    lines.append(f"Validation set  (test) :  RMSE={val_details['rmse']:.4f}  "
                 f"MAE={val_details['mae']:.4f}  "
                 f"Bias={val_details['bias']:+.4f}  "
                 f"R²={val_details['r2']:.4f}  "
                 f"n={val_details['n_cases']}")
    lines.append("")
    lines.append(f"{'Criterion':<6}  {'Description':<35}  {'Result':<45}  {'Pass?':<6}")
    lines.append("-" * 100)
    for r in criteria_results:
        pass_str = "PASS" if r['passed'] is True else "FAIL" if r['passed'] is False else "N/A"
        lines.append(f"  {r['criterion']:<4}  {r['description']:<35}  {r['value']:<45}  {pass_str:<6}")
        if r['note']:
            lines.append(f"        Note: {r['note']}")
    lines.append(sep)
    report = "\n".join(lines)
    print(report)
    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    maxiter = int(OPTIM_SETTINGS.get('maxiter', 30))
    popsize = int(OPTIM_SETTINGS.get('popsize', 8))
    seed    = int(OPTIM_SETTINGS.get('seed', 42))

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "calval_checkpoints"

    print("Cal-Val: Stratified 70/30 Split + Sequential Group Optimization")
    print(f"Train size: {int(TRAIN_SIZE*100)}%  |  Random state: {RANDOM_STATE}")
    print(f"Settings: maxiter={maxiter}, popsize={popsize}, seed={seed}")
    print(f"Warm-start: {WARMSTART_CKPT}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    cases_info = data['cases_info_df']
    print(f"Loaded {len(cases_info)} cases")
    print(f"Groups: {sorted(cases_info['group_calib'].unique().tolist())}")

    baseline = get_baseline_rmse(data)
    baseline_rmse = baseline['rmse']
    print(f"\nBaseline RMSE (all cases, defaults): {baseline_rmse:.4f}")

    # ── Build super-group labels from SUB_RUNS ───────────────────────────────
    # Maps each case's group_calib to the SUB_RUNS super-group name so that
    # small groups (e.g. pruning n=5) are stratified together rather than
    # individually (which would leave them all-train with an 80/20 split).
    group_to_super = {}
    for set_name, _params, target_groups in SUB_RUNS:
        if target_groups is None:
            continue
        for g in target_groups:
            group_to_super[g] = set_name

    super_labels = cases_info['group_calib'].map(group_to_super).fillna(cases_info['group_calib'])

    print("\nSuper-group mapping (for stratification):")
    for sg in sorted(super_labels.unique()):
        members = cases_info[super_labels == sg]['group_calib'].unique().tolist()
        n = (super_labels == sg).sum()
        print(f"  {sg:20s} ({n:2d} cases): {', '.join(sorted(members))}")

    # ── Stratified 70/30 split ────────────────────────────────────────────────
    train_cases, test_cases = supergroup_stratified_split(
        cases_info, super_labels,
        test_frac    = 1.0 - TRAIN_SIZE,
        random_state = RANDOM_STATE,
    )
    train_set   = set(train_cases)

    print(f"\nDataset split: {len(train_cases)} train, {len(test_cases)} test")
    print_split_summary(cases_info, train_cases, test_cases)

    # ── Warm-start from Phase 2 'all' checkpoint ──────────────────────────────
    print("\nLoading warm-start parameters...")
    best_params = load_warmstart_params(BASE_DIR)
    if best_params:
        print("  Values:")
        for p, v in best_params.items():
            print(f"    {p:30s}: {v:.4f}")

    # ── Sequential sub-runs on training set ───────────────────────────────────
    all_results = []

    for set_name, param_names, target_groups in SUB_RUNS:
        if not param_names:
            print(f"\nWARNING: No params defined for set '{set_name}', skipping.")
            continue

        # Resume: use checkpoint if it exists
        existing = load_checkpoint(set_name, ckpt_dir)
        if existing is not None:
            print(f"\n[RESUME] {set_name}: checkpoint found — RMSE={existing['rmse']:.4f}, skipping.")
            for pname, pval in existing['params'].items():
                best_params[pname] = pval
            all_results.append(existing)
            continue

        # Intersect group cases with training set
        case_subset = get_train_case_subset(data, target_groups, train_set)

        if len(case_subset) == 0:
            print(f"\nWARNING: No training cases found for groups {target_groups}, skipping '{set_name}'.")
            continue
        if len(case_subset) < len(param_names):
            print(f"\nWARNING: '{set_name}' has only {len(case_subset)} training cases for "
                  f"{len(param_names)} params — optimization may be unreliable.")

        result = run_de(
            set_name, param_names, data, case_subset, best_params,
            maxiter=maxiter, popsize=popsize, seed=seed,
            polish=(set_name == 'all'),
        )

        for pname, pval in result['params'].items():
            best_params[pname] = pval

        save_checkpoint(result, baseline_rmse, ckpt_dir)
        all_results.append(result)

        print(f"\nCumulative best_params after '{set_name}':")
        for pname, pval in best_params.items():
            print(f"  {pname:30s}: {pval:.4f}")

    # ── Final evaluation: train and test ─────────────────────────────────────
    if not best_params:
        print("ERROR: no parameters calibrated — exiting.")
        sys.exit(1)

    final_param_names  = list(best_params.keys())
    final_param_values = [best_params[p] for p in final_param_names]

    print(f"\n{'='*70}")
    print("FINAL PARAMETER EVALUATION")
    print(f"{'='*70}")

    cal_rmse, cal_details = objective(
        final_param_values, final_param_names, data,
        case_subset=train_cases, return_details=True,
    )
    val_rmse, val_details = objective(
        final_param_values, final_param_names, data,
        case_subset=test_cases, return_details=True,
    )

    print(f"Calibration (train, n={cal_details['n_cases']}):")
    print(f"  RMSE={cal_rmse:.4f}  MAE={cal_details['mae']:.4f}  "
          f"Bias={cal_details['bias']:+.4f}  R²={cal_details['r2']:.4f}")
    print(f"Validation  (test,  n={val_details['n_cases']}):")
    print(f"  RMSE={val_rmse:.4f}  MAE={val_details['mae']:.4f}  "
          f"Bias={val_details['bias']:+.4f}  R²={val_details['r2']:.4f}")

    # ── Acceptance criteria ────────────────────────────────────────────────────
    print("")
    criteria = compute_acceptance_criteria(cal_rmse, cal_details, val_details)
    report   = print_acceptance_report(criteria, cal_details, val_details, train_cases, test_cases)

    # ── Sub-run summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUB-RUN SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sub-run':20s} | {'RMSE (train subset)':>20s} | {'Improve':>8s} | {'Time':>7s}")
    print("-" * 65)
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        t   = r['time_s']
        print(f"{r['set_name']:20s} | {r['rmse']:>20.4f} | {imp:+7.1f}% | {t:6.0f}s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Summary CSV
    summary_rows = []
    for r in all_results:
        imp = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        summary_rows.append({
            'set_name':       r['set_name'],
            'n_cases':        r.get('n_cases'),
            'n_params':       len(r['params']),
            'rmse':           r['rmse'],
            'baseline_rmse':  baseline_rmse,
            'improvement_pct': imp,
            'time_s':         r['time_s'],
            'func_calls':     r['func_calls'],
            'success':        r['success'],
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "calval_summary.csv", index=False)

    # Final params CSV
    param_rows = [{'parameter': p, 'value': v} for p, v in best_params.items()]
    pd.DataFrame(param_rows).to_csv(output_dir / "calval_params.csv", index=False)

    # Train/test case lists
    split_rows = (
        [{'case': c, 'split': 'train'} for c in train_cases] +
        [{'case': c, 'split': 'test'}  for c in test_cases]
    )
    pd.DataFrame(split_rows).to_csv(output_dir / "calval_split.csv", index=False)

    # Acceptance report text
    (output_dir / "calval_report.txt").write_text(report)

    print(f"\nSaved:")
    print(f"  outputs/calval_summary.csv")
    print(f"  outputs/calval_params.csv")
    print(f"  outputs/calval_split.csv")
    print(f"  outputs/calval_report.txt")
    print("Cal-Val complete!")
