"""
Phase 3 Diagnostic: Per-Group and Per-Case Analysis

Uses Phase 3 mean calibrated parameters to identify:
1. Which groups drive negative R²
2. Bias and spread per group
3. Which individual cases are outliers
4. Case-level observed vs predicted scatter

Run with:
  PYTHONPATH=git_code conda run -n terra-plus python3 git_code/run_phase3_diagnostic.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimization import precompute_data, objective

BASE_DIR = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 mean calibrated parameters (from phase3_cv.log)
# ─────────────────────────────────────────────────────────────────────────────
PHASE3_PARAMS = {
    'dr_ratio_annuals':       3.4715,
    'dr_ratio_treegrass':     0.3279,
    'dr_ratio_wood':          0.2094,
    'dr_ratio_amend':         0.7385,
    'plant_cover_modifier':   0.9961,
    'tree_fine_root_ratio':   0.0584,
    'grass_rs_ratio':         1.9730,
    'map_to_prod':            0.0040,
}

# Also keep Phase 2 best (single full-set optimum) for comparison
PHASE2_PARAMS = {
    'dr_ratio_annuals':       3.489367,
    'dr_ratio_treegrass':     0.305041,
    'dr_ratio_wood':          0.27531,
    'dr_ratio_amend':         1.45227,
    'plant_cover_modifier':   0.999,
    'tree_fine_root_ratio':   0.05,
    'grass_rs_ratio':         1.99539,
    'map_to_prod':            0.00404,
}

PARAM_NAMES = list(PHASE3_PARAMS.keys())


def r2_score(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def metrics(obs, pred, label=""):
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae  = np.mean(np.abs(obs - pred))
    bias = np.mean(obs - pred)
    r2   = r2_score(obs, pred)
    if label:
        print(f"  {label}: RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:+.4f}  R²={r2:.4f}  n={len(obs)}")
    return dict(rmse=rmse, mae=mae, bias=bias, r2=r2, n=len(obs))


def get_predictions(param_dict, data):
    names  = list(param_dict.keys())
    values = list(param_dict.values())
    _, details = objective(values, names, data, return_details=True)
    df = details['comparison_df'].copy()
    df = df.rename(columns={'delta_treatment_control_per_year': 'predicted',
                             'delta_soc_t_ha_y': 'observed'})
    df['residual'] = df['observed'] - df['predicted']
    df['abs_error'] = df['residual'].abs()
    return df[['case', 'observed', 'predicted', 'residual', 'abs_error']]


def main():
    print("=" * 70)
    print("PHASE 3 DIAGNOSTIC")
    print("=" * 70)

    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    cases_meta = data['cases_info_df'][['case', 'group_calib', 'duration_years']].copy()

    # ── Get predictions ──────────────────────────────────────────────────────
    print("\nRunning model with Phase 3 mean parameters...")
    pred3 = get_predictions(PHASE3_PARAMS, data).rename(
        columns={'predicted': 'pred_p3', 'residual': 'resid_p3', 'abs_error': 'ae_p3'})

    print("Running model with Phase 2 best parameters...")
    pred2 = get_predictions(PHASE2_PARAMS, data).rename(
        columns={'predicted': 'pred_p2', 'residual': 'resid_p2', 'abs_error': 'ae_p2'})

    # Default parameters for baseline
    from optimization import PARAM_CONFIG
    default_values = [PARAM_CONFIG[p]['default'] for p in PARAM_NAMES]
    print("Running model with default parameters (baseline)...")
    pred_def = get_predictions(dict(zip(PARAM_NAMES, default_values)), data).rename(
        columns={'predicted': 'pred_def', 'residual': 'resid_def', 'abs_error': 'ae_def'})

    # Merge all
    df = pred3.merge(pred2[['case', 'pred_p2', 'resid_p2', 'ae_p2']], on='case')
    df = df.merge(pred_def[['case', 'pred_def', 'resid_def', 'ae_def']], on='case')
    df = df.merge(cases_meta, on='case')

    # ── Overall metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    metrics(df['observed'], df['pred_def'], "Default  ")
    metrics(df['observed'], df['pred_p2'],  "Phase 2  ")
    metrics(df['observed'], df['pred_p3'],  "Phase 3  ")

    # ── Per-group metrics ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-GROUP METRICS (Phase 3 mean params)")
    print("=" * 70)

    group_rows = []
    for grp, gdf in df.groupby('group_calib'):
        obs  = gdf['observed'].values
        p3   = gdf['pred_p3'].values
        p2   = gdf['pred_p2'].values
        pdef = gdf['pred_def'].values
        print(f"\n{grp}  (n={len(gdf)})")
        m_def = metrics(obs, pdef, "Default")
        m_p2  = metrics(obs, p2,   "Phase2 ")
        m_p3  = metrics(obs, p3,   "Phase3 ")
        group_rows.append({'group': grp, 'n': len(gdf),
                           'r2_default': m_def['r2'], 'r2_p2': m_p2['r2'], 'r2_p3': m_p3['r2'],
                           'rmse_default': m_def['rmse'], 'rmse_p2': m_p2['rmse'], 'rmse_p3': m_p3['rmse'],
                           'bias_p3': m_p3['bias']})

    # ── Variance of observed per group ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OBSERVED SOC CHANGE STATISTICS BY GROUP")
    print("=" * 70)
    print(f"\n{'Group':<30} {'n':>4} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}")
    print("-" * 60)
    for grp, gdf in df.groupby('group_calib'):
        obs = gdf['observed']
        print(f"{grp:<30} {len(gdf):>4} {obs.mean():>7.3f} {obs.std():>7.3f} "
              f"{obs.min():>7.3f} {obs.max():>7.3f}")

    # ── Worst-case outliers ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 20 LARGEST ABSOLUTE ERRORS (Phase 3 params)")
    print("=" * 70)
    worst = df.nlargest(20, 'ae_p3')[
        ['case', 'group_calib', 'duration_years', 'observed', 'pred_p3', 'resid_p3', 'ae_p3']
    ]
    print(worst.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # ── Per-fold held-out cases forensics ────────────────────────────────────
    # (Using Phase 3 log fold data from outputs/phase3_cv_folds.csv)
    folds_path = BASE_DIR / "outputs" / "phase3_cv_folds.csv"
    if folds_path.exists():
        folds_df = pd.read_csv(folds_path)
        print("\n" + "=" * 70)
        print("FOLD-LEVEL SUMMARY (from phase3_cv_folds.csv)")
        print("=" * 70)
        print(f"\n{'Fold':>4} {'Train R²':>9} {'Val R²':>8} {'Train RMSE':>11} {'Val RMSE':>10} "
              f"{'Train bias':>11} {'Val bias':>10}")
        print("-" * 70)
        for _, row in folds_df.iterrows():
            print(f"{int(row['fold']):>4} {row['train_r2']:>9.4f} {row['val_r2']:>8.4f} "
                  f"{row['train_rmse']:>11.4f} {row['val_rmse']:>10.4f} "
                  f"{row['train_bias']:>+11.4f} {row['val_bias']:>+10.4f}")
        # Flag folds with negative val R²
        neg_r2_folds = folds_df[folds_df['val_r2'] < 0]
        if not neg_r2_folds.empty:
            print(f"\n⚠  Folds with negative val R²: {neg_r2_folds['fold'].tolist()}")
            print("  These folds have low-variance val subsets where predictions spread > observed spread.")

    # ── Save full case-level results ──────────────────────────────────────────
    out_path = BASE_DIR / "outputs" / "phase3_diagnostic.csv"
    save_cols = ['case', 'group_calib', 'duration_years',
                 'observed', 'pred_def', 'pred_p2', 'pred_p3',
                 'resid_def', 'resid_p2', 'resid_p3',
                 'ae_def', 'ae_p2', 'ae_p3']
    df[save_cols].to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nCase-level diagnostics saved to: {out_path}")

    # ── Group metrics summary table ───────────────────────────────────────────
    gdf_out = BASE_DIR / "outputs" / "phase3_group_metrics.csv"
    pd.DataFrame(group_rows).to_csv(gdf_out, index=False, float_format='%.4f')
    print(f"Group metrics saved to: {gdf_out}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
