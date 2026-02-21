"""
Phase 5: Outlier Removal + Re-optimization

Motivation:
  Phase 4 group-specific optimization improved per-group R² directionally
  but all annuals groups remain negative (annuals_resid R²=-106, caused by
  observed range of only 0.01-0.25 t C/ha/y vs model overestimates of ~0.35+).

Strategy:
  1. Compute per-group residuals using Phase 4 final params
  2. Flag cases where |residual| > 2 × group observed std (objective criterion)
  3. Report flagged cases for review - DO NOT silently drop
  4. Re-run group-specific optimization on cleaned dataset
  5. Compare per-group R² and RMSE before/after removal

Outlier criterion: |resid_p4| > 2 × std(observed_in_group)
  - Conservative: only removes extreme structural mismatches
  - Group-relative: respects that each group has a different scale

Run with:
  PYTHONPATH=git_code mamba run -n terra-plus python3 -u git_code/run_phase5_outlier_filter.py \
    2>&1 | tee outputs/phase5_outlier_filter.log
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, OPTIM_SETTINGS,
    precompute_data, objective, apply_param_updates,
)

BASE_DIR = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 final calibrated parameters (from outputs/phase4_final_params.csv)
# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 v2 final calibrated parameters (from outputs/phase4b_groupopt.log, 2026-02-19)
PHASE4_PARAMS = {
    'dr_ratio_annuals':       3.4995,
    'dr_ratio_treegrass':     0.3020,
    'dr_ratio_wood':          0.1212,
    'dr_ratio_fym':         1.9930,
    'plant_cover_modifier':   0.9961,
    'tree_fine_root_ratio':   0.0339,
    'grass_rs_ratio':         0.7146,
    'map_to_prod':            0.0040,
    'residue_frac_remaining': 0.1462,
    'cover_crop_rs_ratio':    0.2025,
    'tree_turnover_ag':       0.0143,
}

# ─────────────────────────────────────────────────────────────────────────────
# Same sub-run definitions as Phase 4
# ─────────────────────────────────────────────────────────────────────────────
SUBRUN_DEFS = [
    {
        'name':   'Annuals',
        'groups': ['annuals_covercrops', 'annuals_resid'],
        'params': ['dr_ratio_annuals', 'residue_frac_remaining', 'cover_crop_rs_ratio'],
        'maxiter': 100,
        'popsize': 15,
    },
    {
        'name':   'Trees',
        'groups': ['perennials_herb', 'perennials_herb+resid'],
        'params': ['dr_ratio_treegrass', 'dr_ratio_wood', 'tree_fine_root_ratio', 'tree_turnover_ag'],
        'maxiter': 100,
        'popsize': 15,
    },
    {
        'name':   'Pasture',
        'groups': ['annuals_to_pasture'],
        'params': ['grass_rs_ratio', 'map_to_prod'],
        'maxiter': 100,
        'popsize': 15,
    },
    {
        'name':   'Amendments',
        'groups': ['annuals_amend', 'perennials_amend'],
        'params': ['dr_ratio_fym'],
        'maxiter': 100,
        'popsize': 15,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def r2_score(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def get_predictions(param_dict, data, case_subset=None):
    """Run model with param_dict, return DataFrame with case/observed/predicted/residual."""
    names  = list(param_dict.keys())
    values = list(param_dict.values())
    _, details = objective(values, names, data, case_subset, return_details=True)
    df = details['comparison_df'].copy()
    df = df.rename(columns={'delta_treatment_control_per_year': 'predicted',
                             'delta_soc_t_ha_y': 'observed'})
    df['residual'] = df['observed'] - df['predicted']
    df['abs_error'] = df['residual'].abs()
    return df[['case', 'observed', 'predicted', 'residual', 'abs_error']]


def bake_background_params(data, fixed_params):
    """Apply fixed_params into data's base DataFrames."""
    updated = apply_param_updates(fixed_params, data)
    data_baked = dict(data)
    data_baked['ps_general']    = updated['ps_general']
    data_baked['ps_herbaceous'] = updated['ps_herbaceous']
    data_baked['ps_trees']      = updated['ps_trees']
    data_baked['ps_management'] = updated['ps_management']
    return data_baked


def group_metrics(param_dict, data, cases_info, label=""):
    """Print and return per-group RMSE/R²/Bias."""
    pred_df = get_predictions(param_dict, data)
    pred_df = pred_df.merge(cases_info[['case', 'group_calib']], on='case')
    rows = []
    for grp, gdf in pred_df.groupby('group_calib'):
        obs  = gdf['observed'].values
        pred = gdf['predicted'].values
        rmse = np.sqrt(np.mean((obs - pred)**2))
        r2   = r2_score(obs, pred)
        bias = np.mean(obs - pred)
        rows.append({'group': grp, 'n': len(gdf), 'rmse': rmse, 'r2': r2, 'bias': bias})
    df = pd.DataFrame(rows)
    if label:
        print(f"\n  {'Group':<30} {'n':>4} {'RMSE':>7} {'R²':>8} {'Bias':>8}  [{label}]")
        print("  " + "-" * 65)
        for _, r in df.iterrows():
            flag = " ✓" if r['r2'] > 0 else ""
            print(f"  {r['group']:<30} {r['n']:>4} {r['rmse']:>7.4f} {r['r2']:>8.3f} {r['bias']:>+8.4f}{flag}")
    return df


def run_de_subrun(name, param_names, case_subset, data_baked, maxiter, popsize,
                  warmstart_dict, seed=42):
    """DE optimization for one sub-group."""
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    x0 = np.array([warmstart_dict.get(p, PARAM_CONFIG[p]['default']) for p in param_names])
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    n_params = len(param_names)

    print(f"\n{'='*70}")
    print(f"SUB-RUN: {name}  ({len(case_subset)} cases, {n_params} params)")
    print(f"  Params: {param_names}")
    print(f"  Pop={popsize*n_params}, MaxIter={maxiter}")
    print(f"  Warm-start: {dict(zip(param_names, np.round(x0, 4)))}")
    print(f"{'='*70}")

    start = time.time()

    def callback(xk, convergence):
        gen = callback.count + 1
        callback.count = gen
        rmse = objective(xk, param_names, data_baked, case_subset)
        print(f"  step {gen:3d}: f(x)={rmse:.6f}  conv={convergence:.4f}  "
              f"{time.time()-start:.0f}s")
        return False
    callback.count = 0

    result = differential_evolution(
        objective,
        bounds,
        args=(param_names, data_baked, case_subset),
        x0=x0,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.0005,
        atol=0.0001,
        disp=False,
        workers=1,
        polish=True,
        callback=callback,
    )

    elapsed = time.time() - start
    opt_params = dict(zip(param_names, result.x))

    print(f"\n  RMSE={result.fun:.4f}  time={elapsed:.0f}s ({elapsed/60:.1f}min)  "
          f"calls={result.nfev}  converged={result.success}")
    for p, v in opt_params.items():
        bnd = PARAM_CONFIG[p]['bounds']
        at_bnd = " ← LOWER" if abs(v - bnd[0]) < 1e-4 else \
                 " ← UPPER" if abs(v - bnd[1]) < 1e-4 else ""
        print(f"  {p:30s}: {v:.4f}  (P4={warmstart_dict.get(p, PARAM_CONFIG[p]['default']):.4f}){at_bnd}")

    return opt_params, result.fun, elapsed, result.nfev


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 5: OUTLIER REMOVAL + RE-OPTIMIZATION")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    cases_info = data['cases_info_df'][['case', 'group_calib', 'duration_years']].copy()
    n_total = len(cases_info)
    print(f"Loaded {n_total} cases")

    # ── Step 1: Compute Phase 4 residuals ────────────────────────────────────
    print("\n--- Step 1: Phase 4 residuals per case ---")
    pred_p4 = get_predictions(PHASE4_PARAMS, data)
    pred_p4 = pred_p4.merge(cases_info, on='case')

    # ── Step 2: Flag outliers (criterion: |resid| > 2 × group obs std) ────────
    print("\n--- Step 2: Outlier detection (|resid_p4| > 2 × group_obs_std) ---")

    group_stats = pred_p4.groupby('group_calib')['observed'].agg(['mean', 'std']).reset_index()
    group_stats.columns = ['group_calib', 'obs_mean', 'obs_std']
    pred_p4 = pred_p4.merge(group_stats, on='group_calib')
    pred_p4['threshold'] = 2.0 * pred_p4['obs_std']
    pred_p4['is_outlier'] = pred_p4['abs_error'] > pred_p4['threshold']

    outliers = pred_p4[pred_p4['is_outlier']].sort_values('abs_error', ascending=False)
    clean_cases = pred_p4[~pred_p4['is_outlier']]['case'].tolist()

    print(f"\nOutlier threshold = 2 × group observed std:")
    print(f"\n  {'Group':<30} {'obs_std':>8} {'threshold':>10}")
    print("  " + "-" * 52)
    for _, row in group_stats.iterrows():
        print(f"  {row['group_calib']:<30} {row['obs_std']:>8.4f} {2*row['obs_std']:>10.4f}")

    print(f"\n{'='*70}")
    print(f"FLAGGED OUTLIERS ({len(outliers)} cases removed, {len(clean_cases)} retained):")
    print(f"{'='*70}")
    print(f"\n  {'Case':>5} {'Group':<25} {'Dur':>4} {'Obs':>7} {'Pred':>7} "
          f"{'Resid':>7} {'AE':>6} {'Thresh':>7}")
    print("  " + "-" * 72)
    for _, row in outliers.iterrows():
        print(f"  {int(row['case']):>5} {row['group_calib']:<25} {int(row['duration_years']):>4} "
              f"{row['observed']:>7.3f} {row['predicted']:>7.3f} "
              f"{row['residual']:>+7.3f} {row['abs_error']:>6.3f} {row['threshold']:>7.3f}")

    print(f"\nRemoved {len(outliers)} / {n_total} cases ({len(outliers)/n_total*100:.1f}%)")
    print("\nOutliers by group:")
    print(outliers['group_calib'].value_counts().to_string())

    # ── Step 3: Baselines on clean set ───────────────────────────────────────
    print(f"\n--- Step 3: Baselines on clean set (n={len(clean_cases)}) ---")

    default_params = {p: PARAM_CONFIG[p]['default'] for p in PHASE4_PARAMS}
    print("\nDefault params:")
    group_metrics(default_params, data, cases_info[cases_info['case'].isin(clean_cases)],
                  label="default, clean set")

    print("\nPhase 4 params (on clean set):")
    gm_p4_clean = group_metrics(PHASE4_PARAMS, data,
                                cases_info[cases_info['case'].isin(clean_cases)],
                                label="P4, clean set")

    # ── Step 4: Group-specific optimization on clean set ─────────────────────
    print(f"\n--- Step 4: Group-specific optimization on clean set ---")

    all_opt_params = dict(PHASE4_PARAMS)   # warm-start from Phase 4

    for srd in SUBRUN_DEFS:
        name        = srd['name']
        param_names = srd['params']

        # Sub-group cases, intersection with clean set
        grp_cases_all   = cases_info[cases_info['group_calib'].isin(srd['groups'])]['case'].tolist()
        grp_cases_clean = [c for c in grp_cases_all if c in clean_cases]
        removed_n       = len(grp_cases_all) - len(grp_cases_clean)

        print(f"\n  [{name}] {len(grp_cases_clean)} clean cases "
              f"(removed {removed_n} outliers from {len(grp_cases_all)})")

        if len(grp_cases_clean) < 3:
            print(f"  SKIP: too few cases ({len(grp_cases_clean)}) after outlier removal")
            continue

        # Bake background params
        background  = {k: v for k, v in all_opt_params.items() if k not in param_names}
        data_baked  = bake_background_params(data, background)

        opt_params, rmse, elapsed, nfev = run_de_subrun(
            name, param_names, grp_cases_clean, data_baked,
            srd['maxiter'], srd['popsize'],
            warmstart_dict=all_opt_params,
            seed=int(OPTIM_SETTINGS.get('seed', 42))
        )

        # Sub-group eval
        combined = dict(all_opt_params)
        combined.update(opt_params)
        print(f"\n  Sub-group eval (combined params, clean cases):")
        for grp in srd['groups']:
            grp_c = [c for c in cases_info[cases_info['group_calib'] == grp]['case'].tolist()
                     if c in clean_cases]
            if not grp_c:
                print(f"    {grp}: no clean cases")
                continue
            _, d = objective(list(combined.values()), list(combined.keys()),
                             data, grp_c, return_details=True)
            print(f"    {grp}: RMSE={d['rmse']:.4f}  R²={d['r2']:.4f}  "
                  f"Bias={d['bias']:+.4f}  n={d['n_cases']}")

        all_opt_params.update(opt_params)

    # ── Step 5: Global evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: FINAL GLOBAL EVALUATION")
    print("=" * 70)

    param_names_all  = list(all_opt_params.keys())
    param_values_all = list(all_opt_params.values())

    # On full dataset
    _, d_full = objective(param_values_all, param_names_all, data, return_details=True)
    print(f"\nPhase 4 full set: RMSE=1.4740  R²=0.2313  (reference)")
    print(f"Phase 5 full set: RMSE={d_full['rmse']:.4f}  R²={d_full['r2']:.4f}  "
          f"MAE={d_full['mae']:.4f}  Bias={d_full['bias']:+.4f}  n={d_full['n_cases']}")

    # Phase 4 per-group references
    P4_GROUP = {
        'annuals_amend':         (-0.570,  0.5234),
        'annuals_covercrops':    (-3.265,  0.7319),
        'annuals_resid':         (-106.166, 0.6873),
        'annuals_to_pasture':    (-0.589,  0.5936),
        'perennials_amend':      (-0.107,  2.5594),
        'perennials_herb':       (-0.452,  2.2721),
        'perennials_herb+resid': (0.794,   1.2983),
    }

    print(f"\nPer-group performance (Phase 5 params, FULL dataset including outliers):")
    print(f"\n  {'Group':<30} {'n':>4} {'R²_P4':>7} {'R²_P5':>7} {'RMSE_P4':>8} "
          f"{'RMSE_P5':>8} {'Bias_P5':>9}")
    print("  " + "-" * 80)
    group_rows = []
    for grp in sorted(cases_info['group_calib'].unique()):
        grp_cases = cases_info[cases_info['group_calib'] == grp]['case'].tolist()
        _, d = objective(param_values_all, param_names_all, data, grp_cases, return_details=True)
        p4_r2, p4_rmse = P4_GROUP.get(grp, (np.nan, np.nan))
        flag = " ✓" if d['r2'] > 0 else ""
        print(f"  {grp:<30} {len(grp_cases):>4} {p4_r2:>7.3f} {d['r2']:>7.3f} "
              f"{p4_rmse:>8.4f} {d['rmse']:>8.4f} {d['bias']:>+9.4f}{flag}")
        group_rows.append({'group': grp, 'n': len(grp_cases),
                           'r2_p4': p4_r2, 'r2_p5': d['r2'],
                           'rmse_p4': p4_rmse, 'rmse_p5': d['rmse'],
                           'bias_p5': d['bias']})

    print(f"\nPer-group performance (Phase 5 params, CLEAN dataset only):")
    print(f"\n  {'Group':<30} {'n_clean':>7} {'R²_P5_clean':>12} {'RMSE_clean':>11} {'Bias':>8}")
    print("  " + "-" * 65)
    for grp in sorted(cases_info['group_calib'].unique()):
        grp_c = [c for c in cases_info[cases_info['group_calib'] == grp]['case'].tolist()
                 if c in clean_cases]
        if not grp_c:
            print(f"  {grp:<30} {0:>7}  (all removed)")
            continue
        _, d = objective(param_values_all, param_names_all, data, grp_c, return_details=True)
        flag = " ✓" if d['r2'] > 0 else ""
        print(f"  {grp:<30} {len(grp_c):>7} {d['r2']:>12.4f} {d['rmse']:>11.4f} "
              f"{d['bias']:>+8.4f}{flag}")

    # ── Final parameter table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL CALIBRATED PARAMETERS (Phase 5)")
    print(f"{'='*70}")
    print(f"\n  {'Parameter':<30} {'Default':>9} {'Phase4':>9} {'Phase5':>9} {'Δ_dflt':>8}  bounds")
    print("  " + "-" * 82)
    for p in param_names_all:
        cfg     = PARAM_CONFIG.get(p, {})
        default = cfg.get('default', np.nan)
        p4      = PHASE4_PARAMS.get(p, np.nan)
        p5      = all_opt_params[p]
        bnd     = cfg.get('bounds', ('?', '?'))
        pct     = (p5 - default) / default * 100 if default else np.nan
        at_bnd  = " ← LOWER" if abs(p5 - bnd[0]) < 1e-4 else \
                  " ← UPPER" if abs(p5 - bnd[1]) < 1e-4 else ""
        print(f"  {p:<30} {default:>9.4f} {p4:>9.4f} {p5:>9.4f} {pct:>+7.1f}%  {bnd}{at_bnd}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = BASE_DIR / "outputs"

    # Outlier list
    out_cols = ['case', 'group_calib', 'duration_years', 'observed', 'predicted',
                'residual', 'abs_error', 'threshold', 'obs_std']
    outliers[out_cols].to_csv(out_dir / "phase5_outliers.csv", index=False, float_format='%.4f')
    print(f"\nOutlier list saved to: outputs/phase5_outliers.csv")

    # Clean case list
    pd.Series(clean_cases, name='case').to_csv(
        out_dir / "phase5_clean_cases.csv", index=False)
    print(f"Clean case list saved to: outputs/phase5_clean_cases.csv")

    # Per-group metrics
    pd.DataFrame(group_rows).to_csv(
        out_dir / "phase5_group_metrics.csv", index=False, float_format='%.4f')
    print(f"Group metrics saved to: outputs/phase5_group_metrics.csv")

    # Final params
    fp_rows = [{'param': p, 'value': v,
                'default': PARAM_CONFIG.get(p, {}).get('default', np.nan),
                'phase4_value': PHASE4_PARAMS.get(p, np.nan)}
               for p, v in all_opt_params.items()]
    pd.DataFrame(fp_rows).to_csv(
        out_dir / "phase5_final_params.csv", index=False, float_format='%.6f')
    print(f"Final params saved to: outputs/phase5_final_params.csv")

    print(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
