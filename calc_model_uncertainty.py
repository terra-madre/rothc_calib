"""
calc_model_uncertainty.py
--------------------------
Computes two uncertainty metrics for the calibrated RothC model:

1. Model Error (ME)
       ME = SD of model residuals  (predicted − observed Δ SOC)

2. Model Uncertainty Deduction (MUD)
       MUD = (SD(predictions) / mean(predictions)) × t_{n−1, ci_pct}
   where t_{n−1, ci_pct} is the one-sided Student's t value at the ci_pct percentile
   with n−1 degrees of freedom.

Usage
-----
     mamba run -n terra-plus python -u git_code/calc_model_uncertainty.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── make sure git_code/ is on the path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "git_code"))

from optimization import precompute_data, apply_param_updates, PARAM_CONFIG

import calc_carbon_inputs as step2
import run_rothc_model as step5
import calc_soc_deltas as step6


# =============================================================================
# Core function
# =============================================================================

def calc_model_uncertainty(params: dict, data: dict = None,
                           case_subset: list = None, ci_pct: float = 0.90) -> dict:
    """Compute Model Error (ME) and Model Uncertainty Deduction (MUD).

    Parameters
    ----------
    params : dict
        Parameter name → value mapping.  Any parameter not included falls back
        to its default value from param_config.csv.
    data : dict, optional
        Precomputed data dict from precompute_data().  Loaded automatically if
        not provided.
    case_subset : list, optional
        List of case IDs to restrict the evaluation to.  None = all 70 cases.

    Returns
    -------
    dict with keys:
        n          – number of cases
        me         – model error: SD of (pred − obs) residuals  [t C ha⁻¹ yr⁻¹]
        mud        – model uncertainty deduction                 [dimensionless fraction]
        t_value    – Student's t value used in MUD
        df_t       – degrees of freedom for t (= n − 1)
        rmse       – RMSE (for reference)
        bias       – mean(pred − obs)
        sd_pred    – SD of predictions
        mean_pred  – mean of predictions
        cv_pred    – coefficient of variation of predictions (SD / mean)
        comparison – DataFrame with per-case obs, pred, residual
    """
    if data is None:
        data = precompute_data()

    # Fill any missing params with defaults
    full_params = {name: cfg['default'] for name, cfg in PARAM_CONFIG.items()}
    full_params.update(params)
 
    # Apply parameter updates to lookup tables
    updated = apply_param_updates(full_params, data)

    # Filter cases if requested
    cases_info      = data['cases_info_df']
    cases_treatments = data['cases_treatments_df']
    climate_df      = data['climate_df']
    initial_pools_df = data['initial_pools_df']
    plant_cover_df  = data['plant_cover_df']

    if case_subset is not None:
        cases_info       = cases_info[cases_info['case'].isin(case_subset)].copy()
        cases_treatments = cases_treatments[cases_treatments['case'].isin(case_subset)].copy()
        climate_df       = climate_df[climate_df['case'].isin(case_subset)].copy()
        initial_pools_df = initial_pools_df[initial_pools_df['case'].isin(case_subset)].copy()
        subcases         = cases_treatments['subcase'].unique()
        plant_cover_df   = plant_cover_df[plant_cover_df['subcase'].isin(subcases)].copy()

    # Step 2 — C inputs
    carbon_inputs_df = step2.calc_c_inputs(
        cases_treatments_df=cases_treatments,
        cases_info_df=cases_info,
        st_yields_all=data['st_yields_all'],
        ps_herbaceous=updated['ps_herbaceous'],
        ps_management=updated['ps_management'],
        ps_general=updated['ps_general'],
        ps_trees=updated['ps_trees'],
        ps_amendments=data['ps_amendments'],
        cc_yield_mod=full_params.get('cc_yield_mod', 1.0)
    )

    # Step 5 — RothC
    yearly_results, _ = step5.run_rothc(
        cases_treatments_df=cases_treatments,
        cases_info_df=cases_info,
        climate_df=climate_df,
        carbon_inputs_df=carbon_inputs_df,
        initial_pools_df=initial_pools_df,
        plant_cover_df=plant_cover_df,
        soil_depth_cm=data['soil_depth_cm'],
        plant_cover_modifier=full_params.get('plant_cover_modifier', 0.6),
        decomp_mod=full_params.get('decomp_mod', 1.0)
    )

    # Step 6 — Deltas
    deltas_df = step6.calc_deltas(yearly_results)

    comparison = deltas_df.merge(
        cases_info[['case', 'delta_soc_t_ha_y']], on='case'
    ).rename(columns={
        'delta_soc_t_ha_y': 'obs',
        'delta_treatment_control_per_year': 'pred'
    })
    comparison['residual'] = comparison['pred'] - comparison['obs']

    n = len(comparison)

    # ── 1. Model Error ────────────────────────────────────────────────────────
    me = float(np.std(comparison['residual'], ddof=1))

    # ── 2. Model Uncertainty Deduction ───────────────────────────────────────
    # One-sided t at ci_pct percentile, df = n − 1
    df_t    = n - 1
    t_value = float(stats.t.ppf(ci_pct, df=df_t))

    # sd_pred   = float(np.std(comparison['pred'], ddof=1))
    mean_pred = float(np.mean(comparison['pred']))
    cv_pred   = me / mean_pred if mean_pred != 0 else float('nan')
    mud       = cv_pred * t_value

    # Other reference metrics
    residuals = comparison['residual'].values
    rmse = float(np.sqrt(np.mean(residuals**2)))
    bias = float(np.mean(residuals))

    return {
        'n':         n,
        'me':        me,
        'mud':       mud,
        't_value':   t_value,
        'df_t':      df_t,
        'rmse':      rmse,
        'bias':      bias,
        # 'sd_pred':   sd_pred,
        'mean_pred': mean_pred,
        'cv_pred':   cv_pred,
        'comparison': comparison,
    }


# =============================================================================
# Helper — pretty print results
# =============================================================================

def _print_results(label: str, res: dict):
    print(f"\n{'='*60}")
    print(f"  {label}  (n = {res['n']} cases)")
    print(f"{'='*60}")
    print(f"  Model Error (ME)          : {res['me']:+.4f} t C ha⁻¹ yr⁻¹")
    print(f"  Model Uncertainty Deduct. : {res['mud']:+.4f}  (dimensionless)")
    print(f"    = CV({res['cv_pred']:.4f}) × t({res['t_value']:.4f}, df={res['df_t']})")
    print(f"  --- reference metrics ---")
    print(f"  RMSE                      : {res['rmse']:.4f} t C ha⁻¹ yr⁻¹")
    print(f"  Bias (mean pred − obs)    : {res['bias']:+.4f} t C ha⁻¹ yr⁻¹")
    print(f"  Mean prediction           : {res['mean_pred']:.4f} t C ha⁻¹ yr⁻¹")
    # print(f"  SD prediction             : {res['sd_pred']:.4f} t C ha⁻¹ yr⁻¹")


# =============================================================================
# Main — run for Phase 2 (all 70) and Cal-Val (train 47) parameter sets
# =============================================================================

if __name__ == "__main__":
    CHECKPOINT_DIR = ROOT / "outputs"

    phase2_path = CHECKPOINT_DIR / "phase2_sequential_checkpoints" / "all.json"
    calval_path = CHECKPOINT_DIR / "calval_checkpoints" / "all.json"

    print("Loading precomputed data …")
    data = precompute_data()

    # ── Phase 2 params (all 70 cases) ─────────────────────────────────────────
    with open(phase2_path) as f:
        phase2_params = json.load(f)['params']

    res_phase2 = calc_model_uncertainty(phase2_params, data=data)
    _print_results("Phase 2 — calibrated on all 70 cases", res_phase2)

    # ── Cal-Val params (trained on 47 cases); evaluate on all 70 ──────────────
    with open(calval_path) as f:
        calval_params = json.load(f)['params']

    res_calval_all = calc_model_uncertainty(calval_params, data=data)
    _print_results("Cal-Val params — evaluated on all 70 cases", res_calval_all)

    # ── Cal-Val params, training set only (47 cases) ──────────────────────────
    splits_df = pd.read_csv(CHECKPOINT_DIR / "calval_split.csv")
    train_cases = splits_df.loc[splits_df['split'] == 'train', 'case'].tolist()
    test_cases  = splits_df.loc[splits_df['split'] == 'test',  'case'].tolist()

    res_calval_train = calc_model_uncertainty(calval_params, data=data,
                                              case_subset=train_cases)
    _print_results("Cal-Val params — calibration set (n=47 train)", res_calval_train)

    res_calval_test = calc_model_uncertainty(calval_params, data=data,
                                             case_subset=test_cases)
    _print_results("Cal-Val params — validation set (n=23 test)", res_calval_test)
