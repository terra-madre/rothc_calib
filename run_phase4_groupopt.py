"""
Phase 4: Group-Specific Parameter Optimization

Motivation (from Phase 3 diagnostic):
  - Global optimization (single param set, 98 cases) achieves R²=0.25 overall
    but individual groups have R² ranging from -158 to +0.92.
  - The model cannot simultaneously satisfy all groups with one parameter set
    because each group has mechanistically different dominant processes.

Strategy:
  Run 4 independent sub-optimizations, each restricted to its relevant cases
  and parameters. Background parameters are fixed at Phase 3 mean values.

  Sub-run  Cases                              Params (n)         n_cases
  -------  ---------------------------------  -----------------  -------
  Annuals  annuals_covercrops + annuals_resid dr_ratio_annuals,      36
                                              residue_frac_remaining
  Trees    perennials_herb +                  dr_ratio_treegrass,    34
           perennials_herb+resid              dr_ratio_wood,
                                              tree_fine_root_ratio
  Pasture  annuals_to_pasture                 grass_rs_ratio,         8
                                              map_to_prod
  Amend    annuals_amend + perennials_amend   dr_ratio_amend         20

  After all 4 sub-runs, the optimized params are combined and evaluated
  globally (per-group metrics) to confirm improvement over Phase 3.

Fixed context params (Phase 3 mean, not re-optimized here):
  plant_cover_modifier = 0.9961

Run with:
  PYTHONPATH=git_code conda run -n terra-plus python3 git_code/run_phase4_groupopt.py \
    2>&1 | tee outputs/phase4_groupopt.log
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).parent))

from optimization import (
    PARAM_CONFIG, PARAM_SETS, OPTIM_SETTINGS,
    precompute_data, objective, apply_param_updates,
    get_baseline_rmse,
)

BASE_DIR = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 mean params – used as background for each sub-run
# ─────────────────────────────────────────────────────────────────────────────
PHASE3_MEAN = {
    'dr_ratio_annuals':       3.4715,
    'dr_ratio_treegrass':     0.3279,
    'dr_ratio_wood':          0.2094,
    'dr_ratio_amend':         0.7385,
    'plant_cover_modifier':   0.9961,
    'tree_fine_root_ratio':   0.0584,
    'grass_rs_ratio':         1.9730,
    'map_to_prod':            0.0040,
    'residue_frac_remaining': 0.15,   # new target: mostly removed (not grazed)
    'cover_crop_rs_ratio':    0.50,   # new parameter
    'tree_turnover_ag':       0.02,   # new parameter (already in config)
}

# ─────────────────────────────────────────────────────────────────────────────
# Sub-run definitions
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
        'params': ['dr_ratio_amend'],
        'maxiter': 100,
        'popsize': 15,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bake_background_params(data, fixed_params):
    """Return a copy of data with fixed_params already applied to the base DFs.

    This ensures that when the objective function only optimizes a subset of
    params, the remaining parameters use the Phase 3 mean values rather than
    the CSV defaults.
    """
    updated = apply_param_updates(fixed_params, data)
    data_baked = dict(data)           # shallow copy of outer dict
    data_baked['ps_general']    = updated['ps_general']
    data_baked['ps_herbaceous'] = updated['ps_herbaceous']
    data_baked['ps_trees']      = updated['ps_trees']
    data_baked['ps_management'] = updated['ps_management']
    return data_baked


def r2_score(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def full_metrics(param_dict, data, case_subset=None, label=""):
    names  = list(param_dict.keys())
    values = list(param_dict.values())
    rmse, details = objective(values, names, data, case_subset, return_details=True)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}RMSE={rmse:.4f}  MAE={details['mae']:.4f}  "
          f"Bias={details['bias']:+.4f}  R²={details['r2']:.4f}  n={details['n_cases']}")
    return rmse, details


def run_de_subrun(name, param_names, case_subset, data_baked, maxiter, popsize, seed=42):
    """Run differential_evolution for one sub-group."""
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    n_params = len(param_names)
    pop_total = popsize * n_params
    n_cases = len(case_subset)

    print(f"\n{'='*70}")
    print(f"SUB-RUN: {name}")
    print(f"  Cases: {n_cases}  |  Params ({n_params}): {param_names}")
    print(f"  Pop={pop_total}, MaxIter={maxiter}, Seed={seed}")

    # Warm-start from Phase 3 mean
    x0 = np.array([PHASE3_MEAN.get(p, PARAM_CONFIG[p]['default']) for p in param_names])
    print(f"  Warm-start: {dict(zip(param_names, np.round(x0, 4)))}")
    for i, p in enumerate(param_names):
        lo, hi = bounds[i]
        x0[i] = np.clip(x0[i], lo, hi)
    print(f"{'='*70}")

    start = time.time()
    gen_log = []

    def callback(xk, convergence):
        gen = len(gen_log) + 1
        rmse = objective(xk, param_names, data_baked, case_subset)
        elapsed = time.time() - start
        gen_log.append({'gen': gen, 'rmse': rmse, 'convergence': convergence, 'elapsed': elapsed})
        print(f"  step {gen:3d}: f(x)={rmse:.6f}  conv={convergence:.4f}  {elapsed:.0f}s")
        return False

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
        polish=True,        # L-BFGS-B polish at the end for free
        callback=callback,
    )

    elapsed = time.time() - start
    opt_params = dict(zip(param_names, result.x))

    print(f"\n--- {name} Results ---")
    print(f"  RMSE:       {result.fun:.4f}")
    print(f"  Time:       {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Func calls: {result.nfev}  |  Converged: {result.success} ({result.message})")
    for p, v in opt_params.items():
        default = PARAM_CONFIG[p]['default']
        phase3  = PHASE3_MEAN.get(p, default)
        bnd     = PARAM_CONFIG[p]['bounds']
        at_bound = "← AT LOWER BOUND" if abs(v - bnd[0]) < 1e-4 else \
                   "← AT UPPER BOUND" if abs(v - bnd[1]) < 1e-4 else ""
        print(f"  {p:30s}: {v:.4f}  (default={default:.4f}, P3_mean={phase3:.4f})  {at_bound}")

    return opt_params, result.fun, elapsed, result.nfev


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 4: GROUP-SPECIFIC OPTIMIZATION")
    ts_start = time.strftime("%Y-%m-%d %H:%M")
    print(f"Started: {ts_start}")
    print("=" * 70)

    # Load data once
    print("\nLoading data...")
    data = precompute_data(repo_root=BASE_DIR)
    cases_info = data['cases_info_df']
    n_cases = len(cases_info)
    print(f"Loaded {n_cases} cases")
    print("Groups:")
    print(cases_info['group_calib'].value_counts().to_string())

    # Baseline with Phase 3 mean params
    print("\n--- Baselines ---")
    print("Default params (all 98 cases):")
    full_metrics(
        {p: PARAM_CONFIG[p]['default'] for p in PHASE3_MEAN},
        data, label="Default"
    )
    print("Phase 3 mean params (all 98 cases):")
    full_metrics(PHASE3_MEAN, data, label="Phase3")

    # Benchmark: estimate time per call for each sub-run size
    print("\nBenchmarking call times...")
    for srd in SUBRUN_DEFS:
        grp_cases = cases_info[cases_info['group_calib'].isin(srd['groups'])]['case'].tolist()
        t0 = time.time()
        objective(
            [PHASE3_MEAN.get(p, PARAM_CONFIG[p]['default']) for p in srd['params']],
            srd['params'], data, grp_cases
        )
        dt = time.time() - t0
        n_params = len(srd['params'])
        pop = srd['popsize'] * n_params
        est_calls = pop * srd['maxiter']
        est_h = est_calls * dt / 3600
        print(f"  {srd['name']:12s}: {len(grp_cases):3d} cases, {dt:.2f}s/call → "
              f"est {est_calls} calls ≈ {est_h:.1f}h")

    # ── Run all sub-optimisations ─────────────────────────────────────────────
    all_opt_params = dict(PHASE3_MEAN)   # Start from Phase 3 mean; update per sub-run
    subrun_results = []
    total_start = time.time()

    for srd in SUBRUN_DEFS:
        name       = srd['name']
        param_names = srd['params']
        grp_cases  = cases_info[cases_info['group_calib'].isin(srd['groups'])]['case'].tolist()

        # Bake the current best params for ALL other parameters into the data
        background = {k: v for k, v in all_opt_params.items() if k not in param_names}
        data_baked = bake_background_params(data, background)

        opt_params, rmse, elapsed, nfev = run_de_subrun(
            name, param_names, grp_cases, data_baked,
            srd['maxiter'], srd['popsize'],
            seed=int(OPTIM_SETTINGS.get('seed', 42))
        )

        # Evaluate on sub-group cases with FULL combined params (all_opt_params + new)
        combined = dict(all_opt_params)
        combined.update(opt_params)
        print(f"\n  Sub-group eval (combined params, {name} cases):")
        for grp in srd['groups']:
            grp_c = cases_info[cases_info['group_calib'] == grp]['case'].tolist()
            _, d = objective(
                list(combined.values()), list(combined.keys()),
                data, grp_c, return_details=True
            )
            print(f"    {grp}: RMSE={d['rmse']:.4f}  R²={d['r2']:.4f}  "
                  f"Bias={d['bias']:+.4f}  n={d['n_cases']}")

        # Update running best params
        all_opt_params.update(opt_params)

        subrun_results.append({
            'subrun': name,
            'groups': ','.join(srd['groups']),
            'n_cases': len(grp_cases),
            'n_params': len(param_names),
            'params': ','.join(param_names),
            'rmse': rmse,
            'time_s': elapsed,
            'func_calls': nfev,
            **opt_params
        })

    total_elapsed = time.time() - total_start

    # ── Combined global evaluation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMBINED GLOBAL EVALUATION")
    print("=" * 70)

    param_names_all = list(all_opt_params.keys())
    param_values_all = list(all_opt_params.values())
    _, global_details = objective(param_values_all, param_names_all, data, return_details=True)

    print(f"\nPhase 3 mean  → RMSE=1.4585  R²=0.2474  (reference)")
    print(f"Phase 4 group → RMSE={global_details['rmse']:.4f}  "
          f"R²={global_details['r2']:.4f}  "
          f"MAE={global_details['mae']:.4f}  Bias={global_details['bias']:+.4f}")

    print("\nPer-group performance (Phase 4 combined params):")
    group_rows = []
    print(f"\n  {'Group':<30} {'n':>4} {'RMSE_P3':>8} {'RMSE_P4':>8} {'R²_P3':>7} {'R²_P4':>7} {'Bias_P4':>8}")
    print("  " + "-" * 70)

    # Phase 3 per-group cached from diagnostic
    P3_GROUP = {   # from phase3_diagnostic.log
        'annuals_amend':         (0.5606, -0.8017),
        'annuals_covercrops':    (0.9040, -5.5067),
        'annuals_resid':         (0.8374, -158.071),
        'annuals_to_pasture':    (0.6601, -0.9648),
        'perennials_amend':      (2.6806, -0.2146),
        'perennials_herb':       (2.1733, -0.3285),
        'perennials_herb+resid': (0.8112,  0.9197),
    }

    for grp in sorted(cases_info['group_calib'].unique()):
        grp_cases = cases_info[cases_info['group_calib'] == grp]['case'].tolist()
        _, d = objective(param_values_all, param_names_all, data, grp_cases, return_details=True)
        p3_rmse, p3_r2 = P3_GROUP.get(grp, (np.nan, np.nan))
        print(f"  {grp:<30} {len(grp_cases):>4} {p3_rmse:>8.4f} {d['rmse']:>8.4f} "
              f"{p3_r2:>7.3f} {d['r2']:>7.3f} {d['bias']:>+8.4f}")
        group_rows.append({
            'group': grp, 'n': len(grp_cases),
            'rmse_p3': p3_rmse, 'rmse_p4': d['rmse'],
            'r2_p3': p3_r2, 'r2_p4': d['r2'], 'bias_p4': d['bias'],
        })

    # ── Final parameter table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL CALIBRATED PARAMETERS (Phase 4)")
    print("=" * 70)
    print(f"\n  {'Parameter':<30} {'Default':>9} {'Phase3':>9} {'Phase4':>9} {'Δ_dflt':>8} {'bounds'}")
    print("  " + "-" * 80)
    for p in param_names_all:
        cfg = PARAM_CONFIG.get(p, {})
        default = cfg.get('default', np.nan)
        p3      = PHASE3_MEAN.get(p, np.nan)
        p4      = all_opt_params[p]
        bnd     = cfg.get('bounds', ('?', '?'))
        pct     = (p4 - default) / default * 100 if default else np.nan
        at_bnd  = "← LOWER" if abs(p4 - bnd[0]) < 1e-4 else \
                  "← UPPER" if abs(p4 - bnd[1]) < 1e-4 else ""
        print(f"  {p:<30} {default:>9.4f} {p3:>9.4f} {p4:>9.4f} {pct:>+7.1f}%  {bnd}  {at_bnd}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = BASE_DIR / "outputs"

    # Sub-run results
    sr_df = pd.DataFrame(subrun_results)
    sr_path = out_dir / "phase4_subrun_results.csv"
    sr_df.to_csv(sr_path, index=False, float_format='%.6f')
    print(f"\nSub-run results saved to: {sr_path}")

    # Per-group metrics
    grp_df = pd.DataFrame(group_rows)
    grp_path = out_dir / "phase4_group_metrics.csv"
    grp_df.to_csv(grp_path, index=False, float_format='%.4f')
    print(f"Group metrics saved to: {grp_path}")

    # Final params
    fp_rows = []
    for p, v in all_opt_params.items():
        fp_rows.append({'param': p, 'value': v,
                        'default': PARAM_CONFIG.get(p, {}).get('default', np.nan),
                        'phase3_mean': PHASE3_MEAN.get(p, np.nan)})
    fp_df = pd.DataFrame(fp_rows)
    fp_path = out_dir / "phase4_final_params.csv"
    fp_df.to_csv(fp_path, index=False, float_format='%.6f')
    print(f"Final params saved to: {fp_path}")

    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min, {total_elapsed/3600:.1f}h)")
    print(f"Completed: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
