"""
RothC Model Parameter Optimization Module

This module provides functions for calibrating RothC model parameters
using scipy optimization with cross-validation support.

Configuration is loaded from CSV files in inputs/optimization/:
  - param_config.csv: parameter definitions (name, default, bounds, source, tier)
  - param_sets.csv: named groups of parameters to optimize together
  - optim_settings.csv: algorithm settings (method, maxiter, popsize, etc.)

Usage:
    from optimization import precompute_data, objective, run_optimization, cross_validate
    
    data = precompute_data()
    result = run_optimization(['dr_ratio_annuals', 'map_to_prod'], data)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import train_test_split, StratifiedKFold

# Import model steps
import calc_carbon_inputs as step2
import calc_initial_pools as step3
import calc_plant_cover as step4
import run_rothc_model as step5
import calc_soc_deltas as step6


# =============================================================================
# Configuration Loading
# =============================================================================

def _get_config_dir():
    """Return path to inputs/optimization/ config directory."""
    return Path(__file__).resolve().parents[1] / "inputs" / "optimization"


def load_param_config(config_dir=None):
    """Load parameter configuration from param_config.csv.
    
    Returns:
        dict: {param_name: {default, bounds, source, tier, description}}
    """
    if config_dir is None:
        config_dir = _get_config_dir()
    df = pd.read_csv(Path(config_dir) / "param_config.csv")
    config = {}
    for _, row in df.iterrows():
        config[row['name']] = {
            'default': row['default'],
            'bounds': (row['bound_min'], row['bound_max']),
            'source': row['source'],
            'tier': int(row['tier']),
            'description': row['description'],
        }
    return config


def load_param_sets(config_dir=None):
    """Load named parameter sets from param_sets.csv.
    
    Returns:
        dict: {set_name: [param_name, ...]}
    """
    if config_dir is None:
        config_dir = _get_config_dir()
    df = pd.read_csv(Path(config_dir) / "param_sets.csv")
    sets = {}
    for set_name, group in df.groupby('set_name'):
        sets[set_name] = group['param_name'].tolist()
    return sets


def load_optim_settings(config_dir=None):
    """Load optimization settings from optim_settings.csv.
    
    Returns:
        dict: {setting_name: value} with automatic type casting
    """
    if config_dir is None:
        config_dir = _get_config_dir()
    df = pd.read_csv(Path(config_dir) / "optim_settings.csv")
    settings = {}
    for _, row in df.iterrows():
        val = row['value']
        # Auto-cast numeric values
        try:
            if '.' in str(val):
                val = float(val)
            else:
                val = int(val)
        except (ValueError, TypeError):
            pass
        settings[row['setting']] = val
    return settings


# Load configs at module level (used as defaults throughout)
PARAM_CONFIG = load_param_config()
PARAM_SETS = load_param_sets()
OPTIM_SETTINGS = load_optim_settings()

# Cover crop types in ps_herbaceous
COVER_CROP_TYPES = [
    'cover crop - grass-legume mixture',
    'cover crop - single or multiple grasses',
    'cover crop - single or multiple legumes'
]


# =============================================================================
# Parameter Update Helpers
# =============================================================================

def update_param_df(df, param_name, new_value, column='value', key_column='name'):
    """Update a parameter value in a DataFrame (for ps_general style).
    
    Args:
        df: DataFrame with parameters
        param_name: Name of parameter to update
        new_value: New value to set
        column: Column containing values (default: 'value')
        key_column: Column containing parameter names (default: 'name')
    
    Returns:
        DataFrame with updated value
    """
    df = df.copy()
    mask = df[key_column] == param_name
    if mask.any():
        df.loc[mask, column] = new_value
    return df


def update_herbaceous_param(df, group_cover, param_col, new_value):
    """Update a parameter for a specific plant type in ps_herbaceous.
    
    Args:
        df: ps_herbaceous DataFrame
        group_cover: Value in 'group_cover' column to match
        param_col: Column name to update
        new_value: New value to set
    
    Returns:
        DataFrame with updated value
    """
    df = df.copy()
    mask = df['group_cover'] == group_cover
    if mask.any():
        df[param_col] = df[param_col].astype(float)
        df.loc[mask, param_col] = new_value
    return df


def update_tree_param(df, param_col, new_value, species=None):
    """Update a parameter for trees in ps_trees.
    
    Args:
        df: ps_trees DataFrame
        param_col: Column name to update
        new_value: New value to set
        species: Optional species name to filter (None = all species)
    
    Returns:
        DataFrame with updated value
    """
    df = df.copy()
    df[param_col] = df[param_col].astype(float)
    if species is not None:
        mask = df['species'] == species
        df.loc[mask, param_col] = new_value
    else:
        df[param_col] = new_value
    return df


def update_management_param(df, management_type, param_col, new_value):
    """Update a parameter in ps_management.
    
    Args:
        df: ps_management DataFrame
        management_type: Value in 'management' column to match
        param_col: Column name to update ('frac_remaining' or 'prod_modifier')
        new_value: New value to set
    
    Returns:
        DataFrame with updated value
    """
    df = df.copy()
    mask = df['management'] == management_type
    if mask.any():
        df[param_col] = df[param_col].astype(float)
        df.loc[mask, param_col] = new_value
    return df


def apply_param_updates(params, data):
    """Apply parameter updates to all relevant DataFrames.
    
    Args:
        params: Dict of {param_name: value}
        data: Dict from precompute_data()
    
    Returns:
        Dict with updated parameter DataFrames
    """
    ps_general = data['ps_general'].copy()
    ps_herbaceous = data['ps_herbaceous'].copy()
    ps_trees = data['ps_trees'].copy()
    ps_management = data['ps_management'].copy()
    
    for name, value in params.items():
        config = PARAM_CONFIG.get(name)
        if config is None:
            continue
            
        source = config['source']
        
        if source == 'ps_general':
            ps_general = update_param_df(ps_general, name, value)
            
        elif source == 'ps_herbaceous':
            if name == 'cover_crop_rs_ratio':
                for cc_type in COVER_CROP_TYPES:
                    ps_herbaceous = update_herbaceous_param(
                        ps_herbaceous, cc_type, 'r_s_ratio (kg/kg)', value
                    )
            elif name == 'turnover_bg_grass':
                # Grassland belowground biomass turnover rate
                ps_herbaceous = update_herbaceous_param(
                    ps_herbaceous, 'grassland - permanent grasses or shrubs',
                    'turnover_bg (y-1)', value
                )
                
        elif source == 'ps_trees':
            if name == 'tree_fine_root_ratio':
                ps_trees = update_tree_param(ps_trees, 'fine_tot_r_ratio (kg/kg)', value)
            elif name == 'tree_turnover_ag':
                ps_trees = update_tree_param(ps_trees, 'turnover_ag (y-1)', value)
                
        elif source == 'ps_management':
            if name == 'residue_frac_remaining':
                # Targets "crop residues mostly removed (not grazed)" —
                # the typical control management in annuals_resid cases.
                # Range 0–0.15 (default 0.15 = current CSV value).
                ps_management = update_management_param(
                    ps_management, 'crop residues mostly removed (not grazed)',
                    'frac_remaining', value
                )
        
        # 'rothc' source params (e.g. plant_cover_modifier, decomp_mod) are handled
        # directly in the objective function when calling run_rothc
    
    return {
        'ps_general': ps_general,
        'ps_herbaceous': ps_herbaceous,
        'ps_trees': ps_trees,
        'ps_management': ps_management
    }


# =============================================================================
# Data Loading and Precomputation
# =============================================================================

def precompute_data(repo_root=None, soil_depth_cm=30):
    """Load and precompute all data that doesn't depend on calibrated parameters.
    
    Args:
        repo_root: Path to repository root (default: auto-detect)
        soil_depth_cm: Soil depth in cm (default: 30)
    
    Returns:
        Dict with all precomputed data needed for optimization
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    else:
        repo_root = Path(repo_root)
    
    input_dir = repo_root / "inputs"
    loc_data_dir = input_dir / "loc_data"
    proc_data_dir = input_dir / "processed"
    fixed_data_dir = input_dir / "fixed_values"
    
    # Load case data
    cases_info_df = pd.read_csv(proc_data_dir / "cases_info.csv")
    cases_treatments_df = pd.read_csv(proc_data_dir / "cases_treatments.csv")
    climate_df = pd.read_csv(loc_data_dir / "rothc_climate_avg.csv")
    st_yields_all = pd.read_csv(loc_data_dir / "st_yields_selected.csv")
    
    # Precompute initial pools (depends only on SOC and clay, not calibrated params)
    initial_pools_df = step3.get_rothc_pools(cases_info_df, type="transient")
    
    # Precompute plant cover schedule (depends only on treatment type)
    plant_cover_df = step4.plant_cover(cases_treatments_df)
    
    # Load base parameter files (will be modified during optimization)
    ps_herbaceous = pd.read_csv(fixed_data_dir / "ps_herbaceous.csv")
    ps_management = pd.read_csv(fixed_data_dir / "ps_management.csv")
    ps_general = pd.read_csv(fixed_data_dir / "ps_general.csv")
    ps_trees = pd.read_csv(fixed_data_dir / "ps_trees.csv")
    ps_amendments = pd.read_csv(fixed_data_dir / "ps_amendments.csv")
    
    return {
        'cases_info_df': cases_info_df,
        'cases_treatments_df': cases_treatments_df,
        'climate_df': climate_df,
        'st_yields_all': st_yields_all,
        'initial_pools_df': initial_pools_df,
        'plant_cover_df': plant_cover_df,
        'ps_herbaceous': ps_herbaceous,
        'ps_management': ps_management,
        'ps_general': ps_general,
        'ps_trees': ps_trees,
        'ps_amendments': ps_amendments,
        'soil_depth_cm': soil_depth_cm,
    }


# =============================================================================
# Objective Function
# =============================================================================

def objective(param_values, param_names, data, case_subset=None, return_details=False):
    """Compute RMSE between predicted and observed delta SOC.
    
    Args:
        param_values: Array/list of parameter values
        param_names: List of parameter names (same order as param_values)
        data: Dict from precompute_data()
        case_subset: Optional list of case IDs to include (for CV)
        return_details: If True, return (rmse, details_dict) instead of just rmse
    
    Returns:
        float: RMSE (to minimize), or tuple (rmse, details) if return_details=True
    """
    params = dict(zip(param_names, param_values))
    
    # Apply parameter updates to DataFrames
    updated = apply_param_updates(params, data)
    ps_general = updated['ps_general']
    ps_herbaceous = updated['ps_herbaceous']
    ps_trees = updated['ps_trees']
    ps_management = updated['ps_management']
    
    # Filter cases if subset specified
    cases_info = data['cases_info_df']
    cases_treatments = data['cases_treatments_df']
    climate_df = data['climate_df']
    initial_pools_df = data['initial_pools_df']
    plant_cover_df = data['plant_cover_df']
    
    if case_subset is not None:
        cases_info = cases_info[cases_info['case'].isin(case_subset)].copy()
        cases_treatments = cases_treatments[cases_treatments['case'].isin(case_subset)].copy()
        climate_df = climate_df[climate_df['case'].isin(case_subset)].copy()
        initial_pools_df = initial_pools_df[initial_pools_df['case'].isin(case_subset)].copy()
        subcases = cases_treatments['subcase'].unique()
        plant_cover_df = plant_cover_df[plant_cover_df['subcase'].isin(subcases)].copy()
    
    # Step 2: Calculate C inputs
    carbon_inputs_df = step2.calc_c_inputs(
        cases_treatments_df=cases_treatments,
        cases_info_df=cases_info,
        st_yields_all=data['st_yields_all'],
        ps_herbaceous=ps_herbaceous,
        ps_management=ps_management,
        ps_general=ps_general,
        ps_trees=ps_trees,
        ps_amendments=data['ps_amendments']
    )
    
    # Get rothc-level params if being optimized
    plant_cover_modifier = params.get('plant_cover_modifier', 0.6)
    decomp_mod = params.get('decomp_mod', 1.0)
    
    # Step 5: Run RothC
    yearly_results, _ = step5.run_rothc(
        cases_treatments_df=cases_treatments,
        cases_info_df=cases_info,
        climate_df=climate_df,
        carbon_inputs_df=carbon_inputs_df,
        initial_pools_df=initial_pools_df,
        plant_cover_df=plant_cover_df,
        soil_depth_cm=data['soil_depth_cm'],
        plant_cover_modifier=plant_cover_modifier,
        decomp_mod=decomp_mod
    )
    
    # Step 6: Calculate deltas
    deltas_df = step6.calc_deltas(yearly_results)
    
    # Calculate RMSE
    comparison = deltas_df.merge(
        cases_info[['case', 'delta_soc_t_ha_y']], 
        on='case'
    )
    
    residuals = comparison['delta_soc_t_ha_y'] - comparison['delta_treatment_control_per_year']
    rmse = np.sqrt(np.mean(residuals**2))
    
    if return_details:
        mae = np.mean(np.abs(residuals))
        bias = np.mean(residuals)
        r2 = 1 - (np.sum(residuals**2) / np.sum((comparison['delta_soc_t_ha_y'] - comparison['delta_soc_t_ha_y'].mean())**2))
        
        details = {
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'r2': r2,
            'n_cases': len(comparison),
            'comparison_df': comparison,
            'params': params
        }
        return rmse, details
    
    return rmse


# =============================================================================
# Optimization Functions
# =============================================================================

def run_optimization(param_names, data, case_subset=None, method=None, 
                     maxiter=None, popsize=None, x0_warmstart=None, verbose=True):
    """Run parameter optimization.
    
    Settings default to values from OPTIM_SETTINGS (inputs/optimization/optim_settings.csv).
    
    Args:
        param_names: List of parameter names to optimize
        data: Dict from precompute_data()
        case_subset: Optional list of case IDs to use
        method: Optimization method (default: from optim_settings)
        maxiter: Maximum iterations (default: from optim_settings)
        popsize: DE population size multiplier (default: from optim_settings)
        x0_warmstart: Optional dict {param_name: value} to override default initial values
        verbose: Print progress
    
    Returns:
        Dict with optimization results
    """
    if method is None:
        method = OPTIM_SETTINGS.get('method', 'differential_evolution')
    if maxiter is None:
        maxiter = OPTIM_SETTINGS.get('maxiter', 100)
    if popsize is None:
        popsize = OPTIM_SETTINGS.get('popsize', 15)
    # Get initial values and bounds
    x0 = [x0_warmstart.get(p, PARAM_CONFIG[p]['default']) if x0_warmstart else PARAM_CONFIG[p]['default']
          for p in param_names]
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    
    if verbose:
        print(f"Optimizing {len(param_names)} parameters: {param_names}")
        print(f"Initial values: {dict(zip(param_names, x0))}")
        print(f"Bounds: {dict(zip(param_names, bounds))}")
    
    # Track iterations
    iteration_count = [0]
    def callback(xk):
        iteration_count[0] += 1
        if verbose and iteration_count[0] % 10 == 0:
            current_rmse = objective(xk, param_names, data, case_subset)
            print(f"  Iteration {iteration_count[0]}: RMSE = {current_rmse:.4f}")
    
    if method == 'differential_evolution':
        result = differential_evolution(
            objective,
            bounds,
            args=(param_names, data, case_subset),
            maxiter=maxiter,
            popsize=popsize,
            seed=OPTIM_SETTINGS.get('seed', 42), # type: ignore
            disp=verbose,
            workers=1,
            callback=lambda xk, convergence: callback(xk)
        )
    else:
        eps = OPTIM_SETTINGS.get('eps', 1e-8)
        result = minimize(
            objective,
            x0,
            args=(param_names, data, case_subset),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': verbose, 'eps': eps},
            callback=callback
        )
    
    # Get final metrics
    final_rmse, details = objective(result.x, param_names, data, case_subset, return_details=True)
    
    return {
        'success': result.success,
        'params': dict(zip(param_names, result.x)),
        'rmse': final_rmse,
        'mae': details['mae'],
        'bias': details['bias'],
        'r2': details['r2'],
        'n_cases': details['n_cases'],
        'iterations': iteration_count[0],
        'scipy_result': result
    }


def cross_validate(param_names, data, n_splits=None, test_size=None, random_state=None,
                   method=None, maxiter=None, popsize=None, x0_warmstart=None,
                   verbose=True):
    """Stratified k-fold cross-validation with held-out test set.
    
    Settings default to values from OPTIM_SETTINGS (inputs/optimization/optim_settings.csv).
    
    Args:
        param_names: List of parameter names to optimize
        data: Dict from precompute_data()
        n_splits: Number of CV folds (default: from optim_settings)
        test_size: Fraction for held-out test set (default: from optim_settings)
        random_state: Random seed (default: from optim_settings)
        method: 'L-BFGS-B' or 'differential_evolution' (default: from optim_settings)
        maxiter: Maximum iterations per fold (default: from optim_settings)
        popsize: DE population size multiplier (default: from optim_settings)
        x0_warmstart: Optional array of param values to warm-start each fold
                      (for DE: seeded into population via x0; for local: initial guess)
        verbose: Print progress
    
    Returns:
        Tuple of (cv_results_df, test_rmse, test_cases)
    """
    import time as _time
    
    # Apply defaults from settings file
    if n_splits is None:
        n_splits = OPTIM_SETTINGS.get('n_splits', 5)
    if test_size is None:
        test_size = OPTIM_SETTINGS.get('test_size', 0.2)
    if random_state is None:
        random_state = OPTIM_SETTINGS.get('cv_random_state', 42)
    if method is None:
        method = OPTIM_SETTINGS.get('method', 'differential_evolution')
    if maxiter is None:
        maxiter = OPTIM_SETTINGS.get('maxiter', 100)
    if popsize is None:
        popsize = OPTIM_SETTINGS.get('popsize', 15)

    cases_df = data['cases_info_df']
    
    # 1. Reserve held-out test set
    train_val_cases, test_cases = train_test_split(
        cases_df['case'].values,
        test_size=test_size,
        stratify=cases_df['group_calib'].values,
        random_state=random_state
    )
    train_val_df = cases_df[cases_df['case'].isin(train_val_cases)]
    
    # Default initial values
    x0_default = np.array([PARAM_CONFIG[p]['default'] for p in param_names])
    x0 = np.array(x0_warmstart) if x0_warmstart is not None else x0_default
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    
    # DEBUG: Check bounds vs x0
    bounds_arr = np.array(bounds)
    print(f"\nDEBUG bounds check:")
    for i, pname in enumerate(param_names):
        print(f"  {pname}: x0={x0[i]:.6f}, bounds={bounds[i]}, OK={bounds[i][0] <= x0[i] <= bounds[i][1]}")
    
    if verbose:
        print(f"\nDataset split: {len(train_val_cases)} train/val, {len(test_cases)} test")
        print(f"Method: {method}, maxiter: {maxiter}")
        if method == 'differential_evolution':
            print(f"DE popsize multiplier: {popsize} (pop={popsize * len(param_names)})")
        if x0_warmstart is not None:
            print(f"Warm-start: {dict(zip(param_names, x0))}")
        print(f"Running {n_splits}-fold CV on train/val set\n")
    
    # 2. Stratified k-fold on train_val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strata = train_val_df['group_calib'].values
    cases = train_val_df['case'].values
    
    cv_results = []
    cv_start = _time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cases, strata)):
        train_cases = cases[train_idx].tolist()
        val_cases = cases[val_idx].tolist()
        fold_start = _time.time()
        
        if verbose:
            print(f"Fold {fold+1}/{n_splits}: {len(train_cases)} train, {len(val_cases)} val")
        
        # Optimize on training cases
        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds,
                args=(param_names, data, train_cases),
                x0=x0,  # Warm-start with best known solution
                maxiter=maxiter,
                popsize=popsize,
                seed=OPTIM_SETTINGS.get('seed', 42) + fold,  # Different seed per fold
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=0.001,
                atol=0.001,
                disp=verbose,
                workers=1,
                polish=False
            )
        else:
            eps = OPTIM_SETTINGS.get('eps', 1e-8)
            result = minimize(
                objective,
                x0,
                args=(param_names, data, train_cases),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': False, 'eps': eps}
            )
        
        fold_elapsed = _time.time() - fold_start
        
        # Evaluate on validation cases
        train_rmse, train_details = objective(result.x, param_names, data, train_cases, return_details=True)
        val_rmse, val_details = objective(result.x, param_names, data, val_cases, return_details=True)
        
        fold_result = {
            'fold': fold + 1,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_details['r2'],
            'val_r2': val_details['r2'],
            'train_mae': train_details['mae'],
            'val_mae': val_details['mae'],
            'train_bias': train_details['bias'],
            'val_bias': val_details['bias'],
            'n_train': len(train_cases),
            'n_val': len(val_cases),
            'time_s': fold_elapsed,
            'func_calls': result.nfev,
            **dict(zip(param_names, result.x))
        }
        cv_results.append(fold_result)
        
        total_elapsed = _time.time() - cv_start
        if verbose:
            print(f"  Train RMSE: {train_rmse:.4f} (R²={train_details['r2']:.4f}), "
                  f"Val RMSE: {val_rmse:.4f} (R²={val_details['r2']:.4f})")
            print(f"  Fold time: {fold_elapsed:.0f}s ({fold_elapsed/60:.1f}min), "
                  f"Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
            print(f"  Params: {dict(zip(param_names, np.round(result.x, 4)))}\n")
    
    # 3. Final evaluation on held-out test set (using mean params from CV)
    cv_df = pd.DataFrame(cv_results)
    mean_params = [cv_df[p].mean() for p in param_names]
    test_rmse, test_details = objective(mean_params, param_names, data, test_cases.tolist(), return_details=True)
    
    total_time = _time.time() - cv_start
    if verbose:
        print("=" * 50)
        print(f"CV Mean Train RMSE: {cv_df['train_rmse'].mean():.4f} ± {cv_df['train_rmse'].std():.4f}")
        print(f"CV Mean Val RMSE:   {cv_df['val_rmse'].mean():.4f} ± {cv_df['val_rmse'].std():.4f}")
        print(f"Held-out Test RMSE: {test_rmse:.4f} (R²={test_details['r2']:.4f})")
        print(f"Total CV time: {total_time:.0f}s ({total_time/60:.1f}min, {total_time/3600:.1f}h)")
        print(f"\nMean calibrated parameters:")
        for p in param_names:
            print(f"  {p}: {cv_df[p].mean():.4f} ± {cv_df[p].std():.4f}")
    
    return cv_df, test_rmse, test_cases.tolist()


# =============================================================================
# Utility Functions
# =============================================================================

def get_baseline_rmse(data, case_subset=None):
    """Calculate RMSE with default parameter values.
    
    Args:
        data: Dict from precompute_data()
        case_subset: Optional list of case IDs to include
    
    Returns:
        Dict with baseline metrics
    """
    # Use empty param list to get defaults
    param_names = list(PARAM_CONFIG.keys())
    default_values = [PARAM_CONFIG[p]['default'] for p in param_names]
    
    rmse, details = objective(default_values, param_names, data, case_subset, return_details=True)
    
    return {
        'rmse': rmse,
        'mae': details['mae'],
        'bias': details['bias'],
        'r2': details['r2'],
        'n_cases': details['n_cases']
    }


def print_param_config():
    """Print all available parameters and their configuration."""
    print("Available parameters for calibration:\n")
    for tier in sorted(set(cfg['tier'] for cfg in PARAM_CONFIG.values())):
        tier_label = {1: 'Tier 1 (High Priority)', 2: 'Tier 2 (Medium Priority)',
                      3: 'Tier 3 (Lower Priority)'}.get(tier, f'Tier {tier}')
        print(f"{tier_label}:")
        for p, cfg in PARAM_CONFIG.items():
            if cfg['tier'] == tier:
                print(f"  {p}:")
                print(f"    Default: {cfg['default']}, Bounds: {cfg['bounds']}")
                print(f"    Source: {cfg['source']}, {cfg['description']}")
        print()

    print("Available parameter sets:")
    for set_name, params in PARAM_SETS.items():
        print(f"  {set_name}: {params}")
    print()

    print("Optimization settings:")
    for k, v in OPTIM_SETTINGS.items():
        print(f"  {k}: {v}")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    data = precompute_data()
    
    print(f"\nLoaded {len(data['cases_info_df'])} cases")
    print(f"Groups: {data['cases_info_df']['group_calib'].unique().tolist()}")
    
    print("\nBaseline metrics (default parameters):")
    baseline = get_baseline_rmse(data)
    print(f"  RMSE: {baseline['rmse']:.4f}")
    print(f"  MAE: {baseline['mae']:.4f}")
    print(f"  Bias: {baseline['bias']:.4f}")
    print(f"  R²: {baseline['r2']:.4f}")
    
    print("\n" + "=" * 50)
    print_param_config()
