"""
RothC Model Parameter Optimization Module

This module provides functions for calibrating RothC model parameters
using scipy optimization with cross-validation support.

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
import step2_c_inputs as step2
import step3_c_initial as step3
import step4_plant_cover as step4
import step5_run_rothc as step5
import step6_calc_deltas as step6


# =============================================================================
# Parameter Configuration
# =============================================================================

PARAM_CONFIG = {
    # Tier 1: High Priority
    'dr_ratio_annuals': {
        'default': 1.44,
        'bounds': (0.5, 2.5),
        'source': 'ps_general',
        'description': 'DPM/RPM ratio for annual crops'
    },
    'dr_ratio_treegrass': {
        'default': 0.67,
        'bounds': (0.3, 1.5),
        'source': 'ps_general',
        'description': 'DPM/RPM ratio for trees/grass'
    },
    'map_to_prod': {
        'default': 0.006,
        'bounds': (0.004, 0.08),
        'source': 'ps_general',
        'description': 'Productivity from MAP (t/mm)'
    },
    'tree_agb_modifier': {
        'default': 1.0,
        'bounds': (0.8, 1.2),
        'source': 'custom',
        'description': 'Tree biomass uncertainty scaling (±20%)'
    },
    'covercrop_rs_ratio': {
        'default': 0.47,
        'bounds': (0.3, 1.0),
        'source': 'ps_herbaceous',
        'description': 'Root:shoot ratio for cover crops'
    },
    
    # Tier 2: Medium Priority
    'grass_rs_ratio': {
        'default': 0.8,
        'bounds': (0.5, 2.0),
        'source': 'ps_herbaceous',
        'description': 'Root:shoot ratio for permanent grasses'
    },
    'tree_fine_root_ratio': {
        'default': 0.3,
        'bounds': (0.1, 0.6),
        'source': 'ps_trees',
        'description': 'Fine/total root ratio for trees'
    },
    'tree_turnover_ag': {
        'default': 0.02,
        'bounds': (0.01, 0.05),
        'source': 'ps_trees',
        'description': 'Leaf/litter turnover (y⁻¹)'
    },
    'dr_ratio_amend': {
        'default': 1.0,
        'bounds': (0.5, 2.0),
        'source': 'ps_general',
        'description': 'DPM/RPM ratio for amendments'
    },
    'grass_prod_modifier': {
        'default': 1.0,
        'bounds': (0.8, 1.5),
        'source': 'ps_management',
        'description': 'Productivity modifier for grassland'
    },
    
    # Tier 3: Lower Priority
    'plant_cover_modifier': {
        'default': 0.6,
        'bounds': (0.4, 0.8),
        'source': 'rothc',
        'description': 'Decomposition rate with plant cover'
    },
    'residue_frac_remaining': {
        'default': 1.0,
        'bounds': (0.7, 1.0),
        'source': 'ps_management',
        'description': 'Residue fraction under conservation'
    },
}

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
            if name == 'covercrop_rs_ratio':
                for cc_type in COVER_CROP_TYPES:
                    ps_herbaceous = update_herbaceous_param(
                        ps_herbaceous, cc_type, 'r_s_ratio (kg/kg)', value
                    )
            elif name == 'grass_rs_ratio':
                ps_herbaceous = update_herbaceous_param(
                    ps_herbaceous, 'grassland - permanent grasses or shrubs', 
                    'r_s_ratio (kg/kg)', value
                )
                
        elif source == 'ps_trees':
            if name == 'tree_fine_root_ratio':
                ps_trees = update_tree_param(ps_trees, 'fine_tot_r_ratio (kg/kg)', value)
            elif name == 'tree_turnover_ag':
                ps_trees = update_tree_param(ps_trees, 'turnover_ag (y-1)', value)
                
        elif source == 'ps_management':
            if name == 'grass_prod_modifier':
                # Update productivity modifier for grassland management types
                for mgmt in ['natural grasses/shrubs with continuous grazing',
                            'natural grasses/shrubs with planned/rotational grazing']:
                    ps_management = update_management_param(
                        ps_management, mgmt, 'prod_modifier', value
                    )
            elif name == 'residue_frac_remaining':
                # Update for conservation residue management
                ps_management = update_management_param(
                    ps_management, 'crop residues not removed (not grazed)', 
                    'frac_remaining', value
                )
    
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
    cases_treatments_df = pd.read_csv(input_dir / "raw" / "cases_treatments.csv")
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
    
    # Get tree_agb_modifier (custom parameter not in CSVs)
    tree_agb_modifier = params.get('tree_agb_modifier', 1.0)
    
    # Step 2: Calculate C inputs
    carbon_inputs_df = step2.calc_c_inputs(
        cases_treatments_df=cases_treatments,
        cases_info_df=cases_info,
        st_yields_all=data['st_yields_all'],
        ps_herbaceous=ps_herbaceous,
        ps_management=ps_management,
        ps_general=ps_general,
        ps_trees=ps_trees,
        ps_amendments=data['ps_amendments'],
        tree_agb_modifier=tree_agb_modifier
    )
    
    # Step 5: Run RothC
    yearly_results, _ = step5.run_rothc(
        cases_treatments_df=cases_treatments,
        cases_info_df=cases_info,
        climate_df=climate_df,
        carbon_inputs_df=carbon_inputs_df,
        initial_pools_df=initial_pools_df,
        plant_cover_df=plant_cover_df,
        soil_depth_cm=data['soil_depth_cm']
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

def run_optimization(param_names, data, case_subset=None, method='L-BFGS-B', 
                     maxiter=100, verbose=True):
    """Run parameter optimization.
    
    Args:
        param_names: List of parameter names to optimize
        data: Dict from precompute_data()
        case_subset: Optional list of case IDs to use
        method: Optimization method ('L-BFGS-B' or 'differential_evolution')
        maxiter: Maximum iterations
        verbose: Print progress
    
    Returns:
        Dict with optimization results
    """
    # Get initial values and bounds
    x0 = [PARAM_CONFIG[p]['default'] for p in param_names]
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
            seed=42,
            disp=verbose,
            callback=lambda xk, convergence: callback(xk)
        )
    else:
        result = minimize(
            objective,
            x0,
            args=(param_names, data, case_subset),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': verbose},
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


def cross_validate(param_names, data, n_splits=5, test_size=0.2, random_state=42,
                   method='L-BFGS-B', maxiter=100, verbose=True):
    """Stratified k-fold cross-validation with held-out test set.
    
    Args:
        param_names: List of parameter names to optimize
        data: Dict from precompute_data()
        n_splits: Number of CV folds
        test_size: Fraction for held-out test set
        random_state: Random seed
        method: Optimization method
        maxiter: Maximum iterations per fold
        verbose: Print progress
    
    Returns:
        Tuple of (cv_results_df, test_rmse, test_cases)
    """
    cases_df = data['cases_info_df']
    
    # 1. Reserve held-out test set
    train_val_cases, test_cases = train_test_split(
        cases_df['case'].values,
        test_size=test_size,
        stratify=cases_df['group_calib'].values,
        random_state=random_state
    )
    train_val_df = cases_df[cases_df['case'].isin(train_val_cases)]
    
    if verbose:
        print(f"Dataset split: {len(train_val_cases)} train/val, {len(test_cases)} test")
        print(f"Running {n_splits}-fold CV on train/val set\n")
    
    # 2. Stratified k-fold on train_val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strata = train_val_df['group_calib'].values
    cases = train_val_df['case'].values
    
    cv_results = []
    x0 = [PARAM_CONFIG[p]['default'] for p in param_names]
    bounds = [PARAM_CONFIG[p]['bounds'] for p in param_names]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cases, strata)):
        train_cases = cases[train_idx].tolist()
        val_cases = cases[val_idx].tolist()
        
        if verbose:
            print(f"Fold {fold+1}/{n_splits}: {len(train_cases)} train, {len(val_cases)} val")
        
        # Optimize on training cases
        result = minimize(
            objective,
            x0,
            args=(param_names, data, train_cases),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )
        
        # Evaluate on validation cases
        train_rmse, train_details = objective(result.x, param_names, data, train_cases, return_details=True)
        val_rmse, val_details = objective(result.x, param_names, data, val_cases, return_details=True)
        
        fold_result = {
            'fold': fold + 1,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_details['mae'],
            'val_mae': val_details['mae'],
            **dict(zip(param_names, result.x))
        }
        cv_results.append(fold_result)
        
        if verbose:
            print(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
            print(f"  Params: {dict(zip(param_names, result.x))}\n")
    
    # 3. Final evaluation on held-out test set (using mean params from CV)
    cv_df = pd.DataFrame(cv_results)
    mean_params = [cv_df[p].mean() for p in param_names]
    test_rmse, test_details = objective(mean_params, param_names, data, test_cases.tolist(), return_details=True)
    
    if verbose:
        print("=" * 50)
        print(f"CV Mean Train RMSE: {cv_df['train_rmse'].mean():.4f} ± {cv_df['train_rmse'].std():.4f}")
        print(f"CV Mean Val RMSE: {cv_df['val_rmse'].mean():.4f} ± {cv_df['val_rmse'].std():.4f}")
        print(f"Held-out Test RMSE: {test_rmse:.4f}")
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
    for tier, tier_params in [
        ('Tier 1 (High Priority)', ['dr_ratio_annuals', 'dr_ratio_treegrass', 'map_to_prod', 
                                     'tree_agb_modifier', 'covercrop_rs_ratio']),
        ('Tier 2 (Medium Priority)', ['grass_rs_ratio', 'tree_fine_root_ratio', 'tree_turnover_ag',
                                       'dr_ratio_amend', 'grass_prod_modifier']),
        ('Tier 3 (Lower Priority)', ['plant_cover_modifier', 'residue_frac_remaining'])
    ]:
        print(f"{tier}:")
        for p in tier_params:
            cfg = PARAM_CONFIG[p]
            print(f"  {p}:")
            print(f"    Default: {cfg['default']}, Bounds: {cfg['bounds']}")
            print(f"    Source: {cfg['source']}, {cfg['description']}")
        print()


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
