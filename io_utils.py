"""utils/excel_csv_io.py

Helpers for reading Excel and CSV files with consistent column normalization
and a small mapping-file mechanism to map original column labels to canonical
column names used in code.

Features:
- read_csv_or_excel(path, sheet_name=None, use_second_row_as_header=False)
- write_excel_file(path, data, sheet_name='Sheet1')
- copy_excel_file(src, dst)
- sanitize_col, make_unique
- generate_mapping_suggestions(df, mapping_path) -> loads/creates mapping CSV
- apply_column_mapping(df, mapping)

Mapping CSV format: two columns: original,canonical
"""

from pathlib import Path
import pandas as pd
import re
import unicodedata
import glob


def extract_nuts_levels(nuts3_code):
    """Extract NUTS region codes at different administrative levels.
    
    NUTS (Nomenclature of Territorial Units for Statistics) is a hierarchical system
    for dividing up the economic territory of the EU.
    
    Args:
        nuts3_code: NUTS3 region code (e.g., 'ES511', 'PT111')
        
    Returns:
        tuple: (nuts0, nuts1, nuts2) where:
            - nuts0: Country code (2 chars, e.g., 'ES')
            - nuts1: NUTS1 region (3 chars, e.g., 'ES5')
            - nuts2: NUTS2 region (4 chars, e.g., 'ES51')
    """
    nuts0 = nuts3_code[:2]  # Country level
    nuts1 = nuts3_code[:3]  # Major socio-economic regions
    nuts2 = nuts3_code[:4]  # Basic regions for regional policies
    return nuts0, nuts1, nuts2


# Function that takes a dictionary of dataframes and converts the first row of each into column names
def df_dict_first_row_to_header(df_dict):
    """Convert the first row of each DataFrame in the dict to column headers.

    Args:
        df_dict: dict of pandas DataFrames
    Returns:
        dict of pandas DataFrames with updated column headers
    """
    updated_dict = {}
    for sheet_name, df in df_dict.items():
        new_header = df.iloc[0]  # first row as header
        updated_dict[sheet_name] = df[1:].copy()
        updated_dict[sheet_name].columns = new_header
    return updated_dict


def get_default_numeric_columns():
    """Get list of column names that should be converted to numeric type.
    
    Returns:
        list of column names
    """
    return [
        'area_ha', 'percent_cover', 'yield_t_ha', 'dry_frac', 'residue_yield_ratio',
        'fertilizer_kg_ha', 'tillage_passes', 'grazing_days', 'livestock_density',
        'fuel_liters_ha', 'amendment_kg_ha', 'irrigation_l_ha', 'yield_reported_t_ha',
        'percent_shrub_cover', 'number_of_trees', 'trunk_diameter_cm', 'planting_year',
        'predicted_max_trunk_diam_cm', 'predicted_replacement_age_years', 'amount_t_ha',
        'percent_imported', 'amount_kg_ha', 'n_content_perc',
        'amount', 'days_grazing_on_farm', 'days_outside_the_farm',
        'diesel_liters', 'petrol_liters'
    ]


def convert_numeric_columns(df, numeric_columns=None):
    """Convert specified columns to numeric types.
    
    This addresses the common issue where Excel imports numeric data as object dtype,
    causing problems with mathematical operations in pandas eval().
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of column names that should be converted to numeric.
                        If None, uses default list from get_default_numeric_columns()
        
    Returns:
        pandas DataFrame with converted numeric columns
    """
    if numeric_columns is None:
        numeric_columns = get_default_numeric_columns()
    df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Convert to numeric
        # errors='coerce' converts non-numeric values to NaN
        original_dtype = df[col].dtype
        original_non_null = df[col].notna().sum()
        
        df[col] = pd.to_numeric(df[col], errors='coerce')
        converted_non_null = df[col].notna().sum()
        
        print(f"✓ Converted '{col}' from {original_dtype} to {df[col].dtype} ({converted_non_null}/{original_non_null} values)")
        
        if converted_non_null < original_non_null:
            lost_values = original_non_null - converted_non_null
            print(f"  ⚠ Warning: {lost_values} non-numeric values converted to NaN")
    
    return df


def sanitize_col(col, lower=True):
    """Normalize a column name to a machine-friendly identifier.

    Rules:
    - Unicode normalization (strip accents)
    - Replace non-alphanumeric characters with underscore
    - Lowercase (optional)
    - Remove leading digits
    - Collapse multiple underscores and strip
    """
    col = '' if col is None else str(col)
    # Normalize unicode
    col = unicodedata.normalize('NFKD', col)
    col = ''.join(c for c in col if not unicodedata.combining(c))
    if lower:
        col = col.lower()
    # Replace non-alphanumeric/underscore with underscore
    col = re.sub(r'[^0-9a-zA-Z_]+', '_', col)
    # Remove leading digits
    col = re.sub(r'^[0-9]+', '', col)
    # Collapse underscores
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    return col or 'col'


def make_unique(cols):
    """Make column list unique by appending a numeric suffix where needed."""
    seen = {}
    out = []
    for c in cols:
        key = c
        if key not in seen:
            seen[key] = 0
            out.append(key)
        else:
            seen[key] += 1
            new = f"{key}_{seen[key]}"
            # ensure new not used
            while new in seen:
                seen[key] += 1
                new = f"{key}_{seen[key]}"
            seen[new] = 0
            out.append(new)
    return out


def write_excel_file(file_path, data, sheet_name='Sheet1'):
    """Write a DataFrame or dict of DataFrames to an Excel file.

    If data is a dict: keys->sheet names.
    If file exists and is Excel, it will be overwritten.
    """
    file_path = Path(file_path)
    if isinstance(data, dict):
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet, df in data.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
    else:
        # single DataFrame
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)


def save_mapping(mapping, mapping_path):
    """Save column mapping to CSV file, appending new entries and updating existing ones.
    
    Args:
        mapping: dict with original->canonical mappings
        mapping_path: path to mapping CSV file
    """
    mapping_path = Path(mapping_path)
    
    # Load existing mapping if it exists, otherwise start empty
    existing_mapping = load_mapping(mapping_path) if mapping_path.exists() else {}
    
    # Update existing with new mappings (new ones override existing)
    existing_mapping.update(mapping)
    
    # Create directory if needed and save
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(existing_mapping.items()), columns=['original', 'canonical'])
    df.to_csv(mapping_path, index=False)


def apply_column_mapping(df, mapping, sanitize_unmapped=False):
    """Apply mapping dict to DataFrame columns and return renamed DataFrame and mapping used.

    If sanitize_unmapped is True, unmapped original columns will be sanitized.
    Args:
        df: pandas DataFrame
        mapping: dict with original->canonical mappings (can be None or empty)
        sanitize_unmapped: bool, whether to sanitize unmapped columns
    """
    orig_cols = list(df.columns)
    new_cols = []
    missing = []
    
    # Handles cases where mapping is missing and sanitizes if requested
    for c in orig_cols:
        # Convert column name to string for consistent mapping lookup
        c_str = str(c)
        if c_str in mapping and mapping[c_str]:
            new_cols.append(mapping[c_str])
        elif sanitize_unmapped:
            new_cols.append(sanitize_col(c_str))
        else:
            # preserve original name but record missing mapping
            new_cols.append(c)
            missing.append(c)
            
    # make unique
    new_cols = make_unique(new_cols)
    df = df.copy()
    df.columns = new_cols
    used_mapping = dict(zip(orig_cols, new_cols))
    # Warn about unmapped original columns so user can update mapping file
    if missing:
        print("WARNING: The following columns are not present in the mapping file:")
        for m in missing:
            print(f"  - {m}")
        print("Consider running generate_mapping_suggestions(...) to populate suggested canonical names.")
    return df, used_mapping


# Load mapping function
def load_mapping(mapping_path):
    # Read in columns mapping file and convert to dict original->canonical
    column_mappings_df = pd.read_csv(mapping_path)
    column_mappings = dict(zip(column_mappings_df['original'].astype(str), column_mappings_df['canonical'].astype(str)))
    return column_mappings

def apply_column_mapping_and_save(df, mapping_file):
    """Apply column mapping to a DataFrame, sanitize unmapped columns, and enforce numeric types.
    
    This is a convenience function that combines steps:
    1. Load existing mapping from file
    2. Apply mapping to DataFrame (with sanitization of unmapped columns)
    3. Convert numeric columns to proper dtypes
    4. Save any new mappings back to file (DISABLED - manual update only)
    
    Args:
        df: pandas DataFrame to process
        mapping_file: path to column mapping CSV file
        
    Returns:
        DataFrame with mapped column names and enforced numeric types
    """
    mapping = load_mapping(mapping_file) if mapping_file is not None else {}
    df, used_mapping = apply_column_mapping(df, mapping, sanitize_unmapped=True)
    # Automatic saving disabled - mappings must be added manually to prevent
    # unwanted entries like 'nan', '0', '100.0' from being auto-generated
    # save_mapping(used_mapping, mapping_file)
    
    # Enforce numeric column types
    df = convert_numeric_columns(df)
    
    return df


def load_fixed_values(fixed_values_dir, mapping_file):
    """
    Load all fixed value files from the organized parameter directory
    
    Args:
        fixed_values_dir: Directory containing ps_*.csv, cp_*.csv, st_*.csv files
        
    Returns:
        Dict with parameters, common_practice, and standard_values DataFrames
    """
    fixed_values = {
        'parameters': {},
        'common_practice': {},
        'standard_values': {}
    }
    
    # Load parameter files (ps_*.csv) with column mapping
    # Define file prefixes and their corresponding keys in fixed_values dict
    file_types = {
        'ps_': 'parameters',
        'cp_': 'common_practice', 
        'st_': 'standard_values'
    }
    
    for prefix, category in file_types.items():
        fixed_values_path = Path(fixed_values_dir)
        files = list(fixed_values_path.glob(f'{prefix}*.csv'))
        for file_path in files:
            filename = file_path.name
            key = filename.replace(prefix, '').replace('.csv', '')
            
            # Load data and apply column mapping with automatic saving
            df = pd.read_csv(file_path)
            df = apply_column_mapping_and_save(df, mapping_file)
            
            fixed_values[category][key] = df

    print(f"Loaded fixed values")
    return fixed_values

def apply_column_mapping_to_dict(farm_data_raw, mapping_file, numeric_columns=None):
    """Apply column mapping to a dictionary of DataFrames and optionally convert numeric columns.
    
    Args:
        farm_data_raw: dict of DataFrames (typically from Excel sheets)
        mapping_file: path to column mapping CSV file
        numeric_columns: list of column names that should be converted to numeric (after mapping).
                        If None, uses default list. Pass empty list [] to skip conversion.
        
    Returns:
        dict of DataFrames with mapped column names and converted dtypes
    """
    if not mapping_file:
        return farm_data_raw
    
    # apply_column_mapping_and_save now handles numeric conversion internally
    farm_data = {}
    for sheet_name, sheet_df in farm_data_raw.items():
        farm_data[sheet_name] = apply_column_mapping_and_save(sheet_df, mapping_file)
            
    return farm_data

def validate_required_dataframes(farm_data):
    """
    Validate that all required dataframes are present in farm_data and remove any that are not required
    
    Args:
        farm_data: Dict of DataFrames from Excel file (modified in-place)
        
    Raises:
        ValueError: If any required dataframes are missing
        
    Returns:
        Dict of DataFrames containing only required sheets
    """
    # List of required sheet names - update this list as needed
    required_sheets = [
        'Farm Info',
        'Plot Info',
        'Regenerative Practices',
        'Annual Crops - Grasses',
        'Tree Crops - Planted Trees',
        'Soil Amendments',
        'Fertilizer Application',
        'Tillage',
        'Livestock (farm owned)',
        'Livestock (external)',
        'Fuel (Direct Use)',
        'Fuel (External Services)',
        'Unmanaged Felled Trees'
    ]
    
    available_sheets = list(farm_data.keys())
    missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
    extra_sheets = [sheet for sheet in available_sheets if sheet not in required_sheets]
    
    # Remove extra sheets that are not required
    for sheet in extra_sheets:
        del farm_data[sheet]
    
    if extra_sheets:
        print(f"ℹ Removed non-required sheets: {', '.join(extra_sheets)}")
    
    if missing_sheets:
        raise ValueError(
            f"Missing required dataframes: {', '.join(missing_sheets)}. "
            f"Available sheets: {', '.join(available_sheets)}"
        )
    
    print(f"✓ All required dataframes present: {', '.join(required_sheets)}")
    return farm_data

