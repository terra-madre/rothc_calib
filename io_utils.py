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




def get_ipcc_climate_zone_from_tif(lat, lon, climate_zones_file):
    """Read IPCC climate zone from a GeoTIFF raster file.
    
    Args:
        lat (float): Latitude of the location (WGS84)
        lon (float): Longitude of the location (WGS84)
        climate_zones_file (str or Path): Path to the IPCC climate zones TIF file
    
    Returns:
        str: IPCC climate zone label (e.g., 'Tropical Moist', 'Warm Temperate Dry')
             Returns 'Unknown' if coordinates are outside raster bounds or value is invalid
    
    Notes:
        Expected raster values and their mappings:
        1: Tropical Montane, 2: Tropical Wet, 3: Tropical Moist, 4: Tropical Dry
        5: Warm Temperate Moist, 6: Warm Temperate Dry
        7: Cool Temperate Moist, 8: Cool Temperate Dry
        9: Boreal Moist, 10: Boreal Dry
        11: Polar Moist, 12: Polar Dry
    """
    import rasterio
    from pathlib import Path
    
    # Mapping from raster values to IPCC climate zone labels
    ipcc_zone_mapping = {
        1: 'Tropical Montane',
        2: 'Tropical Wet',
        3: 'Tropical Moist',
        4: 'Tropical Dry',
        5: 'Warm Temperate Moist',
        6: 'Warm Temperate Dry',
        7: 'Cool Temperate Moist',
        8: 'Cool Temperate Dry',
        9: 'Boreal Moist',
        10: 'Boreal Dry',
        11: 'Polar Moist',
        12: 'Polar Dry'
    }
    
    try:
        with rasterio.open(climate_zones_file) as src:
            # Get row, col from coordinates
            row_idx, col_idx = src.index(lon, lat)
            
            # Read the value at this location
            zone_value = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0, 0]
            
            # Map value to label
            if zone_value in ipcc_zone_mapping:
                return ipcc_zone_mapping[zone_value]
            else:
                print(f"Warning: Unknown zone value {zone_value} at ({lat}, {lon})")
                return 'Unknown'
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not read climate zone for coordinates ({lat}, {lon}): {e}")
        return 'Unknown'
    except Exception as e:
        print(f"Error reading climate zone file: {e}")
        return 'Unknown'


def get_farm_soil(lat, lon, input_dir, location):
    """Retrieve soil properties data for the farm location from SoilGrids WCS API.
    
    Downloads soil properties data using Web Coverage Service (WCS) if not already cached.
    Variables retrieved (mean values for standard depths):
    - Soil organic carbon content (soc): g/kg (converted from cg/kg)
    - Bulk density (bdod): g/cm³ (converted from cg/cm³)
    - Clay content (clay): % (converted from g/kg)
    - Silt content (silt): % (converted from g/kg)
    - Sand content (sand): % (converted from g/kg)
    
    Standard depths: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm
    
    Args:
        lat (float): Latitude of the farm location (WGS84)
        lon (float): Longitude of the farm location (WGS84)
        input_dir (str): Path to the input data directory
        location (str): Location identifier (e.g., 'TF000')
    
    Returns:
        pd.DataFrame: DataFrame containing soil properties with columns:
            - depth_cm: depth interval as string
            - soil_organic_carbon_g_kg: SOC in g/kg
            - bulk_density_g_cm3: bulk density in g/cm³
            - clay_fraction_perc: clay content in %
            - silt_fraction_perc: silt content in %
            - sand_fraction_perc: sand content in %
    
    Raises:
        ValueError: If coordinates are outside valid range or no data available
        RuntimeError: If WCS request fails
    
    Notes:
        Requires owslib package (install: pip install owslib)
        Uses ISRIC SoilGrids WCS v2.0.1 service
        API documentation: https://docs.isric.org/globaldata/soilgrids/WCS_from_Python.html
    """

    from pathlib import Path
    import numpy as np

    # Define soil data directory and file path
    soil_dir_path = Path(input_dir) / location / 'soil_data'
    soil_dir_path.mkdir(parents=True, exist_ok=True)
    soil_csv_file = soil_dir_path / f'soil_data_soilgrids.csv'
    
    # Check if soil data already exists
    if soil_csv_file.exists():
        print(f"✓ Soil data already exists for location {location}")
        df = pd.read_csv(soil_csv_file)
        return df
    
    print(f"Downloading SoilGrids soil data for location {location}...")
    
    # Validate coordinates
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")
    
    try:
        from owslib.wcs import WebCoverageService
        import rasterio
        from pyproj import Transformer
    except ImportError as e:
        raise RuntimeError(
            "Required packages not found. Please install: pip install owslib rasterio pyproj"
        ) from e
    
    # Properties to retrieve with their WCS service URLs
    # Using mean values (Q0.5 = median/mean for WCS)
    properties = {
        'soc': 'http://maps.isric.org/mapserv?map=/map/soc.map',
        'bdod': 'http://maps.isric.org/mapserv?map=/map/bdod.map',
        'clay': 'http://maps.isric.org/mapserv?map=/map/clay.map',
        'silt': 'http://maps.isric.org/mapserv?map=/map/silt.map',
        'sand': 'http://maps.isric.org/mapserv?map=/map/sand.map'
    }
    
    # Standard depth intervals
    depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
    
    # Transform WGS84 coordinates to Homolosine projection (custom SoilGrids CRS)
    # SoilGrids uses Homolosine equal-area projection (not a standard EPSG code)
    # PROJ string from SoilGrids documentation
    homolosine_proj = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", homolosine_proj, always_xy=True)
    x, y = transformer.transform(lon, lat)
    
    # Create small buffer around point (250m = SoilGrids resolution)
    buffer = 250
    bbox = (x - buffer, y - buffer, x + buffer, y + buffer)
    
    # CRS for WCS 2.0.1 (using opengis.net registry format)
    crs = "http://www.opengis.net/def/crs/EPSG/0/152160"
    
    # Collect data for all properties and depths
    records = []
    
    for depth in depths:
        record = {"depth_cm": depth}
        
        for prop_name, wcs_url in properties.items():
            try:
                # Connect to WCS service
                wcs = WebCoverageService(wcs_url, version='2.0.1')
                
                # Coverage identifier: property_depth_Q0.5 (Q0.5 = mean/median in SoilGrids)
                cov_id = f"{prop_name}_{depth}_Q0.5"
                
                if cov_id not in wcs.contents:
                    print(f"  ⚠ Warning: Coverage {cov_id} not found, skipping")
                    continue
                
                # Get coverage format
                coverage = wcs.contents[cov_id]
                if not coverage.supportedFormats:
                    raise RuntimeError(f"No supported formats for coverage {cov_id}")
                
                # Define subsets for the point location
                subsets = [('X', bbox[0], bbox[2]), ('Y', bbox[1], bbox[3])]
                
                # Get coverage data
                response = wcs.getCoverage(
                    identifier=cov_id,  # Pass as string, not list
                    crs=crs,
                    subsets=subsets,
                    resx=250,
                    resy=250,
                    format=coverage.supportedFormats[0]
                )
                
                # Save temporary GeoTIFF and read value
                temp_tif = soil_dir_path / f'temp_{prop_name}_{depth}.tif'
                with open(temp_tif, 'wb') as f:
                    f.write(response.read())
                
                # Read the raster value at the point
                with rasterio.open(temp_tif) as src:
                    # Read the center pixel
                    data = src.read(1)
                    # Get mean value (excluding nodata)
                    valid_data = data[data != src.nodata]
                    if len(valid_data) > 0:
                        value = float(np.mean(valid_data))
                    else:
                        raise ValueError(f"No valid data for {prop_name} at {depth}")
                
                # Clean up temp file
                temp_tif.unlink()
                
                # Convert units:
                # SoilGrids WCS returns values in cg/kg for soc, cg/cm³ for bdod, g/kg for clay/silt/sand
                # soc: cg/kg -> g/kg (divide by 10)
                # bdod: cg/cm³ -> g/cm³ (divide by 100)
                # clay/silt/sand: g/kg -> % (divide by 10)
                if prop_name == "soc":
                    record["soil_organic_carbon_g_kg"] = round(value / 10, 2)
                elif prop_name == "bdod":
                    record["bulk_density_g_cm3"] = round(value / 100, 3)
                elif prop_name == "clay":
                    record["clay_fraction_perc"] = round(value / 10, 1)
                elif prop_name == "silt":
                    record["silt_fraction_perc"] = round(value / 10, 1)
                elif prop_name == "sand":
                    record["sand_fraction_perc"] = round(value / 10, 1)
                    
            except Exception as e:
                print(f"  ⚠ Warning: Failed to get {prop_name} for {depth}: {e}")
                continue
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Verify we got all expected columns
    expected_cols = [
        'depth_cm', 'soil_organic_carbon_g_kg', 'bulk_density_g_cm3',
        'clay_fraction_perc', 'silt_fraction_perc', 'sand_fraction_perc'
    ]
    
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(
            f"Incomplete soil data received from SoilGrids WCS. "
            f"Expected columns: {expected_cols}, got: {list(df.columns)}"
        )
    
    # Save to CSV
    df.to_csv(soil_csv_file, index=False)
    print(f"✓ Downloaded and saved soil data to {soil_csv_file}")
    
    return df

def get_soilgrids_values(farm_soil, soil_depth = 30):
    """Prepare RothC model inputs for equilibrium run based on farm soil and climate data.
    Args:
        farm_soil (pd.DataFrame): DataFrame with soil properties for the farm location.
        farm_info (pd.Series): Series with farm-level information.
        plots_meta (pd.DataFrame): DataFrame with plot-level information (e.g., measured clay and SOC).
        soil_depth (int): Depth of soil to consider for RothC.
    Returns:
        climate (pd.DataFrame): DataFrame with monthly climate data, averaged for spinup run, else for the period
        plots_rothc (pd.DataFrame): Dataframe with soil clay, soc and carbon inputs for each plot
    """

    import pandas as pd

    ### --- Prepare soil properties --- ###

    # Get aggregated values of soil properties
    # Select soil layers up to soil_depth
    farm_soil = farm_soil.copy()
    
    if soil_depth <= 5:
        farm_soil = farm_soil[farm_soil['depth_cm'] == '0-5cm']
    elif soil_depth <= 15:
        farm_soil = farm_soil[farm_soil['depth_cm'].isin(['5-15cm', '0-5cm'])]
    elif soil_depth <= 30:
        farm_soil = farm_soil[farm_soil['depth_cm'].isin(['15-30cm', '5-15cm', '0-5cm'])]
    elif soil_depth <= 60:
        farm_soil = farm_soil[farm_soil['depth_cm'].isin(['30-60cm', '15-30cm', '5-15cm', '0-5cm'])]
    elif soil_depth <= 100:
        farm_soil = farm_soil[farm_soil['depth_cm'].isin(['60-100cm', '15-30cm', '5-15cm', '0-5cm'])]
    
    # Add depth interval width for weighted averaging
    depth_widths = {
        '0-5cm': 5,
        '5-15cm': 10,
        '15-30cm': 15,
        '30-60cm': 30,
        '60-100cm': 40,
        '100-200cm': 100
    }
    farm_soil['depth_width'] = farm_soil['depth_cm'].map(depth_widths)
    
    # Get the weighted average clay percentage across soil depths 
    clay_perc = (farm_soil['clay_fraction_perc'] * farm_soil['depth_width']).sum() / farm_soil['depth_width'].sum()
    # Get the weighted average bulk density across soil depths
    bulk_density = (farm_soil['bulk_density_g_cm3'] * farm_soil['depth_width']).sum() / farm_soil['depth_width'].sum()
    # Calculate total soil organic carbon in t/ha
    # SOC (g/kg) * depth (cm) * bulk_density (g/cm³=t/m3) * 0.1 = t/ha
    # Explanation: t/t (soc g/kg/ 1000) * m (depth cm / 100) * (area) 10000 m2 * t/m3 (bulk_density) → × 0.1 converts to t/ha
    soc_total = (farm_soil['soil_organic_carbon_g_kg'] / 1000 * farm_soil['depth_width'] / 100 * 10000 * bulk_density).sum()

    # Round values
    clay_perc = round(clay_perc, 1)
    bulk_density = round(bulk_density, 3)
    soc_total = round(soc_total, 2)
    
    return clay_perc, soc_total, bulk_density
