
# Import necessary modules
from pathlib import Path
import pandas as pd
from io_utils import *


def prepare_cases_df(
    cases_info_df,
    input_dir,
    loc_data_dir,
    soil_depth_cm=30,
    nuts_version='2024'
):
    """Enrich the case table with optional preprocessing steps."""

    cases_info_df = fetch_soil_data(cases_info_df, loc_data_dir, soil_depth=soil_depth_cm)

    cases_info_df, climate_df = fetch_climate_data(
        cases_info_df,
        loc_data_dir,
        average_climate=True
    )

    cases_info_df = prepare_initial_soc(
        cases_info_df,
        input_dir=input_dir,
        soil_depth_cm=soil_depth_cm
    )

    cases_info_df = add_nuts3_column(
        cases_info_df,
        input_dir=input_dir,
        nuts_version=nuts_version,
        strict=True,
    )

    return cases_info_df, climate_df


def prepare_variables(cases_info_df, cases_treatments_df):
    """Prepare/clean variables in the cases_info_df as needed."""
    
    # Create 'lonlat' column with format: lon{rounded}_lat{rounded}
    # Example: longitude=12.3456, latitude=45.6789 -> "lon12.35_lat45.68"
    cases_info_df['lonlat'] = cases_info_df.apply(
        lambda row: f"lon{round(row['longitude'], 2)}_lat{round(row['latitude'], 2)}", axis=1
    )
    
    # Recode land_use values to match expected categories
    land_use_mapping = {
        'ANNUAL': 'annuals',
        'PERENNIAL': 'trees'
    }
    cases_info_df['land_use'] = cases_info_df['land_use'].map(land_use_mapping)

    # Create a numeric sampling_depth_cm column from soil_sampling_cm (e.g., "0-30cm" -> 30)
    # Use last number in the string to get the lower depth bound
    cases_info_df['sampling_depth_cm'] = cases_info_df['soil_sampling_cm'].str.extract(r'(\d+)(?:cm)?$').astype(float)

    # Create a 'subcase' column in cases_treatments_df by concatenating 'case' and first letter of group. 
    cases_treatments_df['subcase'] = cases_treatments_df['case'].astype(str) + cases_treatments_df['group'].str[0]

    return cases_info_df, cases_treatments_df


def fetch_soil_data(cases_info_df, loc_data_dir, soil_depth=30, out_csv_path=None):
    cases_info_df['grids_clay_pct'] = None
    cases_info_df['grids_bd_g_cm3'] = None
    cases_info_df['grids_soc30_t_ha'] = None
    cases_info_df['rothc_clay_pct'] = None

    for index, row in cases_info_df.iterrows():
        case_id = row['case']
        print(f"Fetching soil data for case: {case_id}")
        lat = row['latitude']
        lon = row['longitude']
        location = row['lonlat']

        # ------- Get and prepare location soil data --------
        farm_soil_grids = get_soilgrids_data(lat=lat, lon=lon, input_dir=loc_data_dir, location=location)
        grid_clay, grid_soc, grid_bulkdensity = get_soilgrids_values(farm_soil_grids, soil_depth=soil_depth)
        cases_info_df.at[index, 'grids_clay_pct'] = grid_clay
        cases_info_df.at[index, 'grids_soc30_t_ha'] = grid_soc
        cases_info_df.at[index, 'grids_bd_g_cm3'] = grid_bulkdensity

        # Set value of row['rothc_clay_pct'] to row['clay_pct'] if not null, else to grid_clay
        cases_info_df.at[index, 'rothc_clay_pct'] = row['clay_pct'] if pd.notnull(row['clay_pct']) else grid_clay

    if out_csv_path is not None:
        cases_info_df.to_csv(out_csv_path, index=False)

    return cases_info_df


def fetch_climate_data(
    cases_info_df,
    loc_data_dir,
    average_climate=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases_info_df['map_mm'] = None
    climate_rows = []

    for index, row in cases_info_df.iterrows():
        case_id = row['case']
        print(f"Processing case: {case_id}")
        lat = row['latitude']
        lon = row['longitude']
        location = row['lonlat']

        # ------- Get and prepare location climate data ----------
        farm_climate = get_farm_climate(lat=lat, lon=lon, out_dir=loc_data_dir, location=location, start_year=2000, end_year=2020)
        annual_precip = farm_climate.groupby('year')['total_precipitation_mm'].sum().reset_index()
        map_value = round(annual_precip['total_precipitation_mm'].mean())
        cases_info_df.at[index, 'map_mm'] = map_value

        if average_climate:
            # Average farm_climate into a single year of climate data
            farm_climate = farm_climate.groupby('month').mean().reset_index()
            farm_climate['t_year'] = 1  # Dummy year

        # Rename columns to match rothc expected names: 't_month', 't_tmp','t_rain','t_evap' 
        # (units in RothC are mm for t_rain and t_evap, degC for t_tmp)
        # Drop total_evaporation_mm as RothC uses potential evaporation only
        rothc_climate = farm_climate.rename(columns={
            'month': 't_month',
            'temperature_2m_c': 't_tmp',
            'total_precipitation_mm': 't_rain',
            'potential_evaporation_mm': 't_evap'
        })
        rothc_climate = rothc_climate[['t_year', 't_month', 't_tmp', 't_rain', 't_evap']]
        rothc_climate['case'] = case_id
        rothc_climate = rothc_climate[['case', 't_year', 't_month', 't_tmp', 't_rain', 't_evap']]
        climate_rows.append(rothc_climate)

    if climate_rows:
        climate_df = pd.concat(climate_rows, ignore_index=True)
    else:
        climate_df = pd.DataFrame(
            columns=['case', 't_year', 't_month', 't_tmp', 't_rain', 't_evap']
        )
    climate_df.to_csv(loc_data_dir / "rothc_climate_avg.csv", index=False)

    return cases_info_df, climate_df


def add_ipcc_climate_zone(
    cases_info_df,
    input_dir,
    climate_zones_tif="IPCC_Climate_Zones_ts_3.25.tif",
    lat_col='latitude',
    lon_col='longitude',
    out_col='ipcc_climate_zone'
):
    """Add IPCC climate zone to each case using a raster lookup."""
    climate_zones_file = Path(input_dir) / "geo_files" / climate_zones_tif

    cases_info_df[out_col] = ''
    for index, row in cases_info_df.iterrows():
        case_id = row['case'] if 'case' in row else index

        lat = row[lat_col]
        lon = row[lon_col]
        ipcc_zone = get_ipcc_climate_zone_from_tif(
            lat=lat,
            lon=lon,
            climate_zones_file=climate_zones_file,
        )
        cases_info_df.at[index, out_col] = ipcc_zone

    return cases_info_df


def prepare_initial_soc(
    cases_info_df,
    input_dir,
    soil_depth_cm=30,
    climate_zones_tif="IPCC_Climate_Zones_ts_3.25.tif",
    medinet_csv="medinet_soil_carbon.csv"
):
    """End-to-end initial SOC preparation: climate zone -> Medinet merge -> normalize -> choose RothC SOC."""
    cases_info_df = add_ipcc_climate_zone(
        cases_info_df=cases_info_df,
        input_dir=input_dir,
        climate_zones_tif=climate_zones_tif
    )
    cases_info_df = merge_medinet_soc(cases_info_df=cases_info_df, input_dir=input_dir, medinet_csv=medinet_csv)
    cases_info_df = normalize_soc_to_depth(cases_info_df=cases_info_df, soil_depth_cm=soil_depth_cm)
    cases_info_df = set_rothc_soc30(cases_info_df=cases_info_df)
    return cases_info_df


def merge_medinet_soc(
    cases_info_df,
    input_dir,
    medinet_csv="medinet_soil_carbon.csv",
    on_cols=('ipcc_climate_zone', 'land_use'),
):
    """Merge Medinet SOC reference values into the case table."""
    medinet_soc_df = pd.read_csv(Path(input_dir) / "raw" / medinet_csv)
    return pd.merge(cases_info_df, medinet_soc_df, on=list(on_cols), how='left')


def normalize_soc_to_depth(
    cases_info_df,
    soil_depth_cm,
    soc_col='soc_t_ha',
    delta_soc_col='delta_soc_t_ha_y',
    sampling_depth_col='sampling_depth_cm',
    out_soc30_col='soc30_t_ha',
    out_delta_soc30_col='delta_soc30_t_ha_y',
):
    """Normalize SOC and delta-SOC to a fixed soil depth (simple proportional scaling)."""
    required = {soc_col, sampling_depth_col}
    missing = required - set(cases_info_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for SOC normalization: {sorted(missing)}")

    cases_info_df[out_soc30_col] = cases_info_df[soc_col] / cases_info_df[sampling_depth_col] * soil_depth_cm
    cases_info_df[out_soc30_col] = cases_info_df[out_soc30_col].round(2)

    if delta_soc_col in cases_info_df.columns:
        cases_info_df[out_delta_soc30_col] = (
            cases_info_df[delta_soc_col] / cases_info_df[sampling_depth_col] * soil_depth_cm
        )
        cases_info_df[out_delta_soc30_col] = cases_info_df[out_delta_soc30_col].round(2)

    return cases_info_df


def set_rothc_soc30(
    cases_info_df,
    out_col='rothc_soc30_t_ha',
    prefer_cols=('soc30_t_ha', 'medinet_soc30_t_ha', 'grids_soc30_t_ha'),
):
    """Choose a final SOC@30cm input for RothC using a preference order."""
    if not prefer_cols:
        raise ValueError("prefer_cols must contain at least one column name")

    series = None
    for col in prefer_cols:
        if col not in cases_info_df.columns:
            continue
        series = cases_info_df[col] if series is None else series.fillna(cases_info_df[col])

    if series is None:
        raise ValueError(f"None of prefer_cols exist in cases_info_df: {list(prefer_cols)}")

    cases_info_df[out_col] = series
    return cases_info_df


def load_nuts3_regions(
    input_dir,
    nuts_version='2024',
    nuts_geojson_path=None,
    download=True,
):
    """Load NUTS3 polygons as a GeoDataFrame (cached on disk).

    - If `nuts_geojson_path` is not provided, caches under `<input_dir>/geo_files/`.
    - CRS is normalized to EPSG:4326 (WGS84).
    """
    import geopandas as gpd
    import requests

    if nuts_version not in ['2024', '2021']:
        raise ValueError(f"Invalid nuts_version '{nuts_version}'. Must be '2024' or '2021'.")

    if nuts_geojson_path is None:
        cache_dir = Path(input_dir) / 'geo_files'
        cache_dir.mkdir(parents=True, exist_ok=True)
        nuts_geojson_path = cache_dir / f'NUTS_RG_03M_{nuts_version}_4326_LEVL_3.geojson'
    else:
        nuts_geojson_path = Path(nuts_geojson_path)

    if not nuts_geojson_path.exists():
        if not download:
            raise FileNotFoundError(f"NUTS GeoJSON not found: {nuts_geojson_path}")

        print(f"Downloading NUTS {nuts_version} regions data to {nuts_geojson_path}...")
        url = (
            "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
            f"NUTS_RG_03M_{nuts_version}_4326_LEVL_3.geojson"
        )
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(nuts_geojson_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✓ Downloaded NUTS {nuts_version} regions data")

    nuts_gdf = gpd.read_file(nuts_geojson_path)

    if nuts_gdf.crs is None:
        nuts_gdf = nuts_gdf.set_crs(epsg=4326)
    else:
        nuts_gdf = nuts_gdf.to_crs(epsg=4326)

    if 'NUTS_ID' not in nuts_gdf.columns:
        raise ValueError("NUTS GeoJSON does not contain required column 'NUTS_ID'")

    return nuts_gdf.loc[:, ['NUTS_ID', 'geometry']].copy()


def add_nuts3_column(
    cases_info_df,
    input_dir,
    nuts_version='2024',
    nuts_geojson_path=None,
    download=True,
    strict=True,
    out_col='nuts3',
    lat_col='latitude',
    lon_col='longitude',
):
    """Add a NUTS3 code column to a cases DataFrame (loads NUTS polygons once)."""
    import geopandas as gpd
    import pandas as pd

    required_cols = {lat_col, lon_col}
    missing_cols = required_cols - set(cases_info_df.columns)
    if missing_cols:
        raise ValueError(f"cases_info_df is missing required columns: {sorted(missing_cols)}")

    nuts3_polys = load_nuts3_regions(
        input_dir=input_dir,
        nuts_version=nuts_version,
        nuts_geojson_path=nuts_geojson_path,
        download=download,
    )
    nuts3_polys = gpd.GeoDataFrame(nuts3_polys, geometry='geometry', crs='EPSG:4326')

    points_gdf = gpd.GeoDataFrame(
        cases_info_df.copy(),
        geometry=gpd.points_from_xy(cases_info_df[lon_col], cases_info_df[lat_col]),
        crs='EPSG:4326',
    )

    joined = gpd.sjoin(points_gdf, nuts3_polys, how='left', predicate='within')
    nuts_id_by_row = joined.groupby(joined.index)['NUTS_ID'].first()
    points_gdf[out_col] = nuts_id_by_row
    points_gdf = points_gdf.drop(columns=['geometry'])

    if strict and points_gdf[out_col].isna().any():
        missing = points_gdf[points_gdf[out_col].isna()].copy()
        preview_cols = [c for c in ['case', lat_col, lon_col] if c in missing.columns]
        preview = missing[preview_cols].head(10)
        raise ValueError(
            "Some coordinates are not within any NUTS3 region. "
            "Verify coordinates are in Europe and use WGS84 (EPSG:4326). "
            f"Missing rows: {len(missing)}. Preview:\n{preview}"
        )

    return pd.DataFrame(points_gdf)

def get_farm_climate(lat, lon, out_dir, location, start_year=2000, end_year=None):
    """Retrieve ERA5-Land climate data for the farm location from Copernicus CDS.
    
    Downloads monthly climate data from start_year to last year if not already cached.
    Variables retrieved:
    - Total evaporation (e)
    - Potential evaporation (pev)
    - Total precipitation (tp)
    - 2m temperature (t2m)
    
    Args:
        lat (float): Latitude of the farm location
        lon (float): Longitude of the farm location
        out_dir (str): Path to the output data directory
        start_year (int): Start year for climate data retrieval
        end_year (int, optional): End year for climate data retrieval. Defaults to last year.
    
    Returns:
        pd.DataFrame: DataFrame containing climate data with columns for date and climate variables
    
    Notes:
        Requires CDS API credentials configured in ~/.cdsapirc
        See: https://cds.climate.copernicus.eu/how-to-api
    """

    from pathlib import Path
    from datetime import datetime

    # Define climate data directory and file paths
    climate_dir_path = Path(out_dir) / location / 'climate_data'
    climate_dir_path.mkdir(parents=True, exist_ok=True)
    climate_csv_file = climate_dir_path / f'climate_data_era5land.csv'
    climate_nc_file = climate_dir_path / f'climate_data_era5land.nc'

    # Define time range (start_year to last year)
    current_year = datetime.now().year
    if end_year is None:
        end_year = current_year - 1
    years = [str(year) for year in range(start_year, end_year + 1)]
    
    required_years = set(int(y) for y in years)
    needs_download = True

    # Check if climate data already exists and contains all required years
    if climate_csv_file.exists():
        print(f"✓ Climate data already exists for location {location}")
        df = pd.read_csv(climate_csv_file)
        # Check that all required years are present
        present_years = set(df['year'].unique())
        if required_years.issubset(present_years):
            return df
        needs_download = True
    
    # Load required libraries for netcdf processing and CDS API
    import xarray as xr
    import cdsapi
    
    if needs_download and climate_nc_file.exists():
        climate_nc_file.unlink()

    if not climate_nc_file.exists():
        print(f"Downloading ERA5-Land climate data for location {location}...")
        
        # Initialize CDS API client
        client = cdsapi.Client()
        
        # Define spatial bounds (add small buffer around point)
        buffer = 0.1  # degrees
        area = [
            lat + buffer,  # North
            lon - buffer,  # West
            lat - buffer,  # South
            lon + buffer   # East
        ]
        
        # ERA5-Land variable names:
        # - total_evaporation (e): m of water equivalent
        # - potential_evaporation (pev): m of water equivalent
        # - total_precipitation (tp): m
        # - 2m_temperature (t2m): K
        variables = [
            'total_evaporation',
            'potential_evaporation',
            'total_precipitation',
            '2m_temperature'
        ]
        
        try:
            print(f"  Requesting data from CDS API...")
            dataset = "reanalysis-era5-land-monthly-means"
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": variables,
                "year": years,
                "month": [f'{m:02d}' for m in range(1, 13)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area
            }

            client.retrieve(dataset, request, str(climate_nc_file))

            # Check downloaded file size
            file_size = climate_nc_file.stat().st_size
            print(f"  Downloaded {file_size / (1024*1024):.1f} MB")

        except Exception as e:
            raise RuntimeError(
                f"Failed to download climate data for location {location}. "
            ) from e
        
    # Open NetCDF file with xarray
    print(f"  Processing NetCDF file...")
    ds = xr.open_dataset(climate_nc_file)
    
    # Select nearest point to given coordinates
    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    
    # Convert to DataFrame
    df = ds_point.to_dataframe().reset_index()

    # Rename valid_time to time if exists
    if 'valid_time' in df.columns:
        df = df.rename(columns={'valid_time': 'time'})
    
    # Keep only relevant columns: time, e, pev, tp, t2m
    df = df[['time', 'e', 'pev', 'tp', 't2m']] 

    # Add year and month columns (needed for days-in-month calculation)
    df['year'] = df['time'].dt.year # type: ignore
    df['month'] = df['time'].dt.month # type: ignore
    
    # ERA5-Land monthly means have accumulation variables in units "per day"
    # To get monthly totals, multiply by the number of days in each month
    df['days_in_month'] = df['time'].dt.days_in_month # type: ignore
    
    # Convert daily averages to monthly totals and convert to mm
    # for precipitation and evaporation variables
    df['total_evaporation_mm'] = df['e'] * df['days_in_month'] * 1000
    df['potential_evaporation_mm'] = df['pev'] * df['days_in_month'] * 1000
    df['total_precipitation_mm'] = df['tp'] * df['days_in_month'] * 1000
    
    # Change the sign of total_evaporation and potential_evaporation to positive
    df['total_evaporation_mm'] = -df['total_evaporation_mm']
    df['potential_evaporation_mm'] = -df['potential_evaporation_mm']

    # Temperature is instantaneous. Convert to Celsius
    df['temperature_2m_c'] = df['t2m'] - 273.15

    # Round climate variables to 2 decimal places
    df['total_evaporation_mm'] = df['total_evaporation_mm'].round(2)
    df['potential_evaporation_mm'] = df['potential_evaporation_mm'].round(2)
    df['total_precipitation_mm'] = df['total_precipitation_mm'].round(2)
    df['temperature_2m_c'] = df['temperature_2m_c'].round(2)
    
    # Select final columns
    df = df[['year', 'month', 'total_evaporation_mm', 'potential_evaporation_mm',
                'total_precipitation_mm', 'temperature_2m_c']]
    
    # Clean up
    ds.close()
    
    # Save to CSV
    df.to_csv(climate_csv_file, index=False)
    
    print(f"✓ Downloaded and saved climate data to {climate_csv_file}")

    return df



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


def get_soilgrids_data(lat, lon, input_dir, location):
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
            - grids_bd_g_cm3: bulk density in g/cm³
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
        from typing import Any
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
        record: dict[str, float | str] = {"depth_cm": depth}
        
        for prop_name, wcs_url in properties.items():
            try:
                # Connect to WCS service
                wcs: Any = WebCoverageService(wcs_url, version='2.0.1')
                assert wcs is not None
                
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
                    record["grids_bd_g_cm3"] = round(value / 100, 3)
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
        'depth_cm', 'soil_organic_carbon_g_kg', 'grids_bd_g_cm3',
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
    bulk_density = (farm_soil['grids_bd_g_cm3'] * farm_soil['depth_width']).sum() / farm_soil['depth_width'].sum()
    # Calculate total soil organic carbon in t/ha
    # SOC (g/kg) * depth (cm) * bulk_density (g/cm³=t/m3) * 0.1 = t/ha
    # Explanation: t/t (soc g/kg/ 1000) * m (depth cm / 100) * (area) 10000 m2 * t/m3 (bulk_density) → × 0.1 converts to t/ha
    soc_total = (farm_soil['soil_organic_carbon_g_kg'] / 1000 * farm_soil['depth_width'] / 100 * 10000 * bulk_density).sum()

    # Round values
    clay_perc = round(clay_perc, 1)
    bulk_density = round(bulk_density, 3)
    soc_total = round(soc_total, 2)
    
    return clay_perc, soc_total, bulk_density


def get_st_yields(cases_info_df, fixed_data_dir, loc_data_dir):
    """Get yields for each case based on location info."""

    st_yields = pd.read_csv(fixed_data_dir / "st_yields.csv")

    # Fill missing values with 0 for calculating area-weighted yields
    yield_cols = ['yield_dry_rainfed_t_ha', 'yield_dry_irrigated_t_ha', 'area_rainfed_ha', 'area_irrigated_ha']
    st_yields[yield_cols] = st_yields[yield_cols].fillna(0)

    # Calculate area-weighted yields (needed for historical data)
    st_yields['yield_dry_weighted_t_ha'] = (
        (st_yields['yield_dry_rainfed_t_ha'] * st_yields['area_rainfed_ha'] +
        st_yields['yield_dry_irrigated_t_ha'] * st_yields['area_irrigated_ha']) /
        (st_yields['area_rainfed_ha'] + st_yields['area_irrigated_ha'])
    ).where((st_yields['area_rainfed_ha'] + st_yields['area_irrigated_ha']) > 0, 0).round(2)

    # Set 0 values in yield_cols back to missing
    st_yields[yield_cols] = st_yields[yield_cols].replace(0, pd.NA)

    # Drop rows with missing or zero yield values (apply once to entire dataset)
    st_yields = st_yields.dropna(subset=['yield_dry_rainfed_t_ha', 'yield_dry_irrigated_t_ha'], how='all')
    st_yields = st_yields[~((st_yields['yield_dry_rainfed_t_ha'] == 0) & (st_yields['yield_dry_irrigated_t_ha'] == 0))]
    
    all_st_yields_selected = []

    # Process unique NUTS3 regions only to avoid duplicates
    unique_nuts3 = cases_info_df['nuts3'].unique()
    
    for nuts3 in unique_nuts3:
        nuts0 = nuts3[:2]  # Country level
        nuts1 = nuts3[:3]  # Major socio-economic regions
        nuts2 = nuts3[:4]  # Basic regions for regional policies

        # Get yields at each NUTS level
        st_yields_nuts3 = st_yields[st_yields['nuts_code'] == nuts3][['group_name', 'yield_dry_rainfed_t_ha', 'yield_dry_irrigated_t_ha', 'yield_dry_weighted_t_ha']].copy()
        st_yields_nuts2 = st_yields[st_yields['nuts_code'] == nuts2][['group_name', 'yield_dry_rainfed_t_ha', 'yield_dry_irrigated_t_ha', 'yield_dry_weighted_t_ha']].copy()
        st_yields_nuts1 = st_yields[st_yields['nuts_code'] == nuts1][['group_name', 'yield_dry_rainfed_t_ha', 'yield_dry_irrigated_t_ha', 'yield_dry_weighted_t_ha']].copy()
        st_yields_nuts0 = st_yields[st_yields['nuts_code'] == nuts0][['group_name', 'yield_dry_rainfed_t_ha', 'yield_dry_irrigated_t_ha', 'yield_dry_weighted_t_ha']].copy()
        
        # Assign priority to each level (higher number = higher priority/more specific)
        st_yields_nuts3['yield_nuts_level'] = 3
        st_yields_nuts2['yield_nuts_level'] = 2
        st_yields_nuts1['yield_nuts_level'] = 1
        st_yields_nuts0['yield_nuts_level'] = 0

        # Combine all levels and sort by priority (descending) to prefer more specific values
        st_yields_combined = pd.concat([st_yields_nuts3, st_yields_nuts2, st_yields_nuts1, st_yields_nuts0], ignore_index=True)
    
        # For each group_name, keep only the row with highest priority (most specific NUTS level)
        st_yields_selected = st_yields_combined.sort_values('yield_nuts_level', ascending=False).drop_duplicates(subset='group_name', keep='first')

        # Add nuts3 column
        st_yields_selected['nuts3'] = nuts3
        
        # Append to collection
        all_st_yields_selected.append(st_yields_selected)

    # Combine all locations and save once
    st_yields_all = pd.concat(all_st_yields_selected, ignore_index=True)
    st_yields_all.to_csv(loc_data_dir / "st_yields_selected.csv", index=False)

    return st_yields_all