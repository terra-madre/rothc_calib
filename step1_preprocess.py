
# Import necessary modules
from pathlib import Path
import pandas as pd
from io_utils import *


def prepare_cases_df(
    data_clean_df,
    input_dir,
    loc_data_dir,
    soil_depth_cm=30,
    nuts_version='2024'
):
    """Enrich the case table with optional preprocessing steps."""
    data_clean_df = fetch_soil_data(data_clean_df, loc_data_dir, soil_depth=soil_depth_cm)

    data_clean_df = fetch_climate_data(data_clean_df, loc_data_dir, average_climate=True)

    data_clean_df = prepare_initial_soc(
        data_clean_df,
        input_dir=input_dir,
        soil_depth_cm=soil_depth_cm
        )

    data_clean_df = add_nuts3_column(
        data_clean_df,
        input_dir=input_dir,
        nuts_version=nuts_version,
        strict=True,
    )

    return data_clean_df


def fetch_soil_data(data_clean_df, loc_data_dir, soil_depth=30, out_csv_path=None):
    data_clean_df['grids_clay_pct'] = None
    data_clean_df['bulk_density_g_cm3'] = None
    data_clean_df['grids_soc30_t_ha'] = None
    data_clean_df['rothc_clay_pct'] = None
    data_clean_df['rothc_soc30_t_ha'] = None
    for index, row in data_clean_df.iterrows():
        case_id = row['case']
        print(f"Fetching soil data for case: {case_id}")
        lat = row['latitude']
        lon = row['longitude']
        location = row['lonlat']
        # ------- Get and prepare location soil data --------
        farm_soil_grids = get_farm_soil(lat=lat, lon=lon, input_dir=loc_data_dir, location=location)
        grid_clay, grid_soc, grid_bulkdensity = get_soilgrids_values(farm_soil_grids, soil_depth=soil_depth)
        # Set value of row['rothc_clay_pct'] to row['clay_pct'] if not null, else to grid_clay
        data_clean_df.at[index, 'rothc_clay_pct'] = row['clay_pct'] if pd.notnull(row['clay_pct']) else grid_clay
        # data_clean_df.at[index, 'rothc_soc_pct'] = row['soc_pct'] if pd.notnull(row['soc_pct']) else grid_soc # Implement eventually when soc_pct is added to data_clean.csv
        data_clean_df.at[index, 'grids_clay_pct'] = grid_clay
        data_clean_df.at[index, 'grids_soc30_t_ha'] = grid_soc
        data_clean_df.at[index, 'bulk_density_g_cm3'] = grid_bulkdensity

    if out_csv_path is not None:
        data_clean_df.to_csv(out_csv_path, index=False)

    return data_clean_df


def fetch_climate_data(data_clean_df, loc_data_dir, average_climate=True):
    data_clean_df['map_mm'] = None

    for index, row in data_clean_df.iterrows():
        case_id = row['case']
        print(f"Processing case: {case_id}")
        lat = row['latitude']
        lon = row['longitude']
        location = row['lonlat']

        # ------- Get and prepare location climate data ----------
        farm_climate = get_farm_climate(lat=lat, lon=lon, out_dir=loc_data_dir, location=location, start_year=2000, end_year=2020)
        annual_precip = farm_climate.groupby('year')['total_precipitation_mm'].sum().reset_index()
        map_value = round(annual_precip['total_precipitation_mm'].mean())
        data_clean_df.at[index, 'map_mm'] = map_value

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
        # save to csv
        rothc_climate.to_csv(loc_data_dir / location / "climate_data" / f"rothc_climate.csv", index=False)

    return data_clean_df


def add_ipcc_climate_zone(
    data_clean_df,
    input_dir,
    climate_zones_tif="IPCC_Climate_Zones_ts_3.25.tif",
    lat_col='latitude',
    lon_col='longitude',
    out_col='ipcc_climate_zone'
):
    """Add IPCC climate zone to each case using a raster lookup."""
    climate_zones_file = Path(input_dir) / climate_zones_tif

    data_clean_df[out_col] = ''
    for index, row in data_clean_df.iterrows():
        case_id = row['case'] if 'case' in row else index

        lat = row[lat_col]
        lon = row[lon_col]
        ipcc_zone = get_ipcc_climate_zone_from_tif(
            lat=lat,
            lon=lon,
            climate_zones_file=climate_zones_file,
        )
        data_clean_df.at[index, out_col] = ipcc_zone

    return data_clean_df


def prepare_initial_soc(
    data_clean_df,
    input_dir,
    soil_depth_cm=30,
    climate_zones_tif="IPCC_Climate_Zones_ts_3.25.tif",
    medinet_csv="medinet_soil_carbon.csv"
):
    """End-to-end initial SOC preparation: climate zone -> Medinet merge -> normalize -> choose RothC SOC."""
    data_clean_df = add_ipcc_climate_zone(
        data_clean_df=data_clean_df,
        input_dir=input_dir,
        climate_zones_tif=climate_zones_tif
    )
    data_clean_df = merge_medinet_soc(data_clean_df=data_clean_df, input_dir=input_dir, medinet_csv=medinet_csv)
    data_clean_df = normalize_soc_to_depth(data_clean_df=data_clean_df, soil_depth_cm=soil_depth_cm)
    data_clean_df = set_rothc_soc30(data_clean_df=data_clean_df)
    return data_clean_df


def merge_medinet_soc(
    data_clean_df,
    input_dir,
    medinet_csv="medinet_soil_carbon.csv",
    on_cols=('ipcc_climate_zone', 'land_use'),
):
    """Merge Medinet SOC reference values into the case table."""
    medinet_soc_df = pd.read_csv(Path(input_dir) / medinet_csv)
    return pd.merge(data_clean_df, medinet_soc_df, on=list(on_cols), how='left')


def normalize_soc_to_depth(
    data_clean_df,
    soil_depth_cm,
    soc_col='soc_t_ha',
    delta_soc_col='delta_soc_t_ha_y',
    sampling_depth_col='sampling_depth_cm',
    out_soc30_col='soc30_t_ha',
    out_delta_soc30_col='delta_soc30_tC_ha_y',
):
    """Normalize SOC and delta-SOC to a fixed soil depth (simple proportional scaling)."""
    required = {soc_col, sampling_depth_col}
    missing = required - set(data_clean_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for SOC normalization: {sorted(missing)}")

    data_clean_df[out_soc30_col] = data_clean_df[soc_col] / data_clean_df[sampling_depth_col] * soil_depth_cm
    data_clean_df[out_soc30_col] = data_clean_df[out_soc30_col].round(2)

    if delta_soc_col in data_clean_df.columns:
        data_clean_df[out_delta_soc30_col] = (
            data_clean_df[delta_soc_col] / data_clean_df[sampling_depth_col] * soil_depth_cm
        )
        data_clean_df[out_delta_soc30_col] = data_clean_df[out_delta_soc30_col].round(2)

    return data_clean_df


def set_rothc_soc30(
    data_clean_df,
    out_col='rothc_soc30_t_ha',
    prefer_cols=('soc30_t_ha', 'grids_soc30_t_ha', 'medinet_soc30_t_ha'),
):
    """Choose a final SOC@30cm input for RothC using a preference order."""
    if not prefer_cols:
        raise ValueError("prefer_cols must contain at least one column name")

    series = None
    for col in prefer_cols:
        if col not in data_clean_df.columns:
            continue
        series = data_clean_df[col] if series is None else series.fillna(data_clean_df[col])

    if series is None:
        raise ValueError(f"None of prefer_cols exist in data_clean_df: {list(prefer_cols)}")

    data_clean_df[out_col] = series
    return data_clean_df


def load_nuts3_regions(
    input_dir,
    nuts_version='2024',
    nuts_geojson_path=None,
    download=True,
):
    """Load NUTS3 polygons as a GeoDataFrame (cached on disk).

    - If `nuts_geojson_path` is not provided, caches under `<input_dir>/nuts_regions/`.
    - CRS is normalized to EPSG:4326 (WGS84).
    """
    import geopandas as gpd
    import requests

    if nuts_version not in ['2024', '2021']:
        raise ValueError(f"Invalid nuts_version '{nuts_version}'. Must be '2024' or '2021'.")

    if nuts_geojson_path is None:
        cache_dir = Path(input_dir) / 'nuts_regions'
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
    data_clean_df,
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
    missing_cols = required_cols - set(data_clean_df.columns)
    if missing_cols:
        raise ValueError(f"data_clean_df is missing required columns: {sorted(missing_cols)}")

    nuts3_polys = load_nuts3_regions(
        input_dir=input_dir,
        nuts_version=nuts_version,
        nuts_geojson_path=nuts_geojson_path,
        download=download,
    )
    nuts3_polys = gpd.GeoDataFrame(nuts3_polys, geometry='geometry', crs='EPSG:4326')

    points_gdf = gpd.GeoDataFrame(
        data_clean_df.copy(),
        geometry=gpd.points_from_xy(data_clean_df[lon_col], data_clean_df[lat_col]),
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
    
    # Check if climate data already exists and contains all required years
    if climate_csv_file.exists():
        print(f"✓ Climate data already exists for location {location}")
        df = pd.read_csv(climate_csv_file)
        # Check that all required years are present
        required_years = set(int(y) for y in years)
        present_years = set(df['year'].unique())
        if required_years.issubset(present_years):
            return df
    
    # Load required libraries for netcdf processing and CDS API
    import xarray as xr
    import cdsapi
    
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