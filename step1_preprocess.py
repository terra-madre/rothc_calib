# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: terra-plus
#     language: python
#     name: python3
# ---

# %%
# This is the main scrpt for a calibration of the RothC model

# Import necessary modules
from pathlib import Path
import pandas as pd
from io_utils import *

# %%
# Configurations and paths

average_climate = True  # Whether to average climate data into a single year for RothC inputs
soil_depth = 30  # Soil depth in cm for soilgrids data

input_dir = Path("../inputs")
data_raw = input_dir / "data_raw.csv"
data_inputs = input_dir / "data_inputs.csv"
loc_data_dir = input_dir / "loc_data"
data_clean = input_dir / "data_clean.csv"

# %%
# Read data from files
data_raw_df = pd.read_csv(data_raw)
data_clean_df = data_raw_df.copy()  # Start with raw data and add columns to it, then save as data_clean.csv
data_inputs_df = pd.read_csv(data_inputs)

# %% [markdown]
# Data preparation and processing

# %%
# # Loop over each case in data_clean_df, get nut3, add to data_clean_df and save
# data_clean_df['nuts3'] = ''
# for index, row in data_clean_df.iterrows():
#     case_id = row['case']
#     print(f"Processing case: {case_id}")
#     lat = row['latitude']
#     lon = row['longitude']
#     nuts3 = get_farm_nuts3(lat, lon, input_dir, nuts_version='2024') # nuts_version can be '2021' or '2024'
#     data_clean_df.at[index, 'nuts3'] = nuts3
# data_clean_df.to_csv(data_clean, index=False)

# %%
# Data preparation and processing
data_clean_df['map_mm'] = None
data_clean_df['grids_clay_pct'] = None
data_clean_df['grids_soc_pct'] = None
data_clean_df['rothc_clay_pct'] = None

for index, row in data_clean_df.iterrows():
    case_id = row['case']
    print(f"Processing case: {case_id}")
    lat = row['latitude']
    lon = row['longitude']
    location = row['lonlat']

    # ------- Get and prepare location climate data ----------
    farm_climate = get_farm_climate(lat=lat, lon=lon, out_dir=loc_data_dir, location=location, start_year=2000, end_year=2020)
    annual_precip = farm_climate.groupby('year')['total_precipitation_mm'].sum().reset_index()
    map_value = annual_precip['total_precipitation_mm'].mean()
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

    # ------- Get and prepare location soil data --------
    farm_soil_grids = get_farm_soil(lat=lat, lon=lon, input_dir=loc_data_dir, location=location)
    grid_clay, grid_soc, grid_bulkdensity = get_soilgrids_values(farm_soil_grids, soil_depth=soil_depth)
    # Set value of row['rothc_clay_pct'] to row['clay_pct'] if not null, else to grid_clay
    data_clean_df.at[index, 'rothc_clay_pct'] = row['clay_pct'] if pd.notnull(row['clay_pct']) else grid_clay
    # data_clean_df.at[index, 'rothc_soc_pct'] = row['soc_pct'] if pd.notnull(row['soc_pct']) else grid_soc # Implement eventually when soc_pct is added to data_clean.csv
    data_clean_df.at[index, 'grids_clay_pct'] = grid_clay
    data_clean_df.at[index, 'grids_soc_t_ha'] = grid_soc

# %%
# Determine initial soil carbon content. Possible sources in order of preference: from 
# Read the file IPCC_Climate_Zones_ts_3.25.tif and find the climate zone for each case, then add to data_clean_df
climate_zones_file = input_dir / "IPCC_Climate_Zones_ts_3.25.tif"

data_clean_df['ipcc_climate_zone'] = ''
for index, row in data_clean_df.iterrows():
    case_id = row['case']
    print(f"Determining climate zone for case: {case_id}")
    lat = row['latitude']
    lon = row['longitude']
    
    # Get IPCC climate zone from TIF file
    ipcc_zone = get_ipcc_climate_zone_from_tif(lat=lat, lon=lon, climate_zones_file=climate_zones_file)
    data_clean_df.at[index, 'ipcc_climate_zone'] = ipcc_zone
    print(f"  Climate zone: {ipcc_zone}")

# Read in Medinet soc data and merge with data_clean_df by ipcc_climate_zone and land_use
medinet_soc_df = pd.read_csv(input_dir / "medinet_soil_carbon.csv")
data_clean_df = pd.merge(data_clean_df, medinet_soc_df, on=['ipcc_climate_zone', 'land_use'], how='left')
# Set data_clean_df['rothc_soc_t_ha'] to data_clean_df['medinet_soc_t_ha'] if not null, else to data_clean_df['grids_soc_t_ha']
data_clean_df['rothc_soc30_t_ha'] = None
data_clean_df['rothc_soc_t_ha'] = data_clean_df['medinet_soc_t_ha'].fillna(data_clean_df['grids_soc_t_ha'])

# Save
data_clean_df.to_csv(data_clean, index=False)

# %%
# Normalize the deltaC to 30cm soil depth
# Replace this with a more correct method eventually that uses the soil depth distribution of SOC (e.g. from soilgrids)
data_clean_df['delta_soc30_tC_ha_y'] = data_clean_df['delta_soc_t_ha_y'] / data_clean_df['sampling_depth_cm'] * soil_depth
