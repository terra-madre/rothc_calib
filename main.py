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
# Read data from files
input_dir = Path("./inputs")
data_clean = input_dir / "data_clean.csv"
data_inputs = input_dir / "data_inputs.csv"
loc_data_dir = input_dir / "loc_data"

data_clean_df = pd.read_csv(data_clean)
data_inputs_df = pd.read_csv(data_inputs)

# %% [markdown]
# Data preparation and processing

# %%
# Loop over each case in data_clean_df, get nut3, add to data_clean_df and save
data_clean_df['nuts3'] = ''
for index, row in data_clean_df.iterrows():
    case_id = row['case']
    print(f"Processing case: {case_id}")
    lat = row['latitude']
    lon = row['longitude']
    nuts3 = get_farm_nuts3(lat, lon, input_dir, nuts_version='2024') # nuts_version can be '2021' or '2024'
    data_clean_df.at[index, 'nuts3'] = nuts3
data_clean_df.to_csv(data_clean, index=False)

# %%
# Data preparation and processing
for index, row in data_clean_df.iterrows():
    case_id = row['case']
    print(f"Processing case: {case_id}")
    lat = row['latitude']
    lon = row['longitude']
    location = row['lonlat']

    farm_climate = get_farm_climate(lat=lat, lon=lon, out_dir=location, location=location, start_year=2000, end_year=2020)

    map_mm = get_map(farm_climate)
    
    farm_soil = get_farm_soil(lat=lat, lon=lon, input_dir=loc_data_dir, location=location)
