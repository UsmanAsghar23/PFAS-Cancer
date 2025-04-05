import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple


def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df['gm_samp_collection_date'] = pd.to_datetime(df['gm_samp_collection_date'])

    return df[(df['gm_samp_collection_date'] >= start_date) & 
             (df['gm_samp_collection_date'] <= end_date)]

def remove_missing_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['gm_result'])

def extract_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['gm_chemical_vvl', 'gm_result', 'gm_samp_collection_date', 
              'gm_latitude', 'gm_longitude']

    return df[columns].sort_values(by='gm_samp_collection_date', ascending=False)

def load_county_shapefile(file_path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(file_path)

def calculate_county_bounds(gdf: gpd.GeoDataFrame) -> Dict[str, Dict[str, float]]:
    county_bounds = {}
    for _, row in gdf.iterrows():
        county_name = row['NAME']
        bounds = row['geometry'].bounds
        county_bounds[county_name] = {
            'min_lat': bounds[1],
            'max_lat': bounds[3],
            'min_lon': bounds[0],
            'max_lon': bounds[2]
        }
    return county_bounds

def get_california_counties() -> List[str]:
    return ['Alameda', 'Alpine', 'Amador', 'Butte', 'Calaveras', 'Colusa',
            'Contra Costa', 'Del Norte', 'El Dorado', 'Tulare', 'Fresno',
            'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake',
            'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa',
            'Mendocino', 'Merced', 'Modoc', 'Mono', 'Monterey', 'Napa',
            'Nevada', 'Orange', 'Placer', 'Plumas', 'Riverside', 'Sacramento',
            'San Benito', 'San Bernardino', 'San Diego', 'San Francisco',
            'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara',
            'Santa Clara', 'Santa Cruz', 'Shasta', 'Sierra', 'Siskiyou',
            'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Shasta', 'Tehama',
            'Trinity', 'Tulare', 'Tuolumne', 'Ventura', 'Kern', 'Yolo', 'Yuba']

def get_county_from_bounds(lat: float, lon: float, 
                          county_bounds: Dict[str, Dict[str, float]]) -> str:
    for county, bounds in county_bounds.items():
        if (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
            bounds['min_lon'] <= lon <= bounds['max_lon']):
            return county
    return 'Unknown'

def add_county_column(df: pd.DataFrame, 
                     county_bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df['county'] = df.apply(
        lambda row: get_county_from_bounds(row['gm_latitude'], 
                                         row['gm_longitude'], 
                                         county_bounds), 
        axis=1
    )
    return df

def create_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    # Group by county, date, and chemical
    grouped_df = df.groupby(['county', 'gm_samp_collection_date', 'gm_chemical_vvl'])['gm_result'].mean().reset_index()
    
    # Pivot to wide format
    wide_df = grouped_df.pivot(
        index=['county', 'gm_samp_collection_date'],
        columns='gm_chemical_vvl',
        values='gm_result'
    ).reset_index()
    
    # Flatten column names
    wide_df.columns.name = None
    wide_df.columns = ['county', 'gm_samp_collection_date'] + [f'{col}' for col in wide_df.columns[2:]]
    
    return wide_df.sort_values(by=['county', 'gm_samp_collection_date'], ascending=False)

def fill_missing_values(wide_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate annual means
    wide_df['year'] = pd.to_datetime(wide_df['gm_samp_collection_date']).dt.year
    annual_means = wide_df.groupby(['county', 'year']).mean().reset_index()
    
    # Fill with annual means
    def fill_with_annual_mean(row, annual_means):
        county = row['county']
        year = row['year']
        for col in annual_means.columns[2:]:
            if pd.isna(row[col]):
                mean_value = annual_means[(annual_means['county'] == county) & 
                                        (annual_means['year'] == year)][col]
                if not mean_value.empty:
                    row[col] = mean_value.values[0]
        return row
    
    wide_df_filled = wide_df.apply(lambda row: fill_with_annual_mean(row, annual_means), axis=1)
    wide_df_filled = wide_df_filled.drop(columns=['year'])
    
    # Fill remaining NaN values with column means
    columns_to_fill = wide_df_filled.columns.difference(['county', 'gm_samp_collection_date'])
    wide_df_filled[columns_to_fill] = wide_df_filled[columns_to_fill].fillna(
        wide_df_filled[columns_to_fill].mean()
    )
    
    # Fill any remaining NaN values with 0
    return wide_df_filled.fillna(0)

def calculate_total_pfas(df: pd.DataFrame) -> pd.DataFrame:
    pfas_columns = df.columns.difference(['county', 'gm_samp_collection_date'])
    df['total_pfas_concentration'] = df[pfas_columns].sum(axis=1)
    return df
