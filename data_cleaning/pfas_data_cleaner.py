import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple

def load_pfas_data(file_path: str) -> pd.DataFrame:
    """
    Load the PFAS data from CSV file.
    
    Args:
        file_path (str): Path to the PFAS CSV file
        
    Returns:
        pd.DataFrame: Loaded PFAS data
    """
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filter the dataset for a specific date range.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    df['gm_samp_collection_date'] = pd.to_datetime(df['gm_samp_collection_date'])
    return df[(df['gm_samp_collection_date'] >= start_date) & 
             (df['gm_samp_collection_date'] <= end_date)]

def remove_missing_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where gm_result is NaN.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with missing results removed
    """
    return df.dropna(subset=['gm_result'])

def extract_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and sort relevant columns from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with selected columns
    """
    columns = ['gm_chemical_vvl', 'gm_result', 'gm_samp_collection_date', 
              'gm_latitude', 'gm_longitude']
    return df[columns].sort_values(by='gm_samp_collection_date', ascending=False)

def load_county_shapefile(file_path: str) -> gpd.GeoDataFrame:
    """
    Load county shapefile data.
    
    Args:
        file_path (str): Path to the shapefile
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing county boundaries
    """
    return gpd.read_file(file_path)

def calculate_county_bounds(gdf: gpd.GeoDataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate the bounds for each county.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing county boundaries
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of county bounds
    """
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
    """
    Return list of California counties.
    
    Returns:
        List[str]: List of California county names
    """
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
    """
    Determine county name from latitude and longitude coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        county_bounds (Dict[str, Dict[str, float]]): Dictionary of county bounds
        
    Returns:
        str: County name or 'Unknown'
    """
    for county, bounds in county_bounds.items():
        if (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
            bounds['min_lon'] <= lon <= bounds['max_lon']):
            return county
    return 'Unknown'

def add_county_column(df: pd.DataFrame, 
                     county_bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Add county column to DataFrame based on coordinates.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        county_bounds (Dict[str, Dict[str, float]]): Dictionary of county bounds
        
    Returns:
        pd.DataFrame: DataFrame with county column added
    """
    df['county'] = df.apply(
        lambda row: get_county_from_bounds(row['gm_latitude'], 
                                         row['gm_longitude'], 
                                         county_bounds), 
        axis=1
    )
    return df

def create_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame to wide format with PFAS chemicals as columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Wide format DataFrame
    """
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
    """
    Fill missing values using annual means and overall means.
    
    Args:
        wide_df (pd.DataFrame): Input wide format DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with filled missing values
    """
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

def apply_log_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to PFAS columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with log-transformed PFAS values
    """
    pfas_columns = df.columns.difference(['county', 'gm_samp_collection_date'])
    df[pfas_columns] = np.log1p(df[pfas_columns])
    return df

def calculate_total_pfas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total PFAS concentration for each row.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with total PFAS concentration column
    """
    pfas_columns = df.columns.difference(['county', 'gm_samp_collection_date'])
    df['total_pfas_concentration'] = df[pfas_columns].sum(axis=1)
    return df

def merge_with_cancer_data(pfas_df: pd.DataFrame, cancer_file_path: str) -> pd.DataFrame:
    """
    Merge PFAS data with cancer data.
    
    Args:
        pfas_df (pd.DataFrame): PFAS DataFrame
        cancer_file_path (str): Path to cancer data CSV
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    cancer_df = pd.read_csv(cancer_file_path)
    return pd.merge(pfas_df, cancer_df, on='county', how='inner')

def clean_pfas_data(pfas_file_path: str, shapefile_path: str, 
                   cancer_file_path: str, output_path: str) -> None:
    """
    Main function to clean and process PFAS data.
    
    Args:
        pfas_file_path (str): Path to PFAS data CSV
        shapefile_path (str): Path to county shapefile
        cancer_file_path (str): Path to cancer data CSV
        output_path (str): Path to save cleaned data
    """
    # Load and clean PFAS data
    df = load_pfas_data(pfas_file_path)
    df = filter_date_range(df, '2017-01-01', '2021-12-31')
    df = remove_missing_results(df)
    df = extract_relevant_columns(df)
    
    # Process county data
    gdf = load_county_shapefile(shapefile_path)
    county_bounds = calculate_county_bounds(gdf)
    california_counties = get_california_counties()
    california_county_bounds = {name: county_bounds[name] for name in california_counties}
    
    # Add county information
    df = add_county_column(df, california_county_bounds)
    df = df[['gm_chemical_vvl', 'gm_result', 'gm_samp_collection_date', 'county']]
    
    # Create wide format and clean data
    wide_df = create_wide_format(df)
    wide_df = fill_missing_values(wide_df)
    wide_df = apply_log_transformation(wide_df)
    wide_df = calculate_total_pfas(wide_df)
    
    # Merge with cancer data and save
    final_df = merge_with_cancer_data(wide_df, cancer_file_path)
    final_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage
    pfas_file_path = "../data/raw/pfas.csv"
    shapefile_path = "../data/raw/tl_2022_us_county.zip"
    cancer_file_path = "../data/cleaned/cleaned_cancer_dataset.csv"
    output_path = "../data/cleaned/cleaned_pfas_cancer_merged.csv"
    
    clean_pfas_data(pfas_file_path, shapefile_path, cancer_file_path, output_path) 