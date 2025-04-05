import pandas as pd

def load_cancer_data(file_path: str) -> pd.DataFrame:
    """
    Load the cancer data from the specified CSV file.
    
    Args:
        file_path (str): Path to the cancer data CSV file
        
    Returns:
        pd.DataFrame: Loaded cancer data
    """
    return pd.read_csv(file_path)

def clean_cancer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the cancer data by removing missing values, selecting relevant columns,
    filtering for specific sex and year values, and calculating cancer incidents.
    
    Args:
        df (pd.DataFrame): Raw cancer data
        
    Returns:
        pd.DataFrame: Cleaned cancer data
    """
    
    # Filter for specific sex and year values
    df = df[(df['Sex'] != "Both") & (df['Years'] == "05yr")]

    # Remove rows with missing AAIR values
    df = df.dropna(subset=['AAIR'])
    
    # Select relevant columns
    df = df[['Counties', 'Sex', 'Cancer', 'PopTot', 'AAIR']]

    # Calculate cancer incidents
    df.loc[:, 'Cancer_Incidents'] = (df['AAIR'] / 100000) * df['PopTot']
    
    return df

def split_combined_counties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split combined counties into individual counties and adjust their statistics.
    
    Args:
        df (pd.DataFrame): Cancer data with combined counties
        
    Returns:
        pd.DataFrame: Cancer data with split counties
    """
    # California Populations from https://www.california-demographics.com/counties_by_population
    population_dict = {
        "Tulare": 479468,
        "Fresno": 1017162,
        "Shasta": 180366,
        "Tehama": 64896,
        "Ventura": 829590,
        "Kern": 913820,
    }

    # Derived from Information from https://www.states101.com/gender-ratios/california
    female_ratio = {
        "Tulare": 0.5,
        "Fresno": 0.5,
        "Shasta": 0.51,
        "Tehama": 0.5025,
        "Ventura": 0.4926,
        "Kern": 0.4878,
    }
    
    # Mapping of combined counties to individual counties
    combined_counties = {
        'Tulare, Fresno': ['Tulare', 'Fresno'],
        'Shasta, Tehama': ['Shasta', 'Tehama'],
        'Ventura, Kern': ['Ventura', 'Kern']
    }
    
    new_rows = []
    
    # Process each combined county
    for combined_county, counties in combined_counties.items():
        combined_rows = df[df['Counties'] == combined_county]
        
        for _, row in combined_rows.iterrows():
            total_pop = sum(population_dict[c] for c in counties)

            for county in counties:
                new_row = row.copy()
                new_row['Counties'] = county
                
                new_row['PopTot'] = (population_dict[county] / total_pop) * new_row['PopTot']

                # Adjust incidents and AAIR based on sex-specific population ratio
                new_row['Cancer_Incidents'] = (new_row['PopTot'] / total_pop) * row['Cancer_Incidents']
                new_row['AAIR'] = (new_row['Cancer_Incidents'] / new_row['PopTot']) * 100000
                new_rows.append(new_row)
    
    # Create new dataframe with split counties
    new_df = pd.DataFrame(new_rows)
    
    # Remove original combined county rows and add new individual county rows
    df = df[~df['Counties'].isin(combined_counties.keys())]
    final_df = pd.concat([df, new_df], ignore_index=True)
    
    # Rename Counties column to county for consistency
    final_df = final_df.rename(columns={'Counties': 'county'})
    
    return final_df

def process_cancer_data(file_path: str) -> pd.DataFrame:
    """
    Main function to process the cancer data from raw input to cleaned output.
    
    Args:
        file_path (str): Path to the raw cancer data CSV file
        
    Returns:
        pd.DataFrame: Processed and cleaned cancer data
    """
    # Load the data
    df = load_cancer_data(file_path)
    
    # Clean the data
    df = clean_cancer_data(df)
    
    # Split combined counties
    df = split_combined_counties(df)
    
    return df 