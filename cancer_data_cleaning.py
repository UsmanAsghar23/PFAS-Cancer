import pandas as pd


def clean_cancer_data(df: pd.DataFrame) -> pd.DataFrame: 
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
    # California Populations from https://www.california-demographics.com/counties_by_population
    population_dict = {
        "Tulare": 479468,
        "Fresno": 1017162,
        "Shasta": 180366,
        "Tehama": 64896,
        "Ventura": 829590,
        "Kern": 913820,
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