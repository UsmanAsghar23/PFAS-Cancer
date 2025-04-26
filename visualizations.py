import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_top_ten_pfas_counties(df):
    """
    Plots the top 10 counties by average total PFAS concentration.
    """
    # Calculate average PFAS concentration per county
    avg_pfas_by_county = df.groupby('county')['total_pfas_concentration'].mean().reset_index()
    # Sort by concentration in descending order
    avg_pfas_by_county = avg_pfas_by_county.sort_values('total_pfas_concentration', ascending=False)

    # Display the results
    plt.figure(figsize=(6, 4))
    plt.bar(avg_pfas_by_county['county'][:10], avg_pfas_by_county['total_pfas_concentration'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('County')
    plt.ylabel('Average Total PFAS Concentration (ng/L)')
    plt.title('Top 10 Counties by Average Total PFAS Concentration')
    plt.tight_layout()
    return plt


def get_top_ten_cancer_counties(df):
    """
    Plots the top 10 counties by total cancer incidents.
    """
    # Get unique cancer incidents by county by first dropping duplicates
    unique_cancer_df = df.drop_duplicates(subset=['county', 'Cancer', 'Sex', 'Cancer_Incidents'])

    # Now sum up the unique cancer incidents by county
    cancer_by_county = unique_cancer_df.groupby("county")["Cancer_Incidents"].sum().reset_index()
    cancer_by_county = cancer_by_county.sort_values(by="Cancer_Incidents", ascending=False)

        # Filter out AllSite and get top 10 counties
    top_10_counties = cancer_by_county["county"][:10].tolist()

    # Filter data for top 10 counties and exclude AllSite
    plot_data = df[
        (df["county"].isin(top_10_counties)) & 
        (df["Cancer"] != "AllSite")
    ].copy()

    # Apply log transformation to Cancer_Incidents
    plot_data['Cancer_Incidents'] = np.log(plot_data['Cancer_Incidents'])

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=plot_data,
        x="county",
        y="Cancer_Incidents", 
        hue="Cancer",
        errorbar=None,
        palette="deep" # Using husl palette for better color differentiation
    )

    plt.xlabel("County")
    plt.ylabel("Log of Cancer Incidents")
    plt.title("Log-Transformed Cancer Incidents by Type in Top 10 Counties")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return plt

def hypothesis_plot(df):
    """
    Plots the relationship between total PFAS concentration and age-adjusted cancer incident rates (AAIR).
    """
    allsite_data = df[df['Cancer'] == 'AllSite'].drop_duplicates(subset=['county', 'Sex', 'AAIR'])

    aair_by_county = allsite_data.groupby('county')['AAIR'].mean().reset_index()
    pfas_by_county = df.groupby('county')['total_pfas_concentration'].first().reset_index()

    merged_data = aair_by_county.merge(pfas_by_county, on='county')
    merged_data = merged_data.sort_values('total_pfas_concentration')
    merged_data['log_AAIR'] = np.log10(merged_data['AAIR'])

    plt.figure(figsize=(8, 6))


    plt.scatter(merged_data['total_pfas_concentration'], merged_data['log_AAIR'])

  
    z = np.polyfit(merged_data['total_pfas_concentration'], merged_data['log_AAIR'], 1)
    p = np.poly1d(z)
    plt.plot(merged_data['total_pfas_concentration'], p(merged_data['total_pfas_concentration']), "r--", alpha=0.8)

    plt.xlabel('Total PFAS Concentration (ng/L)')
    plt.ylabel('Log10 of Age-Adjusted Cancer Incident Rate (AAIR)')
    plt.title('Higher PFAS Concentration Associated with Increased Age-Adjusted Cancer Incident Rates')

  
    for i, row in merged_data.iterrows():
        plt.annotate(row['county'], 
                    (row['total_pfas_concentration'], row['log_AAIR']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    return plt