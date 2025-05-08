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
    Plots the top 10 counties by total cancer incidents using AAIR directly (no log transformation).
    """
    unique_cancer_df = df.drop_duplicates(subset=['county', 'Cancer', 'Sex', 'Cancer_Incidents'])

    cancer_by_county = unique_cancer_df.groupby("county")["Cancer_Incidents"].sum().reset_index()
    cancer_by_county = cancer_by_county.sort_values(by="Cancer_Incidents", ascending=False)

    top_10_counties = cancer_by_county["county"][:10].tolist()

    plot_data = df[
        (df["county"].isin(top_10_counties)) & 
        (df["Cancer"] != "AllSite")
    ].copy()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_data,
        x="county",
        y="AAIR", 
        hue="Cancer",
        errorbar=None,
        palette="deep",
        width=0.9
    )

    plt.xlabel("County")
    plt.ylabel("AAIR")
    plt.title("Age-Adjusted Incident Rate(AAIR) by Cancer Type in Top 10 Counties")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8}) 
    plt.tight_layout()

    return plt

def hypothesis_plot(df):
    """
    Plots the relationship between age-adjusted cancer incident rates (AAIR) and total PFAS concentration.
    """
    allsite_data = df[df['Cancer'] == 'AllSite'].drop_duplicates(subset=['county', 'Sex', 'AAIR'])

    aair_by_county = allsite_data.groupby('county')['AAIR'].mean().reset_index()
    pfas_by_county = df.groupby('county')['total_pfas_concentration'].first().reset_index()

    merged_data = aair_by_county.merge(pfas_by_county, on='county')
    merged_data = merged_data.sort_values('AAIR')
    merged_data['log_PFAS'] = np.log10(merged_data['total_pfas_concentration'])

    plt.figure(figsize=(8, 5))

    plt.scatter(merged_data['AAIR'], merged_data['log_PFAS'])

    z = np.polyfit(merged_data['AAIR'], merged_data['log_PFAS'], 1)
    p = np.poly1d(z)
    plt.plot(merged_data['AAIR'], p(merged_data['AAIR']), "r--", alpha=0.8)

    plt.xlabel('Age-Adjusted Cancer Incident Rate (AAIR)')
    plt.ylabel('Log10 of Total PFAS Concentration (ng/L)')
    plt.title('Higher AAIR Associated with Increased Total PFAS Concentration')

    for i, row in merged_data.iterrows():
        plt.annotate(row['county'], 
                     (row['AAIR'], row['log_PFAS']), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    return plt