import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(filename):
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns
    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def calculate_summary_stats(df_years, countries, indicators):
    # create a dictionary to store the summary statistics
    summary_stats = {}

    # calculate summary statistics for each indicator and country
    for indicator in indicators:
        for country in countries:
            # summary statistics for individual countries
            stats = df_years.loc[(country, indicator)].describe()
            summary_stats[f'{country} - {indicator}'] = stats

        # summary statistics for the world
        stats = df_years.loc[('World', indicator)].describe()
        summary_stats[f'World - {indicator}'] = stats

    return summary_stats


def print_summary_stats(summary_stats):
    # print the summary statistics
    for key, value in summary_stats.items():
        print(key)
        print(value)
        print()


# create scatter plots
def create_scatter_plots(df_years, indicators, countries):
    for country in countries:
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                x = df_years.loc[(country, indicators[i])]
                y = df_years.loc[(country, indicators[j])]
                plt.scatter(x, y)
                plt.xlabel(indicators[i])
                plt.ylabel(indicators[j])
                plt.title(country)
                plt.show()


def subset_data(df_years, countries, indicators):
    """
    Subsets the data to include only the selected countries and indicators.
    Returns the subsetted data as a new DataFrame.
    """
    df = df_years.loc[(countries, indicators), :]
    df = df.transpose()
    return df


def calculate_correlations(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr = df.corr()
    return corr


def visualize_correlations(corr):
    """
    Plots the correlation matrix as a heatmap using Seaborn.
    """
    sns.heatmap(corr, cmap='spring', annot=True, square=True)
    plt.title('Correlation Matrix of Indicators')
    plt.show()


def plot_line_Cereal_yield(df_years):
    country_list = ['United States', 'India', 'China', 'Canada']
    indicator = 'Cereal yield (kg per hectare)'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Cereal yield (kg per hectare)')
    plt.legend()
    plt.show()


def plot_line_Population_total(df_years):
    country_list = ['United States', 'India', 'China', 'Canada']
    indicator = 'Population, total'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Population, total')
    plt.legend()
    plt.show()


def plot_Cereal_yield(df_years):
    country_list = ['United States', 'India', 'China', 'Canada']

    Cereal_yield_indicator = 'Cereal yield (kg per hectare)'
    years = [1960, 1970, 1980, 1990, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):

        Cereal_yield_values = []
        for country in country_list:

            Cereal_yield_values.append(
                df_years.loc[(country, Cereal_yield_indicator), year])

        rects2 = ax.bar(x + width/2 + i*width/len(years), Cereal_yield_values,
                        width/len(years), label=str(year)+" "+Cereal_yield_indicator)

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(
        'Cereal yield (kg per hectare)')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_Population_total(df_years):
    country_list = ['United States', 'India', 'China', 'Canada']
    Population_total_indicator = 'Population, total'

    years = [1960, 1970, 1980, 1990, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
        Population_total_values = []
        Population_total_values = []
        for country in country_list:
            Population_total_values.append(
                df_years.loc[(country, Population_total_indicator), year])

        rects1 = ax.bar(x - width/2 + i*width/len(years), Population_total_values,
                        width/len(years), label=str(year)+" "+Population_total_indicator)

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(
        'Population, total')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    df_years, df_countries = read_data(
        r"C:\Users\Downloads\wbdata.csv")

    # Call plot_line_Cereal_yield to create the  plot
    plot_line_Cereal_yield(df_years)

    # Call plot_Population_total to create the  plot
    plot_line_Population_total(df_years)

    # Call plot_Cereal_yield
    plot_Cereal_yield(df_years)

    # Call plot_population_total
    plot_Population_total(df_years)

   # select the indicators of interest
indicators = ['Cereal yield (kg per hectare)',
              'Population, total']

# select a few countries for analysis
countries = ['United States', 'China', 'India', 'Canada']

# calculate summary statistics
summary_stats = calculate_summary_stats(df_years, countries, indicators)

# print the summary statistics
print_summary_stats(summary_stats)

# Use the describe method to explore the data for the 'United States'
us_data = df_years.loc[('United States', slice(None)), :]
us_data_describe = us_data.describe()
print("Data for United States")
print(us_data_describe)

# Use the mean method to find the mean Cereal yield (kg per hectare) for each country
Cereal_yield = df_years.loc[(
    slice(None), 'Cereal yield (kg per hectare)'), :]
Cereal_yield_mean = Cereal_yield.mean()
print("\nMean Cereal yield for each country")
print(Cereal_yield)

# Use the mean method to find the mean Population, total for each year
Population_total = df_years.loc[(slice(
    None), 'Population, total'), :]
Population_total_mean = Population_total.mean()
print("\nMean Population total for each country")
print(Population_total)

df = subset_data(df_years, countries, indicators)
corr = calculate_correlations(df)
visualize_correlations(corr)


# create scatter plots
create_scatter_plots(df_years, indicators, countries)
