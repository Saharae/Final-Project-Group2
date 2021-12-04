import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def plot_duration(df):
    '''
    Plots a histogram of the movie duration
    :param df: cleaned dataframe
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
    sns.histplot(data = df, x = 'duration', ax = ax, kde = True, bins = 75)
    ax.set_xlim((0, 300))
    ax.set_title('Distribution of Movie Durrations')
    sns.despine()
    return fig, ax

def plot_budget(df):
    '''
    Plots histogram of budget on log scale
    :param df: cleaned dataframe
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    sns.histplot(data = df, x = 'budget_adjusted', ax = ax, kde = True, log_scale = True)
    ax.set_title('Distribution of Movie Budget')
    ax.set_xlabel('Budget ($)')
    sns.despine()
    return fig, ax

def plot_usa_income(df):
    '''
    plots histogram of domestic income
    :param df: cleaned dataframe
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    sns.histplot(data = df, x = 'usa_gross_income_adjusted', ax = ax, kde = True, log_scale = True)
    ax.set_title('Distribution of Movie Income (USA)')
    ax.set_xlabel('USA Income ($)')
    sns.despine()
    return fig, ax

def plot_worldwide_income(df):
    '''
    plots worldwide income histogram
    :param df: cleaned dataframe
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    sns.histplot(data = df, x = 'worldwide_gross_income_adjusted', ax = ax, kde = True, log_scale = True)
    ax.set_title('Distribution of Movie Income (Worldwide)')
    ax.set_xlabel('Worldwide Income ($)')
    sns.despine()
    return fig, ax


def plot_votes(df):
    '''
    plots target feature histogram
    :param df: cleaned data frame
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
    sns.histplot(data = df, x = 'weighted_average_vote', ax = ax, bins = 25, kde = True)
    ax.set_title('Distribution of Weighted Votes')
    sns.despine()
    return fig, ax

def plot_vote_by_budget(df):
    '''
    plots votes by budget
    :param df: cleaned data frame
    :return: figure and axes object
    '''
    a = sns.lmplot(data = df, x = 'budget_adjusted', y = 'weighted_average_vote')
    a.figure.axes[0].set_title('Weighted Vote By Budget')
    a.figure.axes[0].set_ylabel('Weighted Average Vote')
    a.figure.axes[0].set_xlabel('Budget ($)')
    return a.figure, a.figure.axes[0]

def plot_worldwide_income_by_date(df):
    '''
    plots worldwide income by date
    :param df: cleaned data frame
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
    p = sns.scatterplot(data = df, x = 'date_published_year', y = 'worldwide_gross_income_adjusted', alpha = 0.2, ax = ax)
    p.set(yscale = 'log')
    ax.set_title('Worldwide Gross Income by Date')
    ax.set_ylabel('Worldwide Gross Income ($)')
    ax.set_xlabel('Year Published')
    sns.despine()
    return fig, ax

def plot_USA_income_by_date(df):
    '''
    plots usa income by date
    :param df: cleaned dataframe
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
    p = sns.scatterplot(data = df, x = 'date_published_year', y = 'usa_gross_income_adjusted', alpha = 0.2, ax = ax)
    p.set(yscale = 'log')
    ax.set_title('USA Gross Income by Date')
    ax.set_ylabel('USA Gross Income ($)')
    ax.set_xlabel('Year Published')
    sns.despine()
    return fig, ax

def plot_vote_by_decade(df):
    '''
    plots votes by decade
    :param df: cleaned data frame
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    df['decade'] = pd.cut(df['date_published_year'], np.arange(1910, 2030, 10))
    sns.barplot(data = df, x = 'decade', y = 'weighted_average_vote', ax = ax)
    labs = np.arange(1910, 2300, 10)
    t = ax.set_xticklabels(labs)
    ax.set_title('Average Vote By Decade')
    ax.set_xlabel('Decade')
    ax.set_ylabel("Weighted Average Vote")
    sns.despine()
    return fig, ax

def plot_region_count(df):
    '''
    plots counts of movies by region
    :param df: cleaned data frame
    :return: figure and axes objects
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 5))
    regions = df[['region_Africa', 'region_Americas',
                  'region_Asia', 'region_Europe', 'region_None', 'region_Oceania']].sum()
    labs = [x[x.find('_') + 1:] for x in regions.index]
    sns.barplot(x = labs, y = regions.values, ax = ax)
    ax.set_title("Movie Count by Region")
    ax.set_xlabel('Region')
    ax.set_ylabel('Count')
    sns.despine()
    return fig, ax

def plot_corr(df):
    '''
    plots correlation matrix of all numerical transformed variables
    :param df: cleaned data frame
    :return: correlation matrix and figure and axes objects of plot
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
    hmap = df[['weighted_average_vote', 'duration', 'budget_adjusted',
               'usa_gross_income_adjusted', 'worldwide_gross_income_adjusted',
               'date_published_year', 'date_published_month', 'date_published_day',
               'actors_weighted_frequency', 'director_weighted_frequency',
               'writer_weighted_frequency', 'production_company_frequency', 'title_n_words',
               'title_ratio_long_words', 'title_ratio_vowels',
               'title_ratio_capital_letters', 'description_n_words',
               'description_ratio_long_words', 'description_ratio_vowels',
               'description_ratio_capital_letters', ]].corr()

    labs = ['Vote', 'Duration', 'Budget', 'USA Income', 'World Income', 'Year', 'Month', 'Day', 'Actors', 'Director',
            'Writer', 'Production', 'Title Len', 'Title Long', 'Title Vowels', 'Title Caps', 'Desc Len', 'Desc Long',
            'Desc Vowels', 'Desc Caps']

    hmap.index = labs
    hmap.columns = labs
    sns.heatmap(hmap, vmin = -1, vmax = 1, ax = ax, cmap = 'coolwarm')

    ax.set_title('Correlation Matrix of Transformed Numeric Variables')
    return fig, ax, hmap

def statsdf(df):
    '''
    calculates stats for every variable and puts it in an organized dataframe
    :param df: cleaned dataframe
    :return: stats dataframe
    '''
    stats = df.describe().T
    percent_missing = pd.DataFrame(df.isnull().sum() * 100 / len(df)).reset_index().rename(columns = {'index': 'var', 0: 'perc'})
    stats['perc_null'] = percent_missing['perc'].to_numpy()
    stats['count_encoded'] = df.sum().to_numpy()
    stats.loc[
        ['duration', 'weighted_average_vote', 'budget_adjusted', 'usa_gross_income_adjusted', 'worldwide_gross_income_adjusted', 'date_published_year', 'date_published_month', 'date_published_day', 'actors_weighted_frequency', 'director_weighted_frequency', 'writer_weighted_frequency', 'production_company_frequency', 'title_n_words', 'title_ratio_long_words', 'title_ratio_vowels', 'title_ratio_capital_letters', 'description_n_words', 'description_ratio_long_words', 'description_ratio_vowels',
         'description_ratio_capital_letters'], ['count_encoded']] = np.nan
    stats['corr'] = df.corr()['weighted_average_vote'].to_numpy()
    stats.loc[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6', 'genre_7', 'genre_8', 'genre_9', 'genre_10', 'region_Africa', 'region_Americas', 'region_Asia', 'region_Europe', 'region_None', 'region_Oceania'], ['corr']] = np.nan
    return stats

def decade_anova(df):
    '''
    performs the anova for votes by decade
    :param df: cleaned dataframe
    :return: F statistic and P value for ANOVA
    '''
    df['decade'] = pd.cut(df['date_published_year'], np.arange(1910, 2030, 10))
    df['decade'] = df['decade'].apply(lambda x: x.left)
    melted = df[['weighted_average_vote', 'decade']].pivot(values = 'weighted_average_vote', columns = 'decade')
    F, p = stats.f_oneway(melted[1910].dropna(), melted[1920].dropna(), melted[1930].dropna(), melted[1940].dropna(), melted[1950].dropna(), melted[1960].dropna(), melted[1970].dropna(), melted[1980].dropna(), melted[1990].dropna(), melted[2000].dropna(), melted[2010].dropna())
    return F, p

def moneytary_plots(df):
    '''
    makes the pretty combined monetary plots
    :param df: cleaned data frame
    :return: figure for plot
    '''
    moneycols = df[['budget_adjusted', 'worldwide_gross_income_adjusted', 'usa_gross_income_adjusted', 'weighted_average_vote']]
    melty = moneycols.melt(id_vars = ['weighted_average_vote'], value_vars = ['budget_adjusted', 'worldwide_gross_income_adjusted', 'usa_gross_income_adjusted'], var_name = ['money'])

    a = sns.relplot(data = melty, x = 'value', y = 'weighted_average_vote', col = 'money', color = '#F3880E', kind = 'scatter', alpha = 0.2)
    a.figure.axes[0].set_ylabel('Weighted Average Vote')
    a.figure.axes[0].set_title('Budget')
    a.figure.axes[0].set_xlabel('Budget Adjusted ($)')
    a.figure.axes[0].set_xlim((0, 1e9))

    a.figure.axes[1].set_title('Worldwide Income')
    a.figure.axes[1].set_xlabel('Income Adjusted ($)')

    a.figure.axes[2].set_title('USA Income')
    a.figure.axes[2].set_xlabel('Income Adjusted ($)')

    a.figure.savefig('votebymoney.jpeg')
    return a
