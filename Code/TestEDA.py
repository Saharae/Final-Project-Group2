import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_duration(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
    sns.histplot(data = df, x = 'duration', ax = ax, kde = True, bins = 75)
    ax.set_xlim((0, 300))
    ax.set_title('Distribution of Movie Durrations')
    sns.despine()
    return fig, ax

def plot_budget(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    sns.histplot(data = df, x = 'budget_adjusted', ax = ax, kde = True, log_scale = True)
    ax.set_title('Distribution of Movie Budget')
    ax.set_xlabel('Budget ($)')
    sns.despine()
    return fig, ax

def plot_usa_income(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    sns.histplot(data = df, x = 'usa_gross_income_adjusted', ax = ax, kde = True, log_scale = True)
    ax.set_title('Distribution of Movie Income (USA)')
    ax.set_xlabel('USA Income ($)')
    sns.despine()
    return fig, ax

def plot_worldwide_income(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
    sns.histplot(data = df, x = 'worldwide_gross_income_adjusted', ax = ax, kde = True, log_scale = True)
    ax.set_title('Distribution of Movie Income (Worldwide)')
    ax.set_xlabel('Worldwide Income ($)')
    sns.despine()
    return fig, ax


def plot_votes(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
    sns.histplot(data = df, x = 'weighted_average_vote', ax = ax, bins = 25, kde = True)
    ax.set_title('Distribution of Weighted Votes')
    sns.despine()
    return fig, ax

def plot_vote_by_budget(df):
    a = sns.lmplot(data = df, x = 'budget_adjusted', y = 'weighted_average_vote')
    a.figure.axes[0].set_title('Weighted Vote By Budget')
    a.figure.axes[0].set_ylabel('Weighted Average Vote')
    a.figure.axes[0].set_xlabel('Budget ($)')
    return a.figure, a.figure.axes[0]

def plot_worldwide_income_by_date(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
    p = sns.scatterplot(data = df, x = 'date_published_year', y = 'worldwide_gross_income_adjusted', alpha = 0.2, ax = ax)
    p.set(yscale = 'log')
    ax.set_title('Worldwide Gross Income by Date')
    ax.set_ylabel('Worldwide Gross Income ($)')
    ax.set_xlabel('Year Published')
    sns.despine()
    return fig, ax

def plot_USA_income_by_date(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
    p = sns.scatterplot(data = df, x = 'date_published_year', y = 'usa_gross_income_adjusted', alpha = 0.2, ax = ax)
    p.set(yscale = 'log')
    ax.set_title('USA Gross Income by Date')
    ax.set_ylabel('USA Gross Income ($)')
    ax.set_xlabel('Year Published')
    sns.despine()
    return fig, ax

def plot_vote_by_decade(df):
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