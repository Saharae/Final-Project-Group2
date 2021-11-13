import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

base = '/Users/sahara/Documents/GW/DataMining/Final-Project-Group2'

def load_all(base):
    ratings = pd.read_csv(f'{base}/data/IMDb ratings.csv')
    movies = pd.read_csv(f'{base}/data/IMDb movies.csv')
    names = pd.read_csv(f'{base}/data/IMDb names.csv')
    inflation = pd.read_csv(f'{base}/data/CPIAUCNS_inflation.csv')

    return ratings, movies, names, inflation

def clean_inflation(inflation):
    '''
    Cleans the inflation dataset
    :param inflation: raw inflation dataset from this link https://fred.stlouisfed.org/series/CPIAUCNS
    :return: cleaned inflation dataset
    '''
    # cleaning the inflation data set a bit
    inflation['multiplier'] = inflation['CPIAUCNS'].iloc[-1] / inflation['CPIAUCNS']
    inflation['DATE'] = pd.to_datetime(inflation['DATE'])
    inflation['year'] = inflation['DATE'].apply(lambda x: x.year)
    inflation_simple = inflation.drop_duplicates(subset = 'year', keep = 'first')
    return inflation_simple

def clean_money(x):
    '''
    Function to pass to a column in a pandas dataframe dealing with money
    finds the space after the $ and casts to an int
    :param x:
    :return:
    '''
    if type(x) == float:
        return np.nan
    white = x.find(' ')
    trimmed = x[white+1:]
    return int(trimmed)

def get_primary_country(x):
    if type(x) == float:
        return np.nan
    return x[0]

def merge_and_clean_movies(movies, ratings, inflation):
    '''
    Returns a cleaned and merged movies table merged with the ratings
    :param movies: movies dataset
    :param ratings: ratings dataset
    :return: cleaned movies dataset
    '''

    # removing the TV movie row
    todrop = movies[movies['year'] == 'TV Movie 2019'].index
    movies.drop(todrop, axis = 0, inplace = True)

    # Merging Ratings and Movies (it merges perfectly)
    movies_full = movies.merge(ratings, on = 'imdb_title_id')

    # Merging movies and ratings with inflation to get the multiplier
    movies_full = pd.merge(movies_full, inflation, on = ['year'])

    # Renaming a misnamed column -- 'worlwide' to 'worldwide'
    movies_full.rename(columns = {'worlwide_gross_income' : 'worldwide_gross_income'}, inplace = True)


    good_cols = ['duration', 'budget', 'worldwide_gross_income', 'usa_gross_income', 'title', 'date_published', 'genre',
                 'country', 'director', 'writer', 'production_company', 'actors', 'weighted_average_vote', 'males_allages_avg_vote',
                 'females_allages_avg_vote', 'description', 'multiplier']

    movies_clean = movies_full[good_cols].copy()

    # datetime transformation for the date published
    movies_clean['date_published'] = pd.to_datetime(movies_clean['date_published'])

    # cleaning the money columns
    movies_clean.loc[:,'budget'] = movies_clean['budget'].apply(clean_money)
    movies_clean.loc[:,'usa_gross_income'] = movies_clean['usa_gross_income'].apply(clean_money)
    movies_clean.loc[:,'worldwide_gross_income'] = movies_clean['worldwide_gross_income'].apply(clean_money)

    # filling with 0s so we can apply the multiplier
    movies_clean['budget'] = movies_clean['budget'].fillna(0)
    movies_clean['usa_gross_income'] = movies_clean['usa_gross_income'].fillna(0)
    movies_clean['worldwide_gross_income'] = movies_clean['worldwide_gross_income'].fillna(0)

    # adjusting to current day money
    movies_clean['budget_adjusted'] = movies_clean['budget'] * movies_clean['multiplier']
    movies_clean['usa_gross_income_adjusted'] = movies_clean['usa_gross_income'] * movies_clean['multiplier']
    movies_clean['worldwide_gross_income_adjusted'] = movies_clean['worldwide_gross_income'] * movies_clean['multiplier']

    # putting the nans back in place of 0 for the money
    movies_clean.replace(to_replace = 0, value = np.nan, inplace = True)

    # making 3 new columns with each individual genre (columnn 1 is always populated, 2 and 3 get None if there's only 1 or 2 listed)
    movies_clean[['genre1', 'genre2', 'genre3']] = movies_clean['genre'].str.split(', ', 2, expand = True)

    # converting original genre column to a list
    movies_clean['genre'] = movies_clean['genre'].str.split(', ')
    movies_clean['country'] = movies_clean['country'].str.split(', ')
    movies_clean['primary_country'] = movies_clean['country'].apply(get_primary_country)

    movies_clean.drop(['multiplier', 'budget', 'usa_gross_income', 'worldwide_gross_income'], axis = 1, inplace = True)

    return movies_clean

def US_movies(movies):
    '''
    Separates out only the US movies by the country column
    :param movies: cleaned and merged movie dataframe
    :return: movies dataframe with only the us made movies
    '''
    us_movies = movies[movies['primary_country'] == 'USA']
    return us_movies

def get_train_test_val(data, test_size = 0.3, val_size = 0.2):
    '''
    Performs the train test split
    :param data: dataframe to pull features and values from
    :param test_size: test size to split train test with, default -  0.3
    :param val_size: test size to split train test with, default - 0.2

    :return:
      df_train
      df_val
      df_test
    '''

    df_train, df_test = train_test_split(data, test_size = test_size, random_state=100)
    df_train, df_val = train_test_split(df_train, test_size = val_size, random_state = 100)

    return df_train, df_test, df_val


def expand_date(df, col_to_expand, keep_original = False):

    df[col_to_expand+'_year'] = df[col_to_expand].dt.year
    df[col_to_expand + '_month'] = df[col_to_expand].dt.month
    df[col_to_expand + '_day'] = df[col_to_expand].dt.day

    if not keep_original:
        df.drop([col_to_expand], axis = 1, inplace = True)

    return df

def autobots_assemble(df_train, df_test, df_val, target = ''):
    '''
    Transforms the data
    :param df: data to transform
    :return: transformed data lmao
    '''

    # combine data for some transformations
    df = pd.concat([df_train, df_val, df_test], sort = False)

    # DATE TRANSFORM
    df = expand_date(df, col_to_expand = 'date_published', keep_original = False)

    # ENCODE CAT

    # re-separate data
    df_train = df.loc[df_train.index].copy()
    df_test = df.loc[df_test.index].copy()
    df_val = df.loc[df_val.index].copy()

    # impute


    # standardize --
    ss = StandardScaler()

    #numerical features until the cat is transformed
    vars_to_standardize = np.array(df_train.columns.drop(['genre', 'genre1', 'genre2', 'genre3', 'primary_country', 'description', \
                                                 'actors', 'title', 'country', 'director', 'writer', 'production_company']))
    df_train.loc[:,vars_to_standardize] = ss.fit_transform(df_train[vars_to_standardize])
    df_test.loc[:,vars_to_standardize] = ss.transform(df_test[vars_to_standardize])
    df_val.loc[:,vars_to_standardize] = ss.transform(df_val[vars_to_standardize])

    return df_train, df_test, df_val


def preprocess(test_size = 0.3, val_size = 0.2):

    ratings, movies, names, inflation = load_all(base)
    inflation_clean = clean_inflation(inflation)
    movies = merge_and_clean_movies(movies, ratings, inflation_clean)

    df_train, df_test, df_val = get_train_test_val(movies, test_size = test_size, val_size = val_size)
    df_train, df_test, df_val = autobots_assemble(df_train, df_test, df_val)

    return df_train, df_test, df_val

df_train, df_test, df_val = preprocess()
