import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
    movies['year'] = movies['year'].replace('TV Movie 2019', 2019)

    # Merging Ratings and Movies (it merges perfectly)
    movies_full = movies.merge(ratings, on = 'imdb_title_id')

    # Merging movies and ratings with inflation to get the multiplier
    movies_full = pd.merge(movies_full, inflation, on = ['year'])

    # Renaming a misnamed column -- 'worlwide' to 'worldwide'
    movies_full.rename(columns = {'worlwide_gross_income' : 'worldwide_gross_income'}, inplace = True)

    # cleaning the money columns
    movies_full['budget'] = movies_full['budget'].apply(clean_money)
    movies_full['usa_gross_income'] = movies_full['usa_gross_income'].apply(clean_money)
    movies_full['worldwide_gross_income'] = movies_full['worldwide_gross_income'].apply(clean_money)

    # filling with 0s so we can apply the multiplier
    movies_full['budget'] = movies_full['budget'].fillna(0)
    movies_full['usa_gross_income'] = movies_full['usa_gross_income'].fillna(0)
    movies_full['worldwide_gross_income'] = movies_full['worldwide_gross_income'].fillna(0)

    # adjusting to current day money
    movies_full['budget_adjusted'] = movies_full['budget'] * movies_full['multiplier']
    movies_full['usa_gross_income_adjusted'] = movies_full['usa_gross_income'] * movies_full['multiplier']
    movies_full['worldwide_gross_income_adjusted'] = movies_full['worldwide_gross_income'] * movies_full['multiplier']

    # making 3 new columns with each individual genre (columnn 1 is always populated, 2 and 3 get None if there's only 1 or 2 listed)
    movies_full[['genre1', 'genre2', 'genre3']] = movies_full['genre'].str.split(', ', 2, expand = True)

    # converting original genre column to a list
    movies_full['genre'] = movies_full['genre'].str.split(', ')
    movies_full['country'] = movies_full['country'].str.split(', ')
    movies_full['primary_country'] = movies_full['country'].apply(get_primary_country)
    return movies_full

def US_movies(movies):
    '''
    Separates out only the US movies by the country column
    :param movies: cleaned and merged movie dataframe
    :return: movies dataframe with only the us made movies
    '''
    us_movies = movies[movies['primary_country'] == 'USA']
    return us_movies

def get_train_test(data, features, target, encode_target, test_size):
    '''
    Performs the train test split
    :param data: dataframe to pull features and values from
    :param features: list of columns to train on ex. ['genre', 'duration']
    :param target: column to test ex. ['rating']
    :param encode_target: does the target need to be encoded? boolean True or False
    :param test_size: test size to split train test with, ex. 0.3
    :return:
      feature_vals - the array of only the features
      target_vals - the array of only the target vals (if encoded it will return the encoded vals)
      X_train
      X_test
      y_train
      y_test
    '''
    feature_vals = data[features].values
    target_vals = data[target].values

    if encode_target:
        le = LabelEncoder()
        target_vals = le.fit_transform(target_vals)

    X_train, X_test, y_train, y_test = train_test_split(feature_vals, target_vals, test_size = test_size, random_state = 100)
    return feature_vals, target_vals, X_train, X_test, y_train, y_test


ratings, movies, names, inflation = load_all(base)
inflation_clean = clean_inflation(inflation)
movies = merge_and_clean_movies(movies, ratings, inflation_clean)

feature_vals, target_vals, X_train, X_test, y_train, y_test = get_train_test(movies, features = ['budget', 'country'], target = ['weighted_average_vote'], encode_target = False, test_size = 0.3)

print('finished')