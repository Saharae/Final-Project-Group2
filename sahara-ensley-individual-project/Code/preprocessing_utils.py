import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_repo_root():
    '''
    Function to get the repo base path of '.../Final-Project-Group2' so anyone can run.

    Takes into account the case of running the code directly outside the repo.
    Paramaters
    ----------
    None

    Return
    ----------
    None
    '''
    repo_name = 'Final-Project-Group2'
    current_path = os.path.abspath(__file__)
    current_path_list = current_path.split('/')
    try:
        repo_index = current_path_list.index(repo_name)
    except ValueError as err:
        repo_index = -2
    current_path_list = current_path_list[:repo_index+1]
    current_path = '/'.join(current_path_list)

    if 'Final-Project-Group2' not in current_path:
        current_path+='/Final-Project-Group2'
    
    return current_path

base = get_repo_root()
# base = '/Users/sahara/Documents/GW/DataMining/Final-Project-Group2'

def load_all(base):
    ratings = pd.read_csv(f'{base}/data/IMDb ratings.csv')
    movies = pd.read_csv(f'{base}/data/IMDb movies.csv')
    names = pd.read_csv(f'{base}/data/IMDb names.csv')
    title_principals = pd.read_csv(f'{base}/data/IMDb title_principals.csv')
    inflation = pd.read_csv(f'{base}/data/CPIAUCNS_inflation.csv')

    return ratings, movies, names, inflation, title_principals

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

def clean_money(money):
    '''
    Function to pass to a column in a pandas dataframe dealing with money
    finds the space after the $ and casts to an int
    :param x:
    :return:
    '''
    if type(money) == float:
        return np.nan
    white = money.find(' ')
    trimmed = money[white+1:]
    return int(trimmed)

def get_primary_country(x):
    if type(x) == float:
        return np.nan

    country = x[0]
    country_switch = {
        'USA' : 'United States of America',
        'UK' : 'United Kingdom of Great Britain and Northern Ireland',
        'Russia': 'Russian Federation',
        'Taiwan' : 'Taiwan, Province of China',
        'Czech Republic' : 'Czechia',
        'Vietnam' : 'Viet Nam',
        'North Korea' : "Korea (Democratic People's Republic of)",
        'South Korea' : 'Korea, Republic of',
        'The Democratic Republic Of Congo':'Congo, Democratic Republic of the',
        'Moldova' : 'Moldova, Republic of',
        'Palestine': 'Palestine, State of',
        'Bolivia':'Bolivia (Plurinational State of)',
        'Syria' : 'Syrian Arab Republic',
        'Venezuela':'Venezuela (Bolivarian Republic of)',
        'Iran' : 'Iran (Islamic Republic of)',
        'Isle Of Man' : 'Isle of Man',
        'Republic of North Macedonia' : 'North Macedonia'
    }
    if country in country_switch.keys():
        country = country_switch[country]

    return country

def to_region(df):
    url = 'https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv'
    regions = pd.read_csv(url)
    regions = regions[['name', 'region']].copy()
    regions.rename(columns = {'name':'primary_country'}, inplace = True)
    df_full = df.merge(regions, on = 'primary_country', how = 'left')
    df_full['region'] = df_full['region'].fillna('None')
    return df_full


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
    movies_full = movies_full.merge(inflation, on = ['year'])

    # Renaming a misnamed column -- 'worlwide' to 'worldwide'
    movies_full.rename(columns = {'worlwide_gross_income' : 'worldwide_gross_income'}, inplace = True)


    good_cols = ['duration', 'budget', 'worldwide_gross_income', 'usa_gross_income', 'title', 'date_published', 'genre',
                 'country', 'director', 'writer', 'production_company', 'actors', 'weighted_average_vote', 'males_allages_avg_vote',
                 'females_allages_avg_vote', 'description', 'multiplier', 'imdb_title_id']
    
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
    movies_clean['genre2'] = movies_clean['genre2'].fillna('None')
    movies_clean['genre3'] = movies_clean['genre3'].fillna('None')

    # converting original genre column to a list
    movies_clean['genre'] = movies_clean['genre'].str.split(', ')
    movies_clean['country'] = movies_clean['country'].str.split(', ')

    movies_clean['primary_country'] = movies_clean['country'].apply(get_primary_country)
    movies_clean[movies_clean['primary_country'] != 'USA'][['budget', 'usa_gross_income', 'worldwide_gross_income']] = np.nan


    movies_clean = to_region(movies_clean)

    movies_clean.drop(['primary_country', 'country', 'multiplier', 'budget', 'usa_gross_income', 'worldwide_gross_income'], axis = 1, inplace = True)

    return movies_clean

def merge_and_clean_names(names, title_principals):
    '''
    Function to merge and clean 'names' and 'title_principals' datasets to get the order of importance of each cast member in movie

    Paramaters
    ----------
    names: names DF
    title_principals: title_principals DF

    Return
    ----------
    names_full: Merged and cleaned DF with ['imdb_name_id', 'name', 'ordering', 'imdb_title_id', 'category'] columns
    '''
    names_full = pd.merge(names, title_principals, on = ['imdb_name_id'])
    names_full = names_full[['imdb_name_id', 'name', 'ordering', 'imdb_title_id', 'category']]

    return names_full

def fit_production_company_frequency():

    return

def transform_production_company_frequency():

    return

def binary_encode():

    return

def fit_weighted_popularity_casts(df_train, names):
    '''
    Function to fit frequencies of occurence per cast person from training data.
    We will use a different transform function to actually perform the transformation.

    Cast includes: actors, actresses, writers, and directors

    Parameters
    ----------
    df_train: training dataset
    names: names dataset

    Return
    -------
    names: the 'fitted' names dataset on df_train. Fit in this case means finding the frequency of occurence per cast person in names.
    '''
    df_train['actors_split'] = df_train['actors'].str.split(',', expand=False)
    cast_list = flatten([x for x in df_train['actors_split'].squeeze()])
    cast_list = cast_list + df_train['director'].tolist()
    cast_list = cast_list + df_train['writer'].tolist()

    cast_names_set = set(cast_list)
    total_unique_people = len(cast_names_set)
    
    cast_frequency_dict = dict()
    for i in cast_list:
        cast_frequency_dict[i] = cast_frequency_dict.get(i, 0) + 1

    cast_frequency_dict = {person : count / total_unique_people for person, count in cast_frequency_dict.items()}

    cast_frequency = pd.DataFrame({'name':cast_frequency_dict.keys(), 'frequency':cast_frequency_dict.values()})
    
    names = pd.merge(names, cast_frequency, on='name', how='left')

    return names

def transform_weighted_popularity_casts(df, popularity_fitted):
    '''
    Function to transform cast and crew to a popularity measure weighted by importance on role in movie.
    
    Any missing frequencies for cast occurence is imputed with the median of existing data. 
    
    Missing frequencies can occur from test/val data not being fitted or the original dataset itself did not have enough info to calculate a frequency. Regardless, we will treat these as the median of the fitted dataset. In other words, a new person our model hasn't seen before is treated with the median popularity/frequency of our dataset. 

    Parameters
    -----------
    df: whole dataset
    popularity_fitted: calculated frequencies from fitted training data

    Return
    ----------
    df: transformed actors, directory, writer, and production company into encoded frequencies 
    '''
    popularity_fitted['frequency'] = popularity_fitted['frequency'].fillna(popularity_fitted['frequency'].median())
    good_cols = ['imdb_title_id', 'imdb_name_id', 'category', 'ordering', 'frequency']

    joined_df = pd.merge(popularity_fitted, df, on='imdb_title_id', how='left')
    joined_df = joined_df[good_cols]

    # Calc solution to ordering of importance weight. See function for more info
    solution = solve_linear_transformation([[1,1],[2,1]], [10/10, 9/10])

    # Calculate weighted frequency
    joined_df['weighted_frequency'] = (joined_df['ordering'] * solution[0] + solution[1]) * joined_df['frequency']

    # Actors
    actors_df = joined_df.loc[(joined_df['category'] == 'actor') | (joined_df['category'] == 'actress')]
    actors_df = actors_df.groupby(by=['imdb_title_id']).mean().reset_index()
    actors_df = actors_df[['imdb_title_id', 'weighted_frequency']]

    # Merge actors to df
    df = pd.merge(df, actors_df, on='imdb_title_id', how='left')
    df.drop(['actors'], axis=1, inplace=True)
    df.rename(columns = {'weighted_frequency':'actors_weighted_frequency'}, inplace = True)

    # Director
    director_df = joined_df.loc[(joined_df['category'] == 'director')]
    director_df = director_df.groupby(by=['imdb_title_id']).mean().reset_index()
    director_df = director_df[['imdb_title_id', 'weighted_frequency']]

    # Merge director to df
    df = pd.merge(df, director_df, on='imdb_title_id', how='left')
    df.drop(['director'], axis=1, inplace=True)
    df.rename(columns = {'weighted_frequency':'director_weighted_frequency'}, inplace = True)

    # Writer
    writer_df = joined_df.loc[(joined_df['category'] == 'writer')]
    writer_df = writer_df.groupby(by=['imdb_title_id']).mean().reset_index()
    writer_df = writer_df[['imdb_title_id', 'weighted_frequency']]

    # Merge writer to df
    df = pd.merge(df, writer_df, on='imdb_title_id', how='left')
    df.drop(['writer'], axis=1, inplace=True)
    df.rename(columns = {'weighted_frequency':'writer_weighted_frequency'}, inplace = True)

    # Checks
    # print(len(df))
    # print(len(df[~np.isnan(df['actors_weighted_frequency'])]))
    # print(len(df[~np.isnan(df['director_weighted_frequency'])]))
    # print(len(df[~np.isnan(df['writer_weighted_frequency'])]))
    
    return df

def solve_linear_transformation(X, Y):
    '''
    Function to solve our transformation for weighted frequency of type:
    
    order * m + b = weight multiplier, where m and b's are slope and intercept of our linear transformation

    ie: 1m + b = 10/10, 2m + b = 9/10, 3m + b = 8/10, ...., 10m + b = 1/10

    Parameters
    ----------
    X: list of x values
    Y: list of y values

    Return
    --------
    solution: numpy array of solution
    '''
    X = np.array(X)
    Y = np.array(Y)

    solution = np.linalg.inv(X).dot(Y)
    
    return solution

def flatten(list):
    '''
    Helper function to flatten a list of list

    Parameters
    ----------
    list: some list of list

    Return
    ---------
    dummy_list: flattened list

    '''
    dummy_list = []

    for sublist in list:
        # Try when sublist is a list so can iterate and get item except when np.nan and can't iterate.  
        try:
            for item in sublist:
                dummy_list.append(item)
        except TypeError:
            continue
    return dummy_list

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

def get_missing(df):
    percent_missing = pd.DataFrame(df.isnull().sum() * 100 / len(df)).reset_index().rename(columns = {'index':'var', 0:'perc'})
    percent_missing['dtype'] = percent_missing.apply(lambda x: str(df[x['var']].dtype), axis = 1)
    to_impute_num = percent_missing[(percent_missing['dtype'].isin(['int64', 'float64'])) & (percent_missing['perc'] > 0)]
    to_impute_cat = percent_missing[(~percent_missing['dtype'].isin(['int64', 'float64'])) & (percent_missing['perc'] > 0)]

    return to_impute_num, to_impute_cat

def autobots_assemble(df_train, df_test, df_val, names, target):
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
    # Encode popularity of cast and crew weighted by importance of role in movie
    popularity_fitted = fit_weighted_popularity_casts(df_train, names)
    df = transform_weighted_popularity_casts(df, popularity_fitted)

    # One Hot Encode Region (should be 6 regions)
    df = pd.get_dummies(df, columns = ['region'])

    # Needed this id for getting frequencies, can drop after done.
    df.drop('imdb_title_id', axis=1, inplace=True)

    # transforming description

    # re-separate data
    df_train = df.loc[df_train.index].copy()
    df_test = df.loc[df_test.index].copy()
    df_val = df.loc[df_val.index].copy()

    # impute
    to_impute_num, to_impute_cat = get_missing(df)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    df_train[to_impute_num['var']] = imp_mean.fit_transform(df_train[to_impute_num['var']])
    df_test[to_impute_num['var']] = imp_mean.transform(df_test[to_impute_num['var']])
    df_val[to_impute_num['var']] = imp_mean.transform(df_val[to_impute_num['var']])


    # standardize --
    ss = StandardScaler()

    #numerical features until the cat is transformed
    vars_to_standardize = np.array(df_train.columns.drop(['genre', 'genre1', 'genre2', 'genre3', 'description', 'title', 'production_company']))

    df_train.loc[:,vars_to_standardize] = ss.fit_transform(df_train[vars_to_standardize])
    df_test.loc[:,vars_to_standardize] = ss.transform(df_test[vars_to_standardize])
    df_val.loc[:,vars_to_standardize] = ss.transform(df_val[vars_to_standardize])

    return df_train, df_test, df_val


def preprocess(test_size = 0.3, val_size = 0.2):

    ratings, movies, names, inflation, title_principals = load_all(base)
    inflation_clean = clean_inflation(inflation)
    movies = merge_and_clean_movies(movies, ratings, inflation_clean)
    names = merge_and_clean_names(names, title_principals)

    df_train, df_test, df_val = get_train_test_val(movies, test_size = test_size, val_size = val_size)
    df_train, df_test, df_val = autobots_assemble(df_train, df_test, df_val, names, target = ['weighted_avg_vote'])

    return df_train, df_test, df_val

df_train, df_test, df_val = preprocess()

# just so I don't keep losing this line. delete later
# percent_missing = df.isnull().sum() * 100 / len(df)