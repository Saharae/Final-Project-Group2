import pandas as pd
import numpy as np
import seaborn as sns
import os
import math
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

def get_repo_root_w():
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
    current_path_list = current_path.split('\\')
    try:
        repo_index = current_path_list.index(repo_name)
    except ValueError as err:
        repo_index = -2
    current_path_list = current_path_list[:repo_index+1]
    current_path = '\\'.join(current_path_list)

    if 'Final-Project-Group2' not in current_path:
        current_path+='\\Final-Project-Group2'
    
    return current_path

def load_all(base):
    '''
    Loads data from data folders.

    This is deprecated as the data should be loaded by the data downloader, but as a fall back we're keeping it
    :param base: base directory to pull from
    :return: necessary loaded data frames. ratings, movies, names, inflation, title_principals
    '''
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
    :return: cleaned money value. removes $ and nans out any values that aren't in US dollars
    '''
    if type(money) == float:
        return np.nan
    if money[0]!='$':
        return np.nan
    white = money.find(' ')
    trimmed = money[white+1:]
    return int(trimmed)

def get_primary_country(x):
    '''
    Gets the primary country from the original string of countries.
    Primary country is defined as the first listed countries since they're listed by importance

    To match ISO country codes some countries needed formatting changes

    :param x: country string from movies data frame
    :return: Single primary country with correct name
    '''
    if type(x) == float:
        return np.nan

    country = x[0]

    # reformatting where necessary based off UN country names
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
    '''
    Pulls UN country/region codes and maps the primary country name to the region. If no country then it gets "None"
    :param df: full merged movies dataset
    :return: movies dataframe with region column
    '''
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

    Not completely preprocessed, just the first step. Drops unecessary columns and does initial transformations.
    :param movies: movies dataset
    :param ratings: ratings dataset
    :param inflation: cleaned inflation dataset
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

def fit_production_company_frequency(df_train):
    '''
    Function to calculate frequency of occurence for 'production_company' feature on training data.
    This is the fit and will transform in another function on all (train, test, val) data.

    Parameters
    -----------
    df_train: training dataset

    Return
    -----------
    prod_comp_frequency: pandas df of frequency of 'production_company' column
    '''
    production_company_list = df_train['production_company'].tolist()
    prod_comp_frequency = get_frequencies(production_company_list, 'production_company')

    return prod_comp_frequency

def transform_production_company_frequency(df, production_company_fitted):
    '''
    Function to transform production_company categorical feature to a frequency sampling based numerical feature.
    
    Any missing frequencies occurence is imputed with the median of existing (training) data. 
    
    Missing frequencies can occur from test/val data not being fitted or the original dataset itself did not have enough info to calculate a frequency. 
    Regardless, we will treat these as the median of the fitted dataset. In other words, a new production_company our model hasn't seen before is treated with the median popularity/frequency of our dataset. 
    '''
    production_company_fitted['frequency'] = production_company_fitted['frequency'].fillna(production_company_fitted['frequency'].median())
    
    df = pd.merge(df, production_company_fitted, on='production_company', how='left')
    df.drop(['production_company'], axis=1, inplace=True)
    df.rename(columns = {'frequency':'production_company_frequency'}, inplace = True)

    return df

def binary_encoder_fit(df_train, col_name):
    '''
    Function to fit binary encoder instead of using OHE to reduce dimensionality to log base 2.

    Parameters
    -----------
    df_train: training dataset
    col_name: column name

    Return
    -----------
    bin_df: dataframe of binary codes per label
    '''
    unique_data = [tuple(x) for x in set(tuple(sorted(x)) for x in df_train[col_name].to_list())]
    unique_data.append(('None', 'None', 'None'))

    numbers = [i for i in range(len(unique_data))]
    unique_data_map = {x:y for x,y in zip(unique_data, numbers)}

    dict_map_bin = binary_encoder(unique_data_map)
    
    bin_df = pd.DataFrame({'label':dict_map_bin.keys(), 'binary_code':dict_map_bin.values()})

    bin_df['join_key'] = bin_df['label'].map(lambda x : [i for i in x])
    bin_df['join_key'] = bin_df['join_key'].map(lambda x : ''.join(sorted(''.join(x))))

    return bin_df

def binary_encoder_transform(df, bin_df, col_name):
    '''
    Function to perform the transformation after fitting on binary_encoder_fit.

    Parameter
    ---------
    df: dataset
    bin_df: dataframe with binary encoding per label

    Return
    ---------
    df: dataset with set 'col_name' as binary_encoded where n new columns = log_2(ceil(n of unique labels in col_name))
    '''
    bin_df_copy = bin_df.copy()
    split_df = bin_df_copy['binary_code'].str.split('', expand=True)
    split_df = split_df.iloc[:, 1:-1]
    split_df = split_df.apply(pd.to_numeric)
    
    split_df.rename(columns=lambda x: "{}_{}".format(col_name,(x-1)+1), inplace=True)
    bin_df_copy.rename(columns={'label':col_name}, inplace=True)

    bin_df_copy = pd.concat([bin_df_copy, split_df], axis=1)

    df['join_key'] = df[col_name].map(lambda x : [i for i in x])
    df['join_key'] = df['join_key'].map(lambda x : ''.join(sorted(''.join(x)))) 
    df = pd.merge(df, bin_df_copy, on='join_key', how='left')

    # If missing, impute with 'None' binary encodings
    none_encoding = bin_df_copy[bin_df_copy[col_name] == ('None','None','None')]
    for i in range(len(split_df.columns)):
        col_name_i = '{}_{}'.format(col_name,i+1)
        none_i = none_encoding[col_name_i]
        df[col_name_i].fillna(float(none_i), inplace=True)

    # Drop unnecessary columns
    try:
        df.drop(col_name, axis=1, inplace=True)
    except KeyError:
        df.drop('{}_x'.format(col_name), axis=1, inplace=True)
        df.drop('{}_y'.format(col_name), axis=1, inplace=True)

    df.drop(['binary_code'], axis=1, inplace=True)
    df.drop(['join_key'], axis=1, inplace=True)

    return df

def binary_encoder(dict_map):
    '''
    Function to take mapping dictionary where key = label, value = int category and encode the value to binary
    on log base 2.

    Parameter
    ----------
    dict_map: dictionary object where key = unique label, value = int category corresponding to label

    Return
    ----------
    dict_map_bin: dictionary object where key = unique label, value = binarized string representation of int value
    '''
    num_bits = math.log(len(dict_map),2)
    dict_map_bin = dict()

    for key, value in dict_map.items():
        dict_map_bin[key] = format(value, '0{}b'.format(math.ceil(num_bits)))

    return dict_map_bin

def get_frequencies(all_items_list, col_name):
    '''
    Function to calculate frequency of occurence of each item in a given list.

    Parameters
    ----------
    all_items_list: a list with items of type str, but can work for types int, float, bool
    col_name: str object to use as column name to represent items in list

    Return
    ----------
    frequency_df: Pandas dataframe with column col_name and column 'frequency' of frequency of occurence within input list all_items_list
    '''
    unique_set = set(all_items_list)
    total_unique_items = len(unique_set)
    
    items_frequency_dict = dict()
    for i in all_items_list:
        items_frequency_dict[i] = items_frequency_dict.get(i, 0) + 1

    items_frequency_dict = {item : count / total_unique_items for item, count in items_frequency_dict.items()}

    frequency_df = pd.DataFrame({str(col_name):items_frequency_dict.keys(), 'frequency':items_frequency_dict.values()})

    return frequency_df 

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

    cast_frequency = get_frequencies(cast_list, 'name')

    names = pd.merge(names, cast_frequency, on='name', how='left')

    return names

def transform_weighted_popularity_casts(df, popularity_fitted):
    '''
    Function to transform cast and crew to a popularity measure weighted by importance on role in movie.
    
    Any missing frequencies for cast occurence is imputed with the median of existing data. 
    
    Missing frequencies can occur from test/val data not being fitted or the original dataset itself did not have enough info to calculate a frequency. 
    Regardless, we will treat these as the median of the fitted dataset. In other words, a new person our model hasn't seen before is treated with the median popularity/frequency of our dataset. 

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

def n_words(df, col_name):
    '''
    Function to get number of words in col_name

    Parameter
    ---------
    df: dataset
    col_name: column name to get number of words

    Return
    df: dataset transformed
    '''
    df['{}_n_words'.format(col_name)] = df[col_name].map(lambda x: len(x.split(' ')) if isinstance(x, str) else 0)
    # print(df[[col_name, '{}_n_words'.format(col_name)]].head())
    return df

def ratio_long_words(df, col_name, n_letters):
    '''
    Function to get ratio of words over n_letters

    Parameter
    ---------
    df: dataset
    col_name: column name to get number of words
    n_letters: int of number of letters over to calculate ratio

    Return
    df: dataset transformed
    '''
    df['{}_ratio_long_words'.format(col_name)] = df[col_name].map(lambda x: len([i for i in x.split(' ') if len(i) > n_letters])/len(x.split(' ')) if isinstance(x, str) else 0)
    # print(df[[col_name, '{}_ratio_long_words'.format(col_name)]].head())
    return df

def ratio_vowels(df, col_name):
    '''
    Function to get ratio of vowels in string

    Parameter
    ---------
    df: dataset
    col_name: column name to get number of words

    Return
    df: dataset transformed
    '''
    vowels = ['a', 'e', 'i', 'o', 'u']
    df['{}_ratio_vowels'.format(col_name)] = df[col_name].map(lambda x: len([i for i in x if i in vowels])/len([i for i in x]) if isinstance(x, str) else 0)
    # print(df[[col_name, '{}_ratio_vowels'.format(col_name)]].head())
    return df

def ratio_interesting_characters(df, col_name):
    '''
    Function to get ratio of interesting characters in string

    Parameter
    ---------
    df: dataset
    col_name: column name to get number of words

    Return
    df: dataset transformed
    '''
    char = ['!', '?', '$', '#', '%', '*', '(', ')', '+']
    df['{}_ratio_char'.format(col_name)] = df[col_name].map(lambda x: len([i for i in x if i in char])/len([i for i in x]) if isinstance(x, str) else 0)
    # print(df[[col_name, '{}_ratio_char'.format(col_name)]].head())
    return df

def ratio_capital_letters(df, col_name):
    '''
    Function to get ratio of interesting characters in string

    Parameter
    ---------
    df: dataset
    col_name: column name to get number of words

    Return
    df: dataset transformed
    '''
    df['{}_ratio_capital_letters'.format(col_name)] = df[col_name].map(lambda x: len([i for i in x if i.isupper()])/len([i for i in x]) if isinstance(x, str) else 0)
    # print(df[[col_name, '{}_ratio_capital_letters'.format(col_name)]].head())
    return df

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

    df_train, df_test = train_test_split(data, test_size = (test_size + val_size), random_state = 100)
    df_test, df_val = train_test_split(df_test, test_size = val_size/(test_size + val_size), random_state = 100)

    return df_train, df_test, df_val

def expand_date(df, col_to_expand, keep_original = False):
    '''
    Expands the date column into 3 columns of the year, month, and day
    :param df: dataframe with the column to transform
    :param col_to_expand: name of date column
    :param keep_original: whether or not to keep the original column. if False the original column will be dropped
    :return: dataframe with expanded columns
    '''

    df[col_to_expand+'_year'] = df[col_to_expand].dt.year
    df[col_to_expand + '_month'] = df[col_to_expand].dt.month
    df[col_to_expand + '_day'] = df[col_to_expand].dt.day

    if not keep_original:
        df.drop([col_to_expand], axis = 1, inplace = True)

    return df

def get_missing(df):
    '''
    Given a dataframe this will calculate how much of each column is missing values.
    Separates the numerical and categorical values and constructs 2 dataframes with column name and percentage null
    :param df: dataframe to compute missing values from
    :return: to_impute_num - missing numerical values to_impute_cat - missing categorical values
    '''
    percent_missing = pd.DataFrame(df.isnull().sum() * 100 / len(df)).reset_index().rename(columns = {'index':'var', 0:'perc'})
    percent_missing['dtype'] = percent_missing.apply(lambda x: str(df[x['var']].dtype), axis = 1)
    to_impute_num = percent_missing[(percent_missing['dtype'].isin(['int64', 'float64'])) & (percent_missing['perc'] > 0)]
    to_impute_cat = percent_missing[(~percent_missing['dtype'].isin(['int64', 'float64'])) & (percent_missing['perc'] > 0)]

    return to_impute_num, to_impute_cat

def autobots_assemble(df_train, df_test, df_val, names, target):
    '''
    Main function to transform the data set. does individual column transformations and then imputes and scales
    :param df_train: training dataframe
    :param df_test: testing dataframe
    :param df_val: validation dataframe
    :param names: merged and cleaned names dataframe
    :param target: target variable
    :return: df_train, df_test, df_val, ss_target - scaling object used for the target,
            df - combined dataframe unscaled (for plotting mostly), df_test_untouched (original df test without any modification)
    '''

    # combine data for some transformations
    df = pd.concat([df_train, df_val, df_test], sort = False)

    # Keep copy with IMDB_ids needed later to map back predictions
    df_test_untouched = df_test.copy()
    df_test_untouched.drop('males_allages_avg_vote', axis=1, inplace=True)
    df_test_untouched.drop('females_allages_avg_vote', axis=1, inplace=True)

    # DATE TRANSFORM
    df = expand_date(df, col_to_expand = 'date_published', keep_original = False)

    # ENCODE CAT
    
    # Encode popularity of cast and crew weighted by importance of role in movie
    popularity_fitted = fit_weighted_popularity_casts(df_train, names)
    df = transform_weighted_popularity_casts(df, popularity_fitted)
    
    # Encode production company
    production_company_fitted = fit_production_company_frequency(df_train)
    df = transform_production_company_frequency(df, production_company_fitted)
    
    # Encode genres
    bin_df = binary_encoder_fit(df_train, col_name='genre')
    df = binary_encoder_transform(df, bin_df, col_name='genre')

    df.drop(['genre1'], axis = 1, inplace = True)
    df.drop(['genre2'], axis = 1, inplace = True)
    df.drop(['genre3'], axis = 1, inplace = True)

    # Encode title
    df = n_words(df, col_name='title')
    df = ratio_long_words(df, col_name='title', n_letters=4)
    df = ratio_vowels(df, col_name='title')
    df = ratio_capital_letters(df, col_name='title')

    df.drop(['title'], axis = 1, inplace = True)

    # Encode description
    df = n_words(df, col_name='description')
    df = ratio_long_words(df, col_name='description', n_letters=4)
    df = ratio_vowels(df, col_name='description')
    df = ratio_capital_letters(df, col_name='description')

    df.drop(['description'], axis = 1, inplace = True)

    # One Hot Encode Region (should be 6 regions)
    df = pd.get_dummies(df, columns = ['region'])

    # Needed this id for getting frequencies, can drop after done.
    # df.drop('imdb_title_id', axis=1, inplace=True)
    df.drop('males_allages_avg_vote', axis=1, inplace=True)
    df.drop('females_allages_avg_vote', axis=1, inplace=True)

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
    ss_features = StandardScaler()
    ss_target = StandardScaler()

    #numerical features until the cat is transformed
    vars_to_standardize = np.array(df_train.columns.drop(['weighted_average_vote','imdb_title_id']))

    # Scale features
    df_train.loc[:,vars_to_standardize] = ss_features.fit_transform(df_train[vars_to_standardize])
    df_test.loc[:,vars_to_standardize] = ss_features.transform(df_test[vars_to_standardize])
    df_val.loc[:,vars_to_standardize] = ss_features.transform(df_val[vars_to_standardize])
    
    # Scale target
    df_train['weighted_average_vote'] = ss_target.fit_transform(df_train['weighted_average_vote'].values.reshape(-1,1)).reshape(-1).tolist()
    df_test['weighted_average_vote'] = ss_target.transform(df_test['weighted_average_vote'].values.reshape(-1,1)).reshape(-1).tolist()
    df_val['weighted_average_vote'] = ss_target.transform(df_val['weighted_average_vote'].values.reshape(-1,1)).reshape(-1).tolist()

    return df_train, df_test, df_val, ss_target, df, df_test_untouched

def preprocess(ratings, movies, names, title_principals, inflation):
    '''
    preprocessing wrapper function that calls all necessary functions
    :param ratings: raw ratings dataframe
    :param movies: raw movies dataframe
    :param names: raw names dataframe
    :param title_principals: raw title principals dataframe
    :param inflation: raw inflation dataframe
    :return: df_train - clean and processed training dataset,
            df_test - clean and processed testing dataset,
            df_val - clean and processed validation dataset,
            ss_target - scaling object for the target,
            df - unsclaed combined dataframe for plotting,
            df_test_untouched - testing dataframe un-transformed
    '''
    inflation_clean = clean_inflation(inflation)
    movies = merge_and_clean_movies(movies, ratings, inflation_clean)
    names = merge_and_clean_names(names, title_principals)
    
    test_size = 0.15
    val_size = 0.15

    df_train, df_test, df_val = get_train_test_val(movies, test_size = test_size, val_size = val_size)
    df_train, df_test, df_val, ss_target, df, df_test_untouched= autobots_assemble(df_train, df_test, df_val, names, target = ['weighted_avg_vote'])

    return df_train, df_test, df_val, ss_target, df, df_test_untouched

if __name__ == "__main__":
    '''
        I think this can be cleared out - it will fail if we try to run it
    '''
    print('Executing', __name__)
    df_train, df_test, df_val = preprocess()

    # Josh added these just to double check datasets, can delete once we're confident in dataset
    print(df_train.columns)
    print(len(df_train.columns))

    print(len(df_train))
    print(len(df_test))
    print(len(df_val))

    print(df_train.head())
else:
    print('Importing:', __name__)
