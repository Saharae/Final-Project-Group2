##############################
# In main.py
##############################
import models as mdl

# Modeling
    print('Doing modeling...')
    # For modeling, there are 4 options to run in order of how long you are willing to wait.
    
    # 1) Run all processes from scratch including the time intensive gridsearch validation phase where it is 
    # testing each hyperparameter per model and will take a couple of hours.
        # Set run_model_tuning = True, run_base_estimators = True, fast_gridsearch = False
    
    # [Preferred Method When Running Modeling]
    # 2) Skip model tuning entirely and load already found best model. Roughly 5 mins runtime.
        # Set run_model_tuning = False, run_base_estimators = False, fast_gridsearch = False
    
    # 3) If you still want to go through model tuning but for a much smaller set of hyperparameters. Roughly 10-15 mins runtime.
        # Set run_model_tuning = True, fast_gridsearch = True

    # [Currently Selected Method]
    # 4) Skip modeling entirely to save most amount of time when demoing GUI.
        # Set demo = True
    
    mdl.run_modeling_wrapper(df_train, df_test, df_val, ss_target, df_test_untouched,
                             run_base_estimators = False, #Run base models comparison or not
                             run_model_tuning = False, #Run hyperparameter tuning and gridsearchcv or not
                             fast_gridsearch = False, #Skip most of gridsearchcv to run faster
                             save_model = True, #Save best model results or not
                             demo = True) #Demo = True will skip all of modeling since we already have results

##############################
# In preprocessing_utils.py
##############################
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