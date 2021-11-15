# This file runs our modeling 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from models_helper import Dataset, LinearRegression, RandomForest, GradientBoost, AdaBoost, get_repo_root

############ Set random seed ############
random_seed = 33

# Set random seed in numpy
np.random.seed(random_seed)

############ Get and set current path ############
current_path = get_repo_root()

############ Get Data ############
data_file = 'movies_data_for_model.csv'
movies = Dataset(current_path + '/data/' + data_file)
print(movies.get_dataset().columns)
print(movies.get_dataset()['top1000_voters_votes'].unique())

# Split train, val, test

############ Set Up Models ############
models = {'sgd': LinearRegression()}


# if __name__ == "__main__":
#     print(__name__, 'executed')
# else:
#     print('Importing:', __name__)

# numerical - 
        #Already have: 'duration', 'budget', 'worldwide_gross_income', 'usa_gross_income', '
        #Need to calc: 'age of director at movie release', 'age of writer', 'length of director name',
        #            : 'length of movie title', 

# categorical - 
        #These are going to be too large to try and One Hot Encode. We could also just encode it from 1,2,..,n for each label
        #but we can introduce ordinal bias to non-ordinal data
            #title', 'date_published', 'genre1', 'genre2', 'genre3', country', 'language', 'director', 'writer', 'production_company', 'actors', 
        # Feature engineering and selection: Let's try to capture the essence (signals of the feature) using numerical data.

# Target label - 'weighted_average_vote'