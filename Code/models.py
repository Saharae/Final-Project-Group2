# This file runs our modeling 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from models_helper import Dataset

def run_modeling_wrapper(df_train, df_test, df_val, random_seed = 33):
    '''
    '''
    # Dataset object instantiation and check for most important features to use
    data = Dataset(df_train, df_test, df_val, random_seed, label='weighted_average_vote')
    data.split_features_target()
    data.data_as_arrays()
    data.most_important_features_analysis(show=False)

    return


if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing', __name__)
    
