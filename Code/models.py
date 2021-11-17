# This file runs our modeling 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from models_helper import Dataset, Model
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

def run_modeling_wrapper(df_train, df_test, df_val, ss_target, random_seed = 33):
    '''
    '''
    # 1) Dataset object instantiation and check for most important features to use
    data = Dataset(df_train, df_test, df_val, random_seed, label='weighted_average_vote')
    data.split_features_target()
    data.data_as_arrays()
    # data.most_important_features_analysis(show=False)
    train_X, test_X, val_X, train_Y,  test_Y, val_Y = data.get_data_as_arrays()

    # 2) Create base models
    # Model 1: Baseline Model
    # baseline_model = Model(train_X, train_Y, val_X, val_Y, test_X, test_Y, 'baseline')
    # baseline_model.construct_model(LinearRegression())
    # baseline_model.train()
    # baseline_model.get_val_score()

    # Model 2: SGDRegressor Model
    linear_model = Model(train_X, train_Y, val_X, val_Y, test_X, test_Y, 'linear_sgd', target_scaler=ss_target)
    linear_model.construct_model(SGDRegressor())
    linear_model.train()
    linear_model.get_val_score()
    linear_model.get_error_in_context()

    # Model 3: RandomForestRegressor
    random_forest = Model(train_X, train_Y, val_X, val_Y, test_X, test_Y, 'random_forest', target_scaler=ss_target)
    random_forest.construct_model(RandomForestRegressor())
    random_forest.train()
    random_forest.get_val_score()
    random_forest.get_error_in_context()

    # Model 4: GradientBoostingRegressor
    random_forest = Model(train_X, train_Y, val_X, val_Y, test_X, test_Y, 'gradient_boost', target_scaler=ss_target)
    random_forest.construct_model(GradientBoostingRegressor())
    random_forest.train()
    random_forest.get_val_score()
    random_forest.get_error_in_context()

    # Model 5: AdaBoostingRegressor
    random_forest = Model(train_X, train_Y, val_X, val_Y, test_X, test_Y, 'ada_boost', target_scaler=ss_target)
    random_forest.construct_model(AdaBoostRegressor())
    random_forest.train()
    random_forest.get_val_score()
    random_forest.get_error_in_context()

    # Model 6: KNNRegressor
    random_forest = Model(train_X, train_Y, val_X, val_Y, test_X, test_Y, 'knn_regressor', target_scaler=ss_target)
    random_forest.construct_model(KNeighborsRegressor())
    random_forest.train()
    random_forest.get_val_score()
    random_forest.get_error_in_context()


    # 3) GridSearchCV and Hyperparameter Tuning

    # 4) Ensemble is all

    return


if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing', __name__)
    
