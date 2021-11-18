# This file runs our modeling 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from models_helper import Dataset, Model, Plotter
from preprocessing_utils import get_repo_root

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
    # data.use_most_important_features(10)

    train_X, test_X, val_X, train_Y,  test_Y, val_Y = data.get_data_as_arrays()

    # 2) Create base models
    mse_dict = dict()
    rmse_dict = dict()

    # Model 1: Baseline Model
    ols_regression = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'ols_regression', target_scaler=ss_target)
    ols_regression.construct_model(LinearRegression())
    ols_regression.train()
    ols_regression.get_val_score()
    rmse, mse = ols_regression.get_error_in_context()
    
    mse_dict[ols_regression.name] = mse
    rmse_dict[ols_regression.name] = rmse

    # Model 2: SGDRegressor Model
    linear_model = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'linear_sgd', target_scaler=ss_target)
    linear_model.construct_model(SGDRegressor())
    linear_model.train()
    linear_model.get_val_score()
    rmse, mse = linear_model.get_error_in_context()

    mse_dict[linear_model.name] = mse
    rmse_dict[linear_model.name] = rmse

    # Model 3: RandomForestRegressor
    random_forest = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'random_forest', target_scaler=ss_target)
    random_forest.construct_model(RandomForestRegressor())
    random_forest.train()
    random_forest.get_val_score()
    rmse, mse = random_forest.get_error_in_context()

    mse_dict[random_forest.name] = mse
    rmse_dict[random_forest.name] = rmse

    # Model 4: GradientBoostingRegressor
    gradient_boost = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'gradient_boost', target_scaler=ss_target)
    gradient_boost.construct_model(GradientBoostingRegressor())
    gradient_boost.train()
    gradient_boost.get_val_score()
    rmse, mse = gradient_boost.get_error_in_context()

    mse_dict[gradient_boost.name] = mse
    rmse_dict[gradient_boost.name] = rmse

    # Model 5: AdaBoostingRegressor
    ada_boost = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'ada_boost', target_scaler=ss_target)
    ada_boost.construct_model(AdaBoostRegressor())
    ada_boost.train()
    ada_boost.get_val_score()
    rmse, mse = ada_boost.get_error_in_context()

    mse_dict[ada_boost.name] = mse
    rmse_dict[ada_boost.name] = rmse

    # Model 6: KNNRegressor
    knn_regressor = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'knn_regressor', target_scaler=ss_target)
    knn_regressor.construct_model(KNeighborsRegressor())
    knn_regressor.train()
    knn_regressor.get_val_score()
    rmse, mse = knn_regressor.get_error_in_context()

    mse_dict[knn_regressor.name] = mse
    rmse_dict[knn_regressor.name] = rmse

    # Plots to compare models
    base_models_plotter = Plotter(get_repo_root() + '/data/model_plots/', name='Base Untuned Models, All Features', savename='base_all_features')
    base_models_plotter.model_comparison(mse_dict, show=True, score = 'Val. MSE')
    base_models_plotter.model_comparison(rmse_dict, show=True, score = 'Val. RMSE', alt=1)

    # 3) GridSearchCV and Hyperparameter Tuning
    cv_plotter = Plotter(get_repo_root() + '/data/model_plots/', name='Tuning Model', savename='tuning_model')

    # 4) Ensemble of all

    return


if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing', __name__)
    
