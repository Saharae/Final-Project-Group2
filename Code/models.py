# This file runs our modeling 

# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# import os

from models_helper import Dataset, Model, ModelTuner, Plotter
from preprocessing_utils import get_repo_root

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.pipeline import Pipeline

def run_modeling_wrapper(df_train, df_test, df_val, ss_target, random_seed = 33):
    '''
    '''
    ####################################################
    # 1) Dataset object instantiation and check for most important features to use
    ####################################################
    data = Dataset(df_train, df_test, df_val, random_seed, label='weighted_average_vote')
    data.split_features_target()
    data.data_as_arrays()
    # data.most_important_features_analysis(show=False)
    # data.use_most_important_features(10)

    train_X, test_X, val_X, train_Y,  test_Y, val_Y = data.get_data_as_arrays()

    ####################################################
    # 2) Create base models and compare to narrow down which ones to try tuning
    ####################################################
    mse_dict = dict()
    rmse_dict = dict()

    # Model 1: SGDRegressor Model
    linear_model = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'linear_sgd', target_scaler=ss_target)
    linear_model.construct_model(SGDRegressor())
    linear_model.train()
    linear_model.get_val_score()
    rmse, mse = linear_model.get_error_in_context()

    mse_dict[linear_model.name] = mse
    rmse_dict[linear_model.name] = rmse

    # Model 2: RandomForestRegressor
    random_forest = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'random_forest', target_scaler=ss_target)
    random_forest.construct_model(RandomForestRegressor())
    random_forest.train()
    random_forest.get_val_score()
    rmse, mse = random_forest.get_error_in_context()

    mse_dict[random_forest.name] = mse
    rmse_dict[random_forest.name] = rmse

    # Model 3: GradientBoostingRegressor
    gradient_boost = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'gradient_boost', target_scaler=ss_target)
    gradient_boost.construct_model(GradientBoostingRegressor())
    gradient_boost.train()
    gradient_boost.get_val_score()
    rmse, mse = gradient_boost.get_error_in_context()

    mse_dict[gradient_boost.name] = mse
    rmse_dict[gradient_boost.name] = rmse

    # Model 4: AdaBoostingRegressor
    ada_boost = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'ada_boost', target_scaler=ss_target)
    ada_boost.construct_model(AdaBoostRegressor())
    ada_boost.train()
    ada_boost.get_val_score()
    rmse, mse = ada_boost.get_error_in_context()

    mse_dict[ada_boost.name] = mse
    rmse_dict[ada_boost.name] = rmse

    # Model 5: KNNRegressor
    knn_regressor = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'knn_regressor', target_scaler=ss_target)
    knn_regressor.construct_model(KNeighborsRegressor())
    knn_regressor.train()
    knn_regressor.get_val_score()
    rmse, mse = knn_regressor.get_error_in_context()

    mse_dict[knn_regressor.name] = mse
    rmse_dict[knn_regressor.name] = rmse

    # Plots to compare models
    base_models_plotter = Plotter(get_repo_root() + '/data/model_plots/', name='Base Untuned Models, All Features', savename='base_all_features')
    base_models_plotter.model_comparison(mse_dict, show=False, score = 'Val. MSE', saveplot=False)
    base_models_plotter.model_comparison(rmse_dict, show=False, score = 'Val. RMSE', alt=1, saveplot=False)

    ####################################################
    # 3) GridSearchCV and Hyperparameter Tuning
    ####################################################
    cv_plotter = Plotter(get_repo_root() + '/data/model_plots/', name='Tuning Model', savename='tuning_model')

    X_train_val, Y_train_val, test_X, test_Y, ps = data.get_train_val_predefined_split()

    # Models to tune:
    sgd_regressor_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='sgd_regressor_tuned', target_scaler=ss_target)
    sgd_regressor_tuned.construct_model(SGDRegressor())

    random_forest_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='random_forest_tuned', target_scaler=ss_target)
    random_forest_tuned.construct_model(RandomForestRegressor())
    # rf_params = {'n_estimators':[50, 100, 200, 300], 
    #              'max_depth':[5, 10, 20, 40], 
    #              'min_samples_split':[2, 3, 5, 6, 9], 
    #              'min_samples_leaf':[1,2,4,6], 
    #              'max_features':["auto", "sqrt", "log2"]
    #              }
    rf_params = {'n_estimators':[50, 100], 
                 'max_depth':[10, 20], 
                 'min_samples_split':[2,], 
                 'min_samples_leaf':[1,], 
                 'max_features':["auto",]
                 }
    random_forest_tuned.set_params_to_tune(rf_params)

    gradient_boost_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='gradient_boost_tuned', target_scaler=ss_target)
    gradient_boost_tuned.construct_model(GradientBoostingRegressor())

    knn_regressor_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='knn_regressor_tuned', target_scaler=ss_target)
    knn_regressor_tuned.construct_model(KNeighborsRegressor())

    models = {random_forest_tuned.name: random_forest_tuned.model}
    # models = {random_forest_tuned.name: random_forest_tuned.model, gradient_boost_tuned.name: gradient_boost_tuned.model, knn_regressor_tuned.name:knn_regressor_tuned.model, sgd_regressor_tuned.name: sgd_regressor_tuned.model}
    param_grids = {random_forest_tuned.name: [random_forest_tuned.params_dict]}

    # Hyperparameters to tune:
    # SGD:
    # RF: 
        # n_estimators: number of trees, int default 100
        # max_depth: max depth of tree, int default None
        # min_samples_split: min samples required to split and internal node, int default 2
        # min_samples_leaf: min samples required to be at leaf node, int default 1
        # max_features: n of features to consider when looking for best split, int or {"auto", "sqrt", "log2"}, default = "auto"
        # bootstrap: default = True
        # max_samples: int, if bootsrap is true, then draw max_samples
    # GB:
    # KNN:

    gridsearchcv = ModelTuner(random_seed, X_train_val, Y_train_val, test_x=test_X, test_y=test_Y, name='gridsearchcv', target_scaler=ss_target, ps=ps, models_pipe=models, params=param_grids)
    gridsearchcv.do_gridsearchcv()



    ####################################################
    # 4) Ensemble of all
    ####################################################

    return


if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing', __name__)
    
