# This file runs our modeling 
import pandas as pd

from models_helper import Dataset, Model, ModelTuner, Plotter
from preprocessing_utils import get_repo_root

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

def run_modeling_wrapper(df_train, df_test, df_val, ss_target, random_seed = 33, run_base_estimators = False, run_model_tuning = False, load_model = False):
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
    if run_base_estimators:
        mse_dict = dict()
        rmse_dict = dict()

        # Model 1: SGDRegressor Model
        linear_model = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'linear_sgd', target_scaler=ss_target)
        linear_model.construct_model(SGDRegressor())
        linear_model.train()
        rmse, mse = linear_model.get_error()

        mse_dict[linear_model.name] = mse
        rmse_dict[linear_model.name] = rmse

        # Model 2: RandomForestRegressor
        random_forest = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'random_forest', target_scaler=ss_target)
        random_forest.construct_model(RandomForestRegressor())
        random_forest.train()
        rmse, mse = random_forest.get_error()

        mse_dict[random_forest.name] = mse
        rmse_dict[random_forest.name] = rmse

        # Model 3: GradientBoostingRegressor
        gradient_boost = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'gradient_boost', target_scaler=ss_target)
        gradient_boost.construct_model(GradientBoostingRegressor())
        gradient_boost.train()
        rmse, mse = gradient_boost.get_error()

        mse_dict[gradient_boost.name] = mse
        rmse_dict[gradient_boost.name] = rmse

        # Model 4: AdaBoostingRegressor
        ada_boost = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'ada_boost', target_scaler=ss_target)
        ada_boost.construct_model(AdaBoostRegressor())
        ada_boost.train()
        rmse, mse = ada_boost.get_error()

        mse_dict[ada_boost.name] = mse
        rmse_dict[ada_boost.name] = rmse

        # Model 5: KNNRegressor
        knn_regressor = Model(random_seed, train_X, train_Y, val_X, val_Y, test_X, test_Y, 'knn_regressor', target_scaler=ss_target)
        knn_regressor.construct_model(KNeighborsRegressor())
        knn_regressor.train()
        rmse, mse = knn_regressor.get_error()

        mse_dict[knn_regressor.name] = mse
        rmse_dict[knn_regressor.name] = rmse

        # Plots to compare models
        base_models_plotter = Plotter(get_repo_root() + '/results/model_plots/', name='Base Untuned Models, All Features', savename='base_all_features')
        base_models_plotter.model_comparison(mse_dict, show=False, score = 'Val. MSE', saveplot=True)
        base_models_plotter.model_comparison(rmse_dict, show=False, score = 'Val. RMSE', alt=1, saveplot=True)

    ####################################################
    # 3) GridSearchCV and Hyperparameter Tuning
    ####################################################
    if run_model_tuning:
        # Hyperparameters to tune:
        # RF: 
            # n_estimators: number of trees, int default 100
            # max_depth: max depth of tree, int default None
            # min_samples_split: min samples required to split and internal node, int default 2
            # min_samples_leaf: min samples required to be at leaf node, int default 1
            # max_features: n of features to consider when looking for best split, int or {"auto", "sqrt", "log2"}, default = "auto"
            # bootstrap: default = True
            # max_samples: int, if bootsrap is true, then draw max_samples
        # GB:
            # learning_rate: float, default 0.1
            # n_estimators: int, default 100 larger number better to avoid overfitting
            # min_samples_split: int, default 2
            # min_samples_leaf: int, default 1
            # max_depth: int, default 3
            # max_features: "auto", "sqrt", "log2" 
        # KNN:
            # n_neighbors: int, default 5
            # weights: 'uniform', 'distance'
            # p: Power paramter for Minkowski metric. p=1=manhattan distance, p=2=euclidean distance

        # Combine train_val and get predefined split
        X_train_val, Y_train_val, test_X, test_Y, ps = data.get_train_val_predefined_split()

        # Models to tune and their parameters:
            # 1)
        random_forest_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='random_forest_tuned', target_scaler=ss_target)
        random_forest_tuned.construct_model(RandomForestRegressor())
        rf_params = {'n_estimators':[100, 200, 400, 600], 
                     'max_depth':[10, 20, 40, 60], 
                     'min_samples_split':[2, 5], 
                     'min_samples_leaf':[1,2,4], 
                     }
        # rf_params = { 
        #             'min_samples_leaf':[1,]
        #             }
        random_forest_tuned.set_params_to_tune(rf_params)

            # 2)
        gradient_boost_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='gradient_boost_tuned', target_scaler=ss_target)
        gradient_boost_tuned.construct_model(GradientBoostingRegressor())
        gb_params = {'learning_rate':[0.01, 0.1],
                'n_estimators':[100, 200, 400, 600], 
                'min_samples_split':[2, 6], 
                'max_depth': [10, 20, 40, 60],
                }
        # gb_params = {
        #         'max_depth': [3,]
        #         }
        gradient_boost_tuned.set_params_to_tune(gb_params)

            # 3)    
        knn_regressor_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='knn_regressor_tuned', target_scaler=ss_target)
        knn_regressor_tuned.construct_model(KNeighborsRegressor())
        knn_params = {'n_neighbors':[3, 5, 7, 9],
            'weights':['uniform', 'distance'], 
            'p':[1, 2]
            }
        # knn_params = {
        #     'p':[1,]
        #     }
        knn_regressor_tuned.set_params_to_tune(knn_params)
        
        # Do GridSearchCV of models and params
        models = {random_forest_tuned.name: random_forest_tuned.model, gradient_boost_tuned.name: gradient_boost_tuned.model, knn_regressor_tuned.name:knn_regressor_tuned.model}
        param_grids = {random_forest_tuned.name: [random_forest_tuned.params_dict], gradient_boost_tuned.name: [gradient_boost_tuned.params_dict], knn_regressor_tuned.name: [knn_regressor_tuned.params_dict]}

        gridsearchcv = ModelTuner(get_repo_root() + '/results/', random_seed, X_train_val, Y_train_val, test_x=test_X, test_y=test_Y, name='gridsearchcv', target_scaler=ss_target, ps=ps, models_pipe=models, params=param_grids)
        best_model_df, validation_curve_blob, learning_curve_blob = gridsearchcv.do_gridsearchcv()

        print(best_model_df.head(5))
        mse_dict = {x:y for x,y in zip(best_model_df['model'].to_list(), best_model_df['best_score'].to_list())}

        # Generate validation, learning, and model comparison plots
        cv_plotter = Plotter(get_repo_root() + '/results/model_plots/', name='Tuning Model', savename='tuning_model')
        cv_plotter.validation_curves(validation_curve_blob, show=False, saveplot=True)
        cv_plotter.learning_curves(learning_curve_blob, show=False, saveplot=True)
        cv_plotter.model_comparison(mse_dict, show=False, score = 'Val. MSE', alt=1, saveplot=True)

    ####################################################
    # 4) Choose best model
    ####################################################
        best_params = best_model_df['best_param'][0]
        best_estimator = best_model_df['best_estimator'][0]
        
        print(best_params)
        print(best_estimator)

        best_model = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='best_tuned_model', target_scaler=ss_target)
        best_model.construct_model(best_estimator)
        best_model.set_params(best_params)
        best_model.train()
        
        # Save model
        best_model.save_model(get_repo_root() + '/results/best_model.pkl')

        # Load model
    if load_model:
        my_model = best_model.load_model(get_repo_root() + '/results/best_model.pkl')
    else:
        my_model = best_model

    ####################################################
    # 5) Evaluate best model
    ####################################################
    model_to_use = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='best_tuned_model', target_scaler=ss_target)
    model_to_use.construct_model(my_model)
    model_to_use.evaluate(test_X)
    
    # Calculate MSE & error in context
    rmse, mse = model_to_use.get_error(test_X, test_Y)
    rmse_ctxt, mse_ctxt = model_to_use.get_error_in_context(test_X, test_Y)

    # Save results
    evaluation_results = {'Pure Error RMSE':rmse, 'Pure Error MSE': mse, 'Error in Context RMSE':rmse_ctxt, 'Error in Context MSE':mse_ctxt}
    evaluation_results_df = pd.DataFrame(evaluation_results.items(), columns=['Score Type', 'Score'])
    evaluation_results_df.to_csv(get_repo_root() + '/results/best_model_evaluation_results.csv', index=False)

    ####################################################
    # 6) Better than random? + Look at feature importance(?)
    ####################################################

    return

if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing', __name__)
    
