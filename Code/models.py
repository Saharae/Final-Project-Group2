# This file runs our modeling 
import pandas as pd
import numpy as np

from models_helper import Dataset, Model, ModelTuner, Plotter
from preprocessing_utils import get_repo_root
from preprocessing_utils import get_repo_root_w

from scipy.stats import shapiro, ttest_ind, bartlett, mannwhitneyu

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sys import platform

def run_modeling_wrapper(df_train, df_test, df_val, ss_target, df_test_untouched, random_seed = 33, run_base_estimators = False, run_model_tuning = False, fast_gridsearch = True, save_model = False):
    '''
    '''

    ####################################################
    # 1) Dataset object instantiation and check for most important features to use
    ####################################################
    df_test_ids = df_test[['imdb_title_id', 'weighted_average_vote']]
    df_test_ids['Inverse Transformed Weighted Avg Vote'] = ss_target.inverse_transform(df_test_ids['weighted_average_vote'].values.reshape(-1,1)).reshape(-1)

    df_train.drop('imdb_title_id', axis=1, inplace=True)
    df_val.drop('imdb_title_id', axis=1, inplace=True)
    df_test.drop('imdb_title_id', axis=1, inplace=True)

    data = Dataset(df_train, df_test, df_val, random_seed, label='weighted_average_vote')
    data.split_features_target()
    data.data_as_arrays()

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
        if platform == "darwin":
            base_models_plotter = Plotter(get_repo_root() + '/results/model_plots/', name='Base Untuned Models, All Features', savename='base_all_features')
        elif platform == "win32":
            base_models_plotter = Plotter(get_repo_root_w() + '\\results\\model_plots\\', name='Base Untuned Models, All Features', savename='base_all_features')
        base_models_plotter.model_comparison(mse_dict, show=False, score = 'Val. MSE', saveplot=True)
        base_models_plotter.model_comparison(rmse_dict, show=False, score = 'Val. RMSE', alt=1, saveplot=True)

    ####################################################
    # 3) GridSearchCV and Hyperparameter Tuning
    ####################################################
    
    # Hyperparameters to tune:
    # RF: 
        # n_estimators: number of trees, int default 100
        # max_depth: max depth of tree, int default None
        # min_samples_split: min samples required to split and internal node, int default 2
        # min_samples_leaf: min samples required to be at leaf node, int default 1
        # max_features: n of features to consider when looking for best split, int or {"auto", "sqrt", "log2"}, default = "auto"
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
    rf_params = {'n_estimators':[200, 300, 400, 500],
                    'min_samples_leaf':[2, 4, 8, 12], 
                    'max_features':[0.3, 0.5, 0.7],
                    'max_depth':[5, 10, 15, 25]
                    }
    
    random_forest_tuned.set_params_to_tune(rf_params)

        # 2)
    gradient_boost_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='gradient_boost_tuned', target_scaler=ss_target)
    gradient_boost_tuned.construct_model(GradientBoostingRegressor())
    gb_params = {'learning_rate':[0.01, 0.1],
            'n_estimators':[100, 200, 400], 
            'min_samples_split':[2, 4, 8, 16], 
            }
    
    gradient_boost_tuned.set_params_to_tune(gb_params)

        # 3)    
    knn_regressor_tuned = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='knn_regressor_tuned', target_scaler=ss_target)
    knn_regressor_tuned.construct_model(KNeighborsRegressor())
    knn_params = {'n_neighbors':[3, 5, 7, 9],
        'p':[1, 2]
        }
    
    knn_regressor_tuned.set_params_to_tune(knn_params)
    
    # Do GridSearchCV of models and params
    if fast_gridsearch:
        models = {random_forest_tuned.name: random_forest_tuned.model}
        param_grids = {random_forest_tuned.name: [{'max_depth':[25,], 'max_features':[0.7,], 'min_samples_leaf':[2,], 'n_estimators':[500,]}]}
    else:
        models = {random_forest_tuned.name: random_forest_tuned.model, gradient_boost_tuned.name: gradient_boost_tuned.model, knn_regressor_tuned.name:knn_regressor_tuned.model}
        param_grids = {random_forest_tuned.name: [random_forest_tuned.params_dict], gradient_boost_tuned.name: [gradient_boost_tuned.params_dict], knn_regressor_tuned.name: [knn_regressor_tuned.params_dict]}

    # Skip tuning to run faster once we already have found best model
    if run_model_tuning:
        
        if platform == "darwin":
            gridsearchcv = ModelTuner(get_repo_root() + '/results/', random_seed, X_train_val, Y_train_val, test_x=test_X, test_y=test_Y, name='gridsearchcv', target_scaler=ss_target, ps=ps, models_pipe=models, params=param_grids)
            
        elif platform == "win32":
            gridsearchcv = ModelTuner(get_repo_root_w() + '\\results\\', random_seed, X_train_val, Y_train_val, test_x=test_X, test_y=test_Y, name='gridsearchcv', target_scaler=ss_target, ps=ps, models_pipe=models, params=param_grids)
        
        best_model_df, validation_curve_blob, learning_curve_blob = gridsearchcv.do_gridsearchcv(validation_curves=False, learning_curves=False)

        mse_dict = {x:y for x,y in zip(best_model_df['model'].to_list(), best_model_df['best_score'].to_list())}

        # Generate validation, learning, and model comparison plots
        
        if platform == "darwin":
            cv_plotter = Plotter(get_repo_root() + '/results/model_plots/', name='Tuning Model', savename='tuning_model')            
            
        elif platform == "win32":
            cv_plotter = Plotter(get_repo_root_w() + '\\results\\model_plots\\', name='Tuning Model', savename='tuning_model')
        
        cv_plotter.validation_curves(validation_curve_blob, show=False, saveplot=True)
        cv_plotter.learning_curves(learning_curve_blob, show=False, saveplot=True)
        cv_plotter.model_comparison(mse_dict, show=False, score = 'Val. MSE', alt=1, saveplot=True)

    ####################################################
    # 4) Choose best model
    ####################################################
        best_params = best_model_df['best_param'][0]
        best_estimator = best_model_df['best_estimator'][0]
        
        best_model = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='best_tuned_model', target_scaler=ss_target)
        best_model.construct_model(best_estimator)
        best_model.set_params(best_params)
        best_model.train()
        
        # Save model
        if save_model:
            if platform == "darwin":
                best_model.save_model(get_repo_root() + '/results/best_params.pkl', best_params)
                best_model.save_model(get_repo_root() + '/results/best_estimator.pkl', str(best_estimator))  
            
            elif platform == "win32":
                best_model.save_model(get_repo_root_w() + '\\results\\best_params.pkl', best_params)
                best_model.save_model(get_repo_root_w() + '\\results\\best_estimator.pkl', str(best_estimator))

    else:
        # Load model
        loaded_model = Model(random_seed, name='loaded_model')
         
        if platform == "darwin":
            best_params = loaded_model.load_model(get_repo_root() + '/results/best_params.pkl')
            best_estimator = loaded_model.load_model(get_repo_root() + '/results/best_estimator.pkl')
        
        elif platform == "win32":
            best_params = loaded_model.load_model(get_repo_root_w() + '\\results\\best_params.pkl')
            best_estimator = loaded_model.load_model(get_repo_root_w() + '\\results\\best_estimator.pkl')
        
        best_estimator = eval(best_estimator)

        # Best model already found, don't just load best_model.pkl (already trained model), because the fully trained model is huge so can't upload easily to GitHub.
        # Load the untrained model and parameters and reconstruct and train
        best_model = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='best_tuned_model', target_scaler=ss_target)
        best_model.construct_model(best_estimator)
        best_model.set_params(best_params)
        best_model.train()

    ####################################################
    # 5) Evaluate best model
    ####################################################
    my_model = best_model.get_model()
    model_to_use = Model(random_seed, X_train_val, Y_train_val, val_x=None, val_y=None, test_x=test_X, test_y=test_Y, name='best_tuned_model', target_scaler=ss_target)
    model_to_use.construct_model(my_model)
    test_Y_predicted = model_to_use.evaluate(test_X)
    test_Y_predicted = ss_target.inverse_transform(test_Y_predicted.reshape(-1,1))
    
    # Calculate MSE & error in context
    rmse, mse = model_to_use.get_error(test_X, test_Y)
    rmse_ctxt, mse_ctxt = model_to_use.get_error_in_context(test_X, test_Y)

    ####################################################
    # 6) Results Evaluation:
        # Look at feature importance
        # Better than randomly guessing?
    ####################################################
    # Generate feature importance plot
    if platform == "darwin":
        results_eval = Plotter(get_repo_root() + '/results/model_plots/', name='Results Evaluation', savename='results_eval')
        
    elif platform == "win32":   
        results_eval = Plotter(get_repo_root_w() + '\\results\\model_plots\\', name='Results Evaluation', savename='results_eval')
        
    results_eval.most_important_features(train_df=df_train.iloc[:,:-1], model=model_to_use.model, show=False, saveplot=True)

    # Better than random test
    test_Y_random = np.random.uniform(1, 10, size=test_Y.shape)

    # Check for normality, Shapiro-Wilk Test: Null hypothesis that data was drawn from normal distribution
    test_Y_random_normality = shapiro(test_Y_random.reshape(-1,1))
    test_Y_predicted_normality = shapiro(test_Y_predicted.reshape(-1,1))

    # Check equal variance, Bartlett Test: Null hypothesis that all input samples are from populations with equal variances (Levene's test for non-normal populations)
    bartlett_test = bartlett(test_Y_random.reshape(-1), test_Y_predicted.reshape(-1))

    # Check distributions are independent, 2 Sample T-Test: Null hypothesis that means are equal
    t_test = ttest_ind(test_Y_random.reshape(-1,1), test_Y_predicted.reshape(-1,1), equal_var=True)
    
    # Check distributions are independent: Mann-Whitney Wilcox Test: Null hypothesis that medians are equal, non-parametric
    mannwhitney_test = mannwhitneyu(test_Y_random.reshape(-1,1), test_Y_predicted.reshape(-1,1))

    # Calculate MSE & error for random model
    mse_ctxt_random = mean_squared_error(ss_target.inverse_transform(test_Y.reshape(-1,1)), test_Y_random.reshape(-1,1))
    
    rmse_ctxt_random = np.sqrt(mse_ctxt_random)

    print('RMSE for Random Model: {}'.format(rmse_ctxt_random))
    print('MSE for Random Model: {}'.format(mse_ctxt_random))

    # Save results
    evaluation_results = {'Pure Error RMSE':rmse, 
                          'Pure Error MSE': mse, 
                          'Error in Context RMSE':rmse_ctxt, 
                          'Error in Context MSE':mse_ctxt,
                          'Error in Context RMSE - Random':rmse_ctxt_random, 
                          'Error in Context MSE - Random':mse_ctxt_random,
                          'Shapiro-Wilk Test Statistic - Predicted':test_Y_predicted_normality.statistic,
                          'Shapiro-Wilk Test P Value - Predicted':test_Y_predicted_normality.pvalue,
                          'Shapiro-Wilk Test Statistic - Random':test_Y_random_normality.statistic,
                          'Shapiro-Wilk Test P Value - Random':test_Y_random_normality.pvalue,
                          'Bartlett Test Statistic - Predicted vs Random':bartlett_test.statistic,
                          'Bartlett Test P Value - Predicted vs Random':bartlett_test.pvalue,
                          'T Test Statistic - Predicted vs Random':t_test.statistic,
                          'T Test P Value - Predicted vs Random':t_test.pvalue,
                          'Mann-Whitney U rank test Test Statistic - Predicted vs Random':mannwhitney_test.statistic,
                          'Mann-Whitney U rank test Test P Value - Predicted vs Random':mannwhitney_test.pvalue,
                          }

    # Scores and other model evaluation results
    evaluation_results_df = pd.DataFrame(evaluation_results.items(), columns=['Score Type', 'Score'])
    if platform == "darwin":
        evaluation_results_df.to_csv(get_repo_root() + '/results/best_model_evaluation_results.csv', index=False)

    elif platform == "win32":
        evaluation_results_df.to_csv(get_repo_root_w() + '\\results\\best_model_evaluation_results.csv', index=False)
        
    plot_eval_results = {'Model':['Our Model', 'Random Model'], 'MSE':[mse_ctxt, mse_ctxt_random]}
    results_eval.vs_random_bar(plot_eval_results, show=False, saveplot=True)

    test_Y = ss_target.inverse_transform(test_Y.reshape(-1,1))
    
    # Actual, Predicted, Random
    prediction_results = pd.DataFrame(np.hstack((test_Y.reshape(-1,1), test_Y_predicted.reshape(-1,1), test_Y_random.reshape(-1,1))), columns=['Actual Rating', 'Predicted Rating', 'Random Rating'])
    if platform == "darwin":
        prediction_results.to_csv(get_repo_root() + '/results/prediction_results.csv', index=False)
    elif platform == "win32":
        prediction_results.to_csv(get_repo_root_w() + '\\results\\prediction_results.csv', index=False)
    
    # Un-processed features, Actual, Predicted, Random
    predictions_with_ids = pd.DataFrame(np.hstack((df_test_ids.values, prediction_results.values)), columns=['imdb_title_id','weighted_average_vote','Inverse Transformed Weighted Avg Vote','Actual Rating','Predicted Rating','Random Rating'])
    
    df_test_untouched['imdb_title_id'] = df_test_untouched['imdb_title_id'].astype(str)
    predictions_with_ids['imdb_title_id'] = predictions_with_ids['imdb_title_id'].astype(str)

    predictions_with_ids = predictions_with_ids.merge(df_test_untouched, on='imdb_title_id', how='inner')
    
    if platform == "darwin":
        predictions_with_ids.to_csv(get_repo_root() + '/results/predictions_with_ids.csv', index=False)
    elif platform == "win32":
        predictions_with_ids.to_csv(get_repo_root_w() + '\\results\\predictions_with_ids.csv', index=False)
        
    return

if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing', __name__)
    
