# This file contains the models that we used for our dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV, validation_curve, learning_curve
class Dataset:
    def __init__(self, train_df, test_df, val_df, random_seed, label):
        '''
        Init method, set attributes, set numpy random seed

        Parameters
            self: instance of object
            train_df: training dataframe
            test_df: testing dataframe
            val_df: validation dataframe
            random_seed: random_seed integer number
            label: str of label column name

        Return
            None
        '''
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.label = label
        self.random_seed = random_seed

        # Set random seed in numpy
        np.random.seed(self.random_seed)

    def get_train_test_val_dfs(self):
        '''
        Method to get a train, test, val as pd dataframes

        Parameters
            self: instance of object
        
        Return
            self.train_df: training dataframe attributed to instance of object
            self.test_df: testing dataframe attributed to instance of object
            self.val_df: validation dataframe attributed to instance of object
        '''
        return self.train_df, self.test_df, self.val_df

    def split_features_target(self):
        '''
        Method to split train, test, val data to features and target per dataset

        Parameters
            self: instance of object

        Return
            None
        '''
        self.train_df_X = self.train_df.drop([self.label], axis=1)
        self.test_df_X = self.test_df.drop([self.label], axis=1)
        self.val_df_X = self.val_df.drop([self.label], axis=1)
        
        self.train_df_Y = self.train_df[self.label]
        self.test_df_Y = self.test_df[self.label]
        self.val_df_Y = self.val_df[self.label]

        return

    def data_as_arrays(self):
        '''
        Method to convert data from pandas to np arrays/matrices for training

        Parameters
            self: instance of object
        
        Return
            None
        '''
        self.train_X = self.train_df_X.values
        self.test_X = self.test_df_X.values
        self.val_X = self.val_df_X.values

        self.train_Y = self.train_df_Y.values
        self.test_Y = self.test_df_Y.values
        self.val_Y = self.val_df_Y.values

        return

    def get_data_as_arrays(self):
        '''
        Method to return data as np arrays
        
        Parameters
            self: instance of object
        
        Return
            self.train_X: training X values
            self.test_X: testing X values
            self.val_X: validation X values
            self.train_Y: training Y values
            self.test_Y: testing Y values
            self.val_Y: validation Y values

        '''
        return self.train_X, self.test_X, self.val_X, self.train_Y,  self.test_Y, self.val_Y

    def get_train_val_predefined_split(self):
        '''
        Method to combine train and validation dataset into one and set predefined split for CV

        Parameters:
            self: instance of object

        Return:
            self.X_train_val: Combined train and val dataset X
            self.Y_train_val: Combined train and val dataset Y
            self.test_X: Test dataset X
            self.test_Y: Test dataset Y
            self.ps: predefined split for CV
        '''
        self.X_train_val = np.vstack((self.train_X, self.val_X))
        self.Y_train_val = np.vstack((self.train_Y.reshape(-1,1), self.val_Y.reshape(-1,1))).reshape(-1)

        # Get indexes of training and validation: -1 means not be used in CV
        self.train_val_idxs = np.append(np.full(self.train_X.shape[0], -1), np.full(self.val_X.shape[0], 0))
        self.ps = PredefinedSplit(self.train_val_idxs)

        return self.X_train_val, self.Y_train_val, self.test_X, self.test_Y, self.ps

class Model:
    def __init__(self, random_seed, train_x=None, train_y=None, val_x=None, val_y=None, test_x=None, test_y=None, name=None, target_scaler=None):
        '''
        Init method for Model class

        Parameters
            self: instance of object
            random_seed: random_seed integer number
            train_X: training X values
            test_X: testing X values
            val_X: validation X values
            train_Y: training Y values
            test_Y: testing Y values
            val_Y: validation Y values
            name: nickname (str) for model
            target_scaler: sklearn scaler object used in preprocessing to scale target data
            ps: predefined split for training and validation

        Return
            None   
        '''
        self.random_seed = random_seed
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.name = name
        self.scaler = target_scaler

        return

    def save_model(self, filename, model):
        '''
        Method to save object model

        Parameters
            self: instance of obejct
            model_name: model name to save

        Return
            None
        '''
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

        print('Saved model: {}, {}'.format(self.name, model))
        return
    
    def load_model(self, filename):
        '''
        Method to load object model

        Parameters
            self: instance of obejct
            filename: filename to open

        Return
            self.model: model object
        '''
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)

        print('Loaded model: {}, {}'.format(self.name, self.model))

        return self.model
    
    def evaluate(self, test_x=[]):
        '''
        Method to evaluate model on test data

        Parameters
            self: instance of obejct

        Return
            self.model: model object
        '''
        if len(test_x) > 0:
            self.test_x = test_x

        self.test_y_predict = self.model.predict(self.test_x)
        return self.test_y_predict

    def train(self):
        '''
        Method to train model

        Parameters
            self: instance of object
        
        Return
            None
        '''
        # Set model's random state if param available
        if 'random_state' in self.model.get_params().keys():
            self.model.set_params(random_state=self.random_seed)

        # Train
        self.model.fit(self.train_x, self.train_y)
        return

    def get_val_score(self):
        '''
        Method to get score for this estimator
        Parameters
            self: instance of object
        
        Return
            self.val_score: validation score of model
        '''
        self.val_score = self.model.score(self.val_x, self.val_y)
        print(self.val_score)
        return self.val_score   

    def get_params(self):
        '''
        Method to get parameters for this estimator
        Parameters
            self: instance of object
        
        Return
            self.model.get_params(): dictionary of paramaters and default values
        '''
        print(self.model.get_params())
        return self.model.get_params()

    def get_model_nickname(self):
        '''
        Method to return the model nickname assigned during object instantiation

        Parameters
            self: instance of object
        
        Return
            self.name: nickname (str) for model
        '''
        print('Model: ', self.name)
        return self.name

    def get_model(self):
        '''
        Method to return the model for object

        Parameters
            self: instance of object
        
        Return
            self.model: sklearn model object
        '''
        print(self.model)
        return self.model

    def construct_model(self, model):
        '''
        Method to construct model (instantiate sklearn model)

        Parameters
            self: instance of object
            model: of type sklearn model, ie: model = LinearRegression()
        
        Return
            None
        '''
        self.model = model
        return

    def set_score(self, score):
        '''
        Method to set the score used for this model

        Parameters
            self: instance of object
            score: score used for model
        
        Return
            None
        '''
        self.score = score
        return

    def get_error_in_context(self, val_x=[], val_y=[]):
        '''
        Method to get the error of model in context to target 

        Parameters
            self: instance of object
            val_x: test/validation features, default None to default to self.val_x
            val_y: test/validation target, default None to default to self.val_y
        
        Return
            self.rmse: calculate RMSE after inverse scaling transformation
            self.mse_val: calculate MSE after inverse scaling transformation
        '''
        if len(val_x) > 0:
            self.val_x = val_x
            self.val_y = val_y

        self.val_y_predict = self.model.predict(self.val_x)
        self.inv_val_y_predict = self.scaler.inverse_transform(self.val_y_predict.reshape(-1,1))
        
        self.inv_val_y = val_y.reshape(-1,1)
        self.inv_val_y = self.scaler.inverse_transform(self.val_y.reshape(-1,1))

        self.mse_val = mean_squared_error(self.inv_val_y, self.inv_val_y_predict)

        self.rmse = np.sqrt(self.mse_val)
        print('RMSE for {}: {}'.format(self.name, self.rmse))
        print('MSE for {}: {}'.format(self.name, self.mse_val))
        return self.rmse, self.mse_val

    def get_error(self, val_x=[], val_y=[]):
        '''
        Method to get the error of model 

        Parameters
            self: instance of object
            val_x: test/validation features, default None to default to self.val_x
            val_y: test/validation target, default None to default to self.val_y
        
        Return
            self.rmse: calculate RMSE 
            self.mse_val: calculate MSE
        '''
        if len(val_x) > 0:
            self.val_x = val_x
            self.val_y = val_y
        
        self.val_y_predict = self.model.predict(self.val_x)

        self.mse_val = mean_squared_error(self.val_y, self.val_y_predict)

        self.rmse = np.sqrt(self.mse_val)
        print('RMSE for {}: {}'.format(self.name, self.rmse))
        print('MSE for {}: {}'.format(self.name, self.mse_val))
        return self.rmse, self.mse_val

    def set_params_to_tune(self, params_dict):
        '''
        Method to set the paramaters to test for tuning the model

        Parameters
            self: instance of object
            params_dict: dict of param as key and list of values to try as list

        Return
            None
        '''
        self.params_dict = params_dict

    def set_params(self, params_dict):
        '''
        Method to set parameters to model

        Parameters
            self: instance of object
            params_dict: dict of param as key and list of values to try as list

        Return
            None
        '''
        self.model.set_params(**params_dict)

class ModelTuner(Model):
    def __init__(self, path, random_seed, train_x, train_y, test_x=None, test_y=None, name=None, target_scaler=None, ps=None, models_pipe=None, params=None):
        '''
        Init method for ModelTuner, child of Model

        Parameters
            self: instance of object
            path: path to save results to
            random_seed: random_seed integer number
            train_X: training X values, can include val data if provided ps
            test_X: testing X values
            train_Y: training Y values, can include val data if provided ps
            test_Y: testing Y values
            name: nickname (str) for model
            target_scaler: sklearn scaler object used in preprocessing to scale target data
            ps: predefined split for training and validation
            models_pipe: Pipeline of models
            params: parameter grids for gridsearchCV, dict of {modelname, [param dict]}

        Return
            None   
        '''
        super().__init__(random_seed, train_x, train_y, test_x=test_x, test_y=test_y, name=name, target_scaler=target_scaler)

        self.path = path
        self.ps = ps
        self.models_pipe = models_pipe
        self.params = params

        self.make_directory()
        return
    
    def make_directory(self):
        '''
        Helper method to make directory path if not created

        Parameters
            self: instance of object
            path: path of directory

        Return
            None
        '''
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        return
    
    def do_gridsearchcv(self, save_results=True, validation_curves=True, learning_curves=True):
        '''
        Method to perform GridSearchCV to tune hyperparameters of select models

        Parameters
            self: instance of object

        Return
        '''
        self.validation_curve_blob = dict()
        self.learning_curve_blob = dict()
        best_model = []
        
        for model in self.models_pipe.keys():

            gs = GridSearchCV(estimator=self.models_pipe[model], 
                              param_grid=self.params[model], 
                              scoring='neg_mean_squared_error', 
                              n_jobs=3, 
                              cv=self.ps, 
                              return_train_score=True, 
                              verbose=3)
            
            gs.fit(self.train_x, self.train_y)

            # Update best_model
            best_model.append([gs.best_score_, gs.best_params_, gs.best_estimator_, model])
            
            # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
            cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
            
            # Get the important columns in cv_results
            important_columns = ['rank_test_score',
                                'mean_test_score', 
                                'std_test_score', 
                                'mean_train_score', 
                                'std_train_score',
                                'mean_fit_time', 
                                'std_fit_time',                        
                                'mean_score_time', 
                                'std_score_time']
            
            # Move the important columns ahead
            cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

            # Get Validation Curve data
            if validation_curves:
                param_dict = dict()
                for param in self.params[model][0].keys():
                    train_scores_vc, valid_scores_vc = validation_curve(self.models_pipe[model], 
                                                                self.train_x, 
                                                                self.train_y, 
                                                                param_name=param,
                                                                param_range=self.params[model][0][param],
                                                                cv=self.ps,
                                                                scoring='neg_mean_squared_error')
                    param_dict[param] = [self.params[model][0][param], train_scores_vc, valid_scores_vc]

                # Format: {model_name: [{param: [range, train_scores, valid_scores]}]}
                self.validation_curve_blob[model] = [param_dict]

            # Get Learning Curve Data
            if learning_curves:
                train_sizes, train_scores_lc, valid_scores_lc, fit_times, score_times = learning_curve(self.models_pipe[model], 
                                                                            self.train_x, 
                                                                            self.train_y, 
                                                                            train_sizes=[500,1000,10000,30000], 
                                                                            cv=self.ps,
                                                                            scoring='neg_mean_squared_error',
                                                                            return_times=True)

                # Format: {model_name: [train_sizes, train_scores_lc, valid_scores_lc, fit_times, score_times]}
                self.learning_curve_blob[model] = [train_sizes, train_scores_lc, valid_scores_lc, fit_times, score_times]

        # Sort best_model in descending order of the best_score_
        best_model = sorted(best_model, key=lambda x : x[0], reverse=True)

        self.best_model_df = pd.DataFrame(best_model, columns=['best_score', 'best_param', 'best_estimator', 'model'])
        self.best_model_df['best_score'] = self.best_model_df['best_score']*-1

        if save_results:
            self.best_model_df.to_csv(self.path+'gridsearchcv_results.csv', index=False)
        
        return self.best_model_df, self.validation_curve_blob, self.learning_curve_blob
class Plotter:
    def __init__(self, path, name, savename):
        '''
        Init method for Plotter

        Parameters:
            path: path to save plots to
            name: title of plots
            savename: str to save files to
        
        Return:
            None
        '''
        self.path = path
        self.name = name
        self.savename = savename

        self.make_directory()
        return
    
    def model_comparison(self, score_dict, score, saveplot=True, show=False, alt=0):
        '''
        Method to plot a bar chart of models' avg scores

        Parameters
            self: instance of object
            score_dict: dict of scores {model:score}
            score: score name for ylabel, str
            saveplot: default True, boolean to save plot or not
            show: default True, boolean to show plot or not
            alt: default 0, how many alternates of this method for self object we want to save. alt = 1 will add a '_1' to end of filename.

        Return
            None
        '''
        models = list(score_dict.keys())
        scores = list(score_dict.values())

        plt.figure(figsize = (10,5))
        plt.bar(models, scores, color = 'tab:orange', width = 0.4)
        plt.xlabel("Models")
        plt.ylabel(score)
        plt.title('Model Performance: ' + self.name)
            
        if saveplot:
            if alt == 0:
                plt.savefig(self.path + 'model_comparison' + '_' + self.savename + '.png')
            else:
                plt.savefig(self.path + 'model_comparison' + '_' + self.savename + str(alt) + '.png')
        
        if show:
            plt.show()
        
        plt.close()
        
        return
    
    def learning_curves(self, lc_results, saveplot=True, show=False, alt=0):
        '''
        Method to plot learning curves from CV results

        Parameters
            lc_results: Results to plot learning curves of format: {model_name: [train_sizes, train_scores_lc, valid_scores_lc, fit_times]}
            saveplot: default True, boolean to save plot or not
            show: default True, boolean to show plot or not
            alt: default 0, how many alternates of this method for self object we want to save. alt = 1 will add a '_1' to end of filename.
        '''
        if lc_results:
            for model in lc_results.keys():
                lc_data = lc_results[model]

                # Create subplots
                fig, axs = plt.subplots(3, 1, figsize=(15, 10))
                plt.subplots_adjust(hspace=0.5)
                fig.suptitle("Learning Curve for {}".format(model), fontsize=18, y=0.98)

                train_scores_mean = np.mean(lc_data[1], axis=1)*-1
                train_scores_std = np.std(lc_data[1], axis=1)*-1
                val_scores_mean = np.mean(lc_data[2], axis=1)*-1
                val_scores_std = np.std(lc_data[2], axis=1)*-1
                fit_times_mean = np.mean(lc_data[3], axis=1)
                fit_times_std = np.std(lc_data[3], axis=1)
                train_sizes = lc_data[0]

                # Plot learning curve
                axs[0].grid()
                axs[0].fill_between(
                    train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1,
                    color="r",
                )
                axs[0].fill_between(
                    train_sizes,
                    val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std,
                    alpha=0.1,
                    color="g",
                )
                axs[0].plot(
                    train_sizes, train_scores_mean, "o-", color="tab:blue", label="Training score"
                )
                axs[0].plot(
                    train_sizes, val_scores_mean, "o-", color="tab:orange", label="Validation score"
                )
                axs[0].legend(loc="best")
                axs[0].set_xlabel("Training examples", fontsize=8)
                axs[0].set_ylabel("MSE", fontsize=8)
                axs[0].set_title("Learning Curve", fontsize=10)
                axs[0].patch.set_facecolor('tab:gray')
                axs[0].patch.set_alpha(0.15)

                # Plot n_samples vs fit_times
                axs[1].grid()
                axs[1].plot(train_sizes, fit_times_mean, "o-")
                axs[1].fill_between(
                    train_sizes,
                    fit_times_mean - fit_times_std,
                    fit_times_mean + fit_times_std,
                    alpha=0.1,
                )
                axs[1].set_xlabel("Training examples", fontsize=8)
                axs[1].set_ylabel("fit_times", fontsize=8)
                axs[1].set_title("Scalability of the model", fontsize=10)
                axs[1].patch.set_facecolor('tab:gray')
                axs[1].patch.set_alpha(0.15)

                # Plot fit_time vs score
                axs[2].grid()
                axs[2].plot(fit_times_mean, val_scores_mean, "o-")
                axs[2].fill_between(
                    fit_times_mean,
                    val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std,
                    alpha=0.1,
                )
                axs[2].set_xlabel("fit_times", fontsize=8)
                axs[2].set_ylabel("Score", fontsize=8)
                axs[2].set_title("Performance of the model", fontsize=10)
                axs[2].patch.set_facecolor('tab:gray')
                axs[2].patch.set_alpha(0.15)

                if saveplot:
                    if alt == 0:
                        plt.savefig(self.path + 'learning_curves' + '_' + model + '.png')
                    else:
                        plt.savefig(self.path + 'learning_curves' + '_' + model + str(alt) + '.png')
                
                if show:
                    plt.show()

                plt.close()

        return

    def validation_curves(self, vc_results, saveplot=True, show=False, alt=0):
        '''
        Method to plot validation curves from GridsearchCV results
        
        Parameters
            vc_results: Results to plot learning curves of format: {model_name: [{param: [range, train_scores, valid_scores]}]}
            saveplot: default True, boolean to save plot or not
            show: default True, boolean to show plot or not
            alt: default 0, how many alternates of this method for self object we want to save. alt = 1 will add a '_1' to end of filename.
        '''
        if vc_results:
            for model in vc_results.keys():
                # Dictionary of {param: [range, train_scores, valid_scores]}
                params = vc_results[model][0]

                # Number of parameters that we have
                n_params = len(params.keys())
                
                # Create subplots
                fig, axs = plt.subplots(nrows=math.ceil(n_params/2), ncols=2, figsize=(15, 10))
                plt.subplots_adjust(hspace=0.5)
                fig.suptitle("Validation Curves for {}".format(model), fontsize=18, y=0.98)

                for param, ax in zip(params.keys(), axs.ravel()):
                    train_range = params[param][0]
                    train_scores = params[param][1]*-1
                    valid_scores = params[param][2]*-1

                    train_scores_mean = np.mean(train_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    val_scores_mean = np.mean(valid_scores, axis=1)
                    val_scores_std = np.std(valid_scores, axis=1)

                    # Train scores
                    ax.scatter(train_range, train_scores, c='tab:blue')
                    ax.plot(train_range, train_scores, c='tab:blue', label='Training Score')
                    ax.fill_between(train_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="tab:blue", lw=2)
                    
                    # Test scores
                    ax.scatter(train_range, valid_scores, c='tab:orange')
                    ax.plot(train_range, valid_scores, c='tab:orange', label='Validation Score')
                    ax.fill_between(train_range, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.2, color="tab:orange", lw=2)
                    
                    ax.set_title("{}".format(param), fontsize=10)
                    ax.set_ylabel('')
                    ax.legend(loc='best')
                    ax.patch.set_facecolor('tab:gray')
                    ax.patch.set_alpha(0.15)
                
                fig.text(0.5, 0.04, 'Hyperparameter Values', ha='center')
                fig.text(0.04, 0.5, 'MSE', va='center', rotation='vertical')

                if saveplot:
                    if alt == 0:
                        plt.savefig(self.path + 'validation_curves' + '_' + model + '.png')
                    else:
                        plt.savefig(self.path + 'validation_curves' + '_' + model + str(alt) + '.png')
                
                if show:
                    plt.show()

                plt.close()

        return

    def most_important_features(self, train_df, model, saveplot=True, show=False, alt=0):
        '''
        Method to perform analysis to get most important features from RandomForest.

        Parameters
            self: instance of object
            train_df: training dataframe
            model: model to use
            saveplot: default True, boolean to save plot or not
            show: default True, boolean to show plot or not
            alt: default 0, how many alternates of this method for self object we want to save. alt = 1 will add a '_1' to end of filename.
        
        Return
            None
        '''
        print('Finding most important features...')

        self.feature_importance = pd.DataFrame(np.hstack((np.array(train_df.columns).reshape(-1,1), model.feature_importances_.reshape(-1,1))), columns=['Features', 'Importance'])
        self.feature_importance = self.feature_importance.sort_values(ascending=False, by='Importance').reset_index(drop=True)

        plt.figure(figsize=(10, 5))
        
        plt.bar(self.feature_importance['Features'], self.feature_importance['Importance'], color='tab:orange')
        
        plt.xlabel('Features')
        plt.ylabel('Mean Decrease in Impurity (Gini)')
        plt.title('Feature Importance - Sorted')
        plt.xticks(rotation=90)

        plt.tight_layout()

        if saveplot:
            if alt == 0:
                plt.savefig(self.path + 'most_important_features' + '_' + self.savename + '.png')
            else:
                plt.savefig(self.path + 'most_important_features' + '_' + self.savename + str(alt) + '.png')
        
        if show:
            plt.show()
        
        plt.close()

        return

    def vs_random_bar(self, scores_dict, saveplot=True, show=False, alt=0):
        '''
        Method to plot a bar chart of models' avg scores

        Parameters
            self: instance of object
            scores_dict: dict of scores to plot bar chart of form: {'Model':['Our Model', 'Random Model'], 'MSE':[mse_ctxt, mse_ctxt_random]}
            saveplot: default True, boolean to save plot or not
            show: default True, boolean to show plot or not
            alt: default 0, how many alternates of this method for self object we want to save. alt = 1 will add a '_1' to end of filename.

        Return
            None
        '''
        models = scores_dict[list(scores_dict.keys())[0]]
        scores = scores_dict[list(scores_dict.keys())[1]]

        plt.figure(figsize = (10,5))
        plt.bar(models, scores, color = 'tab:orange', width = 0.4)
        plt.xlabel(list(scores_dict.keys())[0])
        plt.ylabel(list(scores_dict.keys())[1])
        plt.title('Model Performance vs Random')
            
        if saveplot:
            if alt == 0:
                plt.savefig(self.path + 'vs_random' + '_' + self.savename + '.png')
            else:
                plt.savefig(self.path + 'vs_random' + '_' + self.savename + str(alt) + '.png')
        
        if show:
            plt.show()
        
        plt.close()
        
        return
    
    def make_directory(self):
        '''
        Helper method to make directory path if not created

        Parameters
            self: instance of object
            path: path of directory

        Return
            None
        '''
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        return


if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing:', __name__)
    
