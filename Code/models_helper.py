# This file contains the models that we used for our dataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error

class Dataset:
    def __init__(self, train_df, test_df, val_df, random_seed, label):
        '''
        Init method, set attributes, set numpy random seed

        Parameters
            self: instance of object
            train_df: training dataframe
            test_df: testing dataframe
            val_df: validation dataframe
            random_seed: random_seed integet number
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

    def most_important_features_analysis(self, show=False):
        '''
        Method to perform analysis to get most important features using a very basic RandomForest.

        Parameters
            self: instance of object
            show: Boolean value whether to show plot of feature importance or not. Default = False.
        
        Return
            None
        '''
        print('Finding most important features...')
        basic_random_forest = RandomForestRegressor(max_depth=25, random_state=self.random_seed)
        basic_random_forest.fit(self.train_X, self.train_Y)
        
        self.feature_importance = pd.DataFrame(np.hstack((np.array(self.train_df_X.columns).reshape(-1,1), basic_random_forest.feature_importances_.reshape(-1,1))), columns=['Features', 'Importance'])
        self.feature_importance = self.feature_importance.sort_values(ascending=False, by='Importance').reset_index(drop=True)
        print(self.feature_importance.head())

        if show:
            plt.figure(figsize=(10, 5))
            plt.bar(self.feature_importance['Features'], self.feature_importance['Importance'], color='tab:orange')
            
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            plt.show()

        # print('Score: %.2f' % basic_random_forest.score(self.val_X, self.val_Y, sample_weight=None))
        # print(basic_random_forest.score(self.val_X, self.val_Y))

        # y_predict = basic_random_forest.predict(self.val_X)
        # print('Loss: ', mean_squared_error(self.val_Y, y_predict), math.sqrt(mean_squared_error(self.val_Y, y_predict))/(self.val_Y.max() - self.val_Y.min()))

        return
    
    def get_most_important_features(self):
        '''
        Method to perform analysis to get most important features using a very basic RandomForest.

        Parameters
            self: instance of object
        
        Return
            self.feature_importance: dataframe of feature importance by feature
        '''

        return self.feature_importance
    
    def use_most_important_features(self, top_n):
        '''
        Method to choose top n most important features to use from most_important_features_analysis.
        Sets self.

        Parameters
            self: instance of object
            top_n: top number of features to use
        
        Return
            None
        '''
        features_to_use = self.feature_importance['Features'].head(top_n).to_list()

        self.train_df_X = self.train_df_X[features_to_use]
        self.test_df_X = self.test_df_X[features_to_use]
        self.val_df_X = self.val_df_X[features_to_use]

        self.data_as_arrays()
        return

class Model:
    def __init__(self, random_seed, train_x, train_y, val_x, val_y, test_x, test_y=None, name=None, target_scaler=None):
        '''
        Init method for Model parent class

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

    def save_model(self):

        return
    
    def load_model(self):

        return
    
    def predict(self):

        return

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

    def get_error_in_context(self):
        '''
        Method to get the error of model in context to target 

        Parameters
            self: instance of object
        
        Return
            self.rmse: calculate RMSE after inverse scaling transformation
            self.mse_val: calculate MSE after inverse scaling transformation
        '''
        self.val_y_predict = self.model.predict(self.val_x)

        self.inv_val_y_predict = self.scaler.inverse_transform(self.val_y_predict.reshape(-1,1))
        self.inv_val_y = self.scaler.inverse_transform(self.val_y.reshape(-1,1))

        self.mse_val = mean_squared_error(self.inv_val_y, self.inv_val_y_predict)

        self.rmse = np.sqrt(self.mse_val)
        print('RMSE for {}: {}'.format(self.name, self.rmse))
        print('MSE for {}: {}'.format(self.name, self.mse_val))
        return self.rmse, self.mse_val

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
        
        return

    def learning_curves(self):

        return

    def validation_curves(self):

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
    
