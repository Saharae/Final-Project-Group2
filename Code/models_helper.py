# This file contains the models that we used for our dataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from sklearn.linear_model import LinearRegression
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

        return

class Model:
    def __init__(self):
        return

    def save_model(self):

        return
    
    def load_model(self):

        return
    
    def predict(self):

        return

    def train(self):
        return

class LinearRegression(Model):
    def __init__(self):
        return
    
    def construct_model(self):
        return

class RandomForest(Model):
    def __init__(self):
        return

class GradientBoost(Model):
    def __init__(self):
        return

class AdaBoost(Model):
    def __init__(self):
        return        

if __name__ == "__main__":
    print('Executing', __name__)
else:
    print('Importing:', __name__)
    
