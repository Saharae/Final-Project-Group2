# This file contains the models that we used for our dataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

class Dataset:
    def __init__(self, dataset):
        if isinstance(dataset, str):
            self.dataset = pd.read_csv(dataset)
        elif isinstance(dataset, pd.DataFrame):
            self.dataset = dataset

    def get_dataset(self):
        '''
        Method to get a pandas dataframe of the dataset

        Parameters:
            self: instance of object
        
        Return
            self.dataset: pandas dataframe attributed to instance of object
        '''
        return self.dataset

    def split_train_val_test(self, train_split, val_split):
        '''
        Method to split train, val, and test dataset

        Parameters
            self: instance of object
            train_split: float between 0-1, how much portion of dataset for train
            val_split: float between 0-1, how much portion of dataset for validation

        Return
            train_df: DF for training
            val_df: DF for validation
            test_df: DF for testing
        '''
        return

    def split_features_target(self):

        return

class Model(Dataset):
    def __init__(self):
        return

class LinearRegression(Model):
    def __init__(self):
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

def get_repo_root():
    '''
    Function to get the repo base path of '.../Final-Project-Group2' so anyone can run.
    
    Paramaters
    ----------
    None

    Return
    ----------
    None
    '''
    repo_name = 'Final-Project-Group2'
    current_path = os.path.abspath(__file__)
    current_path_list = current_path.split('/')
    repo_index = current_path_list.index(repo_name)
    current_path_list = current_path_list[:repo_index+1]
    current_path = '/'.join(current_path_list)
    
    return current_path


if __name__ == "__main__":
    print(__name__, 'executed')
else:
    print('Importing:', __name__)
    
