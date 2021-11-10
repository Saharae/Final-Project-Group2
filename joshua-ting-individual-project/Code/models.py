# This file contains the models that we used for our dataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.linear_model import LinearRegression

class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def split_train_val_test(self, train_split, val_split):
        '''
        Method to split train, val, and test dataset

        Parameters
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


        
