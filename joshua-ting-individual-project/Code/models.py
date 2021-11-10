# This file contains the models that we used for our dataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.linear_model import LinearRegression

class Model:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def fit(self):
        return
    
    def predict(self):
        return
    
    def get_score(self):
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
    
