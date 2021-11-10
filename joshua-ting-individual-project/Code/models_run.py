# This file runs our modeling 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.linear_model import LinearRegression

from models import Dataset, LinearRegression, RandomForest, GradientBoost

############ Set random seed ############
random_seed = 33

# Set random seed in numpy
np.random.seed(random_seed)

############ Get Data ############

# Load Data


# Split train, val, test

############ Set Up Models ############
models = {'sgd': SGDRegressor(random_state=random_seed)}
