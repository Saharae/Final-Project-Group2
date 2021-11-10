# This file runs our modeling 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.linear_model import LinearRegression

from models import LinearRegression, RandomForest, GradientBoost



model1 = LinearRegression(movies_df)
model2 = RandomForest(movies_df)
model3 = GradientBoost(movies_df)