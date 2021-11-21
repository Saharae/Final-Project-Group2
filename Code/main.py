"""
main.py

Main script to run data mining project on IMDB dataset.

Steps include:
    1. Data download and setup
    2. Data preprocessing
    3. EDA
    4. Modeling
    5. GUI
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import time

import preprocessing_utils as pre
import models as mdl


if __name__ == "__main__":
    startTime = time.time()

    print('Executing', __name__)
    # Run data download and setup

    # Data preprocessing
    print('Doing preprocessing...')
    df_train, df_test, df_val, ss_target, df = pre.preprocess()

    # EDA

    # Modeling
    print('Doing modeling...')
    mdl.run_modeling_wrapper(df_train, df_test, df_val, ss_target, run_base_estimators = False, run_model_tuning = True, load_model = True)

    executionTime = (time.time() - startTime)/60
    print('Execution time in minutes:' + str(executionTime))

    # GUI

    
