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
import sys
from PyQt5.QtWidgets import QApplication

import time

import DataDownloaded as down
import preprocessing_utils as pre
import TestEDA as eda
import models as mdl
import GUI as gui


if __name__ == "__main__":
    startTime = time.time()

    print('Executing', __name__)
    # Run data download and setup
    print('Downloading Data (this may take around a minute)')

    # temp try except block just to prevent hitting limits form data downloader
    try:
        ratings, movies, names, title_principals, inflation, pred = down.downloader()
    except:
        ratings, movies, names, title_principals, inflation = pre.load_all(pre.get_repo_root())

    # Data preprocessing
    print('Doing preprocessing...')
    df_train, df_test, df_val, ss_target, df, df_test_untouched = pre.preprocess(ratings, movies, names, title_principals, inflation)
    
    # EDA
    print('Doing EDA...')
    #ax = eda.plot_duration(df)

    # Modeling
    print('Doing modeling...')
    # Set run_model_tuning = False to save time and use already selected best model, 
    # otherwise it will take a long time to go through all parameters in GridSearchCV.
    
    # If you still want to go through model tuning but for a much smaller set of parameters, 
    # set run_model_tuning = True, fast_gridsearch = True and it will be roughly 10 mins.

    # run_model_tuning = False will take roughly 
    
    # Make sure you have results/ directory from GitHub is unzipped if running run_model_tuning as False
    mdl.run_modeling_wrapper(df_train, df_test, df_val, ss_target, df_test_untouched,
                             run_base_estimators = False, #Run base models comparison or not
                             run_model_tuning = False, #Run hyperparameter tuning and gridsearchcv or not
                             fast_gridsearch = True, #Skip most of gridsearchcv to run faster
                             save_model = True) #Save best model results or not


    # GUI
    print('Creating the GUI')
    gui.unzip_results()
    gui.take(df, pred)
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = gui.Menu()  # Creates the menu
    sys.exit(app.exec_())  # Close the application

    # Execution Time
    executionTime = (time.time() - startTime)/60
    print('Execution time in minutes:' + str(executionTime))
    
