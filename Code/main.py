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

    base_estimators = False
    tuning = False
    fastgrid = False
    demo = True

    args = sys.argv
    if len(args) > 1:
        command = args[1]

        if command == 'nap':
            print("You're running all possible tuning and modeling. 2+ hours")
            base_estimators = True
            tuning = True
            fastgrid = False
            demo = False
        elif command == 'lunch':
            print("You're running with fewer hyperparameters. 10-15 min")
            base_estimators = False
            tuning = True
            fastgrid = True
            demo = False
        elif command == 'coffee':
            print("You're running with no tuning. 5 min")
            base = False
            tuning = False
            fastgrid = False
            demo = False
        elif command == 'demo':
            print("You're skipping the modeling and using pre generated results.")
        else:
            print("You submitted an argument that didn't match any of the options. We're going to run with the DEMO option. Possible options are:\nnap - run all processes and grid searches (2+ hours)\nlunch - run search with fewer hyperparameters (10-15 min)\ncoffee - skip tuning and runs with the best model in the .pkl file (5 min)\ndemo - doesn't run the models and uses the saved results in the GUI (0 min)\n\nif you put no command it will default to demo.\nEnjoy the GUI!")
    else:
        print("You're running with the demo option!")



    print('Executing', __name__)
    # Run data download and setup
    print('Downloading Data (this may take around a minute)')
    ratings, movies, names, title_principals, inflation, pred = down.downloader()

    # Data preprocessing
    print('Doing preprocessing...')
    df_train, df_test, df_val, ss_target, df, df_test_untouched = pre.preprocess(ratings, movies, names, title_principals, inflation)
    
    # EDA
    print('Doing EDA...')
    #ax = eda.plot_duration(df)

    # Modeling
    print('Doing modeling...')
    # For modeling, there are 4 options to run in order of how long you are willing to wait.
    
    # 1) Run all processes from scratch including the time intensive gridsearch validation phase where it is 
    # testing each hyperparameter per model and will take a couple of hours.
        # Set run_model_tuning = True, run_base_estimators = True, fast_gridsearch = False
    
    # [Preferred Method When Running Modeling]
    # 2) Skip model tuning entirely and load already found best model. Roughly 5 mins runtime.
        # Set run_model_tuning = False, run_base_estimators = False, fast_gridsearch = False
    
    # 3) If you still want to go through model tuning but for a much smaller set of hyperparameters. Roughly 10-15 mins runtime.
        # Set run_model_tuning = True, fast_gridsearch = True

    # [Currently Selected Method]
    # 4) Skip modeling entirely to save most amount of time when demoing GUI.
        # Set demo = True
    
    mdl.run_modeling_wrapper(df_train, df_test, df_val, ss_target, df_test_untouched,
                             run_base_estimators = base_estimators, #Run base models comparison or not
                             run_model_tuning = tuning, #Run hyperparameter tuning and gridsearchcv or not
                             fast_gridsearch = fastgrid, #Skip most of gridsearchcv to run faster
                             save_model = True, #Save best model results or not
                             demo = demo) #Demo = True will skip all of modeling since we already have results

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
    
