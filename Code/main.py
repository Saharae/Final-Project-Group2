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
    ratings, movies, names, title_principals, inflation = down.downloader()

    # Data preprocessing
    print('Doing preprocessing...')
    #df_train, df_test, df_val, ss_target, df = pre.preprocess()

    # EDA
    print('Doing EDA...')
    #x = eda.plot_duration(df)

    # Modeling
    print('Doing modeling...')
    #mdl.run_modeling_wrapper(df_train, df_test, df_val, ss_target, run_base_estimators = False, run_model_tuning = True, load_model = True)

    # GUI
    print('Creating the GUI')
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = gui.Menu()  # Creates the menu
    sys.exit(app.exec_())  # Close the application

    # Execution Time
    executionTime = (time.time() - startTime)/60
    print('Execution time in minutes:' + str(executionTime))
    
