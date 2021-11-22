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
    # Set run_model_tuning = False to save time and use already selected best model, 
    # otherwise it will take a long time to go through all parameters in GridSearchCV.
    
    # If you still want to go through model tuning but for a much smaller set of parameters, 
    # set run_model_tuning = True, fast_gridsearch = True and it will be roughly 10 mins.

    # run_model_tuning = False will take roughly 
    
    # Make sure you have results/ directory from GitHub is unzipped if running run_model_tuning as False
    mdl.run_modeling_wrapper(df_train, df_test, df_val, ss_target, 
                             run_base_estimators = False, #Run base models comparison or not
                             run_model_tuning = False, #Run hyperparameter tuning and gridsearchcv or not
                             fast_gridsearch = True, #Skip most of gridsearchcv to run faster
                             save_model = True) #Save best model results or not

    executionTime = (time.time() - startTime)/60
    print('Execution time in minutes:' + str(executionTime))

    # GUI

    
