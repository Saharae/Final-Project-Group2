# Scripts to run:

- main.py: main script to execute for project.

There are 4 option for running our project:  
Make sure you are located in the Code directory. Or run the full path to the main.py file.  
    i. 'demo': This will skip any modeling and run the GUI with previously generated results. It will take 1 minute
     to run.  
    `python3 main.py demo`  
    ii. 'coffee': This will choose the best model based on the saved .pkl file and run it with no tuning. 5 min.  
    `python3 main.py coffee`  
    iii. 'lunch': This will run the modeling code with fewer hyperparameters to speed it up. 10-15 min.  
    `python3 main.py lunch`  
    iv. 'nap': This will run the entire modeling code with the full grid search of all the hyperparameters. 2+ hours.  
    `python3 main.py nap`  
    v. If you run with no argument it will default to the demo mode.  
    `python3 main.py`
#### Example: 

Mac/Linux:
```
cd Final-Project-Group2/Code/
python3 main.py demo
```

Windows:
```
cd Final-Project-Group2\Code\
py -3 main.py demo
```

# Scripts imported into main.py in order
- DataDownloaded.py: script to pull all necessary data from Google Drive
- preprocessing_utils.py: script to perform preprocessing and outputs train, test, validation data.
- TestEDA.py: script with code for individual plots and statistical analysis
- models_helper.py: script to hold objects and methods to help with modeling part of project.
- models.py: script to perform modeling part of project and save results of modeling (plots & csvs).
- GUI code.py: script to create and show GUI.

# Other files
- requirements.txt: Python package requirements to run this project. Install using:
```
pip install -r /path/to/requirements.txt
```

# Our Code Architecture

![Code Architecture](https://github.com/Saharae/Final-Project-Group2/blob/main/assets/code_architecture.png?raw=true)


