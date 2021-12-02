# miscface - Group2
## George Washington University, Introduction to Data Mining - DATS6103, Fall 2021

![misc.face()](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/02/28105325/DF_31.png)

## Project
Predicting Movies' IMDb Ratings with Machine Learning

![Our Model](https://github.com/Saharae/Final-Project-Group2/blob/main/assets/model_results.png?raw=true)

## Table of Contents
1. [Team Members](#team_members)
2. [How to Run](#instructions)
3. [Folder Structure](#structure)
2. [Timeline](#timeline)
3. [Topic Proposal](#topic_proposal)
4. [Datasets](#datasets)
5. [Presentation](#presentation)
6. [Report](#report)
7. [Additional Resources](#resources)

# <a name="team_members"></a>
## Team Members
* [Sahara Ensley](https://github.com/Saharae)
* [Adam Kritz](https://github.com/adamkritz)
* [Joshua Ting](https://github.com/justjoshtings)

# <a name="instructions"></a>
## How to Run
1. Clone this repo
2. Navigate to the Code folder and install the requirements.txt file  
    `pip install -r requirements.txt`
3. Open the results.zip file so there is a results folder in this repo
4. There are 4 option for running our project:  
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

# <a name="structure"></a>
## Folder Structure
- Code: This folder contains all the final code for the project. To run this project navigate to this folder.
- Final-Group-Presentation: This folder contains the presentation we will be giving in class on 12/06/21.
- Final-Group-Project-Report: This folder contains the final report we will be turning in describing the project and
 results.
- Group-Proposal: This folder contains the initial proposal for our project
- results: This folder contains results from the models we tuned. The GUI pulls from this folder.
- *-individual-project: These folders contain the individual code and report for each team member.

# <a name="timeline"></a>
## Timeline
- [X] Dataset and Topic Chosen - Oct 28th
- [X] Group Proposal - Nov 1st
- [X] Code done + rough GUI done - Nov 22nd
- [X] Rough Draft of presentation - Nov 29nd
- [X] Final Presentation and report - Dec 6th

## Things to Remember
- [X] READMEs and documentation
- [X] requirements.txt
 
# <a name="topic_proposal"></a>
## Topic Proposal
* [Topic Proposal Google Doc](https://docs.google.com/document/d/1S1kVV3D69of6toTyy8Y6Jet_hQxWgXRaJ3Osm7h5Meo/edit?usp=sharing)
* [Diagrams and Plan Slides](https://docs.google.com/presentation/d/1S9aHQ0wytiO6Fa-3G47gVMGUPqQ9GgL-u8SgExSSMzY/edit?usp=sharing)
* [Features Selection Tracker](https://docs.google.com/spreadsheets/d/1qrFCjBWOn3emAx8xtMGC3bpg0CQsx3LMaK1O56ReDO0/edit?usp=sharing)

# <a name="datasets"></a>
## Datasets
* [Kaggle IMDB Movies Dataset](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset?select=IMDb+ratings.csv)
# <a name="presentation"></a>
## Presentation
* [Google Slides Presentation](https://docs.google.com/presentation/d/1ovhlTF3I91rgXGHyGw-c16eZaspBXf1D_hGZrHxvosc/edit?usp=sharing)

# <a name="report"></a>
## Report
* [Final Report Google Doc](https://docs.google.com/document/d/15mzM34VmwNzyYF0Mygbi-N_v0qPVQXY8LAIWD_Nz-p8/edit?usp=sharing)

# <a name="resources"></a>
## Additional Resources
* [Feature Engineering for Categorical Data](https://medium.com/geekculture/feature-engineering-for-categorical-data-a77a04b3308)
* [Categorical Data](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
