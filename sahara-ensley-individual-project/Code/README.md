preprocessing_utils
----
Collection of functions for preprocessing.

Overivew of cleaning
---
- Merged the movies and ratings dataset since they share a unique key of `imbd_title_id`. The main dataframe now
 contains all meta data about the movie as well as all the information of the ratings.\
- Also merged an inflation dataset to calculate multiplier and create adjusted money columns and convert to today's
 currency.\
- one column renamed to fix a typo. `worlwide_gross_income` -> `worldwide_gross_income`
 
**By Column**

`year`: removed a single string value, column is now an int
`genre`: originally a string with potential for multiple genre's listed alphabetically. Now a list of 1-3 genres. ex
: `["Action", "Comedy", "Romace"]`.\
`genre1` - `genre3`: 3 columns that expand the list in the genre column to potentially make analysis easier.\
`budget`, `usa_gross_income`, `worldwide_gross_income` : Any symbols for money were removed and it was casted to an int. Any Null values were
 filled in with 0s.\
 `budget_adjusted`, `usa_gross_income_adjusted`, `worldwide_gross_income_adjusted`: monetary columns in today's currency
 . IMPORTANT-- ONLY MAKES SENSE IN US DOLLARS. So use country filter function if using these columns.\
`country`: Similar to genre, converted from a string of multiple countries to a list of all countries.\
`primary_country`: This list seems to be sorted not alphabetically but by importance, so the first country listed
 gets placed in its own column. NAN if there was no country listed (should be 18 in the whole dataset)\
`DATE`, `CAPIAUCNS`: Columns from the inflation dataset that help calculate inflation multiplier.\
`multiplier`: Value that gets multiplied to monetary value to calculate what it is in today's currecy.\



**load_all**
---
Temporary function to stand in place of the download funtion for now.\
'base' is the base directory where the repo is, so it can find the data folder.\
IMPORTANT: make sure you download the conversion dataset: https://fred.stlouisfed.org/series/CPIAUCNS

```python:
base = '/Users/sahara/Documents/GW/DataMining/Final-Project-Group2'
ratings, movies, names, inflation = load_all(base)
```

**clean_inflation**
---
Cleans the inflation dataset.\
Takes the first inflation value for every year and normalizes it to the latest inflation index.\
Creates a multiplier table that can be merged with the movies table on the year column.\

```python:
inflation_clean = clean_inflation(inflation)
```

**clean_money**
---
Utility function to use with pandas to clean the columns associated with money.\
Removes any white space and special characters and casts to an int.

**get_primary_country**
---
Utility function to use with pandas to clean the country column.\
Since the country columns appear to be sorted by most relevant country it takes the first one listed and puts it into
 its own column.
 
**merge_and_clean_movies**
---
Main pre-processing function !!! \
Takes in the movies, ratings, and cleaned inflation columns and returns a fully cleaned and merged movies dataframe
 to use for analysis. 
 
Details about exactly what happens to each column can be found above.\

```python:
movies = merge_and_clean_movies(movies, ratings, inflation_clean)
```

**US_movies**
---
Simple function to return only the movies that were made in the US.\
This is to deal with problems concerning money since finding conversion rates by year for various currencies is
 currently not implemented.\
Make sure you give it the cleaned and merged movies dataframe.

```python:
US_movies = US_movies(movies)
```

**get_train_test**
---
Rough initial function to generate train test splits.\
    :param data: dataframe to pull features and values from\
    :param features: list of columns to train on ex. `['genre1', 'duration']`\
    :param target: column to test ex. `['avg_vote']`\
    :param encode_target: does the target need to be encoded? boolean True or False\
    :param test_size: test size to split train test with, ex. 0.3\
    :return:\
      feature_vals - the array of only the features\
      target_vals - the array of only the target vals (if encoded it will return the encoded vals)\
      X_train, X_test, y_train, y_test\

```python:
feature_vals, target_vals, X_train, X_test, y_train, y_test = get_train_test(movies, features = ['budget', 'country'], target = ['weighted_average_vote'], encode_target = False, test_size = 0.3)
```

EDA.py
---
Code to create plots. Each function creates an individual explanatory plot.  
Also has 2 functions that perform basic statistical analyses.  
More details can be found in the EDA section of the report, in the Final-Report folder.