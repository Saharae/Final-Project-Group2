# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:59:31 2021

@author: adamkritz
"""

# preprocessing code

import pandas as pd
import os

# you will have to change to your directory
os.chdir(r'C:\Users\trash\Desktop\data 6103 work\movies for project')

# I used my directory for now, we wi

movies = pd.read_csv('IMDb movies.csv')
names = pd.read_csv('IMDb names.csv')
ratings = pd.read_csv('IMDb ratings.csv')
TP = pd.read_csv('IMDb title_principals.csv')

# one value in the year column is mislabeled, this causes a warning when importing
movies['year'] = movies['year'].replace('TV Movie 2019', 2019)

# Merging Ratings and Movies (it merges perfectly)
movies_ratings = movies.merge(ratings, on = 'imdb_title_id')

# Merging movies+ratings with title_principal
# this merges with 9 extra from the movies_ratings side (9 movies are in movies_ratings that arent in title_principal)
# and 19 (actually just 2 repeated a lot) extra movies from the title_principal side (19 movies are in title_principal that are not in movies or ratings)
INDmovies_ratings_TP = movies_ratings.merge(TP, on = 'imdb_title_id', how = 'outer', indicator = True)

# list of the 9 movies
INDmovies_ratings_TP.loc[INDmovies_ratings_TP['_merge'] == 'left_only']['imdb_title_id']
onlyin_MR = ['tt10764458', 'tt11010804', 'tt11777308', 'tt3978706', 'tt4045476', 'tt4045478', 'tt4251266', 'tt5440848', 'tt6889806']

# list of the 2 movies repeated 19 times
INDmovies_ratings_TP.loc[INDmovies_ratings_TP['_merge'] == 'right_only']['imdb_title_id']
onlyin_tp = ['tt1860336'] * 10 + ['tt2082513'] * 9

# merge again without indicator
movies_ratings_TP = movies_ratings.merge(TP, on = 'imdb_title_id', how = 'outer')

# Merges movies+ratings+title_principal with names
INDMRtpN = movies_ratings_TP.merge(names, on = 'imdb_name_id', how = 'outer', indicator = True)

# list of 10 movies only in movies, ratings, and title_principal (9 of them only in movies and ratings, not title_principal)
INDMRtpN.loc[INDMRtpN['_merge'] == 'left_only']['imdb_title_id']
onlyin_MRtp = ['tt0091454', 'tt10764458', 'tt11010804', 'tt11777308', 'tt3978706', 'tt4045476', 'tt4045478', 'tt4251266', 'tt5440848', 'tt6889806']

# final product
MRtpN = movies_ratings_TP.merge(names, on = 'imdb_name_id', how = 'outer')
print(MRtpN)

