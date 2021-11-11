# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:32:16 2021

@author: adamkritz
"""

from zipfile import ZipFile
import io
from urllib.request import urlopen
import pandas as pd

def downloader():
    # connect to repository
    r = urlopen('https://github.com/Saharae/Final-Project-Group2/blob/main/data.zip?raw=true').read()
    
    # open the file
    file = ZipFile(io.BytesIO(r))
    
    # read all the data you want
    inflation = pd.read_csv(file.open('data/CPIAUCNS_inflation.csv'))
    movies = pd.read_csv(file.open('data/IMDb movies.csv'))
    names = pd.read_csv(file.open('data/IMDb names.csv'))
    ratings = pd.read_csv(file.open('data/IMDb ratings.csv'))
    movies_data_for_model = pd.read_csv(file.open('data/movies_data_for_model.csv'))
    
    return inflation, movies, names, ratings, movies_data_for_model

downloader()
    
