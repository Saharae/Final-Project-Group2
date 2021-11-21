# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:32:16 2021

@author: adamkritz
"""

import pandas as pd

def downloader():
    
    # downloads all the data
        
    ratingsURL = 'https://drive.google.com/file/d/1CYQE7U9CM1AIt6nMgvIF31X86P-1_wIB/view?usp=sharing'
    path1 = 'https://drive.google.com/uc?export=download&id='+ratingsURL.split('/')[-2]
    ratings = pd.read_csv(path1)
    
    moviesURL = 'https://drive.google.com/file/d/1xLdKMhZ8eSqny6-UcbssxB9PqdROvgOQ/view?usp=sharing'
    path2 = 'https://drive.google.com/uc?export=download&id='+moviesURL.split('/')[-2]
    movies = pd.read_csv(path2)
    
    # names was too big to bypass the virus scanner so i split it into 3 parts
    
    names1URL = 'https://drive.google.com/file/d/1jscP1KWTqZfcv0F-J5DlmJfSbRgu_ptI/view?usp=sharing'
    path31 = 'https://drive.google.com/uc?export=download&id='+names1URL.split('/')[-2]
    names1 = pd.read_csv(path31)
    
    names2URL = 'https://drive.google.com/file/d/1chwblAFN_q8FWNFu1PGXRMhShhY-A_7Y/view?usp=sharing'
    path32 = 'https://drive.google.com/uc?export=download&id='+names2URL.split('/')[-2]
    names2 = pd.read_csv(path32)
    
    names3URL = 'https://drive.google.com/file/d/1hXSEqexkAlLa1nyf8_FszDtl4u_T6cGM/view?usp=sharing'
    path33 = 'https://drive.google.com/uc?export=download&id='+names3URL.split('/')[-2]
    names3 = pd.read_csv(path33)
    
    # and recombined it all
    
    names = pd.concat([names1, names2, names3], ignore_index=True)
    
    TPURL = 'https://drive.google.com/file/d/1lIg_Ty6tbjTaIhnxoacar3cNuK4W0JTS/view?usp=sharing'
    path4 = 'https://drive.google.com/uc?export=download&id='+TPURL.split('/')[-2]
    title_principals = pd.read_csv(path4)
    
    InflationURL = 'https://drive.google.com/file/d/1T4jLbFHXvZEY_CQRQgGyQiDsPhwClUYk/view?usp=sharing'
    path5 = 'https://drive.google.com/uc?export=download&id='+InflationURL.split('/')[-2]
    inflation = pd.read_csv(path5)
    
    return ratings, movies, names, title_principals, inflation
    

    
