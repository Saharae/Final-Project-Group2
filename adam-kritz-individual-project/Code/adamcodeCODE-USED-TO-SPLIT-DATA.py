# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:34:28 2021

@author: adamkritz
"""

# code used to split data

import pandas as pd
import numpy as np

x = pd.read_csv(r'C:\Users\trash\Desktop\data 6103 work\movies for project\IMDb names.csv')

df1, df2, df3 = np.array_split(x, 3)

df1.to_csv(r'C:\Users\trash\Desktop\data 6103 work\movies for project\names1.csv', index = False)
df2.to_csv(r'C:\Users\trash\Desktop\data 6103 work\movies for project\names2.csv', index = False)
df3.to_csv(r'C:\Users\trash\Desktop\data 6103 work\movies for project\names3.csv', index = False)

