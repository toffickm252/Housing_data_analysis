# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 22:04:23 2022

@author: TOFFICK
"""
####Importing and basic investigation
import numpy as np
import pandas as pd 

housing=pd.read_csv("C:\\Users\\TOFFICK\\Documents\\Housing_dataset.csv")

print(housing.head())

#print(housing["size"].head())
print(housing.info())

housing["price"].astype("float")

### Feature Enineering
