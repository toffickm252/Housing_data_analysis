# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 22:04:23 2022

@author: TOFFICK
"""
####Importing and basic investigation
import numpy as np
import pandas as pd 

housing=pd.read_csv("C:\\Users\\TOFFICK\\Documents\\GitHub\\Housing_data_analysis\\Housing_dataset_cleaned.csv")

#print(housing.head())

#print(housing["size"].head())
#print(housing.info())

housing["price"].astype("float")

### Feature Enineering
## Dropping categorical columns
#housing_dropped=pd.drop(['availability','location','size_comp'],axis=1)
housing.drop(['availability', 'location','size_comp'], axis = 1, inplace = True)

#print(housing.head())

## Feature engineering begins 
print(housing.describe())

