# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 22:04:23 2022

@author: TOFFICK
"""
####Importing and basic investigation
import numpy as np
import pandas as pd 

housing=pd.read_csv("C:\\Users\\TOFFICK\\Documents\\GitHub\\Housing_data_analysis\\Housing_dataset_cleaned.csv")

# Basic investigation of the data 
#print(housing.head())

#print(housing["size"].head())
#print(housing.info())
housing['size'].unique()
housing['location'].value_counts()
print(housing.describe())


## More Data cleaning steps 
housing["price"].astype("float")
housing.drop(['availability', 'area_type','size_comp'], axis = 1, inplace = True)

## Initial check of values 
print(housing['location'].value_counts())
housing['location'] = housing['location'].apply(lambda x: x.strip())

# second check after applying lambda function
location_count=housing.location.value_counts()
print(location_count)

# Location counts less than 10 
location_count_10 = location_count[location_count < 10]
print(location_count_10)

# Making a copy of original dataset imported 
housing_edit=housing.copy()

# cleaning location column 
housing_edit['location'] = housing_edit['location'].apply(lambda x: 'other' if x in location_count_10 else x)
print(housing_edit)

# remove outliers from bhk columns // taking out bhk values less than 300 squarefeet
housing_edit = housing_edit[~(housing_edit.total_sqft/housing_edit.bhk < 300)]

# cleaning the price column 
print(housing_edit['price'].unique())

# per square feet price 
housing_edit['per_sq_feet_price'] = (housing_edit['price']*100000)/housing_edit['total_sqft']

# checking for description of the dataset
print(housing_edit['per_sq_feet_price'].describe())

# Remove outliers from per square feet price 
# writing a standard deviation to that effect 
def rmv_outlier(x):
    out_df = pd.DataFrame()
    for key, subdf in x.groupby('location'):
        mean = np.mean(subdf.per_sq_feet_price)
        std = np.std(subdf.per_sq_feet_price)
        reduced_df = subdf[(subdf.per_sq_feet_price > (mean - std)) & (subdf.per_sq_feet_price < (mean + std))]
        out_df = pd.concat([out_df, reduced_df], ignore_index = True)
    return out_df

# Printing out the effect of the function on the dataset
housing_edit_1=rmv_outlier(housing_edit)
print(housing_edit_1)
## Now we can remove those 2 BHK apartments whose per_sq_feet_price is less than mean per_sq_feet_price 
# of 1 BHK apartment
# remove bhk outliers 
def remove_bhk_outliers(x):
    exclude_indices = np.array([])
    for location, location_x in x.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_x in location_x.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_x.per_sq_feet_price),
                'std': np.std(bhk_x.per_sq_feet_price),
                'count': bhk_x.shape[0]
            }
        for bhk, bhk_x in location_x.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_x[bhk_x.per_sq_feet_price<(stats['mean'])].index.values)
    return x.drop(exclude_indices,axis='index')

# applying the function to the set bkn outlier removal
housing_edit_2 = remove_bhk_outliers(housing_edit_1)
print(housing_edit_2)


### Feature Enineering
## Dropping categorical columns
#housing_dropped=pd.drop(['availability','location','size_comp'],axis=1)


#print(housing.head())

## Feature engineering begins 
#print(housing_drop.describe())

### more code to be added 


### Building a model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = housing.drop(['location','price','total_sqft_price'], axis = 1)
Y = housing['price']


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 1)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


pred = model.predict(x_test)
print(pred)


print(np.where(X.columns == '2nd Stage Nagarbhavi')[0][0])
