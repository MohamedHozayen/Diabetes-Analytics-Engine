# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:00:59 2019

@author: MHozayen
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(x, y, pred):
     #degree is unused here

   
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 1, random_state = 0, min_samples_split = 2)
    regressor.fit(x, y)
    
    # Predicting a new result
    y_pred = regressor.predict(pred)
    
    return y_pred