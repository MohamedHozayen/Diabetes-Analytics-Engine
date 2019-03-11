# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:38:24 2019

@author: MHozayen

Simple Linear Regression 
Weighted Linear Regression is commented out
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(x, y, pred):
    
    #degree is unused here
    mu = 0.9
    ns = len(y)
    weights = np.ones(ns)*mu
    for k in range(ns):
        weights[k] = weights[k]**k
    weights = np.flip(weights, 0)
    
    
    # Fitting SVR to the dataset
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    
    #weighted linear regression
    #lr.fit(x, y, sample_weight=weights)
    
    lr.fit(x, y)
    y_pred = lr.predict(pred)
    return y_pred