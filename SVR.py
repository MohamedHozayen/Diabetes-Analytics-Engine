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

    #degree is unused here
    mu = 0.9
    ns = len(y)
    weights = np.ones(ns)*mu
    for k in range(ns):
        weights[k] = weights[k]**k
    weights = np.flip(weights, 0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(x)
    Y = sc_y.fit_transform(y)
    
    # Fitting SVR to the dataset
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, Y)
    
    # Predicting a new result
    y_pred = regressor.predict(pred)
    y_pred = sc_y.inverse_transform(y_pred)
    return y_pred