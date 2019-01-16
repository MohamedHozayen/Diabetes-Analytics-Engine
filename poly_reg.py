# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:57:46 2019

@author: MHozayen
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(x, y, degree, pred):

    # Fitting Linear Regression to the dataset
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    
    # Fitting Polynomial Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree) 
    X_poly = poly_reg.fit_transform(x)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    
    y_pred = lin_reg_2.predict(poly_reg.fit_transform(pred))
    return y_pred
  