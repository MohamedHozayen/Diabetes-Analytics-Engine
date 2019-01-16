# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:38:24 2019

@author: MHozayen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:00:59 2019

@author: MHozayen
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(x, y, degree, pred):
    
    #degree is unused here
    
    # Fitting SVR to the dataset
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(pred)
    return y_pred