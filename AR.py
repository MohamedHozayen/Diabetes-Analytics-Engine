# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:07:36 2019

@author: MHozayen
"""

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
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

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
    lr = AR()
    #lr.fit(x, y, sample_weight=weights)
    lr.fit(y)
    y_pred = lr.predict(pred)
    return y_pred


series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()