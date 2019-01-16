# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:39:22 2018

@author: MHozayen
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 02:49:59 2018

@author: MHozayen
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import SVR

# LOAD DATA
# Read the data files.
#ys = np.genfromtxt(fname='data/ys.csv', delimiter=',')
#ts = np.genfromtxt(fname='data/ts.csv', delimiter=',')
# MODEL FIT AND PREDICTION
# First order polynomial model.
# Importing the dataset
dataset = pd.read_csv('Data//data-points_cases.csv')
gl_h = dataset.iloc[19:, 1:7].values
gl_h = gl_h.astype('U').astype(np.float) 
gl_d = dataset.iloc[19:, 7:].values
gl_d = gl_d.astype('U').astype(np.float)

# Convert numpy array to pandas DataFrame.
ys = pd.DataFrame(gl_h[:,0])

#time axis
dif = 5 #5 min among data points
n_dif = 60/5 #how many 5 min in 1 hour
t = np.linspace(0, dif*12*24, np.size(gl_h, 0))
ts = pd.DataFrame(t)
#plt.plot(ts, ys)

 
# Arrays that will contain predicted values.
tp_pred = np.zeros(len(ys)) 
yp_pred = np.zeros(len(ys))

ws = 6 #window size is one hour (5*12=60min)
st = 1 #shift window by 5 minutes - use i instead 
pred_30 = 6
pred_60 = 12
pred_15 = 3

# Real time data acquisition is here simulated and a prediction of ph minutes forward is estimated.
# At every iteration of the for cycle a new sample from CGM is acquired.
for i in range(ws, len(ys)-1):
    
    ts_tmp = ts[i-ws:i]
    ys_tmp = ys[i-ws:i]
 
      
    if i+pred_30 < ts.size:
        #PREDICTION 6 12
        tp = ts.iloc[i+pred_15,0]
        tp_pred[i] = tp    
        yp_pred[i] = SVR.predict(ts_tmp, ys_tmp, tp.reshape(-1,1))
    
y = gl_h[:,0].astype('U').astype(np.float)
eds = (yp_pred - y)**2
eds = eds[20:]

yplot = yp_pred[12:282]
tplot = tp_pred[12:282]
# PLOT 
fig, ax = plt.subplots()
fig.suptitle('SVR ', fontsize=14, fontweight='bold')
#ax.set_title('Window Size is %g data point' %(ws))
ax.plot(tplot, yplot, '--', label='Prediction') 
ax.plot(ts, ys, label='Measured data') 
ax.set_xlabel('time (min)')
ax.set_ylabel('glucose (mg/dl)')
ax.legend()