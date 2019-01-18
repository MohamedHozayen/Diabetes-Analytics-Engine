# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:39:22 2018

@author: MHozayen
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SVR
import LR
import poly_reg
import abstract_regression

# LOAD DATA
# Read the data files.
dataset = pd.read_csv('Data//data-points_cases.csv')
gl_h = dataset.iloc[19:, 1:7].values
gl_h = gl_h.astype('U').astype(np.float) 
gl_d = dataset.iloc[19:, 7:].values
gl_d = gl_d.astype('U').astype(np.float)

for i in range(0, 6):
    # Convert numpy array to pandas DataFrame.
    ys = pd.DataFrame(gl_d[:,i])
    #time axis
    ts = pd.DataFrame(np.linspace(0, 5*12*24, np.size(gl_d, 0)))
     
    ## Arrays that will contain predicted values.
    #tp_pred = np.zeros(len(ys)) 
    #yp_pred = np.zeros(len(ys))
    
    ws = 12 #window size is one hour (5*12=60min)
    
    
    pred_interval = 6 # 3 is 15min, 6 is 30min, 12 is 60min
    degree = 9
    
    tp_pred, yp_pred_svr = abstract_regression.model(ts, ys, SVR, ws, degree,  pred_interval)
    tp_pred, yp_pred_lr = abstract_regression.model(ts, ys, LR, ws, degree, pred_interval)
    #tp_pred, yp_pred_pr = abstract_regression.model(ts, ys, poly_reg, ws, degree, pred_interval)
    
    
    #y = gl_h[:,0].astype('U').astype(np.float)
    #eds = (yp_pred - y)**2
    #eds = eds[20:]
    
    #yplot = yp_pred[12:282]
    #tplot = tp_pred[12:282]
    #yp_pred_lr = np.trim_zeros(yp_pred_lr)
    #yp_pred_svr = np.trim_zeros(yp_pred_svr)
    #yp_pred_pr = np.trim_zeros(yp_pred_pr)
    name = 'h'*i
    # PLOT 
    fig, ax = plt.subplots()
    fig.suptitle('Window size 60min - Prediction Window 30min' , fontsize=14, fontweight='bold')
    #ax.set_title('Window Size is %g data point' %(ws))
    ax.plot(tp_pred[ws:-20], yp_pred_svr[ws:-20], 'b--', label='svr') 
    ax.plot(tp_pred[ws:-20], yp_pred_lr[ws:-20], 'g--', label='lr') 
    #ax.plot(tp_pred[ws:-10], yp_pred_pr[ws:-10], 'p--', label='pr') 
    ax.plot(ts, ys, 'r',label='Original data') 
    ax.set_xlabel('time (min)')
    ax.set_ylabel('glucose (mg/dl)')
    ax.legend()
    fig.savefig('graphs_d//' + name + 'h.png')
    