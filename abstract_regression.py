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


def model(x, y, model, ws, degree, pred_interval): 

    """
	Model is a function that run predictions model
	Arguments:
		x: time as a dataframe
		y: output values as a dataframe
		model: type of predictive model - must be imported and has a function predict
            for example: SVR is a model based on support vector regression
                        it has predict function
		ws: window size
        pred_interval: prediction time in future t+tau
             if pred-intervak is 6 that means prediction for t+6 (next 30 min)
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
    
    
    # Arrays that will contain predicted values.
    tp_pred = np.zeros(len(y)) 
    yp_pred = np.zeros(len(y))

    for i in range(ws, len(y)-1):
        
        ts_tmp = x[i-ws:i]
        ys_tmp = y[i-ws:i]
          
        if i < x.size-pred_interval:
            #PREDICTION 6 12
            tp = x.iloc[i+pred_interval, 0]
            tp_pred[i] = tp    
            yp_pred[i] = model.predict(ts_tmp, ys_tmp, degree, tp.reshape(-1,1))
    return tp_pred, yp_pred
        
        
#y = gl_h[:,0].astype('U').astype(np.float)
#eds = (yp_pred - y)**2
#eds = eds[20:]

