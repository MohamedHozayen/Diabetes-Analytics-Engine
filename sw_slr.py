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

# Importing the dataset
dataset = pd.read_csv('Data//data-points_cases.csv')
gl_h = dataset.iloc[19:, 1:7].values # Healthry patients
gl_h = gl_h.astype('U').astype(np.float) # convert object to float array 
gl_d = dataset.iloc[19:, 7:].values  # Diabatic patients
gl_d = gl_d.astype('U').astype(np.float) # convert object to float array

# Get the first healthy patient data and convert numpy array to pandas DataFrame.
ys = pd.DataFrame(gl_h[:,0])
# time axis every 5 mins start at 8 AM to 8 AM (24 hours)
t = np.linspace(0, 5*12*24, np.size(gl_h, 0))
ts = pd.DataFrame(t)
#plt.plot(ts, ys)

w_error =  []
for w in range(2, 10):#w is window size
    # Arrays that will contain predicted values.
    tp_pred = np.zeros(len(ys)) 
    yp_pred = np.zeros(len(ys))
    ws = w
    lm_tmp = LinearRegression()
    
    # Real time data acquisition is here simulated and a prediction
    for i in range(ws, len(ys)):
       
        ts_tmp = ts[i-ws:i]
        ys_tmp = ys[i-ws:i]
        
        # MODEL
        # Fit this window to the linear model    
        model_tmp = lm_tmp.fit(X=ts_tmp, y=ys_tmp)
    
        #predict the next t+1 glucose level    
        if i < tp_pred.size-1:
            tp = ts.iloc[i+1,0]
            tp_pred[i+1] = tp    
            yp_pred[i+1] = lm_tmp.predict(tp.reshape(-1,1))
    
    tp_pred = tp_pred[ws+1:] 
    yp_pred = yp_pred[ws+1:]
    
    eds = np.zeros(len(yp_pred))
    actual = np.array(ys[ws+1:])
    for i in range(0, yp_pred.size):
        eds[i] = (yp_pred[i] - actual[i])**2
    s = sum(eds)
    w_error.append([w, s])
    #print(w_error)
 
#pd.DataFrame(w_error).to_csv('list.csv')  #save as csv     
# =============================================================================
#     plt.plot(tp_pred, eds)
#     plt.title('EDST = %g, window size = %g' %(sum(eds), ws))
#     plt.xlabel('Index')
#     plt.ylabel('Magnitude')
#     plt.show()
#     plt.gcf().clear()
# =============================================================================
    
# =============================================================================
#     # PLOT 
#     fig, ax = plt.subplots()
#     fig.suptitle('Glucose prediction', fontsize=14, fontweight='bold')
#     ax.set_title('Window Size is %g data point' %(ws))
#     ax.plot(tp_pred, yp_pred, '--', label='Prediction') 
#     ax.plot(ts, ys, label='Measured data') 
#     ax.set_xlabel('time (min)')
#     ax.set_ylabel('glucose (mg/dl)')
#     ax.legend()
#     plt.gcf().clear()
# =============================================================================
