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
import RFR
import abstract_regression

# LOAD DATA
# Read the data files.
dataset = pd.read_csv('Data//data-points_cases.csv')
gl_h = dataset.iloc[19:, 1:7].values
gl_h = gl_h.astype('U').astype(np.float) 
gl_d = dataset.iloc[19:, 7:].values
gl_d = gl_d.astype('U').astype(np.float)

acc_eds = []

for pred_interval in range(5, 6):
    # Convert numpy array to pandas DataFrame.
    ys = pd.DataFrame(gl_h[:, 2])
    #time axis
    ts = pd.DataFrame(np.linspace(0, 5*12*24, np.size(gl_h, 0)))
     
    ## Arrays that will contain predicted values.
    #tp_pred = np.zeros(len(ys)) 
    #yp_pred = np.zeros(len(ys))
    
    ws = 9 #window size is one hour (5*12=60min)
    #pred_interval = 5 # 3 is 15min, 6 is 30min, 12 is 60min
    
    #tp_svr, yp_svr = abstract_regression.model(ts, ys, SVR, ws,  pred_interval)
    tp_lr, yp_lr = abstract_regression.model(ts, ys, RFR, ws, pred_interval)
    #tp_rfr, yp_rfr = abstract_regression.model(ts, ys, RFR, ws, pred_interval)
    
    #y = gl_h[:,0].astype('U').astype(np.float)
    #eds = (yp_pred - y)**2
    #eds = eds[20:]
      
    #yplot = yp_pred[12:282]
    #tplot = tp_pred[12:282]

    #fix indentation problem
    tp_lr[:] = [i - (pred_interval*5) for i in tp_lr]
    
    eds = np.zeros(len(yp_lr))
    actual = np.array(ys[ws+1:])
    
    for i in range(0, yp_lr.size):
        eds[i] = (yp_lr[i] - actual[i])**2
    s = sum(eds)
    #w_error.append([w, s])
    #print(w_error)
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(yp_lr, ys[ys.size-yp_lr.size:])
    accuracy = (1-MSE)*100
    print('Accuracy %f' %accuracy)
    
    acc_eds.append(accuracy)
    acc_eds.append(s)
     
    #pd.DataFrame(w_error).to_csv('list.csv')  #save as csv     
    plt.plot(tp_lr, eds)
    plt.title('EDS Plot (EDST = %g, Accuracy = %g)' %(sum(eds), accuracy))
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.show()
    #plt.savefig('Graphs//EDS_' + str(ws) + str (pred_interval) + 'h.png')
    #plt.clf()
    
    #name = 'h'*i
    # PLOT 
    fig, ax = plt.subplots()
    fig.suptitle('Random Forest Regression (5 Trees)', fontsize=14, fontweight='bold')
    #ax.set_title('Window Size is %g data point' %(ws))
    #ax.plot(tp_svr, yp_svr, 'b--', label='svr') 
    ax.plot(tp_lr, yp_lr, 'g--', label='rfr') 
    #ax.plot(tp_lr, yp_lr, 'm--', label='lr') 
    
    #ax.plot(tp_pred[ws:-10], yp_pred_pr[ws:-10], 'p--', label='pr') 
    ax.plot(ts, ys, 'r',label='Measured data') 
    ax.set_xlabel('time (min)')
    ax.set_ylabel('glucose (mg/dl)')
    ax.legend()
    #fig.savefig('Graphs//SVR' + str(ws) + str (pred_interval) + 'h.png')
    #fig.clf()
    
        