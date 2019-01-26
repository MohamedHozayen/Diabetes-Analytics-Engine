# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:21:01 2019

@author: MHozayen
"""

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
import sts


# LOAD DATA
# Read the data files.
dataset = pd.read_csv('Data//data-points_cases.csv')
gl_h = dataset.iloc[19:, 1:7].values
gl_h = gl_h.astype('U').astype(np.float) 
gl_d = dataset.iloc[19:, 7:].values
gl_d = gl_d.astype('U').astype(np.float)

#for i in range(0, 6):
# Convert numpy array to pandas DataFrame.
ys = pd.DataFrame(gl_h[:, 2])
data = sts.series_to_supervised(ys, 7)
ys = data['y(t-7)']
ts = data.loc[:, data.columns != 'y(t-7)']

SVR.predict(ts, ys, )

#y = gl_h[:,0].astype('U').astype(np.float)
#eds = (yp_pred - y)**2
#eds = eds[20:]
  
#yplot = yp_pred[12:282]
#tplot = tp_pred[12:282]

#name = 'h'*i
# PLOT 
fig, ax = plt.subplots()
fig.suptitle('Prediction Based on Selected model ', fontsize=14, fontweight='bold')
#ax.set_title('Window Size is %g data point' %(ws))
ax.plot(tp_svr, yp_svr, 'b--', label='svr') 
ax.plot(tp_lr, yp_lr, 'g--', label='lr') 
ax.plot(tp_lr, yp_lr, 'm--', label='lr') 

#ax.plot(tp_pred[ws:-10], yp_pred_pr[ws:-10], 'p--', label='pr') 
ax.plot(ts, ys, 'r',label='Measured data') 
ax.set_xlabel('time (min)')
ax.set_ylabel('glucose (mg/dl)')
ax.legend()
#fig.savefig('graphs//' + name + 'h.png')

    