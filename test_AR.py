# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:07:02 2019

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
#time axis
ts = pd.DataFrame(np.linspace(0, 5*12*24, np.size(gl_h, 0)))
 
## Arrays that will contain predicted values.
#tp_pred = np.zeros(len(ys)) 
#yp_pred = np.zeros(len(ys))

ws = 6 #window size is one hour (5*12=60min)
pred_interval = 4 # 3 is 15min, 6 is 30min, 12 is 60min

#tp_svr, yp_svr = abstract_regression.model(ts, ys, SVR, ws,  pred_interval)
tp_lr, yp_lr = abstract_regression.model(ts, ys, LR, ws, pred_interval)
#tp_rfr, yp_rfr = abstract_regression.model(ts, ys, RFR, ws, pred_interval)

#y = gl_h[:,0].astype('U').astype(np.float)
#eds = (yp_pred - y)**2
#eds = eds[20:]
  
#yplot = yp_pred[12:282]
#tplot = tp_pred[12:282]

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
 
#pd.DataFrame(w_error).to_csv('list.csv')  #save as csv     
plt.plot(tp_lr, eds)
plt.title('EDST = %g, window size = %g' %(sum(eds), ws))
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.show()

#name = 'h'*i
# PLOT 
fig, ax = plt.subplots()
fig.suptitle('Linear Regression', fontsize=14, fontweight='bold')
#ax.set_title('Window Size is %g data point' %(ws))
#ax.plot(tp_svr, yp_svr, 'b--', label='svr') 
ax.plot(tp_lr, yp_lr, 'g--', label='lr') 
#ax.plot(tp_lr, yp_lr, 'm--', label='lr') 

#ax.plot(tp_pred[ws:-10], yp_pred_pr[ws:-10], 'p--', label='pr') 
ax.plot(ts, ys, 'r',label='Measured data') 
ax.set_xlabel('time (min)')
ax.set_ylabel('glucose (mg/dl)')
ax.legend()
#fig.savefig('graphs//' + name + 'h.png')

    