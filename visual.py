# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 02:49:59 2018

@author: MHozayen
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data//data-points_cases.csv')
# Healthy glucose subjects
gl_h = dataset.iloc[19:, 1:7].values
#gl_h = pd.DataFrame(gl_h)
gl_h = gl_h.astype('U').astype(np.float) 
# Diabatic glucos subjects
gl_d = dataset.iloc[19:, 7:].values
#gl_d = pd.DataFrame(gl_d)
gl_d = gl_d.astype('U').astype(np.float)

#time axis
dif = 5 # 5 min among data points
n_dif = 60/5 #how many 5 min in 1 hour
t = np.linspace(0, dif*12*24, np.size(gl_h, 0))

# Visualising 
h = plt.plot(t, gl_h, color = 'green') # , label='Healthy'
d = plt.plot(t, gl_d, color = 'red') # , label='Diabetic'
plt.title('Glucose Level Over Time')
plt.xlabel('Time')
plt.ylabel('Glucose Level (mmol/L)')
plt.legend((d[:-1]), ['Diabetic'])
plt.show()