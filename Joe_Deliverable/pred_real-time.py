# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:39:22 2018

@author: MHozayen
"""
# Importing the libraries
import numpy as np
import pandas as pd
import SVR
import LR
import RFR
import sys, traceback

#def main():
#    try:
#        do main program stuff here
#        ....
#    except KeyboardInterrupt:
#        print "Shutdown requested...exiting"
#    except Exception:
#        traceback.print_exc(file=sys.stdout)
#    sys.exit(0)

#if __name__ == "__main__":
#    result = pred(x, y, RFR, 12, 9)

def pred(x, y, model, ws, pred_interval): 

    """
	Model is a function that run predictions model
    predicts pred_interval based on window size ws
      
    
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
		A single value y-predicted
	"""
        
    #check if x, and y are the same size
    if len(x) != len(y):
        sys.exit('X and Y must be the same length!')
        
    #check if ws is less than x or y    
    if len(x) < ws:
        sys.exit('Window size ws has to be less than length of X and Y!')
    
    ts_tmp = x[len(x)-ws-1:]
    ys_tmp = y[len(y)-ws-1:]
    tp = x.iloc[len(x)-1, 0] + 5; #5 is hard coded - 5 minutes between points in x
    
    i = 0;
    while i <= pred_interval:
    
        y_predicted = model.predict(ts_tmp, ys_tmp, tp.reshape(-1,1))
        #print(y_predicted)
        tp = tp + 5;
        ts_tmp = ts_tmp.append(pd.DataFrame([tp],))
        #print(ts_tmp)
        if model == LR:
            ys_tmp = ys_tmp.append(pd.DataFrame(y_predicted,))
        else:
            ys_tmp = ys_tmp.append(pd.DataFrame([y_predicted],))
            
        i = i +1;
            
    
    #return the last value in array - accumulative predictions
    return y_predicted[0]       

