#!/bin/python3

import sys
import fileinput
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
data=open('trainingdata.txt','r')
dataFile=data.readlines()
data.close()
X=[]
Y=[]
for line in dataFile:
    line = line.strip()
    line = line.split(',')
    X.append([float(line[0])])
    Y.append(float(line[1]))
origY=Y
maxIndex=[]
Xs=[]
for x in range(len(origY)): 
    if origY[x]==8.0:
        maxIndex.append(x)
        
for idx in maxIndex:
    Xs.append(X[idx])
    
min_c_time=min(Xs)

linX=[]
idxlinX=[]
linY=[]
for idxX in range(len(X)):
    feat=X[idxX]
    if feat<min_c_time:
        linX.append(feat)
        idxlinX.append(idxX)
for idx in idxlinX:
    linY.append(origY[idx])
    regressor=Ridge(alpha=0.1)
regressor.fit(linX,linY)
val= float(input().strip())
prediction=regressor.predict(float(val))
prediction=prediction[0]
if prediction>8.0:
    prediction=8.0
print("%.2f" % prediction)