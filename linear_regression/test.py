import csv
import numpy as np
from linear_regression import *

#read data
x1,x2,x3,y=np.loadtxt('train.csv', delimiter=',', usecols=(1,2,3,4), unpack=True) 
x= np.concatenate((x1,x2,x3),axis=0)
x_hat=x.reshape(3,len(x1))
x_hat=np.transpose(x_hat)
b=np.ones((len(x1),1))
x_hat=np.append(x_hat, b,axis = 1)
y_hat=y.transpose()
#train
w=np.array([0,0,0,0]).transpose()
w_hat=linearRegression(w,x_hat,y_hat)
print("result:",w_hat)