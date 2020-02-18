"""
linear  regression model f(x)=wx+b

"""

from performance import*
from optimization import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

def linearRegression(w,x,y):
    """linear regression model
    input training data
    w:weight and bias
    x:variable
    y:actual result
    output expected w
    date:2020/1/18
    """ 

    #set parameters
    learning_rate=0.0000001
    iteration_times=1000



    #iteration
    for i in range(iteration_times):
       
        #g=2X^T(Xw-y)
        gradient=2*np.dot(x.transpose(),np.dot(x,w)-y)
        #gradient iteration
        w=gradient_descent(w,gradient,learning_rate)
        #calculate the cost
        print("R2 cost:",R2(y,np.dot(x,w)))


    return w

