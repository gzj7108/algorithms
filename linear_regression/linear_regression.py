"""
linear  regression model f(x)=wx+b

"""


from optimization import gradient_descent
import numpy as np

def linear_regression(w,x,y):
    """linear regression model
    input training data
    w:weight and bias,ndarray
    x:variable,ndarray
    y:actual result
    output expected w
    date:2020/2/17
    """ 

    #set parameters
    learning_rate=0.001
    convergence_criterion=0.000001
    iteration_times=10000
    #g=2X^T(Xw-y)
    gradient=2*np.transpose(x)*(x*w-y)
    
    #iteration
    i=0
    while i < iteration_times:
        w=gradient_descent(w,gradient,learning_rate)
        print(np.transpose(y-x*w)*(y-x*w))
        i+=1

    return w

w=[0,0]
w=np.array(w)
x=[[1,1],[2,1]]
x=np.array(x)
y=[1,2]
y=np.array(y)
print(linear_regression(w,x,y))