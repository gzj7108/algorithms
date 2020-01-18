"""
linear  regression model f(x)=wx+b

"""


from optimization import gradient_descent
import numpy as np

def linear_regression(w,x,y):
    """linear regression model
    input training data
    w:weight and bias
    x:variable
    y:actual result
    output expected w
    date:2020/2/18
    """ 

    #set parameters
    learning_rate=0.001
    convergence_criterion=0.000001
    iteration_times=40000
    acceptale_error=0.0001

    #iteration
    for i in range(iteration_times):
       
        #g=2X^T(Xw-y)
        gradient=2*np.dot(x.transpose(),np.dot(x,w)-y)
        #gradient iteration
        w=gradient_descent(w,gradient,learning_rate)
        #gradient?0
        if abs(np.dot(gradient.transpose(),gradient)) < acceptale_error:
            print(" gradient convergence!")
            break

    return w

#test
w=[0,0]
w=np.array(w)
x=[[1,1],[2,1]]
x=np.array(x)
#print(x.shape)
y=[1,2]
y=np.array(y).transpose()
print(linear_regression(w,x,y))