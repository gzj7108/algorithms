"""
performance measure

"""

import numpy as np

def MSE(a_r,t_r):
    """mean squared error
    input actual result,trained result,column vector
    output average square
    date:2020/1/18
    """
    length=len(a_r)
    sum=np.dot((a_r-t_r).transpose(),(a_r-t_r))/length

    return sum

def RMSE(a_r,t_r):

    return MSE(a_r,t_r)**0.5

def AMSE(a_r,t_r):
    tmp=np.abs(a_r-t_r)
    length=len(a_r)
    sum=np.dot(tmp.transpose(),tmp)/length

    return sum


def R2(a_r,t_r):
    m_r=np.mean(t_r)
    length=len(a_r)
    tse=np.dot((a_r-m_r).transpose(),(a_r-m_r))/length
    cost=1-MSE(a_r,t_r)/tse
    return cost



#test
a=np.transpose([1,2,4])
print(a,a.shape)
b=np.transpose([1,2,3])
print(abs(-1))