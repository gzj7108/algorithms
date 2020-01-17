"""
performance measure

"""

import numpy as np

def average_square(a_r,t_r):
    """average square measure
    input actual result,trained result,ndarray
    output average square,float 
    date:2020/1/17
    """
    sum=0
    if len(a_r) == len(t_r):
        length=len(a_r)
    else:
        print("average_square parameters don't match!")
        return

    for i in range(0,length) :
        sum+=(a_r[i]-t_r[i])**2
    sum/=length

    return sum

#test
a=np.array([1,2,4])
b=np.array([1,2,3])
print(average_square(a,b))