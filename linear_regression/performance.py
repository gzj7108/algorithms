"""
performance measure

"""

import numpy as np

def average_square(a_r,t_r):
    """average square measure
    input actual result,trained result,column vector
    output average square
    date:2020/1/18
    """
    
    
    if len(a_r) == len(t_r):
        length=len(a_r)
        sum=np.dot((a_r-t_r).transpose(),(a_r-t_r))/length
        print((a_r-t_r).transpose().shape)
    else:
        print("average_square parameters don't match!")
        return

    return sum

#test
a=np.transpose([1,2,4])
print(a,a.shape)
b=np.transpose([1,2,3])


print(average_square(a,b))
print(abs(-1))