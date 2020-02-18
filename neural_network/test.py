import numpy as np 
a=np.random.rand(10)
#print(a)
a=np.array([1,2,3,4,1,2,3,4])
def sigmoid(x):
    """ calculate sigmoid function
    date:2020/2/14"""
    return 1/(1+np.exp(-x))

class T():
    def __init__(self):
        self.a=[]

class V():
    def __init__(self):
        self.b=0


t=T()
v=V()
t.a.append(v)
t.a[0].b=1
t1=t.a[:]
t1[0].b=2
print(t.a[0].b)