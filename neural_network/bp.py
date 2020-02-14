""" 
a neural network with error backpropagation algorithm
you can set the hyper-parameters in convenient ways instead of fixed structures
date:2020/2/13
"""

import numpy as np 

class Layer():
    """
    define a layer model to store information for the convenience of propagation
    date:2020/2/13
    """
    def __init__(self):
        """
        init the node model
        """
        #define the number of each layer
        self.neural_num=0
        #define the matrix w of each layer
        self.w=[]
        self.theta=[]
        #calculate
        self.value=[]
        self.g=[]
        #define the connection
        self.last_layer=[]
        self.next_layer=[]
        #mark the layer
        self.input_layer=False
        self.output_layer=False



def initNeuralNetwork(neurals):
    """
    function as initing the model 
    input:
    neurals=[input-layer_num,hidden-layer1_num,,,hidden-layerN_num,output-layer_num]
    output:the input layer and the output layer
    date:2020/2/13
    """
    #set hyper-parameters
    
    #layer_num=len(neurals)
    ilayer_num=neurals[0]
    olayer_num=neurals[-1]
    hlayer_num=neurals[1:-1]
    #input layer
    ilayer=Layer()
    ilayer.input_layer=True
    ilayer.neural_num=ilayer_num
    ilayer.w=np.random.rand(ilayer_num)
    ilayer.theta=np.random.rand(ilayer_num)
    #output layer
    olayer=Layer()
    olayer.output_layer=True
    olayer.neural_num=olayer_num
    olayer.w=np.random.rand(olayer_num)
    olayer.theta=np.random.rand(olayer_num)
    #hidden layer
    hlayers=[]
    for i in range(len(hlayer_num)):
        layer=Layer()
        layer.neural_num=hlayer_num[i]
        hlayers.append(layer)

    for i in range(len(hlayers)):
        if i == 0:
            hlayers[i].last_layer.append(ilayer)
            if len(hlayers)!=1:
                hlayers[i].next_layer.append(hlayers[i+1])
            ilayer.next_layer.append(hlayers[i])
            #init the parameters
            ilayer.w=np.random.random((hlayer_num[i],ilayer_num))
            ilayer.theta=np.random.random(hlayer_num[i])
            print("input layer to hidden layer w:",ilayer.w)
            print("input layer to hidden layer theta:",ilayer.theta)
        elif i != len(hlayers)-1:
            hlayers[i].last_layer.append(hlayers[i-1])
            hlayers[i].next_layer.append(hlayers[i+1])
            #init the parameters
            hlayers[i].w=np.random.random((hlayer_num[i+1],hlayer_num[i]))
            hlayers[i].theta=np.random.random(hlayer_num[i+1])
        if i == len(hlayers)-1:
            olayer.last_layer.append(hlayers[i])
            hlayers[i].last_layer.append(hlayers[i-1])
            hlayers[i].next_layer.append(olayer)
            #init the parameters
            hlayers[i].w=np.random.random((olayer_num,hlayer_num[i]))
            hlayers[i].theta=np.random.random(olayer_num)
        print("hidden layer",i,"to hidden layer",i+1,"w:",hlayers[i].w)
        print("hidden layer",i,"to hidden layer",i+1,"theta:",hlayers[i].theta)


    return ilayer,olayer

def sigmoid(x):
    """ calculate sigmoid function
    date:2020/2/14"""
    return 1/(1+np.exp(-x))

def forwardPropagation(i_l):
    """forward propagation to calculate the output of each layer
    date:2020/2/14"""
    while(i_l.output_layer != True):
        for i in range(i_l.next_layer[0].neural_num):
            #calculate next layer input
            if len(i_l.next_layer[0].value) != i_l.next_layer[0].neural_num:
                i_l.next_layer[0].value.append(sigmoid(np.dot(i_l.value,i_l.w[i].transpose())-i_l.theta[i]))  
            else:
                i_l.next_layer[0].value[i]=sigmoid(np.dot(i_l.value,i_l.w[i].transpose())-i_l.theta[i])

        i_l=i_l.next_layer[0]

    return i_l.value

def backwardPropagation(o_l,output,learning_rate):
    """backward propagation to update the parameters of each layer
    date:2020/2/14"""
    while(o_l.input_layer != True):
        for i in range(o_l.last_layer[0].neural_num):    
            for j in range(o_l.neural_num):
                if len(o_l.g) != o_l.neural_num:
                    o_l.g.append(0)
                #gradient descent
                if o_l.output_layer == True:
                    #use sigmoid activation function  
                    o_l.g[j]=o_l.value[j]*(o_l.value[j]-1)*(output-o_l.value[j])
                else:                   
                    w=np.array(o_l.w)
                    #print("w:",w,"g:",o_l.next_layer[0].g)
                    w1=w[:,j]                   
                    o_l.g[j]=o_l.value[j]*(o_l.value[j]-1)*np.dot(o_l.next_layer[0].g,w1.transpose())
                #update
                o_l.last_layer[0].w[j][i]=gradientDescent(o_l.last_layer[0].w[j][i],o_l.g[j]*o_l.last_layer[0].value[i],learning_rate)
                o_l.last_layer[0].theta[j]=gradientDescent(o_l.last_layer[0].theta[j],-o_l.g[j]*o_l.last_layer[0].value[i],learning_rate)
            
        #the next layer
        o_l=o_l.last_layer[0]

    return  

def bpNeuralNetwork(input,output,neurals,learning_rate,alteration_times):
    """train the network by using the backpropagation algorithm
    date:2020/2/14
    """
    i_l,o_l=initNeuralNetwork(neurals)
    for i in range(alteration_times):
        for j in range(len(input)): 
            #forward propagation
            i_l.value=input[j][:]    
            forwardPropagation(i_l)
            #backward propagation
            backwardPropagation(o_l,output[j],learning_rate)
        print("iteration times:",i+1)
    return  i_l,o_l

def predict(i_l,x):
    """infact,the model contains w and theta as parameters so
    we can use forward propagation to predict the result
    date:2020/2/14"""
    
    i_l.value=x[:]
    
    return forwardPropagation(i_l)


def gradientDescent(o_x,g,l_r):
    """gradient descent optimization
    input  old parameter,gradient,learning_rate
    output new parameter
    date:2020/2/14
    """
    
    return o_x-g*l_r


#test   
dateSet=np.array([[0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [2, 0, 1, 2, 1],
        [2, 0, 1, 1, 1],
        [2, 1, 0, 1, 1],
        [2, 1, 0, 2, 1],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 2, 1],
            ])
output=dateSet[:,-1]
input=np.delete(dateSet,-1,axis=1)
test=[1,0,0,1]#1
print("input",input)
print("output",output[0])
i_l,o_l=bpNeuralNetwork(input,output,[4,3,1],0.001,1000)
forwardPropagation(i_l)
flag=0.0
for i in range(len(input)):
    if predict(i_l,input[i])[0] < 0.5 and output[i] == 0:
        flag+=1
    elif predict(i_l,input[i])[0] >= 0.5 and output[i] == 1:
        flag+=1

print("accuracy:",100*flag/len(input),"%")
        
#i=0
#while(i_l.output_layer!=True):
#   i+=1
#   print("layer",i,"w:",i_l.w)
#   i_l=i_l.next_layer[0]
#i+=1
#print("layer",i,"w:",i_l.w)
