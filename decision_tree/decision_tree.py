"""
decision tree model 

"""

import math
import numpy as np

class Node():
    """
    define a node model to store information
    """

    def __init__(self):
        """
        init the node model
        """
        #use a list to store the connection for the convenience of extension 
        self.last_node=[]
        self.next_node=[]
        self.value=[]
        #mark the right node category to judge the root and the leaf
        self.root_node=False
        self.leaf_node=False
        #the choosen attribute
        self.attributes=[]
        #the final category
        self.category=0
        
        

def generateTree(d,a,n):
    """
    generate the initial tree by iteration
    d:dataset
    a:attributeset
    """

    #generate the node
    node=Node()
    node.last_node.append(n)
    #if the moment examples contains the same category,return
    c_list=getCategory(d,-1)
    c_list=list(c_list)
    if len(c_list) == 1:
        node.leaf_node=True
        leaf_node.append(node)
        node.category=c_list[0]
        return node

    #if the moment attributeset contains none or the dataset have the same value on attributeset
    d_purity=np.array(d)
    d_purity=np.delete(d_purity,-1,axis=1)
    flag=len(list(set([tuple(d) for d in d_purity])))
    if len(a) == 0 or flag == 1:
        c_best=c_list[0]
        
        for c in c_list:
            if getCategoryNum(d,c_best,-1) < getCategoryNum(d,c,-1):
                c_best=c
        node.leaf_node=True
        leaf_node.append(node)
        node.category=c_best
        return node

    #choose the optimal attribute
    a_optimal=a[0]
   
    for i in a:
        if calculateInfGain(d,a_optimal) < calculateInfGain(d,i):
            a_optimal=i
    node.attributes.append(a_optimal)
   
    a_list=getCategory(d,a_optimal)
    a.remove(a_optimal)
    for i in a_list:
        d_son=getSonData(d,a_optimal,i)
        if len(d_son) == 0:
            c_best=c_list[0]
            for c in c_list:
                if getCategoryNum(d,c_best,-1) < getCategoryNum(d,c,-1):
                     c_best=c
            node.leaf_node=True
            leaf_node.append(node)
            node.category=c_best
            return node
        else:
            node.next_node.append(generateTree(d_son,a,node))
            node.value.append(i)
    

    return node


def postpruning():
    """
    prun the tree to avoid overfitting
    """

def calculateInfGain(d,a_index):
    """
    calculate the information entropy to choose the optimal attribute
    input the moment examples
    d:dataset [attribut1,attribute2,...,category]
    a_c:the chosen attribute index
    """

    #calculation
    sum=calculateInfEnt(d)
    a_list=getCategory(d,a_index)

    for i in a_list:
        d_son=getSonData(d,a_index,i)
        sum+=(-1)*calculateInfEnt(d_son)*(len(d_son)/len(d))

    return sum
    

def getSonData(d,index,a_value):
    """
    get the son dataset which divided by the attribute value

    output the son dataset
    """
    d_son=[]
    for i in range(len(d)):
        if a_value == d[i][index]:
            d_son.append(d[i])
    
    return d_son

def calculateInfEnt(d):
    """
    calculate the information entropy to choose the optimal attribute
    input the moment examples
    d:dataset [attribut1,attribute2,...,category]
    output the information entropy of the dataset
    """
    #get the moment category list
    c_num=getCategory(d,-1)
    #get the numbers of the examples
    e_num=len(d)
    #calculation
    sum=0
    for i in c_num:
        p=getCategoryNum(d,i,-1)/e_num
        sum+=(-1)*p*math.log(p,2)
    
    return sum

def getCategory(d,index):
    """
    get the moment category list to calculate the proportion
    index:determine the attributes or the category
        -1,category
        others,attributes
    output the duplicate removal list
    """

    c=[]
    for i in range(len(d)):
        c.append(d[i][index])

    return set(c)#duplicate removal
        

def getCategoryNum(d,value,index):
    """
    get the moment category number to calculate the proportion
    index:determine the attributes or the category
        -1,category
        others,attributes
    output the number of the chosen attribute or the category

    this function can be replaced by the function getSonData,
    but to help uderstand,I reserve it
    """

    num=0
    e_num=len(d)
    for i in range(e_num):
        if d[i][index] == value:
            num+=1

    return num


def decisionTreeID3(n_model,d_judge):
    """use the model we have trained
    input:
    n_model:the model that has been trained as a root node
    d_judge:the data to be judged

    output:the category of the d_judge
    
    """
    
    if n_model.root_node == True:
        n_model=n_model.next_node[0]
    while n_model.leaf_node == False:
        attribute=n_model.attributes[0]#the chosen attribute
        n_model=n_model.next_node[n_model.value.index(d_judge[attribute])]

    return n_model.category

def getRootNode(dataSet):
    """
    generate the tree and return  a root node to traversal the data
    """
    n=Node()
    n.root_node=True
    #set the number of the attributes
    a=[0,1,2,3]
    n.next_node.append(generateTree(dataSet,a,n))

    return n

#test
dataSet=[[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
testSet=[[0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes']]


#test
leaf_node=[]
root_node=getRootNode(dataSet)
num=0
for i in range(len(testSet)):
    if decisionTreeID3(root_node,testSet[i]) == testSet[i][-1]:
        num+=1
print("accuracy:",100*num/len(testSet),"%")
