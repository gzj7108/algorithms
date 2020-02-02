def getCategoryNum(d,c):
    """
    get the moment category number to calculate the proportion
    """
    num=0
    e_num=len(d)
    for i in range(e_num):
        if d[i][-1] == c:
            num+=1

    return num

d=[[1,2],[1,2]]
c=[]
print(len(c)) 
print(getCategoryNum(d,c))