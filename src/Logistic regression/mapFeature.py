# our vector of two features transformed into a 28-dimensional vector
def mapFeature(X):
    import numpy as np
    degree=6
    n=0
    #out = np.ones(X[:,0].size).reshape([X[:,0].size],1])
    out =[]
    for i in range(degree+1):
        if i>0:
            for j in range(i+1):
                temp=(X[:,0]**(i-j))*(X[:,1]**j)
                out.append(list(temp))
   
    return np.array(out)