def cost_function(theta,X,y):
    from sigmoid_function import sigmoid_function
    import numpy as np
    m=y.size
    J=0
    grad=np.zeros(theta.size)
    H=sigmoid_function(np.dot(X,theta))
    T=y*np.log(H)+(1-y)*np.log(1-H)
    J=-np.sum(T)/m
    # compute the grad
    for i in range(m):
        grad=grad+(H[i]-y[i])*X[i,:].T
    grad=grad/m
    return J,grad