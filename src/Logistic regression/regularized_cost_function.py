def regularize_cost_function(theta,X,y,llambda):
    from sigmoid_function import sigmoid_function
    import numpy as np
    m=y.size
    J=0
    grad=np.zeros([theta.size,1])
    H=sigmoid_function(np.dot(X,theta))
    T=y*np.log(H)+(1-y)*np.log(1-H)
    J=-1*np.sum(T)/m+llambda*np.sum(theta**2)/(2*m)
    # compute the grad
    for i in range(m):
        grad=grad+np.reshape((H[i]-y[i])*X[i,:].T,grad.shape)
    grad=grad/m+llambda*theta/m
    return J,grad