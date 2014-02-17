def batch_gradient_update(theta,X,y,alpha=1,threshold=1e-6,maxIter=1000,llambda=1):
    from sigmoid_function import sigmoid_function
    from cost_function import cost_function
    import numpy as np
    for i in range(maxIter):
        T=np.zeros(np.shape(theta))
        h=sigmoid_function(np.dot(X,theta))
        for i in range(len(X[:,0])):
            T=T+(y[i]-h[i])*(X[i].reshape(np.shape(theta)))
        theta+=alpha*T/len(X[:,0])
    theta+=llambda*theta/len(X[:,0])
    return theta