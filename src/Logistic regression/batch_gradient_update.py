def batch_gradient_update(theta,X,y,alpha=0.1,threshold=1e-6,maxIter=400):
    from sigmoid_function import sigmoid_function
    from cost_function import cost_function
    import numpy as np
    for i in range(maxIter):
        T=np.zeros(np.shape(theta))
        h=sigmoid_function(np.dot(X,theta))
        #J,grad=cost_function(theta,X,y)

        #grad=grad.reshape(np.shape(theta))
        for i in range(len(X[:,0])):
            T=T+(y[i]-h[i])*X[i].T
        theta+=alpha*T/len(X[:,0])
    return theta