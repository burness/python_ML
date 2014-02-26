def batch_gradient_update(theta,X,y,alpha=0.2,threshold=1e-6,maxIter=500):
    from sigmoid_function import sigmoid_function
    from cost_function import cost_function
    import numpy as np
    for i in range(maxIter):
        J,grad=cost_function(theta,X,y)
        theta+=alpha*grad.reshape(np.shape(theta))
    return theta
