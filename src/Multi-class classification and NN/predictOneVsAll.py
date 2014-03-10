def predictOneVsAll(all_theta,X):
    import numpy as np
    m,n=X.shape
    num_labels=all_theta[:,0].size
    p=np.zeros([m,1])
    X0=np.ones([m,1])
    X=np.concatenate((X0,X),axis=1)
    print X.shape
    from sigmoid_function import sigmoid_function
    C=sigmoid_function(np.dot(X,all_theta.T))
    print C.shape
    M=C.max(axis=1)
    p=C.argmax(axis=1)
    return p