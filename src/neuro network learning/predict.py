def predict(Theta1,Theta2,X):
    import numpy as np
    m=len(X[:,0])
    num_labels=len(Theta2[:,0])
    p=np.zeros([m,1])
    from sigmoid_function import sigmoid_function
    import numpy as np
    h1=sigmoid_function(np.dot(np.concatenate((np.ones([m,1]),X),axis=1),Theta1.T))
    h2=sigmoid_function(np.dot(np.concatenate((np.ones([m,1]),h1),axis=1),Theta2.T))
    p=np.max(h2,axis=1)
    p_index=np.argmax(h2,axis=1)
    return p,p_index
    