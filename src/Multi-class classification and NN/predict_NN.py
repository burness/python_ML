def predict(Theta1,Theta2,X):
    import numpy as np
    m,n=X.shape
    num_labels=Theta2[:,0].size
    p=np.zeros([m,1])
    X0=np.ones([m,1])
    X=np.concatenate((X0,X),axis=1)
    from sigmoid_function import sigmoid_function
    # 3 layers in the NN
    a1=X
    a2=sigmoid_function(np.dot(a1,Theta1.T))
    a2=np.concatenate((X0,a2),axis=1)
    # the layer must add all ones when it use for the next layer
    a3=sigmoid_function(np.dot(a2,Theta2.T))
    m=np.max(a3,axis=1)
    p=np.argmax(a3,axis=1)+1
    m=np.reshape(m,[m.size,1])
    p=np.reshape(p,[p.size,1])
    return m,p