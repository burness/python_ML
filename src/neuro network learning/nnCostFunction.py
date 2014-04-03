def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_nn):
    import numpy as np
    from sigmoid_function import sigmoid_function
    from sigmoid_gradient import sigmoid_gradient
    Theta1=np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],[hidden_layer_size,input_layer_size+1])
    Theta2=np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],[num_labels,hidden_layer_size+1])
    m=len(X[:,0])
    J=0
    Theta1_grad=np.zeros(Theta1.size)
    Theta2_grad=np.zeros(Theta2.size)
    X=np.concatenate((np.ones([len(X[:,0]),1]),X),axis=1)
    a1=X
    z2=np.dot(a1,Theta1.T)
    a2=sigmoid_function(z2)
    a2=np.concatenate((np.ones([len(X[:,0]),1]),a2),axis=1)
    a3=sigmoid_function(np.dot(a2,Theta2.T))
    #num_labels_eye=np.eye(num_labels)
    #ry=num_labels_eye[y,:]
    ry=np.zeros([len(X[:,0]),num_labels])
    for i in range(5000):
        ry[i,y[i]-1]=1
    cost=ry*np.log(a3)+(1-ry)*np.log(1-a3)
    J=-np.sum(cost)/m
    reg=np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2)
    J=J+lambda_nn*1.0/(2*m)*reg
    
    # Backpropagation algorithm
    delta3=a3-ry
    temp=np.dot(delta3,Theta2)
    delta2=temp[:,1:]*sigmoid_gradient(z2)

    Delta1=np.dot(delta2.T,a1)
    Delta2=np.dot(delta3.T,a2)

    Theta1_grad=Delta1/m+lambda_nn*np.concatenate((np.zeros([hidden_layer_size,1]),Theta1[:,1:]),axis=1)/m
    Theta2_grad=Delta2/m+lambda_nn*np.concatenate((np.zeros([num_labels,1]),Theta2[:,1:]),axis=1)/m

    Theta1_grad=np.reshape(Theta1_grad,[Theta1_grad.size,1])
    Theta2_grad=np.reshape(Theta2_grad,[Theta2_grad.size,1])
    grad=np.concatenate((Theta1_grad,Theta2_grad),axis=0)
    return J,grad
