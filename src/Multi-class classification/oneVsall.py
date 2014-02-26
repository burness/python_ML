def oneVsall(X,y,num_labels,lamdba):
    from batch_gradient_update import batch_gradient_update
    import numpy as np
    m,n=X.shape
    all_theta=np.zeros([num_labels,n+1])
    X0=np.ones([m,1])
    X=np.concatenate((X0,X),axis=1)
    for c in range(num_labels):
        print 'c:',c
        initial_theta=np.zeros([n+1,1])
        #print y%10
        theta=batch_gradient_update(initial_theta,X,np.array(((y%10)==c),dtype=int))

        all_theta[c,:]=np.reshape(theta,[n+1,])
    return all_theta
