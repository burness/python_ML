# Input: X->a (D+1)xI data matrix, where D is the data dimensionality
# and I is the number of training examples.
# w - a Ix1 vector containing the corresponding world states for each training example,
# nu - degrees of freedom,
# X_test - a data matrix containing training examples for which we need to make prediction
# initial_psi- Ix1 vector that represents the start solution,
# kernel - the kernel function
# lam- the parameter used in the Gaussian kernel.
#
# Output: predictions - 1xI_test row vector which contains the predicted class values for the input data in X_test
# relevant_points - Ix1 boolean vector where a 1 at position i indicates that point X(:,i) remained after
# the elimination phase, that is, it is relevant.



# The first problem, how to set the kernel function a function's parameter,seen in RVM_notebook

def fit_rvc(X,w,nu,X_test,initial_psi,kernel,lam):
    import numpy as np
    from sigmoid_function import sigmoid_function
    from numpy.linalg import inv
    from scipy.optimize import minimize
    I=X[0,:].size
    K=np.zeros([I,I])
    for i in range(I):
        for j in range(I):
            K[i,j]=kernel(X[:,i],X[:,j],lam)

    # Initialize H.
    H=np.ones([I,1])

    # The main loop.
    iterations_count=0
    mu=0
    sig=0
    def costFunction(psi):
        # It is ok to use the H w K int he scope of the function fit_rvc
        # It has an error when the second in this function
        from fit_rvc_cost import fit_rvc_cost
        L,g,Hession=fit_rvc_cost(psi,w,H,K)
        print "cost function: %s"%L
        return L
    def gradientFunction(psi):
        from fit_rvc_cost import fit_rvc_cost
        L,g,Hession=fit_rvc_cost(psi,w,H,K)
        print "gradient : %s" %g
        #print "gradient shape:%s"%(g.shape())
        return g.flatten()
    # what is the psi function in the fit_rvc.m
    while True:
        psi=minimize(costFunction,initial_psi,method='BFGS',jac=gradientFunction)
        #psi_optimize=fmin_cg(costFunction,initial_psi,gradientFunction) have a error
        #psi_optimize=psi_optimize.x
        # error here, no idea about the return of the fmin_bfgs
        #psi=fmin_bfgs(costFunction,initial_psi,fprime=gradientFunction)
        psi=psi.x



        # Compute Hessian S at peak
        # a error here diag()need a 1d array
        S=np.diag(H.flatten())
        # np.dot need 2d array
        #--------debug here in 2014-05-28------------------------------------------#
        ys=sigmoid_function(np.dot(psi.reshape([psi.size,1]).T,K))
        for i in range(I):

            y=ys[0,i]
            S=S+y*(1-y)*np.dot(K[:,i].reshape([K[:,i].size,1]),K[:,i].reshape([K[:,i].size,1]).T)
        # Set mean and variance of Laplace approximation
        mu=psi
        sig=-inv(S)

        # Update H
        H=H*(np.diag(sig).reshape([np.diag(sig).size,1]))
        H=nu+1-H
        H=H/(mu.reshape([mu.size,1])**2+nu)
        iterations_count=iterations_count+1
        print "iteration: %d"%iterations_count
        if(iterations_count==3):
            break

    threshold=1000
    selector=(H.flatten()<threshold)
    X=X[:,selector]
    mu=mu[selector]
    mu=mu.reshape([mu.size,1])
    sig=sig[selector,selector]
    sig=sig.reshape([sig.shape[0],sig.shape[0]])
    #sig=sig.reshape([sig[:,0].size,sig[0,:].size])
    relevant_points=selector
    
    print "Hessian:%s"%H

    # Recompute K[X,X]
    I=X[0,:].size
    K=np.zeros([I,I])
    for i in range(I):
        for j in range(J):
            K[i,j]=kernel(X[:,i],X[:,j],lam)

    # Recompute K[X,X_test]
    I_test=X_test[0,:].size
    K_test=np.zeros([I,I_test])
    for i in range(I):
        for j in range(I_test):
            K_test[i,j]=kernel(X[:,i],X_test[:,j],lam)
    # Compute mean and variance of activation
    mu_a=np.dot(mu.T,K_test)
    var_a_temp=np.dot(sig,K_test)
    var_a=np.zeros([1,I_test])
    for i in range(I_test):
        var_a[:,i]=np.dot(K_test[:,i].T,var_a_temp[:,i])

    # Approximate the integral to get the Bernoulli parameter.
    predictions=np.sqrt(1+np.pi/8*var_a)
    predictions=mu_a/predictions
    predictions=sigmoid_function(predictions)
    return predictions,relevant_points

        
        
        
        


