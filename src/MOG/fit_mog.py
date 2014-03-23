def fit_mog(x,K,precision=0.01):
    import numpy as np
    import numpy.matlib as nm
    from scipy.stats import multivariate_normal
    # initialize all values in lambda to 1/K
    lambda_mog=nm.repmat(1.0/K,K,1)
    I=x[:,0].size
    K_random_unique_integers=np.random.permutation(I)
    K_random_unique_integers=K_random_unique_integers[0:K]
    mu=x[K_random_unique_integers,:]
    
    # Initialize the var in sig to the var of the dataset
    dimensionality=x[1,:].size
    dataset_mean=np.sum(x,axis=0)/I
    dataset_variance=np.zeros([dimensionality,dimensionality])
    sig=np.zeros([K,dimensionality,dimensionality])
    for i in range(I):
        mat=x[i,:]-dataset_mean
        mat=np.reshape(mat,[mat.size,1])
        mat=np.dot(mat,mat.T)
        dataset_variance=dataset_variance+mat
    dataset_variance=dataset_variance/I
    for i in range(K):
        sig[i,:,:]=dataset_variance
    
    # The main loop.
    iterations=0
    previous_L=100000
    while True:
        l=np.zeros([I,K])
        r=np.zeros([I,K])

        for i in range(K):
            # compute by Bayes' rule
            l[:,i]=lambda_mog[i]*multivariate_normal.pdf(x,mu[i,:],sig[i,:,:])
        # compute the responsibilities by normalizing
        s=np.sum(l,axis=1)
        for i in range(I):
            r[i,:]=l[i,:]/s[i]

        # Maximizattion step
        r_summed_rows=np.sum(r,axis=0)
        r_summed_all=np.sum(r_summed_rows)

        for k in range(K):
            # update lambda
            lambda_mog[k]=r_summed_rows[k]*1.0/r_summed_all
            # Update mu.
            new_mu=np.zeros([1,dimensionality])
            for i in range(I):
                new_mu+=np.dot(r[i,k],x[i,:])
            mu[k,:]=new_mu/r_summed_rows[k]

            # Update sigma
            new_sigma=np.zeros([dimensionality,dimensionality])
            for i in range(I):
                mat=x[i,:]-mu[k,:]
                mat=np.reshape(mat,[mat.size,1])
                mat=np.dot(r[i,k],np.dot(mat,mat.T))
                new_sigma+=mat
            sig[k,:,:]=new_sigma*1.0/r_summed_rows[k]

        # Compute the log likelihood L.
        temp=np.zeros([I,K])
        for k in range(K):
            temp[:,k]=lambda_mog[k]*multivariate_normal.pdf(x,mu[k,:],sig[k,:,:])
        temp=np.sum(temp,axis=1)
        temp=np.log(temp)
        L=np.sum(temp)

        iterations+=1

        if np.abs(L-previous_L) < precision:
            break

        previous_L=L

        return lambda_mog,mu,sig







    




