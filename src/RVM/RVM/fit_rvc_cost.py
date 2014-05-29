def fit_rvc_cost(psi,w,Hd,K):
    import numpy as np
    from scipy.stats import multivariate_normal
    from sigmoid_function import sigmoid_function
    I=K[:,0].size
    # ensure that Hd is 1-D array
    Hd=np.reshape(Hd,[Hd.size,])
    Hd_diag=np.diag(Hd)
    # error in mvnpdf
    psi=psi.flatten()
    mvnpdf=multivariate_normal.pdf(psi,np.zeros([40,]),np.diag(1/Hd))
    L=I*(-1)*np.log(mvnpdf)
    psi=np.reshape(psi,[psi.size,1])
    g=I*np.dot(Hd_diag,psi)
    H=I*Hd_diag
    predictions=sigmoid_function(np.dot(psi.T,K))
    predictions=np.reshape(predictions,[predictions.size,1])
    for i in range(I):
        # update L
        y=predictions[i,0]
        if w[i]==1:
            L=L-np.log(y)
        else:
            L=L-np.log(1-y)

        # update g and H, debug here 2014/05/26
        K_temp=np.reshape(K[:,i],[K[:,i].size,1])
        g=g+(y-w[i])*K_temp
        H=H+y*(1-y)*np.dot(K_temp,K_temp.T)

    return L,g,H
