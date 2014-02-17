def featureNormalize(X):
    import numpy as np
    X_mu=np.mean(X,axis=0)
    X_norm=X-X_mu
    X_sigma=np.std(X,axis=0)
    for i in range(len(X_sigma)):
        X_norm[:,i]=X_norm[:,i]/X_sigma[i]

    return X_norm,X_mu,X_sigma