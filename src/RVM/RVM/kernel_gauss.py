def kernel_gauss(x_i,x_j,lam):
    import numpy as np
    x_diff=x_i-x_j
    temp=np.dot(x_diff.T,x_diff)
    f=np.exp(-0.5*temp/(lam**2))
    return f
