def read_data(Z):
    import numpy as np
    data=np.loadtxt(Z,delimiter=",")
    # data=np.loadtxt("ex2data1.txt",delimiter=",")
    X=data[:,0:2]
    y=data[:,-1]
    y=np.reshape(y,(y.size,1))
    return X,y