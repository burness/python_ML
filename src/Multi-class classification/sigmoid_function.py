import numpy as np
def sigmoid_function(Z):
    G=1.0/(1+np.exp(-Z))
    return G

