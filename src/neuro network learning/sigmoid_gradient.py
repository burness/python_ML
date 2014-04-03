import numpy as np
from sigmoid_function import sigmoid_function
def sigmoid_gradient(z):
    g=np.zeros(z.size)
    g=sigmoid_function(z)*(1-sigmoid_function(z))
    return g
    