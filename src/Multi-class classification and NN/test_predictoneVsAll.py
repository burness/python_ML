import numpy as np
all_theta=np.load('theta.npy')
from predictOneVsAll import predictOneVsAll
p=predictOneVsAll(all_theta,X)