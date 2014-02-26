import scipy.io as sio
import numpy as np
## read data from ex3data1.mat
a=sio.loadmat('ex3data1.mat')
data=a['X']
data=np.array(data)# 5000*400
labels=a['y']
labels=np.array(labels)# 5000*1

#from sklearn.datasets.samples_generator import make_blobs
# display is not realted of the methods, realize it latter
#data, labels = make_blobs(n_samples=5000, centers=10, random_state=0, cluster_std=0.5)

m,n=data.shape
#from featureNormalize import featureNormalize
#data,data_mu,data_sigma=featureNormalize(data)

# test cost_function
#from cost_function import cost_function
#theta=np.zeros([n,1])
#J,cost=cost_function(theta,data,np.array(labels==4,dtype=int))

from oneVsall import oneVsall
all_theta=oneVsall(data,labels,10,0.1)
np.save("theta_50.npy",all_theta)
#all_theta=np.load('theta_50.npy')
from predictOneVsAll import predictOneVsAll
p=np.reshape(predictOneVsAll(all_theta,data),[m,1])
pred=np.array(p==labels,dtype=int)
print float(np.sum(pred))/5000.0