# load data(this implementation exclude training step)
import scipy.io as sio
import numpy as np
## read data from ex3data1.mat
a=sio.loadmat('ex3data1.mat')
data=a['X']
data=np.array(data)# 5000*400
labels=a['y']
labels=np.array(labels)# 5000*1
# load W from ex3weights.mat
b=sio.loadmat('ex3weights.mat')
Theta1=b['Theta1']
Theta2=b['Theta2']
from predict_NN import predict
m,p=predict(Theta1,Theta2,data)
# compute its accuary
y=np.mean(np.array(p==labels,dtype=int),axis=0)
str='Training Set Accuracy: %f'%(y)
print str

