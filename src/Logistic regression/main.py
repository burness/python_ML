from read_data import read_data
from plot_data import plot_data
from cost_function import cost_function
from batch_gradient_update import batch_gradient_update
from sigmoid_function import sigmoid_function
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
import scipy as sp
#X,y=read_data("ex2data1.txt")

X, y = make_blobs(n_samples=400, centers=2, random_state=0, cluster_std=1)

#plot_data(X,y)
y=np.reshape(y,(y.size,1))
m,n=X.shape
X=np.concatenate((np.ones([len(X[:,0]),1]),X),axis=1)
initial_theta=np.zeros([n+1,1])


#initial_theta=np.array([1,1,1])
# test is the cost_function  ok?
cost,grad=cost_function(initial_theta,X,y)


# batch_gradient_update error!!! wrong theta
theta=batch_gradient_update(initial_theta,X,y)
print theta

prob=sigmoid_function(np.dot(X,theta))
print prob
prob[prob>0.5]=1.0
prob[prob<0.5]=0.0
print prob
y=np.reshape(y,prob.shape)
print "accuracy:",tuple(1-sum(abs(prob-y))/100)


