from read_data import read_data
from plot_data import plot_data
from cost_function import cost_function
from batch_gradient_update import batch_gradient_update
from sigmoid_function import sigmoid_function
from featureNormalize import featureNormalize
import numpy as np
from mapFeature import mapFeature
from regularized_cost_function import regularize_cost_function
import scipy as sp
X,y=read_data("ex2data2.txt")
# after featureNormalize it accuarcy could get 89%
X,X_mu,X_sigma=featureNormalize(X)

#plot_data(X,y)
y=np.reshape(y,(y.size,1))
#*********** mapFeature 2D-->28D
X=mapFeature(X)
X=np.concatenate((np.ones([len(X[0,:]),1]),X.T),axis=1)
llambda=1
m,n=X.shape
initial_theta=np.zeros([n,1])
cost,grad=regularize_cost_function(initial_theta,X,y,llambda)
theta=batch_gradient_update(initial_theta,X,y,llambda)
print theta
prob=sigmoid_function(np.dot(X,theta))
print prob
prob[prob>0.5]=1.0
prob[prob<0.5]=0.0
print prob
y=np.reshape(y,prob.shape)
print "accuracy:",tuple(1-sum(abs(prob-y))/100)
