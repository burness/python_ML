from read_data import read_data
from plot_data import plot_data
from cost_function import cost_function
from batch_gradient_update import batch_gradient_update
from sigmoid_function import sigmoid_function
import numpy as np
X,y=read_data("ex2data1.txt")
#plot_data(X,y)
m,n=X.shape
X=np.concatenate((np.ones([len(X[:,0]),1]),X),axis=1)
#initial_theta=np.zeros([n+1,1])
initial_theta=np.array([1,1,1])
# test is the cost_function  ok?
cost,grad=cost_function(initial_theta,X,y)
# batch_gradient_update error!!! wrong theta
theta=batch_gradient_update(initial_theta,X,y)
print theta
#cost_save=cost_save[cost_save!='nan']
#print cost_save
#prob=sigmoid_function(np.dot(np.array([1,3,43]).T,theta))
#print prob
prob=sigmoid_function(np.dot(X,theta))
print prob

