# create a grid. Set granlarity to 100/1000, for low/high quality plots
import numpy as np
from fit_rvc import fit_rvc
from kernel_gauss import kernel_gauss
granularity=100
a=-5
b=5
domain=np.linspace(a,b,granularity)
X,Y=np.meshgrid(domain,domain)
x=X.reshape([X.size,1])
y=Y.reshape([Y.size,1])
n=X[1,:].size

# Generate 2D data from normal distributions
mu=np.array([[-1,2.5],[1,-2.5]])
sig=np.array([[0.5,0],[0,0.5]])
points_per_class=20
X_data1=np.random.multivariate_normal(mu[0,:],sig,points_per_class)
X_data2=np.random.multivariate_normal(mu[1,:],sig,points_per_class)
X_data=np.concatenate((X_data1,X_data2))
# Prepare the training input
X_train=np.concatenate((np.ones([1,X_data[:,0].size]),X_data.T))
w=np.concatenate((np.zeros([points_per_class,1]),np.ones([points_per_class,1])))
var_prior=6
X_test=np.concatenate((np.concatenate((np.ones([1,granularity*granularity]),x.T)),y.T))

lam=0.3

# Fit a relevance vector classification model.
initial_psi=np.zeros([X_train[0,:].size,1])
nu=0.0005
predictions,relevant_points=fit_rvc(X_train,w,nu,X_test,initial_psi,kernel_gauss,lam)


