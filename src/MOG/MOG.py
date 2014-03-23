# genearte multi-norm data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
mu1=np.array([1,2])
sig1=np.array([[2,0],[0,0.5]])
mu2=np.array([1,5])
sig2=np.array([[1,0],[0,1]])

data1=np.random.multivariate_normal(mu1,sig1,5000)
data2=np.random.multivariate_normal(mu2,sig2,5000)
data=np.concatenate((data1,data2))
plt.scatter(data[:,0],data[:,1])
plt.axis([-6,6,-1,9])
plt.xlabel('x')
plt.ylabel('y')
plt.title('(a)')
# Fit Mog using our funtion fit_mog (EM)
from fit_mog import fit_mog
lambda_mog,mu,sig=fit_mog(data,2,0.01)
# Plot the mixture of Gaussians as contour plot
xx=np.arange(-10,10,0.01)
yy=np.arange(-10,10,0.01)
n=xx.size
xx,yy=np.meshgrid(xx,yy)
x=np.reshape(xx,[xx.size,1])
y=np.reshape(yy,[yy.size,1])
x_y_matrix=np.concatenate((x,y),axis=1)
# compute the 1st Gaussian as a matrix
temp1=multivariate_normal.pdf(x_y_matrix,mu[0,:],sig[0,:,:])
gaussian1=np.reshape(temp1,[n,n])
# compute the 2nd Gaussian as a matrix
temp2=multivariate_normal.pdf(x_y_matrix,mu[1,:],sig[1,:,:])
gaussian2=np.reshape(temp2,[n,n])
# combine the two Gaussian to obtain the final mixture of Gaussians
mog=lambda_mog[0]*gaussian1+lambda_mog[1]*gaussian2
# create the contour plot
plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.axis([-6,6,-1,9])
plt.xlabel('x')
plt.ylabel('y')
plt.title('(a)')
plt.contour(xx,yy,mog,10)
plt.show()