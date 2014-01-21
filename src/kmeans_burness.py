import os
import scipy as sp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import pylab
def dist(x,y): # arg is points
    return np.sqrt(np.sum((x-y)**2))

def dist_Point_centroids(X,Y):
    dist_P_C=[]
    i=0
    #for y in Y:
    #    for x in X:
    #        dist_P_C.append(dist(x,y))
    #dist_P_C=np.array(dist_P_C).reshape([60,3])
    for x in X:
        dist_P_C.append(map(lambda y:dist(x,y),Y))
    dist_P_C=np.array(dist_P_C)
    return dist_P_C
def update_centroids(points,labels,num_clusters):
    centroids=[]
    for i in range(k):
        centroids.append(np.mean(points[labels==i],axis=0))   
    centroids=np.array(centroids)
    return centroids

k=3
seed=2
threshold=1.0e-6
maxIter=200
sp.random.seed(seed)
xw1=norm(loc=0.3,scale=.15).rvs(20)
yw1=norm(loc=0.3,scale=.15).rvs(20)

xw2=norm(loc=0.7,scale=.15).rvs(20)
yw2=norm(loc=0.7,scale=.15).rvs(20)

xw3=norm(loc=0.2,scale=.15).rvs(20)
yw3=norm(loc=0.8,scale=.15).rvs(20)

x=sp.append(sp.append(xw1,xw2),xw3)
y=sp.append(sp.append(yw1,yw2),yw3)
c=[]
# init cluster centroids
for i in range(k):
    x1=np.random.random_sample()
    y1=np.random.random_sample()
    c.append([x1,y1])
c=np.array(c)
p=np.array(zip(x,y))

#plt.plot(x,y,'bo')
#plt.plot(c[:,0],c[:,1],'r+')
#plt.show()
dist_p_c=dist_Point_centroids(p,c)
dist_p_c_min=np.amin(dist_p_c,axis=1)
J=np.sum(dist_p_c_min)
dist_p_c_min_index=np.argmin(dist_p_c,axis=1)
centroids_update=update_centroids(p,dist_p_c_min_index,k)
n=0
while 1:
    n=n+1
    dist_p_c=dist_Point_centroids(p,centroids_update)
    dist_p_c_min=np.amin(dist_p_c,axis=1)
    J1=np.sum(dist_p_c_min)
    dist_p_c_min_index=np.argmin(dist_p_c,axis=1)
    centroids_update=update_centroids(p,dist_p_c_min_index,k)
    print centroids_update
    if J-J1<=threshold:
        break
    J=J1
    if n>maxIter:
        break
print centroids_update
plt.plot(x,y,'bo')
plt.plot(centroids_update[:,0],centroids_update[:,1],'r+')
plt.show()