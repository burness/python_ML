# It is a Linear Regression with python by Burness from SHU
# read the data from ex1data2.txt
import numpy as np
def cost_function(data,prices,W):
    J=0.5*np.sum((Hypoth(data,W)-prices)**2)/len(prices)
    return J

def Hypoth(data,W):
    H=np.dot(data,W.T)
    return H

def batch_gradient_update(W,data,prices,alpha,threshold,maxIter):
    n=0
    while(1):
        n+=1
        T=np.array(np.zeros(np.shape(W)))
        for i in range(len(training_data[:,0])):
            T=T+np.dot((prices[i]-Hypoth(data,W)[i]).T,data[i].reshape(np.shape(W)))
        W=W+(alpha*T)/len(training_data[:,0])
            #W+=alpha*np.dot((prices-Hypoth(data,W)).T,data)
        #J1=cost_function(data,prices,W)
        #if J-J1<threshold:
        #    break
        if n>maxIter:
            break
        #J1=J
    return W

if __name__=="__main__":
    training_data_prices=np.loadtxt("ex1data2.txt",delimiter=",")
    training_data=training_data_prices[:,:2]
    prices=training_data_prices[:,2:]
    for i in range(len(training_data[1,:])):
        training_data[:,i]=(training_data[:,i]-np.min(training_data[:,i]))/(np.max(training_data[:,i])-np.min(training_data[:,i]))
    threshold=1e-6
    maxIter=100
    # append all ones col to training_data 
    # W0=np.ones(len(training_data[:,0]),1)
    X0=np.ones([len(training_data[:,0]),1])
    training_data=np.concatenate((X0,training_data),axis=1)
    W=np.array(np.ones(np.size(training_data[1,:])).reshape(len(training_data[1,:]),1)).T
    #W=np.array([[1,1,1]])
    alpha=0.1
    #W=W.T
    W_LR=batch_gradient_update(W,training_data,prices,alpha,threshold,maxIter)
    #test=np.array([[1,0.3252,0.5]])
    test=np.array([[1,0.3252,0.5]])
    print Hypoth(test,W_LR)
