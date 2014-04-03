input_layer_size=400
hidden_layer_size=25
num_labels=10

# load data form ex4data1.mat
import scipy.io as sio
import numpy as np
from nnCostFunction import nnCostFunction
from sigmoid_gradient import sigmoid_gradient
a=sio.loadmat('ex4data1.mat')
data=a['X']
data=np.array(data)
labels=a['y']
labels=np.array(labels)
# the 10s in labels convert to 0s
#labels[np.nonzero(labels==10)]=0
# load weight
b=sio.loadmat('ex4weights.mat')
Theta1=b['Theta1']
Theta2=b['Theta2']
Theta1=np.reshape(Theta1,[Theta1.size,1])
Theta2=np.reshape(Theta2,[Theta2.size,1])
nn_params=np.concatenate((Theta1,Theta2))
# compute cost(FeedForward) with lambda_nn=0
print "FeedForwad Using Neural Network..."
lambda_nn=0
J,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,data,labels,lambda_nn)
print "Cost at parameters (loaded from ex4weight): (%s)"%(J)

# Compute the cost with lambda_nn=1
print "FeedForwad Using Neural Network...(with lambda=1)"
lambda_nn=1
J,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,data,labels,lambda_nn)
print "Cost at parameters (loaded from ex4weight): (%s)"%(J)

# Sigmoid Gradient
print "Evaluating sigmoid gradient..."
test=np.array([1,-0.5,0,0.5,1])
g=sigmoid_gradient(test)
print "sigmoid gradient with :[1,-0.5,0,0.5,1] (%s)"%(g)

# Initializing Pameters
print "Initializing Neural Network Parameters..."
from randInitializeWeights import randInitializeWeights
initial_Theta1=randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2=randInitializeWeights(hidden_layer_size,num_labels)
initial_Theta1=np.reshape(initial_Theta1,[initial_Theta1.size,1])
initial_Theta2=np.reshape(initial_Theta2,[initial_Theta2.size,1])
initial_nn_params=np.concatenate((initial_Theta1,initial_Theta2))

def costFunction(p):
    J,gradient=nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,data,labels,lambda_nn)
    print "training"
    print J
    return J
def gradientFunction(p):
    J,gradient=nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,data,labels,lambda_nn)
    gradient=np.ndarray.flatten(gradient)
    return gradient
# Training Neural Newwork...
print "Training Neural Network"
from scipy.optimize import fmin_cg
#res=fmin_cg(costFunction,initial_nn_params,gradientFunction)
# fmin_bfgs would get an memory error
res=fmin_cg(costFunction,initial_nn_params,gradientFunction,maxiter=100)
res=np.ndarray.flatten(res)
Theta1=np.reshape(res[0:hidden_layer_size*(input_layer_size + 1)],[hidden_layer_size,(input_layer_size+1)])
Theta2=np.reshape(res[hidden_layer_size*(input_layer_size+1):],[num_labels,(hidden_layer_size+1)])
from predict import predict
p_index,p=predict(Theta1,Theta2,data)
labels=np.ndarray.flatten(labels)
aa=np.array(p==(labels-1),dtype=int)
accuracy=np.mean(aa)
print "accuracy is (%s)"%(accuracy)