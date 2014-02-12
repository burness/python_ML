from sigmoid_function import sigmoid_function
import numpy as np
import matplotlib.pyplot as plt
Zz=np.linspace(-5,5,100)
Gg=sigmoid_function(Zz)
plt.figure(1)
plt.plot(Zz,Gg)
plt.title('Sigmoid function')
plt.show()