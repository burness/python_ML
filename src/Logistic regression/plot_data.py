def plot_data(X,y):
    import matplotlib.pyplot as plt
    plt.figure(1)
    pos=(y==1)
    neg=(y==0)
    plt.plot(X[pos,0],X[pos,1],'k+',linewidth=2,label='Admitted')
    plt.plot(X[neg,0],X[neg,1],'ro',label='Not Admitted')
    plt.legend(loc=1)
    plt.show()
# test plot_data()    
