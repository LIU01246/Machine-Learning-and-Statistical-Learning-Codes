## CSCI_5521, HW_2, Question_3
## Xiang Liu



import numpy as np

##def meanX(dataX):
##    return np.mean(dataX,axis=0)

#def myPCA(X,k):
    

def myPCA(X, k):
    
    #average = meanX(X) 
    m, n = np.shape(X) #
    
    
    avgs = np.mean(X, axis=0)
    
    data_adjust = X - avgs
    
    covX = np.cov(data_adjust.T)   # calculate cov.
    
    featValue, featVec =  np.linalg.eig(covX)  
    # the first element represents eigenvalues and the second one represents eigenvectors
    
    index = np.argsort(-featValue) #rank featValue
    newVec = featVec[:,index] # rank featVec  
    print(index)
    print(newVec)
    selectVec = newVec[:,0:k] # Tanspose
 
    return selectVec, avgs

#print(myPCA(Xt,2)
    

def ProjectDatapoints(X,W,mu_bar):
    Xd = X - mu_bar
    return np.matmul(Xd,W)