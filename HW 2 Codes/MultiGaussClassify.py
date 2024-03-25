## CSCI_5521, HW_2, Question_2
## Xiang Liu

import numpy as np
import math



class MultiGaussClassify:
    
    def __init__(self, k, d, diag):
        self.__k = k
        self.__d = d
        self.__diag = diag
        self.__prior = [1 / k] * k 
        self.__mean = np.array([[0.0] * d] * k) ###
        self.__cov = np.array([np.identity(d)] * k)
    
    
    def fit(self, X, r):
        # calculate the imensions of array X
        N = X.shape[0]
        
        for i in range(self.__k):
            # clculate prior of class i
            
            
            r_i = 0
            for j in range(N):
                
                if r[j] == i:
                    
                    r_i = r_i + 1
                    
            self.__prior[i] = r_i / N
            
            # calculate mean of class i
            
            for a in range(self.__d):
                
                sum_x_i = 0
                
                for j in range(N):
                    if r[j] == i:
                        sum_x_i = sum_x_i + X[j][a]
                        
                        
                self.__mean[i][a] = sum_x_i / r_i
                
                
            # calculate covariance matrix of class i
            #cov_mat = np.array([np.zeros(self.__d)]*self.__d) ###
                
            cov_mat = np.array([np.zeros(self.__d)] * self.__d)
            
            for j in range(N):
                
                if r[j] == i:
                    
                    xj = X[j]
                    mi = self.__mean[i]
                    
                    xj_mi = np.array(xj - mi)[np.newaxis]
                    xj_mi_t = np.transpose(xj_mi)
                    vvt = np.matmul(xj_mi_t, xj_mi)
                    cov_mat = cov_mat + vvt
                    
            self.__cov[i] = cov_mat/ r_i
            
            if self.__diag:
                for row in range(self.__d):
                    for col in range(self.__d):
                        if row != col:
                            self.__cov[i][row][col] = 0

    ## Define predict function. Input X is the feature matrix corresponding to the validation set 
    ## and the output is the predicted labels for each point in the validation set.

    def predict(self, X):
        # the validation set has M observations
        # create an empty np.array to store the predicted output
        # dimensions of the given validation set
        
        M = X.shape[0]
        
        predicted = np.array([0] * M)
        
        # create an empty matrix of M by k to store the possibility
        
        # set a gi (discriminant function) matrix and calculate every entry of it
        g_disc = np.zeros((M, self.__k))
        
        for j in range(M):
            
            for i in range(self.__k):
                
                
                g_disc[j][i] = - 0.5 * math.log(np.linalg.det(self.__cov[i])) - 0.5 * (np.matmul(np.matmul((X[j] - self.__mean[i]).transpose(), np.linalg.inv(self.__cov[i])), (X[j] - self.__mean[i]))) + math.log(self.__prior[i])
                
        # find max gi, find the maximum of gi and make prediction
                
        for z in range(M):
            max_g = 0
            pos = 0
            for q in range(self.__k):
                max_g = g_disc[z][0]
                if g_disc[z][q] > max_g:
                    max_g = g_disc[z][q]
                    pos = q
            predicted[z] = pos
        return predicted
            



    def getk(self):
        return self.__k

    def getd(self):
        return self.__d

    def getdiag(self):
        return self.__diag

    def getprior(self):
        return self.__prior

    def getmean(self):
        return self.__mean

    def getcov(self):
        return self.__cov




