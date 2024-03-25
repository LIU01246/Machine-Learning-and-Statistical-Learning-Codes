#liu xiang

import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


class MyLogisticRegression:

    def __init__(self, d, max_iter=1000, eta=1e-3):

        '''
        d: [int], feature dimension used to initialize weights
        max_iter: maximum number of iterations to run gradient descent
        eta: learning rate for gradient descent
        '''
        ### Your Code starts here
        self.d = d
        self.max_iter = max_iter
        self.eta = eta
        ### Initialize w_j with random values close to 0: [-0.01, 0.01]
        self.w_vec = (np.random.rand(1, d) - 0.5) / 50
        self.w_0 = (np.random.rand() - 0.5) / 50
        self.error_vec = np.zeros(self.max_iter)


    def fit(self, X, r):
        #X = self.zscoring(X)
        ### Your Code starts here
        ### Don't forget to check convergence!
        #iter_change = 1
        N = X.shape[0]
        self.error_vec = np.zeros(self.max_iter)
        prev_E = 0.0
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        for i in range(self.max_iter):
            # print("---i: ", i)
            delta_w_vec = np.zeros(self.d)
            
            y = np.zeros(N)
            for t in range(N):
                o = np.dot(self.w_vec,X[t]) + self.w_0
                y[t] = self.sigmoid(o)
            total = 0
            for j in range(N):
                total = total + (r[j]*np.log(y[j])+(1-r[j])*np.log(1-y[j]))
            total = -total
            current_E = total
            

            if abs(current_E - prev_E) < 10 ** (-6):
                print("BREAK i: ", i)
                break

            prev_E = current_E

            for k in range(self.d):
                self.w_vec[0][k] = self.w_vec[0][k] + self.eta * np.dot(X[:,k],r-y)
            self.w_0 = self.w_0 + self.eta*sum(r-y)
            """#print("w_vec = ", self.w_vec)
            #print("y: ", y)
            for t in range(N):
                E = E - (r[t]*np.log(y[t])+(1-r[t])*np.log(1-y[t]))
            #print("E: ", E)
            self.error_vec[i] = E
            #print("error_vec 12: ", error_vec[0:12])"""


    def predict(self, X):
        #X = self.zscoring(X)
        ### Your Code starts here
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        N = X.shape[0]
        r_pred = []
        for t in range(N):
            pred = np.dot(self.w_vec,X[t]) + self.w_0
            if pred > 0.5:
                r_pred.append(1)
            else:
                r_pred.append(0)
        return r_pred

    def sigmoid(self, y):
        #X = self.zscoring(X)
        epsilon = 10 ** (-10)
        
        
        sigmoid_val = np.exp(y) / (1 + np.exp(y))
        if sigmoid_val < epsilon:
            clipped_sigmoid_val = epsilon
        elif sigmoid_val > 1 - epsilon:
            clipped_sigmoid_val = 1 - epsilon
        else:
            clipped_sigmoid_val = sigmoid_val
        return clipped_sigmoid_val
