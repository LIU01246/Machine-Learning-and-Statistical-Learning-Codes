import numpy as np
from scipy import stats
import random
import math

from sklearn.tree import DecisionTreeClassifier


class MyAdaboost:
    def __init__(self, num_iters, max_depth=None):

        """
            num_iters: [int],
                Number of tree classifiers (i.e., number of iterations or rounds-the variable T from the lecture notes) in the ensemble.
                Must  be  an  integer >=1.

            max_depth: [int, default=None],
                The maximum depth of the trees. If None, then nodes are expanded until all leaves are pure.
        """
        ### Your code starts here ###
        self.num_iters = num_iters
        self.max_depth = max_depth
        self.alphas = []
        
        



    def fit(self, X, r):
        """
            Build a AdaBoost classifier from the training set (X, r)
            X: the feature matrix 
            r: class labels 
        """

        ### Your code starts here ###
        self.trees = []
        #self.alphas = [] 
        
        N = X.shape[0] # number of samples in X
        #d = X.shape[1] # number of features in X
        
        # initialize the sample weight
        sample_weights = np.repeat(1/N, N) 
        
        for i in range(self.num_iters):
            
            sample = random.choices(np.arange(N), sample_weights, k=N)
            
            boot_X = X[sample]
            boot_r = r[sample]
            
            clf= DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=self.max_depth)
            clf.fit(boot_X, boot_r, sample_weights)
            r_hat_i = clf.predict(X)
            
            e_i = sum(sample_weights*(r_hat_i != r))/sum(sample_weights)
            alpha = 0.5*np.log((1-e_i)/e_i)
            new_weights = np.zeros(N)
            
            for j in range(len(r)):
                new_weights[j] = sample_weights[j]*math.exp(-alpha*r_hat_i[j]*r[j])
            
            Z_t = sum(new_weights)
            sample_weights = new_weights/Z_t
            
            self.trees.append(clf)
            self.alphas.append(alpha)




    def predict(self, X):    

        """
            Predict class(es) for X as the number of tree classifiers in the ensemble grows

            X: the feature matrix

            Return:

            r_pred: [list], contains predictions as the number of tree classifiers in the ensemble grows
            The list should have the same size of num_trees.
            Each element in the list has the same dimension of X (number of data points in the test set),
            and the prediction is made based on the first k tree classifiers in the ensemble as k grows from 1 to num_trees.
            E.g., when k = 1, the Bagging classifier makes predictions only based on the first tree classifier you built from self.fit function;
            when k = num_tress, the Bagging classifier makes predictions based on all tree classifiers you built from self.fit function
            

        """
        ### Your code starts here ###
        
        n_test_obs = X.shape[0]
        r_pred = np.zeros((self.num_iters, n_test_obs))
        all_pred = np.zeros((self.num_iters, n_test_obs))
        
        
        for i in range(self.num_iters):
            all_pred[i] = self.trees[i].predict(X)
        
        
        for k in range(self.num_iters):
            
            # make predictions based on the first k tree classifiers
            r_pred[k] = np.zeros((n_test_obs))
            for j in range(k+1):
                r_pred[k] += self.alphas[j]*all_pred[j]
            for m in range(n_test_obs):
                if r_pred[k][m] >= 0:
                    r_pred[k][m] = 1
                else:
                    r_pred[k][m] = -1
            


        return r_pred
