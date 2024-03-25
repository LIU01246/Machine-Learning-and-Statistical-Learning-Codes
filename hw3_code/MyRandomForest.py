import numpy as np
from scipy import stats
import random


from sklearn.tree import DecisionTreeClassifier


class MyRandomForest:

    def __init__(self, num_features, num_trees, max_depth=None):

        """
            num_features: [int]
                Number of random features to choose at each split in the decision tree
                (the variable m from the lecture notes, and must be 1 <= num_features <= d),

            num_trees: [int],
                Number of tree classifiers in the ensemble (must be an integer >=1)


            max_depth: [int, default=None],
                The maximum depth of the trees. If None, then nodes are expanded until all leaves are pure.
        """
        ### Your code starts here ###
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.num_features = num_features
        
        


    def fit(self, X, r):
        """
            Build a RandomeForest classifier from the training set (X, r)
            X: the feature matrix 
            r: class labels 
        """

        ### Your code starts here ###
        self.trees = []
        
        
        N = X.shape[0] #number of samples in X
        


        for i in range(self.num_trees):
            sample = np.random.choice(np.arange(N), size=N, replace=True)
            boot_X = X[sample]
            boot_r = r[sample]
            tree = DecisionTreeClassifier(criterion='entropy', random_state=0,
                                            max_depth=self.max_depth,
                                            max_features=self.num_features)
            tree.fit(boot_X, boot_r)
            self.trees.append(tree)


    def predict(self,X):
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
        
        r_pred_init = np.zeros((self.num_trees, X.shape[0]))
        r_pred = np.zeros((self.num_trees, X.shape[0]))
        
        
        for i in range(self.num_trees):
            r_pred_init[i] = self.trees[i].predict(X)
            
        
        for k in range(self.num_trees):
            # make predictions based on the first k tree classifiers
            
            r_pred[k] = stats.mode(r_pred_init[0:k+1], axis=0)[0]
            
        # for i in range(self.num_trees):
        #     r_pred_init[i] = self.trees[i].predict(X)
        
        
        # for k in range(self.num_trees):
            
        #     # make predictions based on the first k tree classifiers
        #     r_pred[k] = np.zeros((X.shape[0]))
        #     for j in range(k+1):
        #         r_pred[k] += self.trees[j]*r_pred_init[j]
        #     for m in range(X.shape[0]):
        #         if r_pred[k][m] >= 0:
        #             r_pred[k][m] = 1
        #         else:
        #             r_pred[k][m] = -1
            

        return r_pred

