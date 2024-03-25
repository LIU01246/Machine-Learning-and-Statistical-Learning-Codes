## CSCI_5521, HW_2, Question_2
## Xiang Liu

import numpy as np

import sklearn as sk
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from MultiGaussClassify import MultiGaussClassify

from my_cross_val import my_cross_val





X, t = datasets.load_boston(return_X_y=True)

t_50 = np.percentile(t, 50)

# for boston50
r = np.zeros(t.shape)
r[t>=t_50]=1


N = X.shape[0] # number of samples in X
d = X.shape[1] # number of features in X


### YOUR CODE STARTS HERE ###

# class MultiGaussClassify:
#     def init (self, k, d, diag):
        
        
#     def fit(self, X, r):
    
    
#     def predict(self, X):

def print_error(outcome, length, method, data):
    print("Error rate for", method, " on ", data, " :")
    for i in range(length):
        print("Fold", i+1, ":", outcome[i])
    print("Mean:", round(np.mean(outcome), 4))
    print("Standard Deviation", round(np.std(outcome), 4))
    return


oc1 = my_cross_val(MultiGaussClassify(2, 13, False), X, r, 5)
print_error(oc1, 5, "MultiGaussClassify with full covariance matrix", "Boston50")
oc2 = my_cross_val(MultiGaussClassify(2, 13, True), X, r, 5)
print_error(oc2, 5, "MultiGaussClassify with diagonal covariance matrix", "Boston50")
oc3 = my_cross_val(LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000), X, r, 5)
print_error(oc3, 5, "LogisticRegression", "Boston50")