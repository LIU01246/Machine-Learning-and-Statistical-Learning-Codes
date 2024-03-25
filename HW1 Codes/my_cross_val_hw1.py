# CSCI5521 HW_1 Problem 4
# Xiang Liu 
# Discussed and colabrated with Shuang Liu & Han Lu




#import sklearn as sk
# X, t = sk.datasets.load_boston(return_X_y = True)

import numpy as np

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def q4():
    
    X, t = load_boston(return_X_y=True)

    # Boston50
    n = len(t)
    r50 = []
    m_50 = np.median(t)
    for i in range(n):
        if t[i] >= m_50:
            r50.append(1)
        else:
            r50.append(0)

    # Boston75
    r75 = []
    m_75 = np.percentile(t, 75)
    for i in range(n):
        if t[i] >= m_75:
            r75.append(1)
        else:
            r75.append(0)

    # Digits
    X_d, r_d = load_digits(return_X_y=True)


    # Call
    arr1, mean1, std1 = my_cross_val('LinearSVC', X, r50, 10)
    print_error((1-arr1), 10, 'LinearSVC', 'Boston50')
    
    arr2, mean2, std2 = my_cross_val('LinearSVC', X, r75, 10)
    print_error((1-arr2), 10, 'LinearSVC', 'Boston70')
    
    arr3, mean3, std3 = my_cross_val('LinearSVC', X_d, r_d, 10)
    print_error((1-arr3), 10, 'LinearSVC', 'Digits')
    
    arr4, mean4, std4 = my_cross_val('SVC', X, r50, 10)
    print_error((1-arr4), 10, 'SVC', 'Boston50')
    
    arr5, mean5, std5 = my_cross_val('SVC', X, r75, 10)
    print_error((1-arr5), 10, 'SVC', 'Boston70')
    
    arr6, mean6, std6 = my_cross_val('SVC', X_d, r_d, 10)
    print_error((1-arr6), 10, 'SVC', 'Digits')
    
    arr7, mean7, std7 = my_cross_val('LogisticRegression', X, r50, 10)
    print_error((1-arr7), 10, 'LogisticRegression', 'Boston50')
    
    arr8, mean8, std8 = my_cross_val('LogisticRegression', X, r75, 10)
    print_error((1-arr8), 10, 'LogisticRegression', 'Boston70')
    
    arr9, mean9, std9 = my_cross_val('LogisticRegression', X_d, r_d, 10)
    print_error((1-arr9), 10, 'LogisticRegression', 'Digits')
    
# Note: I didn't round the output here because I think in this way the results are more accurrate.
# However, the results in my summary tables were rounded.

    
def print_error(outcome, length, method, data):
    print("Error rates for", method, " with ", data, " :")
    for i in range(length):
        print("Fold", i+1, ":", outcome[i])
    print("Mean:", round(np.mean(outcome),4))
    print("Standard Deviation:", round(np.std(outcome),4))
    return


def my_cross_val(method,X,r,k):
# For each data set and each method, using my_cross_val, report the error rates in each of the 
# k=10 folds as well as the mean and standard deviation of error rates across folds for the 
# different methods applied to the three classication datasets. The table of the results for each method
# and each dataset is on the pdf file. Please chaeck that.

    if method == 'LinearSVC':
        model = LinearSVC(max_iter=2000, dual=False)

    elif method == 'SVC':
        model = SVC(gamma='scale', C=10)

    elif method == 'LogisticRegression':
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=5000)

    scores = cross_val_score(model, X, r, cv=k)
    return scores, scores.mean(), scores.std()


if __name__ == '__main__':
    q4()
