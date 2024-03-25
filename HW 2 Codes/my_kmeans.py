## CSCI_5521, HW_2, Question_4
## Xiang Liu


import numpy as np

np.random.seed(0)



def kmeans(X,k=5):
    '''The kmeans performs the Kmeans algorithm
    # The parameters received are:
    # - X (N x 3): Matrix for a color image RGB, where N is the number of pixels. 
    # - k (1 x 1): Number of clusters (number of colors in the compression)
    # 
    # The function should return:
    # - r (N x K): Contains 0 or 1, where r(n,k) contains 1 if pixel n belongs to cluster k, otherwise 0
    # - mu (k x 3): Contains the k centroids found, representing the k colors learned
    # - E : reconstruction errors during training
    '''



    max_iter = 100 ###
    '''initial mu'''
    N = len(X) # number of samples
    m_idx = np.random.permutation(N)
    mu = X[m_idx[:k],:] # (k x 3) <-- mu starts with random rows in X
    
    r = np.zeros((N,k))
    E = []


    for i in range(max_iter):

        # update r
        ### YOUR CODE STARTS HERE ###
        
        # define N*k matrix for marking cluster 
        
        # r = np.array([[0 for i in range(N)] for j in range(k)])
        
        
        l2_norm = np.zeros((N,k))
        
        for p in range(N):
            # min_dist = np.sqrt(np.sum((X[p] - mu[0])**2))
            # q_idx = 0
            
            for q in range(k):
                l2_norm[p][q] = np.linalg.norm(X[p] - mu[q])
                
                # solve for argmin problem of the l2-norm and find coresponding class
                min_l2 = np.argmin(l2_norm[p])
            for q in range(k):
                if q == min_l2:
                    r[p][q] = 1
                else:
                    r[p][q] = 0
            
        
                           
        


        # Calculate the total reconstruction error define in Textbook Eq.(7.3) 
        ### YOUR CODE STARTS HERE ###
        
        
        # E = np.array([])
        e = 0
        for m in range(N): 
            
            for n in range(k):
                if r[m][n] == 1:
                    
                    e = e + l2_norm[m][n]
                # else:
                #     mid = 0
                #     e = e
        E.append(e)
            
        
        
        

        print('Iteration {}: Error {}'.format(i,E))

        

        # update mu
        ### YOUR CODE STARTS HERE ###
        
        for z in range(k):
            c = 0
            s1 = 0
            s2 = 0
            s3 = 0
            
            for a in range(N):
                
                if r[a][z]==1:
                    c = c + 1
                    
                    s1 += X[a][0]
                    s2 += X[a][1]
                    s3 += X[a][2]
                    
                    mu[z][0] = s1/c
                    mu[z][1] = s2/c
                    mu[z][2] = s3/c
                    
        # print(np.shape(r))  
        # print(np.shape(mu))


        # check convergence
        # by checking if the the error function decreased less than 1e-6 from the previous iteration.
        # Break loop if converged
        ### YOUR CODE STARTS HERE ###
        
        if i >= 1:
        
            if E[i-1] - E[i] < 1e-6:
                break

    return r,mu,E


