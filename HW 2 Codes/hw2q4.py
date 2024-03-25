## CSCI_5521, HW_2, Question_4
## Xiang Liu

import numpy as np

from my_kmeans import kmeans

# for plot
import matplotlib.pyplot as plt
plt.switch_backend('agg')





fig1,ax1 = plt.subplots(2,2,figsize=(14, 8)) # to plot images
ax1 = ax1.flatten()

fig2,ax2 = plt.subplots(1,3,figsize=(14, 4),sharey=True) # to plot reconstruction error E
ax2 = ax2.flatten()


# load image
img = plt.imread('stadium.png')
w,h,c = img.shape # image width x heigth x channels
X=img.reshape(-1,c) # Nx3


ax1[0].imshow(img)
ax1[0].set_title('Original Image')
ax1[0].axis('off')


Ks = [3,5,7]

for i in range(len(Ks)):
    k = Ks[i]

    '''Executing kmeans'''
    # run kmeans on X, for K clusters, using kmeans
    ### YOUR CODE STARTS HERE ###
    r,mu,E = kmeans(X,k)
    


    '''Generate compressed image X_new with k colors resulting from kmeans'''
    # X_new will have the same size as X, but with k colors only
    ### YOUR CODE STARTS HERE ###
    
    #X_new = img.reshape(w,h,c)
    array_X_new = r.dot(mu)
    X_new = array_X_new.reshape(w,h,c)
    # print(np.shape(X_new))
    # print(X_new)
   



    # plot compressed image
    ax1[i+1].imshow(X_new)
    ax1[i+1].set_title('Compressed Image: k={}'.format(k))
    ax1[i+1].axis('off')

    # plot reconstruction error E
    ax2[i].plot(E,'*-')
    ax2[i].set_xlabel('No. of Iterations',fontsize=12)
    ax2[i].set_ylabel('Reconstruction Error',fontsize=12)    
    ax2[i].set_title('k={}'.format(k),fontsize=14)



fig1.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05,wspace=0.2,hspace=0.2)
fig1.savefig('hw2q4_image.png')

fig2.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.15,wspace=0.1)
fig2.savefig('hw2q4_error.png')
