import numpy as np

import torch
import torch.nn as nn


# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_epochs, learning_rate=0.1):

        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        max_epochs: maximum number of epochs to run stochastic gradient descent
        learning_rate: learning rate for SGD
        '''
        ### Your Code starts

        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.W1 = nn.Linear(self.input_size, self.hidden_size)
        self.W2 = nn.Linear( self.hidden_size, self.output_size)
        ### You want to construct your MLP Here (consider the recommmended functions in HW4 writeup)

        # super(MLP, self).__init__()

    def forward(self, x):
        ### To do feed-forward pass
        ### Your Code starts here
        ### Use the layers you constructed in __init__ and pass x through the network
        A1 = nn.ReLU()
        hiddenraw = self.W1(x)
        hidden = A1(hiddenraw)
        outraw = self.W2(hidden)
        out = A1(outraw)
        return out

    def fit(self, dataloader, criterion, optimizer):

        '''
        Function used to training the MLP

        Inputs:
        dataloader: includes the feature matrix and classlabels corresponding to the training set
        criterion: the loss function used. Set to cross_entropy loss!
        optimizer: which optimization method to train the model. Use SGD here!

        Returns:
        Training loss: cross-entropy loss evaluated on the training dataset
        Training Accuracy: Prediction accuracy (0-1 loss) evaluated on the training dataset
        '''

        train_loss = []
        train_acc = []

        for i in range(self.max_epochs):
            for j, (images, labels) in enumerate(dataloader):
                X = images.reshape(-1, self.input_size)
                # Forward pass (consider the recommmended functions in HW4 writeup)
                #images.resize_(128,784)
                y = self.forward(X)
                # Backward pass and optimize (consider the recommmended functions in HW4 writeup)
                loss = criterion(y, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Track the accuracy
                train_loss.append(loss.item)
                count = 0
                for i in range(128):
                    max = 0
                    maxnum = y[i][0]
                    for j in range(10):
                        if y[i][j]>=maxnum:
                            maxnum=y[i][j]
                            max = j
                    if max == labels[i]:
                        count = count + 1
                train_acc.append(count/128)
            #train_loss = train_loss.sum()
            #train_acc = np.count_nonzero(train_acc) / len(train_acc)
        return train_loss, train_acc

    def predict(self, dataloader, criterion):
        '''
        Function used to evaluate the MLP

        Inputs:
        dataloader: includes the feature matrix and classlabels corresponding to the validation/test set
        criterion: the loss function used. Set to cross_entropy loss!

        Returns:
        Test loss: cross-entropy loss evaluated on the validation/test set
        Test Accuracy: Prediction accuracy (0-1 loss) evaluated on the validation/test set
        '''
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for j, (images, labels) in enumerate(dataloader):

                # compute output and loss
                y = self.forward(images)
                test_loss = criterion(y,labels).item
                count = 0
                for i in range(len(labels)):
                    max = 0
                    maxnum = y[i][0]
                    for j in range(10):
                        if y[i][j] >= maxnum:
                            maxnum = y[i][j]
                            max = j
                    if max == labels[i]:
                        count = count + 1
                test_acc = count / len(labels)
        return test_loss, test_acc

