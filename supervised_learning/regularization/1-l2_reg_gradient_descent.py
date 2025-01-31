#!/usr/bin/env python3
'''
NOTE:
To update the weights and biases
of a neural network using
gradient descent with L2 regu.
'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    ''' Updates the weights and biases of
        a neural network using gradient
        descent with L2 regularization

    Args:
        Y (ndarrray): one hot matrix with the correct labels
                for the data.
        weights (dict): weights and biases of the neural network
        cache (dict): output of each layer
        alpha (float): learning rate
        lambtha (float): L2 regularization parameter
        L (int): layers of the neural network
    '''
    # initialize the number of datapoints
    m = Y.shape[1]
    # initialize the activatioon for current layer
    A = cache['A' + str(L)]
    # initialize the gradient of the loss
    # respect to the output of the last layer
    dz = A - Y

    for i in range(L, 0, -1):
        # initialize previous actrivation
        A_prev = cache['A' + str(i - 1)]
        # initialize weight and biase of current layer
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        # compute derivative weight and biases
        # of each layer
        dW = (np.matmul(dz, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m
        # apply the learning rate to weights dictionary
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
        # Apply backpropagation to all layers except the first one
        if i > 1:
            # compute the gradient activation with
            # respect to the previous layer
            DA_prev = np.matmul(W.T, dz)
            # compute gradient of the loss with
            # respect to the input of the previous layer
            dz = DA_prev * (1 - A_prev ** 2)
