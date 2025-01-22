#!/usr/bin/env python3
'''
NOTE:
to make a mini batch creator function
using gradient descent, we need to iterate
repeatedly through the training set data
mini-batch gradient descent is useful when
dealing with large datasets and complex models
some useful scenarios where mini-batch shines
are: deep learning, online learning and
resource-constrained environments.
'''

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    '''creates mini-batches to be used for
    training a neural network using mini-batch
    gradient descent

    Args:
        X (ndarray): representing input data
        Y (ndarray): representing the labels
        batch_size (int): number of data points in a batch
    Returns:
    list: mini-batches containing tuples (X_batch, Y_batch)
    '''
    # shuffle the data
    X, Y = shuffle_data(X, Y)

    mini_batches = []
    m = X.shape[0]
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
