#!/usr/bin/env python3
'''
NOTE:
in order to shuffle in the same
way we need to use the same
permutation of an array so
we used X's permutation to shuffle
the datapoints in the same way
'''

import numpy as np


def shuffle_data(X, Y):
    '''Shuffles the data points in two matrices the same way

    Args:
        X (ndarray): first matrix to shuffle
        Y (ndarray): second matrix to shuffle

    Returns:
        tuple: shuffled X and Y matrices
    '''
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
