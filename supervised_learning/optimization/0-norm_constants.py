#!/usr/bin/env python3
'''
NOTES:
to normalize constants of a matrix you
need to use the features in order to do so.
That's why this function takes the data points
on each column to calculate the normalization
constants by using the mean and standard deviation
'''

import numpy as np


def normalization_constants(X):
    '''Calculates the normalization constants of a matrix

    Args:
        X (ndarray): matrix to normalize with (m, nx) shape

    Returns:
        mean (ndarray): mean of each feature
        std (ndarray): standard deviation of each feature
    '''
    # use mean function to calculate the mean along the columns
    mean = np.mean(X, axis=0)
    # std = standard deviation function in numpy
    # this cclaculates the standard devition along the columns
    std = np.std(X, axis=0)

    return mean, std
