#!/usr/bin/env python3
'''
NOTES:
normalization formula
normie x = (matrix - mean) / standard deviation
'''
# import numpy as np


def normalize(X, m, s):
    '''normalizes (standardizes) a matrix

    Args:
        X (ndarray): numpy.ndarray of shape (d, nx) to normalize
        m (ndarray): contains the mean of all features of X
        s (ndarray): contains the standard deviation of all features of X

    Returns:
       List: The normalized X matrix
    '''
    return (X - m) / s
