#!/usr/bin/env python3
'''THis model has a function that converts a one-hot
matrix into a vector of labels'''

import numpy as np


def one_hot_decode(one_hot):
    '''Converts a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): One-hot encoded matrix with shape (classes, m)

    Returns:
        numpy.ndarray: Vector of labels with shape (m,)
    '''
    return np.argmax(one_hot, axis=0)
