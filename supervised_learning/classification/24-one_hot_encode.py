#!/usr/bin/env python3
'''This module converts a numeric label vector into a one-hot matrix'''

import numpy as np


def one_hot_encode(Y, classes):
    '''Converts a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): Numeric label vector with shape (m,)
        classes (int): Number of classes

    Returns:
        numpy.ndarray: One-hot encoding matrix with shape (classes, m)
    '''
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1

    return one_hot
