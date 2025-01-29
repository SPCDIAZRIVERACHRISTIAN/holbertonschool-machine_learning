#!/usr/bin/env python3
"""Tis module calculate the sensitivity for each
class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """This function calculates the sensitivity for each
    class in a confusion matrix

    Args:
        confusion (numpy.ndarray): Is a confusion matrix
        of shape (classes, classes)
        where row indices represent the correct labels and
        column indices represent
        the predicted labels

    Returns:
        numpy.ndarray: Contains the sensitivity of each class
    """
    # true positives are located in the diagolan so we can use np.diag
    # then we get the total number of false positives by summing the columns
    return np.diag(confusion) / np.sum(confusion, axis=1)
