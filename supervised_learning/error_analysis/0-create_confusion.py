#!/usr/bin/env python3
'''
This function creates a confusion matrix
'''

import numpy as np


def create_confusion_matrix(labels, logits):
    '''Creates a confusion matrix.

    Args:
    labels (numpy.ndarray): The true labels.
    logits (numpy.ndarray): The predicted labels.

    Returns:
    numpy.ndarray: The confusion matrix.
    '''
    return np.dot(labels.T, logits)
