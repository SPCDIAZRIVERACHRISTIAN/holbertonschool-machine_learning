#!/usr/bin/env python3
"""This module calculates the F1 score for each class in a
confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """This function calculates the F1 score for each class in a
    confusion matrix

    Args:
        confusion (numpy.ndarray): A confusion matrix of
        shape (classes, classes)
        where row indices represent the correct labels and
        column indices represent
        the predicted labels

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,)
        containing the F1 score of each class
    """
    numerator = 2 * (sensitivity(confusion) * precision(confusion))
    denominator = sensitivity(confusion) + precision(confusion)
    return numerator / denominator
