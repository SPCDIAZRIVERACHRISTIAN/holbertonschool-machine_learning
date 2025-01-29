#!/usr/bin/env python3
"""this module calculates the specificity for each
class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix

    Args:
        confusion (numpy.ndarray): confusion matrix of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels

    Returns:
        numpy.ndarray: numpy.ndarray of shape (classes,) containing
        the specificity
        of each class
    """
    # We need the True Negatives (TN) for each class
    #  For that wee need the sum of the matrix,
    #  the sum of the diagonal will give us the True Positives (TP)
    total = np.sum(confusion)
    # We can get the True Positives (TP) by summing the diagonal
    TP = np.diag(confusion)
    # Sum each row (actual positives)
    actual_positives = np.sum(confusion, axis=1)
    # Sum each column (predicted positives)
    predicted_positives = np.sum(confusion, axis=0)
    # True Negatives (TN) are all values that are not in the
    # actual or predicted
    TN = total - (actual_positives + predicted_positives - TP)
    # False Positives (FP) are the values that are not in the actual
    # but in the predicted
    FP = predicted_positives - TP
    # Calculate the specificity for each class
    return TN / (TN + FP)
