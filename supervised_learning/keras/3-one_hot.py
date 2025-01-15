#!/usr/bin/env python3
'''This function converts a label vector into a one-hot matrix'''

import tensorflow.keras as K  # type: ignore


def one_hot(labels, classes=None):
    '''makes a one hot from a vector
        Args:
            labels: matrix data
            classes: classes from data. Default to None
        Returns:
            matrix: one-hot matrix
    '''
    matrix = K.utils.to_categorical(labels, num_classes=classes)
    return matrix
