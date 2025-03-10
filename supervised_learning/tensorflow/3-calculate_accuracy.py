#!/usr/bin/env python3
'''This function calculates the accuracy of a prediction'''

import tensorflow.compat.v1 as tf  # type:ignore


def calculate_accuracy(y, y_pred):
    '''Calculates the accuracy of a prediction

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions

    Returns:
        tensor containing the decimal accuracy of the prediction
    '''
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
