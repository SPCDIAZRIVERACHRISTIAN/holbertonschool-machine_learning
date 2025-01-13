#!/usr/bin/env python3
'''This function calculates the softmax cross-entropy loss of a prediction'''

import tensorflow.compat.v1 as tf  # type: ignore


def calculate_loss(y, y_pred):
    '''Calculates the softmax cross-entropy loss of a prediction

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network’s predictions

    Returns:
        tensor containing the loss of the prediction
    '''
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
