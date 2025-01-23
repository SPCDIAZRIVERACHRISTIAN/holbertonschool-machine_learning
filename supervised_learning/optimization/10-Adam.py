#!/usr/bin/env python3
"""This module contains the function for updating a variable using the Adam
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon_):
    """
    Set up the Adam optimization algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The first moment weight.
    beta2 (float): The second moment weight.
    epsilon (float): A small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.Adam: The Adam optimizer object.
    """
    # Create an Adam optimizer object with the specified
    # learning rate, first moment weight, second moment weight,
    # and epsilon
    optimizer = \
        tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1,
                                 beta_2=beta2, epsilon=epsilon_)

    # Return the optimizer
    return optimizer
