#!/usr/bin/env python3
"""This module contains the function for updating a variable using the RMSProp
optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Set up the RMSProp optimization algorithm in TensorFlow.

    Parameters:
    loss (tf.Tensor): The loss of the network.
    alpha (float): The learning rate.
    beta2 (float): The RMSProp weight (Discounting factor).
    epsilon (float): A small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.RMSprop: The RMSprop optimizer object.
    """
    # Create a RMSprop optimizer object with the specified
    # learning rate, RMSProp weight, and epsilon
    optimizer = \
        tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2,
                                    epsilon=epsilon)

    # Return the optimizer
    return optimizer
