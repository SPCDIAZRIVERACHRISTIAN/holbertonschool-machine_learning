#!/usr/bin/env python3
"""This modlue contains the function for creating a batch normalization
layer in Tensorflow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in tensorflow.

    Parameters:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (callable): The activation function that should be
    used on the output of the layer.

    Returns:
    tf.Tensor: The activated output for the layer.
    """
    # Create a dense layer
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"),
        use_bias=False,
    )(prev)

    # Calculate mean and variance
    batch_mean, batch_var = tf.nn.moments(dense, [0])

    # Create gamma and beta variables
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    # Calculate batch normalization
    norm = tf.nn.batch_normalization(
        dense, batch_mean, batch_var, beta, gamma, 1e-7)

    # Apply the activation function
    out = activation(norm)

    return out
