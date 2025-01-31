#!/usr/bin/env python3
"""This module contains a function that creates a neural
network layer in TensorFlow using dropout.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Parameters:
    prev: tensor - the output of the previous layer.
    n: int - the number of nodes the new layer should contain.
    activation: function - the activation function for the new layer.
    keep_prob: float - the probability that a node will be kept.
    training: boolean - indicating whether the model is in training mode.

    Returns:
    The output of the new layer.
    """
    # Initialize weights with variance scaling initializer.
    # This initializer is designed to scale the variance of the weights
    # to ensure a uniform variance across all weights in the network.
    initializer = \
        tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    # Create a dense (fully connected) layer with 'n' units and the
    # specified activation function.
    # The weights for the layer are initialized using the variance
    # scaling initializer defined above.
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=initializer)

    # Apply dropout to the layer.
    # Dropout randomly sets a fraction 'rate' of the input units to 0
    # at each update during training time,
    # which helps prevent overfitting. The fraction is 1 - keep_prob,
    # so keep_prob fraction of the units are kept.
    drop = tf.keras.layers.Dropout(rate=1 - keep_prob)

    # Apply the dropout to the layer and return the result.
    # The 'training' argument determines whether dropout will be applied.
    return drop(layer(prev), training=training)
