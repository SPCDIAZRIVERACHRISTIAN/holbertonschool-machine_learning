#!/usr/bin/env python3
'''
NOTE:
In this portion we create a layer
that will be regularized by
l2 regularization. we need the
specified number of nodes
activation function and
L2 parameter.
'''

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''creates a neural network
        layer in tensorFlow that
        includes L2 regularization

    Args:
        prev (tf.tensor): tensor containing the output of the previous layer
        n (int): number of nodes the new layer should contain
        activation (function): activation function
            that should be used on the layer
        lambtha (float): L2 regularization parameter

    Returns:
        tf.tensor: the output of the new layer
    '''
    # initialize the weight using a kernel regularizer
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg")
    regularizer = tf.keras.regularizers.l2(lambtha)

    # Create the layer
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)

    # Connect the new layer to the previous layer
    output = layer(prev)

    return output
