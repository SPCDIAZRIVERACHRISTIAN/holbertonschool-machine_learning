#!/usr/bin/env python3
'''This function creates tensorflow layers for a neural network'''

import tensorflow.compat.v1 as tf  # type: ignore


def create_layer(prev, n, activation):
    '''This function creates layers for a neural network

    Args:
        prev: last layer created
        n: number of nodes in layer
        activation: activation used for layer

    Returns:
        tensor output of layer
    '''
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer(prev)
