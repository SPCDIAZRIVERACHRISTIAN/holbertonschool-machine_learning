#!/usr/bin/env python3
'''
NOTE
the keras library has a module for optimizing
momentum the way to access it is
tensorflow.keras.optimizers.
in this case we are using SGD
which stands for Stochastic
Gradient Descent.
'''

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    '''
    Set up the gradient descent with momentum
    optimization algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.

    Returns:
    tf.keras.optimizers.SGD: The SGD optimizer object with momentum.
    '''
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
