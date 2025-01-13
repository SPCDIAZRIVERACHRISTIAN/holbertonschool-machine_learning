#!/usr/bin/env python3
'''this function creates the forward propagation graph for the neural network'''

import tensorflow.compat.v1 as tf  # type: ignore
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''Creates the forward propagation graph for the neural network

    Args:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer of the network
        activations: list containing the activation functions for each layer of the network

    Returns:
        the prediction of the network in tensor form
    '''
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    return output
