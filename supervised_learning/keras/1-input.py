#!/usr/bin/env python3
'''This function builds a neural network with the Keras library'''

import tensorflow.keras as K  # type: ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Builds a neural network model using Keras Functional API

    Args:
        nx (int): number of input features
        layers (list): list containing the
            number of nodes in each layer of the network
        activations (list): list containing the activation
            functions used for each layer of the network
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout

    Returns:
        Keras model
    '''
    # make sure the layer and activations are =
    assert (len(layers) == len(activations))
    # initialize the inputs usign the input function
    inputs = K.Input(shape=(nx,))
    # initialize tensor
    x = inputs
    # iterate through layers to build the layers
    for i in range(len(layers)):
        # assign the output to x with Dense function
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
        )(x)
        # add the Dropout regularization to inner layers
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    # create the model
    model = K.Model(inputs=inputs, outputs=x)

    return model
