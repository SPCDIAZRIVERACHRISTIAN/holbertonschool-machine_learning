#!/usr/bin/env python3
'''This is a function :)'''

import tensorflow.keras as K  # type:ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''this function  builds a neural network with the Keras library

    Args:
        nx (ndarray): number of input features to the network
        layers (list): the number of nodes in each layer of the network
        activations (list): containing the activation functions
            used for each layer of the network
        lambtha (func):  L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout
    '''

    assert (len(layers) == len(activations))

    model = K.models.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(layers[i],
                               input_dim=nx,
                               activation=activations[i],
                               kernel_regularizer=K.regularizers.l2(lambtha
                                                                    )
                               )
                )
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                    )
                )
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
