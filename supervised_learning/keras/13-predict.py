#!/usr/bin/env python3
'''This function that makes a prediction using a neural network'''

import tensorflow.keras as K  # type: ignore


def predict(network, data, verbose=False):
    '''makes a prediction using a neural network

    Args:
        network: is the network model to make the prediction with
        data: is the input data to make the prediction with
        verbose: is a boolean that determines if output should be
            printed during the prediction process
    '''
    return network.predict(data, verbose=verbose)
