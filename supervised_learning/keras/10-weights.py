#!/usr/bin/env python3
'''This functions saves a model wights and loads the weihgts'''

import tensorflow.keras as K  # type: ignore


def save_weights(network, filename, save_format='keras'):
    '''Saves a model's weights

    Args:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: the format in which the weights
            should be saved. Defaults to 'h5'.
    '''
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    '''loads a model’s weights

    Args:
        network: is the model to which the weights should be loaded
        filename: is the path of the file that the
            weights should be loaded from
    '''

    network.load_weights(filename)
