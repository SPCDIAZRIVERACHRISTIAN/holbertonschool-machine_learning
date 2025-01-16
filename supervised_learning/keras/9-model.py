#!/usr/bin/env python3
'''This file contains function to save a
    model and another that lods a model'''

import tensorflow.keras as K  # type: ignore


def save_model(network, filename):
    '''saves a model

    Args:
        network: is the model to save
        filename: is the path of the file that the
            model should be saved to
    '''
    network.save(filename)


def load_model(filename):
    '''loads an entire model

    Args:
        filename: is the path of the file that
            the model should be loaded from
    '''
    return K.models.load_model(filename)
