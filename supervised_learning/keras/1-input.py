#!/usr/bin/env python3
'''This function builds a neural network with the Keras library'''

import tensorflow.keras as K  # type: ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    assert (len(layers) == len(activations))

    model = K.models.Model()

    for i in layers:
        if i == 0:
            

