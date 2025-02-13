#!/usr/bin/env python3
"""This module builds an inception block as
described in Going Deeper with Convolutions (2014):
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going
    Deeper with Convolutions (2014)

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3, F5R, F5, FPP
            F1: number of filters in the 1x1 convolution
            F3R: number of filters in the 1x1 convolution
            before the 3x3 convolution
            F3: number of filters in the 3x3 convolution
            F5R: number of filters in the 1x1 convolution
            before the 5x5 convolution
            F5: number of filters in the 5x5 convolution
            FPP: number of filters in the 1x1 convolution
            after the max pooling

    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1 = K.layers.Conv2D(F1, (1, 1), activation="relu")(A_prev)

    # 3x3 convolution
    conv3R = K.layers.Conv2D(F3R, (1, 1), activation="relu")(A_prev)
    conv3 = K.layers.Conv2D(F3, (3, 3),
                            padding="same", activation="relu")(conv3R)

    # 5x5 convolution
    conv5R = K.layers.Conv2D(F5R, (1, 1), activation="relu")(A_prev)
    conv5 = K.layers.Conv2D(F5, (5, 5), padding="same",
                            activation="relu")(conv5R)

    # Max pooling
    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                 padding="same")(A_prev)
    poolP = K.layers.Conv2D(FPP, (1, 1), activation="relu")(pool)

    # Concatenate the outputs of the inception block
    output = K.layers.concatenate([conv1, conv3, conv5, poolP])

    return output
