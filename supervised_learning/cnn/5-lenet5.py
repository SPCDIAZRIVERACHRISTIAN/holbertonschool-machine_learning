#!/usr/bin/env python3
"""
Module to create a modified LeNet-5 architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5
    architecture using Keras

    Parameters:
    X is a K.Input of shape (m, 28, 28, 1) containing the
    input images for the network

    Returns:
    a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal(seed=0)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=5, padding='same',
                            activation='relu', kernel_initializer=init)(X)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5,
                            padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Flatten the pool2 output
    flat = K.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flat)

    # Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(fc1)

    # Fully connected softmax output layer with 10 nodes
    softmax = K.layers.Dense(units=10, activation='softmax',
                             kernel_initializer=init)(fc2)

    # Create model
    model = K.Model(inputs=X, outputs=softmax)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
