#!/usr/bin/env python3
"""This module contains a function that updates
the weights of a neural with Dropout regularization
using gradient descent.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout
    regularization using gradient descent.

    Parameters:
    Y: numpy.ndarray - the correct labels for the data.
    weights: dict - the weights and biases of the neural network.
    cache: dict - the outputs and dropout masks of each layer
    of the neural network.
    alpha: float - the learning rate.
    keep_prob: float - the probability that a node will be kept.
    L: int - the number of layers in the network.

    Returns:
    None
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            D = cache["D" + str(i - 1)]
            dA_prev = np.matmul(weights_copy["W" + str(i)].T, dZ)
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - np.power(A_prev, 2))
        weights["W" + str(i)] = weights_copy["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights_copy["b" + str(i)] - alpha * db
