#!/usr/bin/env python3
"""This module contains the function for
testing a model"""
import tensorflow.keras as K  # type: ignore


def test_model(network, data, labels, verbose=True):
    """
    Test a neural network.

    Parameters:
    network (keras model): The network model to test.
    data (numpy.ndarray): The input data to test the
    model with.
    labels (numpy.ndarray): The correct one-hot labels
    of data.
    verbose (bool): Determines if output should be printed
    during the testing process.

    Returns:
    The loss and accuracy of the model with the testing data,
    respectively.
    """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return [loss, accuracy]
