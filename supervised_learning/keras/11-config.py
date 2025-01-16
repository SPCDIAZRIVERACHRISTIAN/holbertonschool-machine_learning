#!/usr/bin/env python3
"""This module saves and loads the configuration
of a model using Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Save a model's configuration in JSON format.

    Parameters:
    network (keras model): The model whose
    configuration should be saved.
    filename (str): The path of the file that
    the configuration should be saved to.

    Returns:
    None
    """
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """
    Load a model with a specific configuration.

    Parameters:
    filename (str): The path of the file containing
    the model configuration in JSON format.

    Returns:
    The loaded model.
    """
    with open(filename, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model
