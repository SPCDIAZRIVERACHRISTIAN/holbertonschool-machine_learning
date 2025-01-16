#!/usr/bin/env python3
'''This functions save and load configuration in json'''

import json


def save_config(network, filename):
    '''saves a model’s configuration in JSON format

    Args:
        network: is the model whose configuration should be saved
        filename: is the path of the file that
            the configuration should be saved to
    '''
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)


def load_config(filename):
    '''loads a model with a specific configuration

    Args:
        filename: is the path of the file containing the
            model’s configuration in JSON format
    '''
    with open(filename, 'r') as json_file:
        json_config = json_file.read()
    return json.loads(json_config)
