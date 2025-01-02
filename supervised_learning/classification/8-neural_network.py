#!/usr/bin/env python3
'''This file contains a class Neuron that defines a
    single neuron performing binary classification'''

import numpy as np


class NeuralNetwork:
    '''This class is to create neurons'''
    def __init__(self, nx, nodes):
        '''initialize the neural network

        Args:
            nx (list): list containing integers
            nodes(int): is the number of nodes found in the hidden layer

        Raises:
            TypeError: if not an integer
            ValueError: if not positive
        '''
        # verify if is int
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        # verify if its positive
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        # use np.random to initialize weight
        # with nodes as row and nx as columns
        self.W1 = np.random.randn(nodes, nx)
        self.W2 = np.random.randn(nodes, nx)
        # initialize bias at 0
        self.b1 = 0
        self.b2 = 0
        # initialize activated output at 0
        self.A1 = 0
        self.A2 = 0
