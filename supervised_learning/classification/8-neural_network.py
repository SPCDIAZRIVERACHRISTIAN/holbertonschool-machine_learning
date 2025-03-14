#!/usr/bin/env python3
'''This file contains a class that defines
    a neural network with one hidden layer performing binary classification'''

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
        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        # initialize bias at 0
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        # initialize activated output at 0
        self.__A1 = 0
        self.__A2 = 0
