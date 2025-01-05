#!/usr/bin/env python3
'''This is a deep neural network class'''

import numpy as np


class DeepNeuralNetwork:
    '''defines a deep neural network performing binary classification'''
    def __init__(self, nx, layers):
        '''this is the class constructor

        Args:
            nx (ndarray): number of input features
            layers (list):  represents the number of
                nodes in each layer of the network
        '''
        # verify if nx is int and a positive one
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        # verify if layers is a list and not empty
        # and check that all elements in layers are positive integers
        if not isinstance(layers, list) or len(
                layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        # initialize the number of layers
        self.__L = len(layers)
        # makes a temporary cache to store
        # all intermediary values
        self.__cache = {}
        # stores the weight of nodes
        self.__weights = {}

        # iterate over layers to add weights and biases to dictionary
        for k in range(self.__L):
            # f0r the initial number in layers
            # the first layer is the number of nodes in the first layer
            # we need to store it differently because the
            # dimensions of the weight
            # depend on the number of features(nx)
            if k == 0:
                # use the He et al. method
                self.__weights['W' + str(k + 1)] =\
                    np.random.randn(layers[k], nx) * np.sqrt(2 / nx)
            # store the rest of the weights using pnly the layers
            else:
                # use he et al. method to initialize the weight
                self.__weights['W' + str(k + 1)] = np.random.randn(
                    layers[k], layers[k - 1]) * np.sqrt(2 / layers[k - 1])
            # store the biase f0r every weight to
            # be a ndarray of 0
            self.__weights['b' + str(k + 1)] = np.zeros((layers[k], 1))
