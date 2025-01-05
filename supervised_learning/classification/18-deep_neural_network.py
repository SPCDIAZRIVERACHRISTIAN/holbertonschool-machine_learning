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

    @property
    def L(self):
        """ This method retrieves the number of layers"""
        return self.__L

    @property
    def cache(self):
        """ This method retrieves the intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """ This method retrieves the weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        '''calculates the forward propagation

        Args:
            X (ndarray): with shape (nx, m) that
                contains the input data nx=number of features
                m=number of examples

        Returns:
            ndarray: returns cache and output of the neural network
        '''
        # store X in cache dictionary
        self.__cache['A0'] = X
        # initialize activity output with X
        # to start forward propagation
        A = X
        # iterate through layers starting in the second
        # hidden layer node
        for k in range(1, self.__L + 1):
            # add the weight of each node
            W = self.__weights['W' + str(k)]
            # add the biases
            b = self.__weights['b' + str(k)]
            # use matmul to calculate the output neuron
            z = np.matmul(W, A) + b
            # calculate the sigmoid function on every layer
            A = 1 / (1 + np.exp(-z))
            # store output of each layer in cache
            self.__cache['A' + str(k)] = A
        # return the output and cache
        return A, self.__cache
