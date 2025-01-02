#!/usr/bin/env python3
'''This file contains a class Neuron that defines a
    single neuron performing binary classification'''

import numpy as np


class Neuron:
    '''This class is to create neurons'''
    def __init__(self, nx):
        '''initialize the neuron

        Args:
            nx (list): list containing integers

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
        # use np.random to initialize weight with 1 row and nx as columns
        self.__W = np.random.randn(1, nx)
        # initialize bias at 0
        self.__b = 0
        # initialize activated output at 0
        self.__A = 0

    @property
    def W(self):
        '''gets weight'''
        return self.__W

    @property
    def b(self):
        '''gets bias'''
        return self.__b

    @property
    def A(self):
        '''gets activity output'''
        return self.__A

    def forward_prop(self, X):
        '''calculate the forward propagation

        Args:
            X (list): ndarray with the shape of nx, m

        Returns:
            list: returns the activity product of neuron
        '''
        # X is a ndarray with shape of number of input
        # and number of examples
        # get the dot product of weight and ndarray X
        z = np.dot(self.__W, X) + self.__b
        # calculate the output using sigmoid function
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        '''calculates the cost function in a neuron

        Args:
            Y (ndarray): numpy.ndarray with shape (1, m) that contains
                the correct labels for the input data
            A (ndarray): numpy.ndarray with shape (1, m) containing the
                activated output of the neuron for each example

        Returns:
            float: Returns the result of cost calculation
        '''
        # you need the number of examples
        # to complete the cost function formula
        # in this case is one example
        m = Y.shape[1]
        # Apply the cost formula to get the result using m
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
