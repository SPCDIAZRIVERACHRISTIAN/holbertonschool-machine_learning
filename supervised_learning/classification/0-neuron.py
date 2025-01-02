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
        self.W = np.random.randn(1, nx)
        # initialize bias at 0
        self.b = 0
        # initialize activated output at 0
        self.A = 0
