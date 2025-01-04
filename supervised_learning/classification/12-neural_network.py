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

    @property
    def W1(self):
        '''gets the W1 attribute'''
        return self.__W1

    @property
    def W2(self):
        '''gets the W2 attribute'''
        return self.__W2

    @property
    def b1(self):
        '''get the b1 attribute'''
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        '''gets the attribute A1'''
        return self.__A1

    @property
    def A2(self):
        '''gets the attribute A2'''
        return self.__A2

    def forward_prop(self, X):
        '''calculates forward propagation

        Args:
            X (ndarray): numpy.ndarray with shape
                (nx, m) that contains the input data

        Returns:
            ndarray: return activity output of hidden and output layer
        '''
        # compute the hidden layer
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        # Calculate the output neuron
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''claculates the cost of the model

        Args:
            Y (ndarray): True label of input data
            A (ndarray): activated output of the neuron

        Returns:
            float: float number that tells the model
                how much did the prediction match
                the true label
        '''
        # get the labels using shape to grab
        # the 2nd index because Y is a ndarray
        # with shape (1,m) and m is the total of labels
        m = Y.shape[1]
        # loss A.K.A. cost is the formula that gives us how close
        # the prediction was to the label
        # if it was close the result is small
        loss = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return loss

    def evaluate(self, X, Y):
        '''evaluates the prediction of the Neural Network

        Args:
            X (ndarray): has the input data with
                shape (nx, m) where nx is the number of
                    input features and m is the number
                    of examples
            Y (ndarray): it has the true label of the data

        Returns:
            pred: returns the prediction of the output layer
            cost: returns how close the prediction was to the true layer
        '''
        # Because we only need the forward
        # propagation of the output layer
        # we assigned _ to the hidden layer
        # to be used in the prediciton
        _, A2 = self.forward_prop(X)
        # the prediction consist on using numpy's
        # where function that looks the elements
        # in ndarray given to the function and organizes
        # it in 0 and 1 where 1 will be every
        # element that is greater or equal than 0.5
        # and 0 where it's not
        pred = np.where(A2 >= 0.5, 1, 0)
        # calls the loss function
        cost = self.cost(Y, A2)
        return pred, cost
