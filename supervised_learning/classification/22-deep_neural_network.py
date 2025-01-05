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

    def cost(self, Y, A):
        '''calculate the loss of the model

        Args:
            Y (ndarray): True label of input data
            A (ndarray): activated output of the neuron

        Returns:
            float: float number that tells the model
                how much did the prediction match
                the true label
        '''
        # gets the labels of true label data
        m = Y.shape[1]
        # loss A.K.A. cost is the formula that gives us how close
        # the prediction was to the label
        # if it was close the result is small
        loss = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        # return findings which will be float
        return loss

    def evaluate(self, X, Y):
        '''Evaluates the neural network’s predictions

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
        # only use the activity output of
        # the value forward prop gives
        A, _ = self.forward_prop(X)
        # the prediction consist on using numpy's
        # where function that looks the elements
        # in ndarray given to the function and organizes
        # it in 0 and 1 where 1 will be every
        # element that is greater or equal than 0.5
        # and 0 where it's not
        pred = np.where(A >= 0.5, 1, 0)
        # get the loss function on act out
        cost = self.cost(Y, A)
        # return prediction and cost
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''calculates the gradient descent on the deep
            neural network

        Args:
            Y (ndarray): true labels of data
            cache (_type_): dictionary containing intermediary values
            alpha (float, optional): learning rate. Defaults to 0.05.
        '''
        # Number of examples in input data
        m = Y.shape[1]
        # Calculate the gradients of the output data
        dZ = cache['A' + str(self.L)] - Y
        for i in range(self.L, 0, -1):
            # Get the cached activations
            A_prev = cache['A' + str(i - 1)]
            # Calculate the derivatives of the weights and biases
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            # Calculate the derivative of the cost
            # with respect to the activation
            dZ_step1 = np.dot(self.weights['W' + str(i)].T, dZ)
            dZ = dZ_step1 * (A_prev * (1 - A_prev))
            # Update the weights and biases
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''trains the model

        Args:
            X (ndarray): input data
            Y (ndarray): true label
            iterations (int, optional): number of iterations. Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.

        Returns:
            prediction, cost: returns the cost and prediction
        '''
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
