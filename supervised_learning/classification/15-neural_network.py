#!/usr/bin/env python3
'''This file contains a class that defines
    a neural network with one hidden layer performing binary classification'''

import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''calculates the gradient descent on the
            Neural Network

        Args:
            X (ndarray): has the input data with
                shape (nx, m) where nx is the number of
                    input features and m is the number
                    of examples
            Y (ndarray): it has the true label of the data
                with shape (1, m) m being the number of examples
            A1 (ndarray): output of hidden layer
            A2 (ndarray): predicted output
            alpha (float, optional): learning rate. Defaults to 0.05.
        '''
        # get the number of examples given in Y
        # as the second index
        m = Y.shape[1]
        # calculate the error in the output layer
        # finding the difference between predicted output
        # and true labels
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - alpha * dw1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b1 = self.__b1 - alpha * db1
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        '''Trains the model by iterations

        Args:
            X (ndarray): input data
            Y (ndarray): True Label data
            iterations (int, optional): number of iterations to train
                over. Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.
            verbose (bool): defines whether or not to print info
                about the training
            graph (bool): defines whether or not display graph info
                about the training
            step (int): include data of every nth iteration

        Raises:
            TypeError: if iteration is not an integer
            ValueError: if iteration not positive
            TypeError: if alpha is not a float
            ValueError: if alpha is not positive

        Returns:
            ndarray, float: result of evaluation method
        '''
        # validate iterations
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        # validate learning rate
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        # initialize cost list for graphing
        costs = []

        # iterate over the number in iterations
        for i in range(iterations + 1):
            # store the activity output
            # of inner nodes and output respectively
            A1, A2 = self.forward_prop(X)

            cost = self.cost(Y, self.__A2)
            # check if the current iteratin is a multiple of
            # step or the last iteration
            if i % step == 0 or i == iterations:
                if verbose is True:
                    print(f'Cost after {i} iterations: {cost}')
                # check if graph is true to
                # append cost to costs list
                if graph is True:
                    costs.append(cost)
            # check if it's not the last iteration
            if i < iterations:
                # call gradient method to get the descent
                self.gradient_descent(X, Y, A1, A2, alpha)
        # if graph true plot it
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
        # evaluate the output using input data and the True
        # labels
        return self.evaluate(X, Y)
