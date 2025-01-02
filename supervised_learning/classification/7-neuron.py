#!/usr/bin/env python3
'''This file contains a class Neuron that defines a
    single neuron performing binary classification'''

import numpy as np

import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        '''evaluuates the neuron's prediction

        Args:
            X (ndarray): numpy.ndarray with shape (nx, m)
                that contains the input data
            Y (ndarray): numpy.ndarray with shape (1, m)
                that contains the correct labels for the input data

        Returns:
            pred: ndarray of shape (1, m) with binary values (0 or 1)
            cost: the cost of prediction
        '''
        # calls forward propagation method
        A = self.forward_prop(X)
        # call where numpy method to get elements
        # that are greater or equal to 0.5 1=true and 0=false
        pred = np.where(A >= 0.5, 1, 0)
        # calls the cost method on Y and A
        cost = self.cost(Y, A)

        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Calculates one pass of gradient descent on the neuron

        Args:
            X (ndarray): numpy.ndarray with shape (nx, m)
                that contains the input data
            Y (_type_): numpy.ndarray with shape (1, m)
                that contains the correct labels for the input data
            A (_type_): numpy.ndarray with shape (1, m)
                containing the activated output of the neuron for each example
            alpha (float, optional): learning rate. Defaults to 0.05.

        return
            __W: updated weight
            __b updated bias
        '''

        # get numberof examples in X
        m = Y.shape[1]
        # Calculate the gradient
        dz = A - Y
        # Derivative of the weight
        dw = np.matmul(X, dz.T) / m
        # Derivative of the bias
        db = np.sum(dz) / m
        # set the new weights and bias
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        This function trains the neuron

        Args:
            X (numpy.ndarray): contains the input data
            Y (numpy.ndarray): contains the correct labels
            iterations (int, optional): Defaults to 5000.
            alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        # Validate iterations
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        # Validate alpha
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize cost list for graphing
        costs = []

        # Loop over the number of iterations + 1
        for i in range(iterations + 1):
            # Perform forward propagation
            self.forward_prop(X)
            # Calculate the cost
            cost = self.cost(Y, self.__A)
            # If the current iteration is a multiple of
            # step or the last iteration
            if i % step == 0 or i == iterations:
                # If verbose is True, print the cost
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                # If graph is True, append the cost to the
                # costs list for later graphing
                if graph is True:
                    costs.append(cost)
            # If not the last iteration, perform gradient
            # descent to update the weights and bias
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)

        # If graph is True, plot the costs over the iterations
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')  # Label the x-axis as 'iteration'
            plt.ylabel('cost')  # Label the y-axis as 'cost'
            plt.title('Training Cost')  # Title the plot as 'Training Cost'
            plt.show()  # Display the plot

        # Return the evaluation of the training data
        # after iterations of training
        return self.evaluate(X, Y)
