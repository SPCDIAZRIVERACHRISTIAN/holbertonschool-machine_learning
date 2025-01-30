#!/usr/bin/env python3
'''
NOTE:
To create an L2 regularization function
you need to add the L2 reg. term to the
original cost
'''

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''Calculates the cost of a neural
        of a neural network using L2 regularization

    Args:
        cost (ndarray): the cost of the network without L2 regularization
        lambtha (float): regularization parameter
        weights (dict): weight and biases of the neural network
        L (int): number of layers of the neural network
        m (int): number of data points used

    Returns
        cost_with_l2: the cost of the network
    '''
    # initialize the l2 term to 0
    l2_term = 0

    # make a loop to iterat over data points
    for i in range(1, L + 1):
        # access the weight matrix for the i-th layer
        # compute the element-wise square of weight
        # sum it all with  all squared elements
        l2_term += np.sum(np.square(weights['W' + str(i)]))
    # Scale the L2 term by (\frac{\lambda}{2m})
    l2_term = (lambtha / (2 * m)) * l2_term
    # add l2 term to original cost
    cost_with_l2 = cost + l2_term
    # return l2 cost
    return cost_with_l2
