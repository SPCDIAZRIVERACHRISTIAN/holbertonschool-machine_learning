#!/usr/bin/env python3
'''
NOTE:
Gradient descent with momentum
keeps track of an exponentially
decaying average of past gradients
(often called the “velocity”).
Instead of updating parameters
purely by the current gradient,
momentum “remembers” previous
updates to smooth out the
optimization path.
In order to do that we need
to compute the gradient then
update the velocity and
finally update the parameter.
'''

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    Update a variable using the gradient descent
    with momentum optimization algorithm.

    Args:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.
    var (numpy.ndarray): The variable to be updated.
    grad (numpy.ndarray): The gradient of var.
    v (numpy.ndarray): The previous first moment of var.

    Returns:
    numpy.ndarray, numpy.ndarray: The updated variable and
    the new moment, respectively.
    '''
    # computes the new velocity with
    # this formula: v_t = \beta_1 v_{t-1} +
    # (1 - \beta_1) \nabla f(\theta_{t-1})
    v = beta1 * v + (1 - beta1) * grad
    # update the variable using this formula:
    # \theta_t = \theta_{t-1} - \alpha v_t
    var = var - alpha * v

    return var, v
