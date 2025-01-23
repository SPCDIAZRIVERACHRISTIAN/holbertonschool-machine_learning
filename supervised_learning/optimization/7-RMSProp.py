#!/usr/bin/env python3
'''
NOTE:
to implement RMSProp in deep
learning we need to compute
the new second moment and update
the variable using these formulas:
new second moment:
s_t = [\beta_2 s_{t-1} + (1 - \beta_2)
(\nabla f(\theta_{t-1}))^2]
updated variable:
\theta_t = [\theta_{t-1} -
\alpha \frac{\nabla f(\theta_{t-1})}
{\\sqrt{s_t} + \\epsilon}]
'''

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''Update a variable using the RMSProp optimization algorithm.

    Args:
    alpha (float): The learning rate.
    beta2 (float): The RMSProp weight.
    epsilon (float): A small number to avoid division by zero.
    var (numpy.ndarray): The variable to be updated.
    grad (numpy.ndarray): The gradient of var.
    s (numpy.ndarray): The previous second moment of var.

    Returns:
    numpy.ndarray, numpy.ndarray: The updated variable and
    the new moment, respectively.
    '''
    s_t = beta2 * s + (1 - beta2) * np.square(grad)
    theta_t = var - alpha * grad / (np.sqrt(s_t) + epsilon)

    return theta_t, s_t
