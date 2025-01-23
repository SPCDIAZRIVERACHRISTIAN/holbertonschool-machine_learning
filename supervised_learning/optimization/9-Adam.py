#!/usr/bin/env python3
'''
NOTE:
Adam (Adaptive Moment Estimation) is
an optimization algorithm that combines
the advantages of two other popular
optimization techniques: AdaGrad and
RMSProp. It computes adaptive learning
rates for each parameter by maintaining
running averages of both the gradients
and their squared values.

this are the steps to follow:

compute first moment:
[ v_t = \beta_1 v_{t-1} +
(1 - \beta_1) \nabla f(\theta_{t-1}) ]

compute the second moment:
[ s_t = \beta_2 s_{t-1} +
(1 - \beta_2) (\nabla f(\theta_{t-1}))^2 ]

bias correction:
[ \\hat{v}_t = \frac{v_t}{1 - \beta_1^t} ]
[ \\hat{s}_t = \frac{s_t}{1 - \beta_2^t} ]

update parameters:
[ \theta_t = \theta_{t-1} -
\alpha \frac{\\hat{v}_t}{\\sqrt{\\hat{s}_t}
+ \\epsilon} ]
'''

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''updates a variable in place using the Adam optimization algorithm

    Args:
    alpha (float): The learning rate.
    beta1 (float): The exponential decay rate
        for the first moment estimates.
    beta2 (float): The exponential decay rate
        for the second moment estimates.
    epsilon (float): A small number to avoid
        division by zero.
    var (numpy.ndarray): The variable to be updated.
    grad (numpy.ndarray): The gradient of var.
    v (numpy.ndarray): The previous first moment of var.
    s (numpy.ndarray): The previous second moment of var.
    t (int): The time step used for bias correction.

    Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray:
    The updated variable, the new first moment,
    and the new second moment, respectively.
    '''
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.square(grad)
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
