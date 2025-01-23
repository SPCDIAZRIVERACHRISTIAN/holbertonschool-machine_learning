#!/usr/bin/env python3
'''
NOTE:
Batch normalization helps to
stabilize and accelerate the
training process by reducing
internal covariate shift.
The process involves normalizing
the outputs to have a mean
of zero and a variance of
one, and then scaling and
shifting them using
learnable parameters.
following are the formulas
to normalize a batch:

compute the mean and variance
of the batch:
[ \\mu = \frac{1}{m} \\sum_{i=1}^{m} Z_i ]
[ \\sigma^2 = \frac{1}{m} \\sum_{i=1}^{m}
(Z_i - \\mu)^2 ]

normalize the batch:
[ \\hat{Z} = \frac{Z - \\mu}{\\sqrt{\\sigma^2
+ \\epsilon}} ]

scale and shift the normalized batch:
[ Z_{\text{norm}} = \\gamma \\hat{Z} + \beta ]
'''

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
    Normalizes an unactivated output of a
    neural network using batch normalization.

    Args:
    Z (numpy.ndarray): The unactivated output to be normalized.
    gamma (numpy.ndarray): The scale parameter.
    beta (numpy.ndarray): The shift parameter.
    epsilon (float): A small number to avoid division by zero.

    Returns:
    numpy.ndarray: The normalized and scaled output.
    '''
    mu = np.mean(Z, axis=0)
    sigma2 = np.var(Z, axis=0)

    Z_hat = (Z - mu) / np.sqrt(sigma2 + epsilon)

    Z_norm = gamma * Z_hat + beta

    return Z_norm
