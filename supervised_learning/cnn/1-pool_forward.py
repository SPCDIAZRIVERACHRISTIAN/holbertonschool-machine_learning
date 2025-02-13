#!/usr/bin/env python3
"""This moduleperforms forward propagation over
a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters:
    A_prev (numpy.ndarray): The output of the previous layer
    kernel_shape (tuple): The size of the kernel for the pooling
    stride (tuple): The strides for the pooling
    mode (str): The pooling mode ('max' or 'avg')

    Returns:
    The output of the pooling layer.
    """
    # Extract dimensions from A_prev's shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    # Extract kernel shape
    kh, kw = kernel_shape
    # Extract strides
    sh, sw = stride

    # Calculate output dimensions
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Initialize output with zeros
    output = np.zeros((m, h_out, w_out, c_prev))

    # Pooling
    for i in range(h_out):
        for j in range(w_out):
            slice_A_prev = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(slice_A_prev, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(slice_A_prev, axis=(1, 2))

    return output
