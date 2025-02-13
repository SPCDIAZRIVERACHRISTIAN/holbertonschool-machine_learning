#!/usr/bin/env python3
"""This module preforms forwarsd propagation over a
convutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a
    neural network.

    Parameters:
    A_prev (numpy.ndarray): The output of the previous layer
    with shape (m, h_prev, w_prev, c_prev).
    W (numpy.ndarray): The kernels for the convolution with
    shape (kh, kw, c_prev, c_new).
    b (numpy.ndarray): The biases applied to the convolution with
    shape (1, 1, 1, c_new).
    activation (function): An activation function applied to the convolution.
    padding (str): The type of padding used, either 'same' or 'valid'.
    stride (tuple): The strides for the convolution, (sh, sw).

    Returns:
    The output of the convolutional layer.
    """
    # Extract dimensions from A_prev's shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Extract dimensions from W's shape
    kh, kw, _, c_new = W.shape

    # Extract strides
    sh, sw = stride

    # Calculate padding for 'same' and 'valid'
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0

    # Initialize output with zeros
    output = np.zeros((m, int((h_prev - kh + 2 * ph) // sh) + 1,
                       int((w_prev - kw + 2 * pw) // sw) + 1, c_new))

    # Pad A_prev
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Convolve the input with the kernel and apply activation function
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            for k in range(c_new):
                slice_A_prev = \
                    A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
                conv = (W[..., k] * slice_A_prev).sum(axis=(1, 2, 3))
                output[:, i, j, k] = activation(conv + b[..., k])

    return output
