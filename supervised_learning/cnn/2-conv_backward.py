#!/usr/bin/env python3
"""This module preforms back propagation over a
convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters:
    dZ -- numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the unactivated output of the
    convolutional layer
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
    output of the previous layer
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
    for the convolution
    b -- numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied
    to the convolution
    padding -- string that is either same or valid, indicating the type
    of padding used
    stride -- tuple of (sh, sw) containing the strides for the convolution

    Returns:
    dA_prev -- the partial derivatives with respect to the previous layer
    dW -- the partial derivatives with respect to W
    db -- the partial derivatives with respect to b
    """
    # Retrieve dimensions from dZ's shape
    m, h_new, w_new, c_new = dZ.shape

    # Retrieve dimensions from A_prev's shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Retrieve dimensions from W's shape
    kh, kw, _, _ = W.shape

    # Retrieve information from "stride"
    sh, sw = stride

    # Compute the padding dimensions
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:  # padding == 'valid':
        ph = pw = 0

    # Initialize dA_prev, dW, db with the correct shapes
    A_prev_pad = \
        np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), "constant")
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over each position of the output
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Find the corners of the current slice
                slice_A_prev = A_prev_pad[
                    :, i * sh: i * sh + kh, j * sw: j * sw + kw, :
                ]

                # Update gradients for the window and the
                # filter's parameters using the code formulas given above
                dA_prev_pad[:, i * sh: i * sh + kh,
                            j * sw: j * sw + kw, :] += (
                    W[..., k] * dZ[:, i, j, k, np.newaxis,
                                   np.newaxis, np.newaxis]
                )
                dW[..., k] += np.sum(
                    slice_A_prev * dZ[:, i, j, k,
                                      np.newaxis, np.newaxis, np.newaxis],
                    axis=0,
                )

    # Set the ith training example's dA_prev to the
    # unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:  # padding == 'valid':
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
