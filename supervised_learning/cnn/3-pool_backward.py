#!/usr/bin/env python3
"""This module preforms back propagation over a p
ooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters:
    dA -- numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c) containing
    the output of the previous layer
    kernel_shape -- tuple of (kh, kw) containing the size of the kernel
    for the pooling
    stride -- tuple of (sh, sw) containing the strides for the pooling
    mode -- string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively

    Returns:
    dA_prev -- the partial derivatives with respect to the previous layer
    """
    # Retrieve dimensions from A_prev's shape and dA's shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dA.shape

    # Retrieve dimensions from kernel_shape
    kh, kw = kernel_shape

    # Retrieve strides from stride
    sh, sw = stride

    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)

    # Loop over the training examples
    for i in range(m):
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Find the corners of the current "slice"
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current
                        # slice from a_prev
                        a_prev_slice = a_prev[
                            vert_start:vert_end, horiz_start:horiz_end, c
                        ]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = a_prev_slice == np.max(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied
                        # by the correct entry of dA)
                        dA_prev[
                            i, vert_start:vert_end, horiz_start:horiz_end, c
                        ] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "avg":
                        # Get the value a from dA
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf
                        shape = (kh, kw)
                        # Distribute it to get the correct slice of
                        # dA_prev. i.e. Add the distributed value of da.
                        average = da / (kh * kw)
                        a = np.ones(shape) * average
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += a

    return dA_prev
