#!/usr/bin/env python3
"""This module preforms a valid convolution
on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Parameters:
    images (numpy.ndarray): The grayscale images
    with shape (m, h, w).
    kernel (numpy.ndarray): The kernel for the
    convolution with shape (kh, kw).

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = \
                (images[:, y: y + kh, x: x + kw] * kernel).sum(
                axis=(1, 2)
            )

    return output
