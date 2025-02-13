#!/usr/bin/env python3
"""This module preforms a same convolution on
grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images.

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
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    images_padded = np.pad(images, ((0, 0), (ph, ph),
                                    (pw, pw)), "constant")
    output = np.zeros((m, h, w))

    for x in range(w):
        for y in range(h):
            output[:, y, x] = \
                (images_padded[:, y: y + kh, x: x + kw] * kernel).sum(
                axis=(1, 2)
            )

    return output
