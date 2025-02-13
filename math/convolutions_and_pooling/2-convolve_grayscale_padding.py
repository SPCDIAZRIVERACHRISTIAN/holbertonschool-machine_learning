#!/usr/bin/env python3
"""This module preforms a convolution on grayscale images
with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images with
    custom padding.

    Parameters:
    images (numpy.ndarray): The grayscale images
    with shape (m, h, w).
    kernel (numpy.ndarray): The kernel for the
    convolution with shape (kh, kw).
    padding (tuple): The padding for the height
    and width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), "constant")
    output_h = h - kh + 2 * ph + 1
    output_w = w - kw + 2 * pw + 1
    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = \
                (images_padded[:, y: y + kh, x: x + kw] * kernel).sum(
                axis=(1, 2)
            )

    return output
