#!/usr/bin/env python3
"""Pooling function"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    images (numpy.ndarray): The images with shape (m, h, w, c).
    kernel_shape (tuple): The shape of the kernel for pooling.
    stride (tuple): The stride for the height and width of the image.
    mode (str): The mode for pooling, either 'max' or 'avg'.

    Returns:
    A numpy.ndarray containing the pooled images.
    """
    # Extract dimensions from images' shape
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize output with zeros
    output = np.zeros((m, output_h, output_w, c))

    # Apply pooling to images
    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j] = np.max(
                    images[:, i * sh: i * sh + kh, j * sw: j * sw + kw],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j] = np.mean(
                    images[:, i * sh: i * sh + kh, j * sw: j * sw + kw],
                    axis=(1, 2)
                )

    return output
