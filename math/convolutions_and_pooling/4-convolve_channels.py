#!/usr/bin/env python3
"""Convolution with channels"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple
    channels with custom padding and stride.

    Parameters:
    images (numpy.ndarray): The images with shape (m, h, w, c).
    kernel (numpy.ndarray): The kernel for the convolution
    with shape (kh, kw, c).
    padding (str or tuple): The padding for the height and
    width of the image.
    stride (tuple): The stride for the height and width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    # Extract dimensions from images' and kernel's shapes
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Calculate padding for 'same' and 'valid'
    if padding == 'same':
        ph = int(((h - 1) * sh - h + kh) / 2) + 1
        pw = int(((w - 1) * sw - w + kw) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Calculate output dimensions
    output_h = int((h - kh + (2 * ph)) / sh) + 1
    output_w = int((w - kw + (2 * pw)) / sw) + 1

    # Initialize output with zeros
    output = np.zeros((m, output_h, output_w))

    # Pad images
    image_padded = \
        np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Apply kernel to images
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                image_padded[:, i * sh: i * sh + kh,
                             j * sw: j * sw + kw, :] * kernel,
                axis=(1, 2, 3),
            )

    return output
