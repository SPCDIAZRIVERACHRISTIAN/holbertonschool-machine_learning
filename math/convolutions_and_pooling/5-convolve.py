#!/usr/bin/env python3
"""Multiple Kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple
    kernels with custom padding and stride.

    Parameters:
    images (numpy.ndarray): The images with shape (m, h, w, c).
    kernels (numpy.ndarray): The kernels for the convolution
    with shape (kh, kw, kc, nc).
    padding (str or tuple): The padding for the height and
    width of the image.
    stride (tuple): The stride for the height and width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    # Extract dimensions from images' and kernels' shapes
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
    output = np.zeros((m, output_h, output_w, nc))

    # Pad images
    image_padded = \
        np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Reshape kernels for convolution
    kernels = kernels.reshape((1, *kernels.shape))

    # Apply kernels to images
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                image_padded[:, i * sh: i * sh + kh, j *
                             sw: j * sw + kw, :, None] * kernels,
                axis=(1, 2, 3),
            )

    return output
