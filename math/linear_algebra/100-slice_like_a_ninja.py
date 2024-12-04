#!/usr/bin/env python3
'''this function slices a matrix along specific axes'''


def np_slice(matrix, axes={}):
    '''this function returns a new numpy.ndarray of sliced matrix'''
    # find the number of axis
    num_axes = matrix.ndim

    # slice
    slices = [slice(None)] * num_axes

    # update
    for axis, slice_range in axes.items():
        slices[axis] = slice(*slice_range)


    return matrix[tuple(slices)]


