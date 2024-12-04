#!/usr/bin/env python3
'''this function slices a matrix along specific axes'''


def np_slice(matrix, axes={}):
    '''this function returns a new numpy.ndarray of sliced matrix'''
    # find the number of axis
    num_axes = matrix.ndim
    matrix_new = matrix.copy()
    print(f"dimentions {num_axes}")

    # slice
    slices = [slice(None)] * num_axes
    print(f"slice {slices}")

    # update
    for axis, slice_range in axes.items():
        print(f"axis {axis}")
        print(f"slice range {slice_range}")
        slices[axis] = slice(*slice_range)


    return matrix_new[tuple(slices)]


