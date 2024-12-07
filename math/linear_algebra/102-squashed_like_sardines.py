#!/usr/bin/env python3
'''This function concatenates two matrices along a specific axis'''


def cat_matrices(mat1, mat2, axis=0):
    '''
    Concatenates two matrices (nested lists) along a specified axis.
    Returns a new matrix if possible, otherwise returns None.
    '''

    # A helper function to determine the shape of the matrix recursively
    def get_shape(matrix):
        shape = []
        current = matrix
        while isinstance(current, list):
            shape.append(len(current))
            if len(current) == 0:
                # Empty list found, can't go deeper
                break
            current = current[0]
        return tuple(shape)

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check that both have the same number of dimensions
    if len(shape1) != len(shape2):
        return None

    # Check that all dimensions match except for the one we are concatenating along
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    # Now perform the concatenation
    # Base case: If axis == 0, we concatenate along the first dimension
    if axis == 0:
        # Make sure that all sub-dimensions are compatible (already checked above)
        return mat1 + mat2

    # If axis > 0, we must recursively go deeper
    # Both mat1 and mat2 must be lists of lists at this point
    if not (isinstance(mat1, list) and isinstance(mat2, list)):
        # Can't go deeper if they're not lists
        return None

    if len(mat1) != len(mat2):
        # They must match in this dimension to zip correctly
        return None

    new_matrix = []
    for sub1, sub2 in zip(mat1, mat2):
        merged = cat_matrices(sub1, sub2, axis=axis-1)
        if merged is None:
            return None
        new_matrix.append(merged)

    return new_matrix
