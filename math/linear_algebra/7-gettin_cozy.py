#!/usr/bin/env python3
'''this function concatenates two matrices along a specific axis'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''This returns concatenated matrices
    by their axis

    Args:
        mat1 (list): _description_
        mat2 (list): _description_
        axis (int, optional): specifying if it
        will be concatenated by rows(0) or columns(1).
        Defaults to 0.

    Returns:
        list: concatenated matrices
    '''
    mat_cp1 = [row[:] for row in mat1]
    mat_cp2 = [row[:] for row in mat2]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            result = mat_cp1 + mat_cp2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            result = [row1 + row2 for row1, row2 in zip(mat_cp1, mat_cp2)]
    return result
