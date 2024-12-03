#!/usr/bin/env python3
'''this function multiplies two matrices'''


def mat_mul(mat1, mat2):
    '''return the result of multiplying two matrices

    Args:
        mat1 (list): an integer matrix
        mat2 (list): an integer matrix

    Returns:
        list: result of the 2 matrices multiplied
    '''
    if len(mat1[0]) != len(mat2):
        return None

    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
