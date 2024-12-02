#!/usr/bin/env python3
'''This function returns the shape of a matrix'''


def matrix_shape(matrix):
    '''returns the shape of a matrix

    Args:
        matrix (_type_): matrix with the same size of columns

    Returns:
        shape (list): the size of the matrix
    '''
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
