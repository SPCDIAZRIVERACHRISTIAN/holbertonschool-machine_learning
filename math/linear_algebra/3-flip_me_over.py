#!/usr/bin/env python3
'''This function transposes a matrix '''


def matrix_transpose(matrix):
    '''returns a transposed matrix

    Args:
        matrix (list[list[int]]): a variable that exp[pects a matrix
        transposed (list): variable with transposed matrix
        return: a transpose matrix
    '''
    transposed = [[matrix[j][i] for j in range(len(matrix))]
                  for i in range(len(matrix[0]))]

    return transposed
