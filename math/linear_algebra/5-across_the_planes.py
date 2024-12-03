#!/usr/bin/env python3
'''This function adds two matrices if they are the same'''


def add_matrices2D(mat1, mat2):
    '''_summary_

    Args:
        mat1 (list): a 2d matrix
        mat2 (list): a 2d matrix
        sum (list): sum of mat1 and mat2
        return: a none if they are not the same
        if they are return the sum
    '''
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        sum = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
               for i in range(len(mat1))]
        return sum
