#!/usr/bin/env python3
'''This function performs element-wise addition, subtraction,
    multiplication, and division'''


def np_elementwise(mat1, mat2):
    '''adds substract divides and multiplies the matrices given'''

    added = mat1 + mat2
    substracted = mat1 - mat2
    multiplied = mat1 * mat2
    divided = mat1 / mat2

    return added, substracted, multiplied, divided
