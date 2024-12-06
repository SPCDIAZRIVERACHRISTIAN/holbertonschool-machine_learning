#!/usr/bin/env python3
'''This function adds two matrices'''


def add_matrices(mat1, mat2):
    '''This function uses recursion to iterate through
        multiple dimension in a matrix. If the
        matrix is only 1D it adds all elements.

    Args:
        mat1 (list): Arrays that can be multidimensional
        mat2 (list)): Arrays that can be multidimensional

    Returns:
        _type_: Returns a matrix with
    '''
    # Checks if mat1 is a int or float
    # this is the part that adds lists of ints or floats
    if isinstance(mat1, (int, float)):
        return mat1 + mat2
    # checks if length of mat1 and 2 are the same
    if len(mat1) != len(mat2):
        return None
    # this part iterates over mat1 to recursively
    # go through the different layers of lists
    # until it gets to where the int and floats are
    sum = []
    for i in range(len(mat1)):
        sum_element = add_matrices(mat1[i], mat2[i])
        if sum_element is None:
            return None
        sum.append(sum_element)
    return sum
