#!/usr/bin/env python3
'''This function returns a concatenated matrix in a specific axis'''

import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''Returns concatenated matrix in axis specified'''

    return np.concatenate((mat1, mat2), axis)
