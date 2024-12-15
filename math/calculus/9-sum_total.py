#!/usr/bin/env python3
'''this function does a sigma notation and returns the result'''


def summation_i_squared(n):
    '''returns the sum of the range of n'''
    if isinstance(n, int):
        # derive the sigma notation and return it
        return n * (n + 1) * (2 * n + 1) // 6
    else:
        return None
