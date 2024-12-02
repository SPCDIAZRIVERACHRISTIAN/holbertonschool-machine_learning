#!/usr/bin/env python3
'''This function adds two arrays element-wise'''


def add_arrays(arr1, arr2):
    '''this returns two added arrays

    Args:
        arr1 (list): a list contyining integers
        arr2 (list): a list containing integers
        return: if arrays are the same returns
        the sum of them else retirns None
    '''
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    else:
        return None
