#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a matrix
"""

determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix

    Args:
        matrix (list of lists): Matrix
            whose adjugate matrix should be calculated

    Returns:
        list of lists: The adjugate matrix of matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Get the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # Calculate the adjugate matrix (transpose of cofactor matrix)
    adjugate_matrix = []
    for j in range(n):
        adjugate_row = []
        for i in range(n):
            adjugate_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
