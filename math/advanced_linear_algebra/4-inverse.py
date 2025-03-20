#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""

determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor
cofactor = __import__('2-cofactor').cofactor
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Calculates the inverse of a matrix

    Args:
        matrix (list of lists): Matrix whose inverse should be calculated

    Returns:
        list of lists or None: The inverse
            of matrix, or None if matrix is singular

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

    # Calculate the determinant
    det = determinant(matrix)

    # Check if matrix is singular (determinant is zero)
    if det == 0:
        return None

    # Calculate the adjugate matrix
    adj = adjugate(matrix)

    # Calculate the inverse matrix
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            # Divide each element of the adjugate by the determinant
            inverse_row.append(adj[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
