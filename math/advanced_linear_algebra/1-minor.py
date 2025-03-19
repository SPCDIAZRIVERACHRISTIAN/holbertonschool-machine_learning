#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix
"""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    Args:
        matrix (list of lists): Matrix whose minor matrix should be calculated

    Returns:
        list of lists: The minor matrix of matrix

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

    # Special case for 1x1 matrix
    if n == 1:
        return [[1]]

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create the submatrix excluding the current row and column
            submatrix = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            # Calculate the determinant of the submatrix
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
