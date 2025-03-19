#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix (list of lists): Matrix whose determinant should be calculated

    Returns:
        float or int: The determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Special case for 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is not empty
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices
    det = 0
    for j in range(n):
        # Create the submatrix excluding the current row and column
        submatrix = [
            [matrix[i][k] for k in range(n)
             if k != j] for i in range(1, n)]
        # Calculate the cofactor and add it to the determinant recursively
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)

    return det
