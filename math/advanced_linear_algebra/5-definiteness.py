#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    Args:
        matrix (numpy.ndarray): Matrix whose definiteness should be calculated

    Returns:
        str or None: The definiteness category of the matrix
        ('Positive definite', 'Positive semi-definite',
         'Negative semi-definite', 'Negative definite', 'Indefinite')
         or None if the matrix does not fit any category

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is empty
    if matrix.size == 0:
        return None

    # Check if matrix is square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric (M = M^T)
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues to determine definiteness
        eigenvalues = np.linalg.eigvals(matrix)

        # Check for positive definiteness: all eigenvalues > 0
        if np.all(eigenvalues > 0):
            return "Positive definite"

        # Check for positive semi-definiteness: all eigenvalues >= 0
        if np.all(eigenvalues >= 0):
            return "Positive semi-definite"

        # Check for negative definiteness: all eigenvalues < 0
        if np.all(eigenvalues < 0):
            return "Negative definite"

        # Check for negative semi-definiteness: all eigenvalues <= 0
        if np.all(eigenvalues <= 0):
            return "Negative semi-definite"

        # If some eigenvalues are positive and some are negative
        if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
            return "Indefinite"

    except np.linalg.LinAlgError:
        # Handle case where eigenvalue calculation fails
        return None

    # If we get here, the matrix doesn't fit any category
    return None
