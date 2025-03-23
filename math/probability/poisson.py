#!/usr/bin/env python3
"""
Module for representing a Poisson distribution
"""


class Poisson:
    """
    Class that represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Args:
            data (list, optional): List of data to estimate distribution
            lambtha (float, optional): Expected number
                of occurrences in a time frame

        Raises:
            ValueError: If lambtha is not positive or equals to 0
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        Args:
            k: Number of successes

        Returns:
            float: PMF value for k
        """
        # Convert k to integer if it's not already
        k = int(k)

        # Check if k is out of range (k must be non-negative for Poisson)
        if k < 0:
            return 0

        # Calculate PMF: P(X = k) = (λ^k * e^(-λ)) / k!
        e = 2.7182818285  # Euler's number approximation

        # Calculate λ^k
        lambda_k = 1
        for _ in range(k):
            lambda_k *= self.lambtha

        # Calculate e^(-λ)
        e_neg_lambda = e ** (-self.lambtha)

        # Calculate k!
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        # Calculate PMF
        return (lambda_k * e_neg_lambda) / k_factorial

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes

        Args:
            k: Number of successes

        Returns:
            float: CDF value for k
        """
        # Convert k to integer if it's not already
        k = int(k)

        # Check if k is out of range (k must be non-negative for Poisson)
        if k < 0:
            return 0

        # Calculate CDF: P(X <= k) = sum(P(X = i)) for i from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
