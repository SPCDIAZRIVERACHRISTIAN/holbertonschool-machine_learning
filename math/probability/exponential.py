#!/usr/bin/env python3
"""
Module for representing an Exponential distribution
"""


class Exponential:
    """
    Class that represents an Exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        Args:
            data (list, optional): List of data to estimate distribution
            lambtha (float, optional): Expected number of occurrences in a time frame

        Raises:
            ValueError: If lambtha is not positive
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
            # For exponential distribution, lambtha is 1/mean
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period

        Args:
            x: Time period

        Returns:
            float: PDF value for x, 0 if x is out of range
        """
        # Check if x is out of range (x must be non-negative for Exponential)
        if x < 0:
            return 0

        # Calculate PDF: f(x) = λ * e^(-λx)
        e = 2.7182818285  # Euler's number approximation

        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period

        Args:
            x: Time period

        Returns:
            float: CDF value for x, 0 if x is out of range
        """
        # Check if x is out of range (x must be non-negative for Exponential)
        if x < 0:
            return 0

        # Calculate CDF: F(x) = 1 - e^(-λx)
        e = 2.7182818285  # Euler's number approximation

        return 1 - (e ** (-self.lambtha * x))
