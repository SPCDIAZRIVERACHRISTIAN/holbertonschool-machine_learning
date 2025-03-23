#!/usr/bin/env python3
"""
Module for representing a Normal distribution
"""


import math


class Normal:
    """
    Class that represents a Normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        Args:
            data (list, optional): List of data to estimate distribution
            mean (float, optional): Mean of the distribution
            stddev (float, optional): Standard deviation of the distribution

        Raises:
            ValueError: If stddev is not positive or equals to 0
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = float(sum(data) / len(data))

            # Calculate standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Args:
            x: The x-value

        Returns:
            float: The z-score of x
        """
        # Z-score formula: z = (x - μ) / σ
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Args:
            z: The z-score

        Returns:
            float: The x-value of z
        """
        # X-value formula: x = z * σ + μ
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value

        Args:
            x: The x-value

        Returns:
            float: The PDF value for x
        """
        # Constants
        e = 2.7182818285  # Euler's number
        pi = 3.1415926536  # Pi

        # Calculate the exponent term
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2

        # PDF formula: f(x) = (1 / (σ√(2π))) * e^(-0.5 * ((x-μ)/σ)²)
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)

        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value

        Args:
            x: The x-value

        Returns:
            float: The CDF value for x
        """
        # Calculate z-score for the given x-value
        z = self.z_score(x)

        # Use the error function to approximate the CDF
        # CDF(x) = 0.5 * (1 + erf(z/√2))
        return 0.5 * (1 + self._erf(z / 2**0.5))

    def _erf(self, x):
        """
        Approximates the error function

        Args:
            x: Input value

        Returns:
            float: Approximation of the error function at x
        """
        # Constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # Save the sign of x
        sign = 1
        if x < 0:
            sign = -1
        x = abs(x)

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return sign * y
