#!/usr/bin/env python3
"""
Module for representing a Binomial distribution
"""


class Binomial:
    """
    Class that represents a Binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor

        Args:
            data (list, optional): List of data to estimate distribution
            n (int, optional): Number of Bernoulli trials
            p (float, optional): Probability of a success

        Raises:
            ValueError: If n is not positive or p is not a valid probability
            TypeError: If data is not a list
            ValueError: If data does not contain at least two data points
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Calculate p first (p = 1 - variance/mean)
            # For binomial distribution, variance = n*p*(1-p) and mean = n*p
            # So, variance/mean = (n*p*(1-p))/(n*p) = 1-p
            # Therefore, p = 1 - variance/mean
            p = 1 - (variance / mean)

            # Calculate n using the formula: mean = n*p, so n = mean/p
            n = mean / p

            # Round n to the nearest integer
            self.n = round(n)

            # Recalculate p using the rounded value of n
            # This ensures p is adjusted based on the integer n
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes

        Args:
            k: Number of successes

        Returns:
            float: PMF value for k
        """
        # Convert k to integer
        k = int(k)

        # Check if k is out of range for binomial distribution
        if k < 0 or k > self.n:
            return 0

        # Calculate binomial coefficient (n choose k)
        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i

        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i

        n_minus_k_fact = 1
        for i in range(1, self.n - k + 1):
            n_minus_k_fact *= i

        binomial_coef = n_fact / (k_fact * n_minus_k_fact)

        # Calculate PMF: P(X = k) = (n choose k) * p^k * (1-p)^(n-k)
        p_k = self.p ** k
        q_n_minus_k = (1 - self.p) ** (self.n - k)

        return binomial_coef * p_k * q_n_minus_k

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes

        Args:
            k: Number of successes

        Returns:
            float: CDF value for k
        """
        # Convert k to integer
        k = int(k)

        # Check if k is out of range for binomial distribution
        if k < 0:
            return 0

        # If k is greater than n, the probability is 1
        if k >= self.n:
            return 1.0

        # Calculate CDF: F(k) = P(X ≤ k) = sum(P(X = i)) for i from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
