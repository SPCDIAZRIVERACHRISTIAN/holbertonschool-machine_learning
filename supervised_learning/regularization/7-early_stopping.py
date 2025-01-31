#!/usr/bin/env python3
"""This module determines if you should stop
gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Parameters:
    cost: float - the current validation cost of the neural network.
    opt_cost: float - the lowest recorded validation cost of the
    neural network.
    threshold: float - the threshold used for early stopping.
    patience: int - the patience count used for early stopping.
    count: int - the count of how long the threshold has not been met.

    Returns:
    A boolean of whether the network should be stopped early, followed
    by the updated count.
    """
    # If the current cost is less than the optimal cost by the threshold,
    # then the count is reset to 0, as the threshold has been met.
    if cost < opt_cost - threshold:
        count = 0
    else:
        # If the current cost is not less than the optimal cost
        # by the threshold,
        # then the count is incremented, as the threshold has not been met.
        count += 1

    # If the count is equal to the patience, then early stopping should occur,
    # so the function returns True. Otherwise, it returns False.
    if count >= patience:
        return True, count
    else:
        return False, count
