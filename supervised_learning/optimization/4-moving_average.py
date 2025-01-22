#!/usr/bin/env python3
'''
NOTE:
moving average is a technique
that can be used to smooth out time
series data to reduce the “noise” in
the data and more easily identify patterns
and trends.

The  idea behind a moving average is to
take the average of a certain number of
previous periods to come up with an “moving
average” for a given period.
'''


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a
    data set with bias correction.

    Parameters:
    data (list): The list of data to calculate the moving average of.
    beta (float): The weight used for the moving average.

    Returns:
    list: A list containing the moving averages of data.
    """
    # initialize exponentially weighted moving average
    v = 0
    # initialize storage for bias-corrected moving averages
    moving_averages = []
    # iterate over data
    for t in range(len(data)):
        # update the moving average for each data[t]
        # using this formula: v_t = \beta v_{t-1} + (1 - \beta) x_t )
        v = beta * v + (1 - beta) * data[t]
        # apply the bias correction to the moving average
        # using this formula: ( \hat{v}_t = \frac{v_t}{1 - \beta^t} )
        bias_corrected_v = v / (1 - beta**(t + 1))
        # append the bias corrected moving average to the list
        moving_averages.append(bias_corrected_v)
    # return it pretty simple stuff... :')
    return moving_averages
