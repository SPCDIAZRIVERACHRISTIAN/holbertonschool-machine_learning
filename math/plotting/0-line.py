#!/usr/bin/env python3
'''This function shows a plotted line graph '''
import numpy as np
import matplotlib.pyplot as plt


def line():
    '''shows a plotted linear graph'''

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, "r-")
    plt.xlim(0, 10)
    plt.show()
