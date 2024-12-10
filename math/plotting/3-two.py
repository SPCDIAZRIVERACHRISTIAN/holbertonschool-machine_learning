#!/usr/bin/env python3
'''This function shows two line graphs in one plot'''
import numpy as np
import matplotlib.pyplot as plt


def two():
    '''This function shows two lines in a graph'''

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.ylabel('Fraction Remaining')
    plt.xlabel('Time (years)')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g', label='Ra-226')
    plt.legend(loc='upper right')
    plt.show()
