#!/usr/bin/env python3
'''This creates a scatter plot of sampled elevations on a mountain'''
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    '''this makes a scatter plot with elevation on a mountain'''

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))
    # normalizes the range of z to a range between 0 and 1
    # formula z range(0, 1) = (z - z min number) /
    # (z max number - z min number)
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))

    # Creates a colormap for elevation values
    cmap = plt.get_cmap('viridis')

    # Create the scatter plot
    plt.scatter(x, y, c=z_normalized, cmap=cmap)

    # Add labels and titles to the plot
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    plt.colorbar(label='Elevation (m)')
    # shows the plot
    plt.show()
