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
    z_min, z_max = np.min(z), np.max(z)
    z_normalized = (z - z_min) / (z_max - z_min)

    # Creates a colormap for elevation values
    cmap = plt.get_cmap('viridis')

    # Create the scatter plot
    scatter = plt.scatter(x, y, c=z_normalized, cmap=cmap)
    scale = [5, 10, 15, 20, 25, 30, 35, 40]
    # normalizes scale to be the same as z
    scale_normalized = (np.array(scale) - z_min) / (z_max - z_min)

    cbar = plt.colorbar(scatter)
    cbar.set_ticks(scale_normalized)
    cbar.set_ticklabels(scale)
    # Add labels and titles to the plot
    cbar.set_label('elevation (m)')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    # shows the plot
    plt.show()
