#!/usr/bin/env python3
'''This script returns a iris flower data set'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Scatter plot, with color determined by the labels
scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels, cmap='plasma')

# Set the labels for the x, y, and z axes
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

# Set the title of the plot
plt.title('PCA of Iris Dataset')

# Display the plot
plt.show()
