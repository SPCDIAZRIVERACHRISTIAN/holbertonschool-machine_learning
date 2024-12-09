#!/usr/bin/env python3
'''This function plots a histogram'''

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    '''This function shows a histogram'''

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    bin_edges = np.arange(0, 100, 10)
    plt.hist(student_grades, bins=bin_edges, edgecolor='black')
    plt.title("Project A")
    plt.show()
