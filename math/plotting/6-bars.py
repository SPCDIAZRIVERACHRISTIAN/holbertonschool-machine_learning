#!/usr/bin/env python3
'''This function plots a stacked bar graph'''

import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''Shows a stacked bar graph'''
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    #  x axis string columns
    x = ['Farrah', 'Fred', 'Felicia']
    #  y axis are the rows in the matrix fruit
    y1 = fruit[0]  # apple
    y2 = fruit[1]  # bananas
    y3 = fruit[2]  # oranges
    y4 = fruit[3]  # peaches
    #  set the width of the bars
    width = 0.5
    #  plot bars in stack manner
    plt.bar(x, y1, color='red', width=width, label='apples')
    plt.bar(x, y2, color='yellow', bottom=y1, width=width, label='bananas')
    plt.bar(x, y3, color='#ff8000', bottom=np.array(y1) +
            np.array(y2), width=width, label='oranges')
    plt.bar(x, y4, color='#ffe5b4', bottom=np.array(y1) +
            np.array(y2) + np.array(y3), width=width, label='peaches')
    #  move legend to upper right corner
    plt.legend(loc='upper right')
    #  label y axis
    plt.ylabel('Quantity of Fruit')
    #  sets the range shown in y axis
    plt.yticks(range(0, 81, 10))
    #  title of the plot
    plt.title('Number of Fruit per Person')
    #  shows the bar graph
    plt.show()
