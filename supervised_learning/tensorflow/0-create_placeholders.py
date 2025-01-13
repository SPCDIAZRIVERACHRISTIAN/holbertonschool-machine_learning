#!/usr/bin/env python3
'''This is a function that returns two placeholders'''

import tensorflow.compat.v1 as tf # type: ignore
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    '''Returns two placeholders x and y

    Args:
        nx (int): the number of feature columns in our data
        classes (int): the number of classes in our classifier

    Returns:
        array: placeholder for the input data to the neural network(X)
                and placeholder for the one-hot labels for the input data(Y)
    '''
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
