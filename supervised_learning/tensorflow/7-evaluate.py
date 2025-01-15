#!/usr/bin/env python3
'''This function evaluates the output of a neural network'''

import tensorflow.compat.v1 as tf  # type:ignore
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    '''Evaluates the output of a neural network

    Args:
        X: np.ndarray containing the input data
        Y: np.ndarray containing the labels
        save_path: str, path to the saved model

    Returns:
        tuple: (prediction, accuracy, loss)
    '''
    with tf.Session() as sesh:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sesh, save_path)

        x = tf.get_collection("X")[0]
        y = tf.get_collection("Y")[0]
        y_pred = tf.get_collection("Y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        prediction = sesh.run(y_pred, feed_dict={x: X, y: Y})
        accuracy_value = sesh.run(accuracy, feed_dict={x: X, y: Y})
        loss_value = sesh.run(loss, feed_dict={x: X, y: Y})

        return prediction, accuracy_value, loss_value
