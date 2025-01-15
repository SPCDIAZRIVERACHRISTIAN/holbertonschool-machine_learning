#!/usr/bin/env python3
'''This is a function that builds,
    trains, and saves a neural network classifier'''

import tensorflow.compat.v1 as tf  # type: ignore
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''the function trains a model from a given dataset

    Args:
        X_train (np.ndarray): containing the training input data
        Y_train (np.ndarray): containing the training labels
        X_valid (np.ndarray): containing the validation input data
        Y_valid (np.ndarray): containing the validation labels
        layer_sizes (list): containing the number of nodes in each
            layer of the network
        activations (list): containing the activation
            functions for each layer of the network
        alpha (float):  learning rate
        iterations (int): number of iterations to train over
        save_path (str, optional): designates where to
            save the model. Defaults to "/tmp/model.ckpt".

    Returns:
        str: path where the model is saved
    '''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    y_pred = forward_prop(x, layer_sizes, activations)

    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()

    tf.add_to_collection("X", x)
    tf.add_to_collection("Y", y)
    tf.add_to_collection("Y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            train_loss, train_accuracy = sess.\
                    run([loss, accuracy], feed_dict={x: X_train, y: Y_train})

            valid_loss, valid_accuracy = sess.\
                run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)
    return save_path
