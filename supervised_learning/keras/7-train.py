#!/usr/bin/env python3
"""This module implements learning rate decay
"""
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    early_stopping=False,
    patience=0,
    learning_rate_decay=False,
    alpha=0.1,
    decay_rate=1,
    verbose=True,
    shuffle=False,
):
    """
    Train a model using mini-batch gradient descent and
    also analyze validation data.

    Parameters:
    network (keras model): The model to train.
    data (numpy.ndarray): The input data, of shape (m, nx).
    labels (numpy.ndarray): The labels of the data,
    one-hot encoded, of shape (m, classes).
    batch_size (int): The size of the batch used for
    mini-batch gradient descent.
    epochs (int): The number of passes through data for
    mini-batch gradient descent.
    validation_data (tuple): Data to validate the model with, if not None.
    early_stopping (bool): Whether to use early stopping mechanism.
    patience (int): Number of epochs to wait for improvement before stopping.
    learning_rate_decay (bool): Whether to use learning rate decay.
    alpha (float): Initial learning rate.
    decay_rate (float): The decay rate.
    verbose (bool, optional): Determines if output should
    be printed during training. Default is True.
    shuffle (bool, optional): Determines whether to shuffle
    the batches every epoch. Default is False.

    Returns:
    history: The History object generated after training the model.
    """

    callbacks = []

    # If early_stopping is True and validation_data is provided,
    # use EarlyStopping callback
    if early_stopping and validation_data:
        callbacks.append(
            K.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
        )

    # If learning_rate_decay is True and validation_data is provided,
    # use LearningRateScheduler callback
    if learning_rate_decay and validation_data:

        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(scheduler,
                                                           verbose=1))

    # Fit the model using the provided parameters. If validation_data
    # is not None, it will be used for validation.
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks,
    )

    return history
