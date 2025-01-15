#!/usr/bin/env python3
'''This function trains a model using mini-batch gradient descent'''

import tensorflow.keras as K  # type:ignore


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                verbose=True,
                shuffle=False
                ):
    '''trains a model using mini-batch gradient descent

    Args:
        network (model): is the model to train
        data (ndarray):  containing the input data
        labels (ndarray): containing the labels of data
        batch_size (int): size of the batch used for
            mini-batch gradient descent
        epochs (int): number of passes through data
            for mini-batch gradient descen
        verbose (bool, optional): determines if
            output should be printed during training. Defaults to True.
        shuffle (bool, optional): determines whether
            to shuffle the batches every epoch. Normally,
            it is a good idea to shuffle,
            but for reproducibility. Defaults to False.

    Returns:
        History object: a record of training loss values and metrics values at
            successive epochs, as well as validation loss values and validation
            metrics values (if applicable).
    '''
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle
                          )
    return history
