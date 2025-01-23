#!/usr/bin/env python3
'''
NOTE:
To update the learning rate
using inverse time decay in NumPy,
you can use the following formula:

[ \alpha_t = \frac{\alpha}{1 + \text{decay_rate}
\times \\left\\lfloor \frac{\text{global_step}}
{\text{decay_step}} \right\rfloor} ]
'''

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Updates the learning rate using inverse time decay.

    Args:
    alpha (float): The initial learning rate.
    decay_rate (float): The decay rate.
    global_step (int): The current step in the training process.
    decay_step (int): The number of steps
    after which the learning rate is decayed.

    Returns:
    float: The updated learning rate.
    '''
    update_alpha = alpha / (1 + decay_rate
                            * np.floor(global_step / decay_step))

    return update_alpha
