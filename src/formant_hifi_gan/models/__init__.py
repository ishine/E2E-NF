import numpy as np

from .discriminator import *  # NOQA
from .generator import *  # NOQA
from .neuralformants import *  # NOQA


def dilated_factor(batch_f0, fs, dense_factor):
    """Pitch-dependent dilated factor

    Args:
        batch_f0 (ndarray): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle

    Return:
        dilated_factors(np array):
            float array of the pitch-dependent dilated factors (T)

    """
    batch_f0[batch_f0 == 0] = fs / dense_factor
    dilated_factors = np.ones(batch_f0.shape) * fs / dense_factor / batch_f0
    assert np.all(dilated_factors > 0)

    return dilated_factors
