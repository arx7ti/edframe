from __future__ import annotations

import numpy as np


def fft_amplitudes(x):
    x = np.fft.rfft(x) * 2 / len(x)
    x = np.abs(x)
    return x


def fft_amplitudes_check_fn(x):
    if len(x.shape) > 1:
        raise ValueError