from __future__ import annotations

import numpy as np


def spectral_centroid(x):
    x = np.abs(np.fft.rfft(x))[1:]
    x = (x * np.arange(1, len(x) + 1)).sum() / x.sum()
    return x


def temporal_centroid(x):
    x = np.abs(np.fft.rfft(x))[2:]
    x = (x * 60 * np.arange(1, len(x) + 1)).sum() / x.sum()
    return x


def spf(x):
    x = np.abs(np.fft.rfft(x))[2:]
    x = np.power(np.prod(x), 1 / len(x)) / np.mean(x)
    return x


def rms(x):
    return np.sqrt(np.mean(np.square(x)))