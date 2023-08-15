from __future__ import annotations

import numpy as np
from scipy.stats import gmean


def spectral_centroid(x: np.ndarray, normalized: bool = True) -> float:
    a = np.abs(np.fft.rfft(x))[1:]
    s = (a * np.arange(1, len(a) + 1)).sum() / a.sum()

    if normalized:
        s /= len(x)

    return s


def temporal_centroid(x):
    x = np.abs(np.fft.rfft(x))[2:]
    x = (x * 60 * np.arange(1, len(x) + 1)).sum() / x.sum()
    return x


def spectral_flatness(x):
    x = np.abs(np.fft.rfft(x))[1:]
    x = gmean(x) / np.mean(x)
    return x


def rms(x):
    return np.sqrt(np.mean(np.square(x)))