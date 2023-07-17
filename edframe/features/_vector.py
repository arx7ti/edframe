from __future__ import annotations

from sklearn.decomposition import PCA, FastICA

import numpy as np


def fft_amplitudes(x):
    x = np.fft.rfft(x) * 2 / len(x)
    x = np.abs(x)
    return x


class PrincipalComponents(PCA):
    pass


class IndependentComponents(FastICA):
    pass