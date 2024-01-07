from __future__ import annotations

import numpy as np

from scipy.stats import gmean
from edframe.utils.hf import zero_crossings


def spectral_centroid(x, normalized):
    a = abs(np.fft.rfft(x, axis=-1))[..., 1:]
    arange = np.arange(1, a.shape[-1] + 1)
    arange = np.expand_dims(arange, axis=tuple(range(len(a.shape) - 1)))
    s = (a * arange).sum(-1) / a.sum(-1)

    if normalized:
        s /= x.shape[-1]

    return s


def temporal_centroid(x):
    x = abs(np.fft.rfft(x))[2:]
    x = (x * 60 * np.arange(1, len(x) + 1)).sum() / x.sum()
    return x


def spectral_flatness(x):
    x = abs(np.fft.rfft(x))[1:]
    x = gmean(x) / x.mean()
    return x


def rms(x, axis=-1, keepdims=False):
    return np.sqrt(np.square(x).mean(axis, keepdims=keepdims))


def zero_crossing_rate(x, mode='median'):
    x0 = zero_crossings(x)

    if len(x0) <= 1:
        raise ValueError

    if mode == 'mean':
        x0_rate = np.diff(x0).mean()
    elif mode == 'median':
        x0_rate = np.median(np.diff(x0))
    else:
        raise ValueError

    return x0_rate


def fundamental(x, fs, mode='median'):
    x0_rate = 2 * zero_crossing_rate(x, mode=mode)
    f0 = fs / x0_rate

    return f0


def spectrum(x, fs, f0=None, **kwargs):
    z = np.fft.rfft(x)
    a, phi = abs(z), np.angle(z)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)

    if f0 is None:
        f0 = fundamental(x, fs, **kwargs)

    f0_idx = np.argmin(abs(freqs - f0))
    f_idxs = f0_idx * np.arange(1, len(a) // f0_idx)

    a0, a, phi = a[0], a[f_idxs], phi[f_idxs]

    return a0, a, phi


def thd(x, fs, f0=None, **kwargs):
    _, a, _ = spectrum(x, fs, f0=f0, **kwargs)
    v = np.sqrt((a[1:]**2).sum()) / a[0]

    return v
