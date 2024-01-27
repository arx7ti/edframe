from __future__ import annotations

from typing import Union, Any, Callable
from scipy.interpolate import interp1d
from scipy.signal import resample, find_peaks
from statsmodels.tsa.ar_model import AutoReg

import math
import inspect
import numpy as np

from edframe.features import rms, spectral_centroid
from edframe.data.generators import make_hf_cycles_from
from edframe.signals._fitps import FITPS


def identity(x: np.ndarray):
    return x


def downsample(x: np.ndarray, fs0: int, fs: int, axis: int = -1) -> np.ndarray:
    if fs > fs0:
        raise ValueError

    n = x.shape[-1]
    n_new = round(n / fs0 * fs)
    # assert n_new < n

    if n_new == n:
        return x

    if axis < 0:
        axis = len(x.shape) + axis

    if axis >= len(x.shape):
        raise ValueError

    x = resample(x, n_new, axis=axis)

    return x


def upsample(
    x: np.ndarray,
    fs0: int,
    fs: int,
    kind: str = 'linear',
    axis: int = -1,
) -> np.ndarray:
    if fs < fs0:
        raise ValueError

    n = x.shape[-1]
    n_new = round(n / fs0 * fs)
    # assert n_new > n

    if n_new == n:
        return x

    t0 = np.linspace(0, 1, n, dtype=x.dtype)
    t1 = np.linspace(0, 1, n_new, dtype=x.dtype)
    mean = x.mean(axis, keepdims=True)
    x = x - mean
    upsampler = interp1d(t0, x, kind=kind, axis=axis)
    x = upsampler(t1)
    x += mean

    return x


def pad(
    x: np.ndarray,
    n: int | tuple[int, int],
    axis: int = -1,
) -> np.ndarray:
    # TODO pad with values
    if isinstance(n, tuple):
        a, b = n
    else:
        a, b = n - n // 2, n // 2

    if axis < 0:
        axis = len(x.shape) + axis

    x = np.swapaxes(x, axis, -1)
    xa = np.zeros((*x.shape[:-1], a))
    xb = np.zeros((*x.shape[:-1], b))
    x = np.concatenate((xa, x, xb), axis=-1)
    x = np.swapaxes(x, -1, axis)

    return x


def roll(x: np.ndarray, n: int, axis: int = -1) -> np.ndarray:
    x = np.roll(x, n, axis=axis)
    return x


def crop(x: np.ndarray, a: int, b: int, axis=-1) -> np.ndarray:
    if a < 0 or b < 0:
        raise ValueError

    if b <= a:
        raise ValueError

    if b > x.shape[axis] or a > x.shape[axis]:
        raise ValueError

    if axis < 0:
        axis = len(x.shape) + axis

    x = np.swapaxes(x, axis, -1)
    x = x[..., a:b]
    x = np.swapaxes(x, -1, axis)

    return x


def replicate(
    x: np.ndarray,
    n: int | tuple[int, int],
    axis: int = -1,
) -> np.ndarray:
    if isinstance(n, int):
        a, b = 0, n
    else:
        a, b = n

    if axis < 0:
        axis = len(x.shape) + axis

    x = np.swapaxes(x, axis, -1)

    if a > 0:
        xa = np.repeat(x[..., :1], a, axis=-1)
    else:
        xa = np.empty((*x.shape[:-1], 0))

    if b > 0:
        xb = np.repeat(x[..., -1:], b, axis=-1)
    else:
        xb = np.empty((*x.shape[:-1], 0))

    x = np.concatenate((xa, x, xb), axis=-1)
    x = np.swapaxes(x, -1, axis)

    return x


def extrapolate2d(x, n, **kwargs):
    assert len(x.shape) == 2

    n_orig, n_samples = x.shape

    # Replicate if only one cycle presents
    if len(x) == 1:
        x0 = 0
        x_extra = np.repeat(x, n, axis=0)
    else:
        # Generate cycles for extrapolation interval
        x_extra = make_hf_cycles_from(x, n_samples=n)
        x_extra = x_extra / abs(x_extra).max(1, keepdims=True)

        # Estimate amplitude envelope on the extrapolation interval
        x0 = x.mean(-1, keepdims=True)
        x = x - x0
        a = abs(x).max(1)
        t = np.linspace(0, 1, n_orig)
        kind = kwargs.get('kind', 'linear')
        n_orig_flat, n_extra_flat = np.product(x.shape), np.product(
            x_extra.shape)
        t_flat = np.linspace(0, 1, n_orig_flat)
        a_flat = interp1d(t, a, kind=kind)(t_flat)
        a_extra = AutoReg(a_flat,
                          1).fit().predict(n_orig_flat,
                                           n_orig_flat + n_extra_flat - 1)
        a_extra = a_extra.reshape(-1, n_samples)

        # Apply envelope
        x_extra = a_extra * x_extra

    # Combine original and extrapolated signals
    # FIXME x0 for x_extra?
    x = np.concatenate((x + x0, x_extra))

    return x


# @staticmethod
# @njit
def sync_cycles(
    x: np.ndarray,
    x0: np.ndarray,
    dx: np.ndarray,
    ns_int: int,
) -> np.ndarray:
    ns_float = x.dtype.type(ns_int)
    n_periods = len(dx) - 1
    mat = np.zeros((n_periods, ns_int), dtype=x.dtype)
    x0 = x0.astype(x.dtype)

    for j in np.arange(n_periods, dtype=np.int32):
        length = (x0[j + 1] + dx[j + 1]) - (x0[j] + dx[j])
        dist = length / ns_float

        for k in np.arange(ns_int, dtype=np.int32):
            k1 = x0[j] + dx[j] + dist * x.dtype.type(k)
            k2 = np.int32(np.floor(k1))
            k3 = np.int32(np.ceil(k1))
            mat[j, k] = x[k2] + (x[k3] - x[k2]) * dx[j]
    return mat


def compute_shifts(v: np.ndarray, v0: np.ndarray) -> np.ndarray:
    dv = np.empty(0, dtype=v.dtype)

    for j in np.arange(len(v0) - 1, dtype=np.int32):
        k = v0[j]
        dv = np.append(dv, -v[k] / (v[k + 1] - v[k]))

    return dv


def extrapolate(x, n, x0=None, fs=None, f0=None, **kwargs):
    '''
    Main assumption: constant frequency
    '''
    if isinstance(n, tuple):
        a, b = n
    else:
        a, b = n - n // 2, n // 2

    is_aligned = len(x.shape) == 2

    if is_aligned:
        n_orig = np.product(x.shape)
        xs = x
        da, db = 0, 0
    else:
        n_orig = len(x)

        if x0 is None:
            assert fs is not None
            assert f0 is not None
            T = int(round(fs / f0))
            # TODO if not whole cycles (amp issue)
            x0 = np.arange(0, len(x), T)
            dx = np.zeros(len(x0) - 1)
        else:
            dx = compute_shifts(x, x0)
            T = round(np.diff(x0).mean())

        # Obtain synchronized signal with equal number of samples per cycle
        xs = sync_cycles(x, x0, dx, T)

        # Compensate missing cycles
        da = x0[0]
        db = n_orig - x0[-2]

    # Number of cycles to extrapolate
    n_samples = xs.shape[1]
    na = math.ceil((a + da) / n_samples)
    nb = math.ceil((b + db) / n_samples)

    # Generate extra cycles
    xe = make_hf_cycles_from(xs, n_samples=na + nb)
    xe /= abs(xe).max(1, keepdims=True)
    xa, xb = xe[:na], xe[na:]

    amp = abs(xs).max(1)
    n_cycles = len(xs)
    areg1 = AutoReg(amp[::-1], 1).fit()
    areg2 = AutoReg(amp, 1).fit()

    if na > 0:
        ampa = areg1.predict(n_cycles, n_cycles + na - 1)[::-1]
        xa = (xa * ampa[:, None])

    if nb > 0:
        ampb = areg2.predict(n_cycles, n_cycles + nb - 1)
        xb = (xb * ampb[:, None])

    x = np.concatenate((xa, xs, xb)).ravel()

    # Restrict the number of samples in the final signal
    x = x[(n_samples - a) % n_samples:]
    x = x[:n_orig + a + b]

    return x


def fryze(v, i):
    s = v * i
    ia = s.mean(-1, keepdims=True) * v / rms(v, axis=-1, keepdims=True)**2
    ir = i - ia

    return ia, ir


def budeanu(v, i):
    # TODO multiple cycles support
    # Fourier Transform
    zv = np.fft.rfft(v, axis=-1)
    zi = np.fft.rfft(i, axis=-1)

    # Compute RMS amplitudes
    const = 2 / v.shape[-1] / np.sqrt(2)
    V, I = abs(zv), abs(zi)
    V[..., 1:], I[..., 1:] = const * V[..., 1:], const * I[..., 1:]
    V[..., 0], I[..., 0] = V[..., 0] / v.shape[-1], I[..., 0] / v.shape[-1]

    # Compute phase differences between current and voltage
    phi = np.angle(zv) - np.angle(zi)

    # Perform Budeanu decomposition
    P = V * I * np.cos(phi)
    Q = V * I * np.sin(phi)
    P = P.sum(-1)
    Q = Q.sum(-1)

    # Compute active component of current
    Vrms = np.sqrt((V**2).sum(-1))
    ia = P / Vrms**2 * v

    # Compute reactive component of current
    zv[..., 1:] *= 1j
    u = np.fft.irfft(zv, axis=-1, n=v.shape[-1])
    iq = Q / Vrms**2 * u

    # Compute distortion component of current
    id = i - ia - iq

    return ia, iq, id
