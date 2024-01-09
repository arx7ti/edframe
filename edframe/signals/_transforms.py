from __future__ import annotations

from scipy.signal import resample
from scipy.interpolate import interp1d
from typing import Union, Any, Callable
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
    n = x.shape[-1]
    n_new = round(n / fs0 * fs)
    assert n_new < n

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
    n = x.shape[-1]
    n_new = round(n / fs0 * fs)
    assert n_new > n

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
    if isinstance(n, int):
        a, b = 0, n
    else:
        a, b = n

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

    # Generate cycles for extrapolation interval
    x_extra = make_hf_cycles_from(x, n_samples=n)
    x_extra = x_extra / abs(x_extra).max(1, keepdims=True)

    # Estimate amplitude envelope on the extrapolation interval
    x0 = x.mean(-1, keepdims=True)
    x = x - x0
    a = abs(x).max(1)
    t = np.linspace(0, 1, n_orig)
    kind = kwargs.get('kind', 'linear')
    n_orig_flat, n_extra_flat = np.product(x.shape), np.product(x_extra.shape)
    t_flat = np.linspace(0, 1, n_orig_flat)
    a_flat = interp1d(t, a, kind=kind)(t_flat)
    a_extra = AutoReg(a_flat, 1).fit().predict(n_orig_flat,
                                               n_orig_flat + n_extra_flat - 1)
    a_extra = a_extra.reshape(-1, n_samples)

    # Apply envelope
    x_extra = a_extra * x_extra

    # Combine original and extrapolated signals
    x = np.concatenate((x + x0, x_extra))

    return x


def extrapolate(x, n, v=None, **kwargs):
    # FIXME if only one signal and non-aligned
    is_aligned = len(x.shape) == 2

    # if not is_aligned and v is None:
    #     z = np.fft.rfft(x, axis=)
    #     v =

    if not is_aligned:
        n_orig = len(x)

        # Obtain synchronized signal with equal number of samples per cycle
        fitps = FITPS()
        v2d, x2d = fitps(v, x)
        v0 = fitps.zero_crossings

        # Compensate missing cycles from the right after FITPS
        dright = n_orig - v0[-2]
    else:
        n_orig = np.product(x.shape)
        v2d, x2d = v, x
        v0 = None
        dright = 0

    # Number of cycles that are basis for extrapolation
    n_obs, n_samples = x2d.shape
    # Number of cycles to extrapolate
    n_extra = math.ceil((n + dright) / n_samples)
    # Number of samples in the final signal
    n_full = n + n_orig

    v2d = v2d if v2d is None else extrapolate2d(v2d, n_extra, **kwargs)
    x2d = extrapolate2d(x2d, n_extra, **kwargs)

    if not is_aligned:
        # Keep only extrapolated interval
        v2d = v2d[-n_extra:]
        x2d = x2d[-n_extra:]

        # Extrapolate frequency fluctuations
        dv0 = np.diff(v0)
        dv0 = AutoReg(dv0, 1).fit().predict(n_obs, n_obs + n_extra - 1)
        dv0 = np.round(dv0).astype(int)

        # Apply frequency fluctuations over extrapolated signal
        t_aligned = np.linspace(0, 1, n_samples)
        crop = True

        for vi, xi, n_samples_i in zip(v2d, x2d, dv0):
            if n_samples_i != n_samples:
                ti = np.linspace(0, 1, n_samples_i)
                vi = interp1d(t_aligned, vi)(ti)
                xi = interp1d(t_aligned, xi)(ti)

            if dright > 0 and crop:
                vi = vi[dright:]
                xi = xi[dright:]
                crop = False

            v = np.concatenate((v, vi))
            x = np.concatenate((x, xi))
    else:
        v = v if v is None else v2d.ravel()
        x = x2d.ravel()

    del v2d, x2d

    if n_full < len(x):
        v = v if v is None else v[:n_full]
        x = x[:n_full]

    if v is not None:
        return v, x

    return x


# class F:

#     def __init__(self, fn: Callable, map_out_args, **map_in_args) -> None:

#         if len(map_out_args) == 0:
#             raise ValueError

#         params = inspect.signature(fn).parameters.keys()

#         for param in map_in_args.keys():
#             if param not in params:
#                 raise ValueError

#         self._fn = fn
#         self._map_out_args = map_out_args
#         self._map_in_args = map_in_args

#     def __call__(self, ps: PowerSample) -> PowerSample:

#         data = {}

#         for fparam, attr in self._map_in_args.items():
#             if hasattr(ps, attr):
#                 data.update({fparam: getattr(ps, attr)})
#             else:
#                 raise ValueError("Parameter \"%s\" was not found" % attr)

#         result = self._fn(**data)

#         if not isinstance(result, tuple):
#             result = (result, )

#         if len(result) < len(self._map_out_args):
#             raise ValueError

#         data = {}

#         for attr, v in zip(self._map_out_args, result):
#             if hasattr(ps, attr):
#                 data.update({attr: v})
#             else:
#                 raise ValueError

#         ps = ps.update(**data)

#         return ps


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
    zv[1:] *= 1j
    u = np.fft.irfft(zv, axis=-1, n=v.shape[-1])
    iq = Q / Vrms**2 * u

    # Compute distortion component of current
    id = i - ia - iq

    return ia, iq, id
