from __future__ import annotations

from scipy.signal import resample
from scipy.interpolate import interp1d
from typing import Union, Any, Callable
from statsmodels.tsa.ar_model import AutoReg

import math
import inspect
import numpy as np

from ..features import rms, spectral_centroid
from ..data.generators import make_periods_from


def _align_variation(v1, v2):
    order = []

    for k in range(len(v1)):
        tmp = []

        for j in range(len(v2)):
            d = abs(v1[k] - v2[j])
            tmp.append((d, j))

        tmp = list(sorted(tmp, key=lambda x: x[0]))
        tmp = [x for _, x in tmp]

        for j in tmp:
            if j not in order:
                order.append(j)
                break

    order = np.asarray(order)

    return order


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


def extrapolate(x0, n, **kwargs):
    assert n > 0
    assert len(x0.shape) == 2

    n0 = len(x0)
    n_max = n
    n = math.ceil(n / x0.shape[1])

    sc0 = spectral_centroid(x0, True)
    sc0 = AutoReg(sc0, 1).fit().predict(start=n0, end=n0 + n - 1)

    x = make_periods_from(x0, n_samples=n)
    x = x / abs(x).max(1, keepdims=True)
    sc = spectral_centroid(x, False)

    order = _align_variation(sc0, sc)
    x = x[order]

    a0 = abs(x0).max(1)
    t0 = np.linspace(0, 1, n0)
    kind = kwargs.get('kind', 'linear')
    interp = interp1d(t0, a0, kind=kind)

    x0 = x0.ravel()
    x = x.ravel()
    n0, n = len(x0), len(x)
    t0 = np.linspace(0, 1, n0)
    a0 = interp(t0)

    a = AutoReg(a0, 1).fit().predict(start=n0, end=n0 + n - 1)

    x = a * x

    if n_max < len(x):
        x = x[:n_max]

    x = np.concatenate((x0, x))

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
    raise NotImplementedError
