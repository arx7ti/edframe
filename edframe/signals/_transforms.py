from __future__ import annotations

import inspect
from scipy.signal import resample
from scipy.interpolate import interp1d
from typing import Union, Any, Callable
from statsmodels.tsa.ar_model import AutoReg

import numpy as np


def identity(x: np.ndarray):
    return x


def downsample(x: np.ndarray, n_samples: int) -> np.ndarray:
    if n_samples > x.shape[-1]:
        raise ValueError

    x = resample(x, n_samples, axis=-1)

    return x


def enhance(x: np.ndarray, n_samples: int, kind: str = 'linear') -> np.ndarray:
    if n_samples < x.shape[-1]:
        raise ValueError

    t0 = np.linspace(0, 1, x.shape[-1], dtype=x.dtype)
    t1 = np.linspace(0, 1, n_samples, dtype=x.dtype)
    mean = x.mean(axis=-1, keepdims=True)
    x = x - mean
    enhancer = interp1d(t0, x, kind=kind, axis=-1)
    x = enhancer(t1)
    x += mean

    return x


def pad(
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
    xa = np.zeros((*x.shape[:-1], a))
    xb = np.zeros((*x.shape[:-1], b))
    x = np.concatenate((xa, x, xb), axis=-1)
    x = np.swapaxes(x, -1, axis)

    return x


def roll(
    x: np.ndarray,
    n_periods: int,
    mode: str = 'constant',
    **kwargs: Any,
) -> np.ndarray:
    if n_periods > 0:
        slicer = np.s_[:n_periods]
    elif n_periods < 0:
        slicer = np.s_[n_periods:]
    else:
        slicer = None

    if slicer is not None:
        x = np.roll(x, n_periods, axis=-2)

        if mode == 'constant':
            fill_values = kwargs.get('constant_value', 0)
        elif mode == 'noise':
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            fill_values = np.random.randn(*x.shape[:-2], abs(n_periods),
                                          x.shape[-1])
            fill_values = mean + std * fill_values.astype(x.dtype)
        else:
            raise ValueError

        x[slicer] = fill_values

    return x


def crop(x: np.ndarray, a: int, b: int, axis=-1) -> np.ndarray:
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


def extrapolate(x: np.ndarray, n: int, lags: int) -> np.ndarray:
    if len(x.shape) > 1:
        raise NotImplementedError

    if x.shape[0] == 1:
        raise NotImplementedError

    if x.shape[0] == 2:
        raise NotImplementedError

    a = x.size
    b = x.size + n - 1
    autoreg = AutoReg(x, lags, trend='ct').fit()
    xe = autoreg.predict(a, b).view()
    x = np.concatenate((x, xe))

    return x


class F:

    def __init__(self, fn: Callable, map_out_args, **map_in_args) -> None:

        if len(map_out_args) == 0:
            raise ValueError

        params = inspect.signature(fn).parameters.keys()

        for param in map_in_args.keys():
            if param not in params:
                raise ValueError

        self._fn = fn
        self._map_out_args = map_out_args
        self._map_in_args = map_in_args

    def __call__(self, ps: PowerSample) -> PowerSample:

        data = {}

        for fparam, attr in self._map_in_args.items():
            if hasattr(ps, attr):
                data.update({fparam: getattr(ps, attr)})
            else:
                raise ValueError("Parameter \"%s\" was not found" % attr)

        result = self._fn(**data)

        if not isinstance(result, tuple):
            result = (result, )

        if len(result) < len(self._map_out_args):
            raise ValueError

        data = {}

        for attr, v in zip(self._map_out_args, result):
            if hasattr(ps, attr):
                data.update({attr: v})
            else:
                raise ValueError

        ps = ps.update(**data)

        return ps