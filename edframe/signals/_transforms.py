from __future__ import annotations

import inspect
from typing import Union, Any
from scipy.signal import resample
from scipy.interpolate import interp1d
# from statsmodels.tsa.ar_model import AutoReg

import numpy as np


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


def pad(x: np.ndarray, padwidth: Union[int, tuple[int, int]],
        **kwargs: Any) -> np.ndarray:
    if isinstance(padwidth, tuple):
        padwidth = ((0, 0), padwidth)
    if len(kwargs) == 0:
        kwargs.update(mode='constant', constant_values=0)
    x = np.pad(x, padwidth, **kwargs)
    return x


def roll(x: np.ndarray,
         n_periods: int,
         mode: str = 'constant',
         **kwargs: Any) -> np.ndarray:
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


def crop(x: np.ndarray, on: int, off: int) -> np.ndarray:
    x = x[on:off + 1]
    return x


def replicate(x: np.ndarray, repwidth: Union[int, tuple[int,
                                                        int]]) -> np.ndarray:
    if isinstance(repwidth, int):
        repwidth = (0, repwidth)
    if repwidth[0] > 0:
        x0 = x[..., :1, :]
    else:
        x0 = np.empty((*x.shape[:-2], 0, x.shape[-1]))
    if repwidth[1] > 0:
        x1 = x[..., -1:, :]
    else:
        x1 = np.empty((*x.shape[:-2], 0, x.shape[-1]))
    x = np.concatenate((x0, x, x1), axis=-2)
    return x


def extrapolate(x: np.ndarray, extrawidth: int) -> np.ndarray:
    if len(x.shape) > 2:
        raise NotImplementedError
    if x.shape[0] == 1:
        raise NotImplementedError
    if x.shape[0] == 2:
        raise NotImplementedError
    start = x.size
    end = x.size + x.shape[1] * extrawidth - 1
    autoreg = AutoReg(x.ravel(), x.shape[1]).fit()
    extra = autoreg.predict(start, end).view()
    extra = extra.reshape(extrawidth, x.shape[1])
    x = np.concatenate((x, extra), axis=-2)
    return x

class T:

    def __init__(self, f: Callable, out_args, **in_args) -> None:

        if len(out_args) == 0:
            raise ValueError

        fparams = inspect.signature(f).parameters.keys()
        for fprop in in_args.keys():
            if fprop not in fparams:
                raise ValueError

        self._f = f
        self._out_args = out_args
        self._in_args = in_args

    def __call__(self, x: PowerSample) -> dict[str, Any]:

        data = {}

        for fprop, xprop in self._in_args.items():
            if hasattr(x, xprop):
                data.update({fprop: getattr(x, xprop)})
            else:
                raise ValueError

        result = self._f(**data)
        data = {}
        for xprop, v in zip(self._out_args, result):
            if hasattr(x, xprop):
                data.update({xprop: v})
            else:
                raise ValueError

        x = x.update(**data)

        return x