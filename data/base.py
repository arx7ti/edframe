import math
import numpy as np
import types
import os
import torch
import pandas as pd

from scipy.signal import resample
import warnings


class Datman:

    @classmethod
    def read(cls, reader):
        sampling_range = reader.__sampling_range__
        data = list(reader)

        return cls(data, sampling_range)

    def __init__(self, data=None, sampling_range=None):
        self.__sampling_range__ = sampling_range
        self.data = data

        for name, method in self.__sampling_range__.__dict__.items():
            if callable(method):
                setattr(self, name, types.MethodType(method, self))

    def is_read(self):
        return self._data__ is not None


import numpy as np
import torch
import warnings


class HighFreqSample:

    def __init__(self, v, i, devices, fs, f0, locs=None):
        assert isinstance(v, (list, np.ndarray, torch.Tensor))
        assert isinstance(i, (list, np.ndarray, torch.Tensor))
        assert isinstance(devices, (list, tuple, np.ndarray, torch.Tensor))
        assert isinstance(fs, int)
        assert isinstance(f0, (int, float))
        assert isinstance(locs, (list, np.ndarray, torch.Tensor))

        assert v.dtype == i.dtype

        # Convert to NumPy if needed
        if isinstance(v, list):
            v = np.asarray(v)
        if isinstance(i, list):
            i = np.asarray(i)
        if isinstance(locs, list):
            locs = np.asarray(locs)

        if isinstance(v, torch.Tensor):
            warnings.warn("Input 'v' was a PyTorch tensor, cast to NumPy.",
                          UserWarning)
            v = v.detach().cpu().numpy()

        if isinstance(i, torch.Tensor):
            warnings.warn("Input 'i' was a PyTorch tensor, cast to NumPy.",
                          UserWarning)
            i = i.detach().cpu().numpy()

        if isinstance(locs, torch.Tensor):
            warnings.warn("Input 'locs' was a PyTorch tensor, cast to NumPy.",
                          UserWarning)
            locs = locs.detach().cpu().numpy()

        self._check_vi_data(v, i)

        self._v = v
        self._i = i
        self.devices = devices
        self.fs = fs
        self.f0 = f0
        self.locs = locs

        self.__v_modified__ = None
        self.__i_modified__ = None

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        self.__v_modified__ = v

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, i):
        self.__i_modified__ = i

    @property
    def voltage(self):
        return self.v

    @property
    def current(self):
        return self.i

    @property
    def apparent(self):
        return self.v[np.newaxis, :] * self.i

    @property
    def active(self):
        raise NotImplementedError

    @property
    def reactive(self):
        raise NotImplementedError

    @property
    def n_samples(self):
        if self.is_invariant():
            return np.prod(self.v.shape[-2:])

        return self.v.shape[-1]

    @property
    def n_cycles(self):
        if self.is_invariant():
            return self.v.shape[-2]

        raise AttributeError

    @property
    def n_components(self):
        return self.i.shape[0]

    def save(self):
        v = self.__v_modified__ if self.__v_modified__ is not None else self.v
        i = self.__i_modified__ if self.__i_modified__ is not None else self.i

        self._check_vi_data(v, i)
        self._v, self._i = v, i

        self.__v_modified__, self.__i_modified__ = None, None

    def _check_vi_data(self, v, i):
        cond1 = v.ndim == 1 and i.ndim == 1
        cond2 = v.ndim == 1 and i.ndim == 2
        cond3 = v.ndim == 2 and i.ndim == 2
        cond4 = v.ndim == 2 and i.ndim == 3
        cond5 = v.shape[-1] == i.shape[-1]

        if not any([cond1, cond2, cond3, cond4]) or not cond5:
            raise ValueError("Invalid voltage/current data shape.")

    def is_invariant(self):
        return len(self.i.shape) == 3

    def apparent_power(self, multicomponent=False):
        s = self.apparent
        axis = tuple(range(1, len(self.i.shape))) if multicomponent else None

        return np.sqrt(np.power(s, 2).mean(axis, keepdims=multicomponent))

    def active_power(self, multicomponent=False):
        axis = tuple(range(1, len(self.i.shape))) if multicomponent else None

        return self.apparent.mean(axis, keepdims=multicomponent)

    def reactive_power(self, multicomponent=True):
        axis = tuple(range(1, len(self.i.shape))) if multicomponent else None
        S = np.sqrt(
            np.power(self.apparent, 2).mean(axis, keepdims=multicomponent))
        P = self.apparent.mean(axis, keepdims=multicomponent)

        return np.sqrt(np.maximum(S**2 - P**2, 0))

    def resample(self, fs):
        k = int(round(fs / self.fs) * self.v.shape[-1])
        v = resample(self.v, k, axis=-1)
        i = resample(self.i, k, axis=-1)

        if self.locs is not None:
            locs = fs / self.fs * self.locs
            locs[:, 0], locs[:, 1] = np.floor(locs[:, 0]), np.ceil(locs[:, 1])
            locs = locs.astype(int)
            locs = np.clip(locs, a_min=0, a_max=np.prod(v.shape))

        return self.__class__(v, i, self.devices, fs, self.f0, locs=self.locs)


class HighFreqDataset:

    @staticmethod
    def assert_format(output):
        pass
