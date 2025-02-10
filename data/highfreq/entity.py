import math
import numpy as np
import types
import os
import torch
import pandas as pd
import random

from scipy.signal import resample
from tqdm import tqdm
import warnings


class HighFreqSample:

    def __init__(self, v, i, fs, f0, devices=None, locs=None, brands=None):
        assert isinstance(v, (list, np.ndarray, torch.Tensor))
        assert isinstance(i, (list, np.ndarray, torch.Tensor))
        assert isinstance(fs, int)
        assert isinstance(f0, (int, float))

        if devices is not None:
            assert isinstance(devices, (list, tuple, np.ndarray, torch.Tensor))

        if locs is not None:
            assert isinstance(locs, (list, np.ndarray, torch.Tensor))

        if brands is not None:
            assert isinstance(brands, (list, tuple, np.ndarray, torch.Tensor))

        assert v.dtype == i.dtype

        # Convert to NumPy if needed
        if isinstance(v, list):
            v = np.asarray(v)
        if isinstance(i, list):
            i = np.asarray(i)
        if not isinstance(devices, list) and devices is not None:
            devices = list(devices)
        if isinstance(locs, list) and locs is not None:
            locs = np.asarray(locs)
        if not isinstance(brands, list) and brands is not None:
            brands = list(brands)

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

        if (v.ndim == 1 and i.ndim == 1) or (v.ndim == 2 and i.ndim == 2):
            i = i[None]

        self._v = v
        self._i = i
        self.devices = devices
        self.fs = fs
        self.f0 = f0
        self.locs = locs
        self.brands = brands

        self.__v_modified__ = None
        self.__i_modified__ = None

    def __superpose__(self, sample):
        """
        Superposes another HighFreqSample instance onto the current one.

        Conditions:
        - `fs` (sampling frequency) must be the same.
        - `f0` (nominal frequency) must be the same.
        - `v` and `i` must have compatible shapes.

        Returns:
        - A new `HighFreqSample` instance with superposed voltage and current signals.
        """
        if self.is_unsaved():
            warnings.warn(
                "Instance has unsaved changes. Use method .save() first to apply changes.",
                UserWarning)

        if not isinstance(sample, HighFreqSample):
            raise TypeError(
                "Superposition requires another HighFreqSample instance.")

        if self.fs != sample.fs:
            raise ValueError(
                f"Cannot superpose: Sampling frequencies (fs) must match ({self.fs} != {sample.fs})."
            )

        if self.f0 != sample.f0:
            raise ValueError(
                f"Cannot superpose: Nominal frequencies (f0) must match ({self.f0} != {sample.f0})."
            )

        if self.v.shape != sample.v.shape or self.i.shape != sample.i.shape:
            raise ValueError(
                "Cannot superpose: Voltage and current signal shapes must be the same."
            )

        # Perform superposition
        v = self.v
        i = self.i + sample.i

        # Merge devices if both instances have devices
        if self.devices and sample.devices:
            devices = self.devices + sample.devices
        elif self.devices:
            devices = self.devices
        elif sample.devices:
            devices = sample.devices
        else:
            devices = None

        # Merge locations if both instances have locs
        if self.locs is not None and sample.locs is not None:
            locs = np.concatenate((self.locs, sample.locs), axis=0)
        elif self.locs is not None:
            locs = self.locs
        elif sample.locs is not None:
            locs = sample.locs
        else:
            locs = None

        # Merge brands if both instances have brands
        if self.brands and sample.brands:
            brands = self.brands + sample.brands
        elif self.brands:
            brands = self.brands
        elif sample.brands:
            brands = sample.brands
        else:
            brands = None

        # Return a new instance with the superposed values
        return HighFreqSample(v,
                              i,
                              self.fs,
                              self.f0,
                              devices=devices,
                              locs=locs,
                              brands=brands)

    def __add__(self, sample):
        return self.__superpose__(sample)

    def __radd__(self, sample):
        return self.__superpose__(sample)

    def is_unsaved(self):
        cond = self.__v_modified__ is not None
        cond |= self.__i_modified__ is not None

        return cond

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

    @property
    def n_devices(self):
        return len(self.devices)

    @property
    def n_types(self):
        return len(set(self.devices))

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

        return HighFreqSample(v, i, fs, self.f0, self.devices, locs=self.locs)

    def is_submetered(self):
        return self.n_components == 1

    def is_aggregated(self):
        return self.n_components > 1

    def copy(self):
        """
        Creates a copy of the instance without copying temporary modifications
        (i.e., v_modified and i_modified remain None).
        
        Returns:
            HighFreqSample: A new instance with the same data but without modifications.
        """

        return HighFreqSample(
            v=self.v.copy(),
            i=self.i.copy(),
            fs=self.fs,
            f0=self.f0,
            devices=self.devices.copy() if self.devices else None,
            locs=self.locs.copy() if self.locs is not None else None)

    def is_transient(self, thresh=0.1):
        if self.locs is not None:
            if self.locs.min() > 0:
                return True

        _E0 = 1e-9

        u, l = self.i.sum(0).max(1), self.i.sum(0).min(1)
        u = u / (u.max() + _E0)
        l = l / (l.min() + _E0)
        scores = [abs(u.max() - u.min())]
        scores += [abs(l.max() - l.min())]
        scores += [abs(u.max() - l.min())]
        scores += [abs(l.max() - u.min())]
        score = max(scores)
        is_transient = score > thresh

        return is_transient

    def compute_locs(self, I_on=0.05, I_min=0.1):
        # TODO to check
        if not self.is_invariant():
            raise AttributeError

        i = self.i.sum(0)
        I = abs(i).max(1)

        s = (I > I_min).astype(int)
        ds = np.diff(s, prepend=s[0])
        on = (ds > 0).nonzero()[0]
        off = (ds < 0).nonzero()[0]

        if on.size == 0:
            return None

        if off.size == 0:
            off = np.append(off, len(s))

        if off[0] < on[0]:
            on = np.insert(on, 0, 0)

        if off[-1] < on[-1]:
            off = np.append(off, len(s))

        assert len(on) == len(off)

        locs = np.stack((on, off)).T

        assert (locs[:, 1] > locs[:, 0]).all()

        for k, (a, b) in enumerate(locs):
            ia = i[a]
            on = (abs(np.diff(ia)) > I_on).nonzero()[0]

            if on.size > 0:
                on = on[0]
            else:
                on = 0

            if on > 0 and abs(ia[:on]).mean() > I_min:
                on = 0

            ib = i[b - 1]
            off = (abs(np.diff(ib[::-1])) > I_on).nonzero()[0]

            if off.size > 0:
                off = off[0]
            else:
                off = 0

            if off > 0 and abs(ib[:off]).mean() > I_min:
                off = 0

            off = len(i[a:b].ravel()) - off - 1

            locs[k] = [a * i.shape[1] + on, a * i.shape[1] + off]

        return locs
