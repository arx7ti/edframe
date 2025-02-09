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
import fitps


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


class HighFreqDataset:

    @property
    def fs(self):
        return sorted(set(sample.fs for sample in self.data))

    @property
    def f0(self):
        return sorted(set(sample.f0 for sample in self.data))

    @property
    def devices(self):
        return sorted(set(sample.devices for sample in self.data))

    def is_homogeneous(self):
        return len(set([x.i.shape for x in self.data])) == 1

    def v(self):
        return self.data[0].v

    def i(self):
        if self.is_homogeneous():
            return np.concatenate([x.i for x in self.data])

        raise AttributeError

    def __len__(self):
        return len(self.data)

    def map(self, fn):
        data = list(map(fn, self.data))

        return HighFreqDataset(data)

    def _check_if_invariant(self):
        if not all([sample.is_invariant() for sample in self.data]):
            raise ValueError

    def split_by_cycles(self, n_cycles):
        self._check_if_invariant()

        data = []

        for sample in self.data:
            agg_fmt = True

            if isinstance(app, str):
                app = [app]
                agg_fmt = False

            for j in range(0, sample.n_cycles, n_cycles):
                vj = sample.v[..., j:j + n_cycles, :]
                ij = sample.i[..., j:j + n_cycles, :]

                if len(vj) == n_cycles:
                    if sample.locs is not None:
                        _locs = []
                        _devices = []

                        for device, (on, off) in zip(app, sample.locs):
                            on = max(on - j * n_cycles * sample.i.shape[-1], 0)
                            off = max(
                                min(
                                    off - on // sample.i.shape[-1] *
                                    sample.i.shape[-1],
                                    n_cycles * sample.i.shape[-1] - 1), 0)

                            if off == on:
                                continue

                            _devices.append(device)
                            _locs.append([on, off])

                        if not agg_fmt:
                            _devices = _devices[0]

                        data.append(
                            HighFreqSample(vj, ij, sample.fs, sample.f0,
                                           _devices, _locs))
                    else:
                        if not agg_fmt:
                            app = app[0]

                        data.append(
                            HighFreqSample(vj, ij, sample.fs, sample.f0, app))

        return HighFreqDataset(data)

    def split_by_ncomponents(self):
        data = []

        for sample in tqdm(self.data):
            Q = np.zeros((len(sample.devices), sample.i.shape[1]), dtype=int)

            for j, (a, b) in enumerate(sample.locs):
                a = a // sample.i.shape[1]
                b = b // sample.i.shape[1] + 1
                Q[j, a:b] += 1

            n_components = Q.sum(0)
            dn = np.diff(n_components, prepend=0, append=0)
            chpts = (dn != 0).nonzero()[0].ravel()

            a = chpts[0]

            for b in chpts[1:]:
                ids = Q[:, a:b].sum(1).nonzero()[0]
                assert len(ids) > 0

                _locs = []
                _devices = []

                for j in ids:
                    device = sample.devices[j]
                    on, off = sample.locs[j]

                    on = max(0, on - a * sample.v.shape[-1])
                    off = max(
                        0,
                        min((b - a) * sample.v.shape[-1] - 1,
                            off - a * sample.shape[-1]))
                    assert off >= on

                    if off == on:
                        continue

                    _devices.append(device)
                    _locs.append([on, off])

                data.append(
                    HighFreqSample(sample.v[a:b], sample.i[a:b], sample.fs,
                                   sample.f0, _devices, _locs))
                a = b

        return data

    def to_invariant(
        self,
        tol=20,
        buff_size=1.2,
        progress_bar=True,
    ):
        data = []

        for sample in tqdm(self.data, disable=not progress_bar):
            cycle_size = int(sample.fs / sample.f0)
            fitps = fitps.FITPS(cycle_size, int(buff_size * cycle_size), tol)
            v, i = fitps.transform(sample.v, sample.i)
            sample = HighFreqSample(v, i, sample.fs, sample.f0, sample.devices,
                                    sample.locs)
            data.append(sample)

        return HighFreqDataset(data)

    def submetered(self):
        data = list(filter(lambda sample: sample.n_components == 1, self.data))

        return HighFreqDataset(data)

    def aggregated(self):
        data = list(filter(lambda sample: sample.n_components > 1, self.data))

        return HighFreqDataset(data)

    def random(self, random_seed=None):
        random.seed(random_seed)
        idx = random.sample(range(len(self)))

        return self.data[idx]

    def similarity(self, dataset, metric='cosine'):
        I1, I2 = self.i, dataset.i

        if metric == 'cosine':
            I1n = I1 / np.linalg.norm(I1, axis=1, keepdims=True)
            I2n = I2 / np.linalg.norm(I2, axis=1, keepdims=True)
            scores = I1n @ I2n.T
        else:
            raise ValueError

        return scores


class HighFreqSample:

    def __init__(self, v, i, fs, f0, devices=None, locs=None):
        assert isinstance(v, (list, np.ndarray, torch.Tensor))
        assert isinstance(i, (list, np.ndarray, torch.Tensor))
        assert isinstance(fs, int)
        assert isinstance(f0, (int, float))

        if devices is not None:
            assert isinstance(devices, (list, tuple, np.ndarray, torch.Tensor))

        if locs is not None:
            assert isinstance(locs, (list, np.ndarray, torch.Tensor))

        assert v.dtype == i.dtype

        # Convert to NumPy if needed
        if isinstance(v, list):
            v = np.asarray(v)
        if isinstance(i, list):
            i = np.asarray(i)
        if not isinstance(devices, list):
            devices = list(devices)
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

        # Return a new instance with the superposed values
        return HighFreqSample(v,
                              i,
                              self.fs,
                              self.f0,
                              devices=devices,
                              locs=locs)

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

        return HighFreqSample(v, i, self.devices, fs, self.f0, locs=self.locs)
