from __future__ import annotations

import os
import math
import pickle
import random
import numpy as np
import pandas as pd
import itertools as it
from copy import deepcopy
from numbers import Number
from inspect import isfunction
from scipy.signal import resample
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.tools.sm_exceptions import MissingDataError

from datetime import datetime
from pickle import HIGHEST_PROTOCOL
from .decorators import feature, safe_mode, check_empty
from ..features import fundamental, spectrum, thd, spectral_centroid, temporal_centroid, rms
from ..utils.exceptions import NotEnoughCycles, SingleCycleOnly, OrthogonalsMismatch, NSamplesMismatch, SamplingRateMismatch, MainsFrequencyMismatch
from ..signals import FITPS, downsample, upsample, fryze, budeanu, extrapolate, pad
from ..utils.common import nested_dict


class Recording:
    __recording_type__ = None

    @property
    def typeof(self):
        return self.__recording_type__

    @property
    def n_channels(self):
        return self.data.shape[0]

    @property
    # TODO rename to appliances
    def n_components(self):
        if self.is_empty():
            return 0

        return self.data.shape[1]

    @property
    def n_orthogonals(self):
        if self.is_empty():
            return 0

        return self.data.shape[2]

    @property
    def n_samples(self):
        return self.data.shape[3]

    @property
    def feature_names(self):
        features = sorted([
            k for k, v in self.__class__.__dict__.items()
            if getattr(v, 'is_feature', False)
        ])

        return list(features)

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    def fs(self):
        return self._fs

    @property
    def data(self):
        return self._data

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, data, fs, appliances=None, locs=None) -> None:
        if isinstance(appliances, str | int):
            appliances = [appliances]

        has_apps = appliances is not None and hasattr(appliances, '__len__')
        has_locs = locs is not None and hasattr(locs, '__len__')

        if has_locs:
            locs = np.asarray(locs)
            assert len(locs.shape) == 2

        if has_apps and has_locs:
            assert len(locs) == len(appliances)

        # User-defined data
        self._data = data
        self._fs = fs

        # TODO -1 for unknown appliances
        if appliances is not None:
            appliances = [-1 if a is None else a for a in appliances]

        self._appliances = appliances
        self._locs = locs

        # Default attributes
        self._require_components = True
        self._is_empty = False

    def is_empty(self):
        return self._is_empty

    def empty(self):
        raise NotImplementedError

    def require_components(self, required=True):
        if required:
            rec = self.copy()
        else:
            rec = self.aggregate()

        rec._require_components = required

        return rec

    def aggregate(self):
        raise NotImplementedError


class BackupMixin:

    def save(self, filepath=None, make_dirs=False):
        if filepath is None:
            today = datetime.now().date()
            filename = f'{self.__class__.__name__}-{today}-H{self.hash()}.pkl'
            filepath = os.path.join(os.getcwd(), filename)
        elif make_dirs:
            dirpath = os.path.dirname(filepath)
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, 'wb') as fb:
            pickle.dump(self, fb, protocol=HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as fb:
            instance = pickle.load(fb)

        return instance


class L(Recording):
    pass


class VI(Recording, BackupMixin):
    '''
    Assumes a whole number of cycles and synchronized voltage 
    '''
    __recording_type__ = 'high_sampling_rate'

    @property
    def values(self):
        '''
        Source signal used for features calculation
        '''

        return self.s

    @staticmethod
    def __iaggrule__(i, keepdims=False):
        return i.sum((0, 1), keepdims=keepdims)

    @staticmethod
    def __vaggrule__(v, keepdims=False):
        return v.mean((0, 1), keepdims=keepdims)

    @property
    def f0(self):
        return self._f0

    @property
    def v(self):
        return self.__vaggrule__(self.data[0])

    @property
    def i(self):
        return self.__iaggrule__(self.data[1])

    @property
    def s(self):
        return self.v * self.i

    @property
    def vc(self):
        return self.data[0]

    @property
    def ic(self):
        return self.data[1]

    @property
    def sc(self):
        return self.vc * self.ic

    @property
    def vfold(self):
        return self.v.reshape(self.n_cycles, self.cycle_size)

    @property
    def ifold(self):
        return self.i.reshape(self.n_cycles, self.cycle_size)

    @property
    def vcfold(self):
        return self.vc.reshape(self.n_components, self.n_cycles,
                               self.cycle_size)

    @property
    def icfold(self):
        return self.ic.reshape(self.n_components, self.n_cycles,
                               self.cycle_size)

    @property
    def sfold(self):
        return self.sc.reshape(self.n_components, self.n_cycles,
                               self.cycle_size)

    @property
    def appliances(self):
        if self.is_empty():
            return None

        if self.has_appliances():
            # TODO test for drop if locs dropped
            return self._appliances.copy()

        return [-1]

    @property
    def labels(self):
        return self.appliances

    @property
    def n_appliances(self):
        return self.n_components

    @property
    def n_types(self):
        return list(set(self.appliances))

    @property
    def cycle_size(self):
        return math.ceil(self.fs / self.f0)

    @property
    def n_cycles(self):
        return self.n_samples // self.cycle_size

    @property
    def dt(self):
        return self.n_samples / self.fs

    @property
    def dims(self):
        return self.n_channels, self.n_components, self.n_orthogonals, self.n_samples

    @property
    def datafold(self):
        return self.data.reshape(*self.dims[:-1], self.n_cycles,
                                 self.cycle_size)

    @property
    def components(self):
        if self.is_empty():
            return []

        components = []

        for k in range(self.n_components):
            if self.has_locs():
                locs = self.locs[k, None]
            else:
                locs = None

            vi = self.new(self.vc[k],
                          self.ic[k],
                          self.fs,
                          self.f0,
                          appliances=self.appliances[k],
                          locs=locs)
            components.append(vi)

        return components

    @property
    def orthogonality(self):
        return self._orthogonality

    @property
    def locs(self):
        if self.is_empty():
            return None

        if not self.has_locs():
            return np.asarray([[0, self.n_samples] * self.n_components])

        return self._locs.copy()

    def split_locs(self, mode='onchange'):
        if self.is_empty():
            return None

        if not self.has_locs():
            return self.locs

        locs = self.locs
        x = np.zeros(self.n_samples)
        ids = np.unique(np.sort(locs.ravel()))

        for a, b in locs:
            x[a:b] = 1

        locs = []

        if mode == 'onchange':

            for i in range(len(ids) - 1):
                a, b = ids[i], ids[i + 1]

                if (x[a:b] != 0).all():
                    locs.append([a, b])

            locs = np.asarray(locs)
        elif mode == 'running':
            for id in ids:
                if id == 0 or id == self.n_samples:
                    locs.append(id)
                elif x[id - 1] == 0 or x[id] == 0:
                    locs.append(id)

            locs = np.asarray(locs)
            locs = locs.reshape(-1, 2)
        else:
            raise ValueError

        return locs

    @feature
    def phase_shift(self):
        # TODO move to `..features`
        v, i = self.v, self.i
        zv = np.fft.rfft(v)
        zi = np.fft.rfft(i)

        x = zv * np.conj(zi)
        a, phi = np.abs(x), np.angle(x)
        dphi = phi[np.argmax(a)]

        return dphi

    @feature
    def thd(self, **kwargs):
        return thd(self.values, self.fs, f0=self.f0, **kwargs)

    @feature
    def power_factor(self, **kwargs):
        # TODO move to `..features`
        pf = np.cos(self.phase_shift()) / np.sqrt(1 + self.thd(**kwargs)**2)
        return pf

    @feature
    def spec(self, **kwargs):
        return spectrum(self.values, self.fs, f0=self.f0, **kwargs)

    @feature
    def vspec(self, **kwargs):
        return spectrum(self.v, self.fs, f0=self.f0, **kwargs)

    @feature
    def ispec(self, **kwargs):
        return spectrum(self.i, self.fs, f0=self.f0, **kwargs)

    @feature
    def spectral_centroid(self):
        return spectral_centroid(self.values)

    @feature
    def temporal_centroid(self):
        return temporal_centroid(self.values)

    def trajectory(self, n_bins=50):
        '''
        ref: github.com/sambaiga/MLCFCD
        '''
        j = 0
        V = self.v
        I = self.i
        V = V / abs(V).max()
        I = I / abs(I).max()
        x_bins = np.linspace(-1, 1, num=n_bins + 1)
        y_bins = np.linspace(-1, 1, num=n_bins + 1)
        T = np.zeros((n_bins, n_bins))

        for x1, x2 in zip(x_bins[:-1], x_bins[1:]):
            i = n_bins - 1

            for y1, y2 in zip(y_bins[:-1], y_bins[1:]):
                T[i, j] = sum((x1 <= self.i) & (self.i < x2) & (y1 <= self.v)
                              & (self.v < y2))
                i -= 1

            j += 1

        T = T / T.max()

        return T

    def copy(self):
        return deepcopy(self)

    def aggregate(self):
        return self.new(self.v, self.i, self.fs, self.f0)

    def __init__(
        self,
        v,
        i,
        fs,
        f0,
        appliances=None,
        locs=None,
        **kwargs,
    ) -> None:
        assert v.shape == i.shape

        T = math.ceil(fs / f0)
        with_components = len(v.shape) == 2
        with_orthogonals = len(v.shape) == 3

        if not with_orthogonals and with_components:
            v, i = v[None], i[None]
        elif not with_components and not with_orthogonals:
            v, i = v[None, None], i[None, None]

        self._check_voltage(v, T)

        data = np.stack((v, i))
        self._f0 = f0
        self._T = T
        # TODO which decomposition if passed with orthogonals
        self._orthogonality = kwargs.get('orthogonality', None)

        super().__init__(data, fs, appliances=appliances, locs=locs)

    @staticmethod
    def _check_voltage(v, T):
        # Voltage should be synchronous and starting with positive semi-cycle
        if v.shape[-1] // T == 0 and v.shape[-1] > 0:
            raise ValueError

        # TODO reverse case support
        if v.shape[-1] >= T // 2:
            vhalf = v[..., :T // 2]
            # ihalf = i[..., :T // 2]

            if np.mean(vhalf) < 0:
                raise ValueError

    def __len__(self):
        return self.n_samples

    def __getitem__(self, indexer):
        if not isinstance(indexer, slice):
            raise ValueError

        if indexer.step is not None:
            raise ValueError

        locs = None
        a0 = indexer.start if indexer.start is not None else 0
        a = a0 // self.cycle_size * self.cycle_size
        b0 = indexer.stop if indexer.stop is not None else self.n_samples
        b = math.ceil(b0 / self.cycle_size) * self.cycle_size

        if b0 <= a0:
            raise ValueError

        if a > self.n_samples or b > self.n_samples:
            raise ValueError

        if b - a < self.cycle_size:
            raise NotImplementedError

        data = self.data[..., a:b].copy()
        da, db = a0 - a, b - b0

        if da > 0:
            data[1, ..., :da] = 0

        if db > 0:
            data[1, ..., -db:] = 0

        assert not np.may_share_memory(data, self.data)

        v, i = data
        xa, xb = self.locs.T
        drop = np.argwhere((a0 >= xb) | (b0 <= xa)).ravel()

        if data.shape[1] != len(self.appliances) and v.shape[0] == 1:
            appliances, xa, xb = self._drop_components(drop, self.appliances,
                                                       xa, xb)
        else:
            v, i, appliances, xa, xb = self._drop_components(
                drop, v, i, self.appliances, xa, xb)

            if v is None:
                return self.empty()

        locs = np.stack((xa, xb)).T
        locs = np.clip(locs - a, a_min=da, a_max=data.shape[-1] - db)

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=appliances,
                        locs=locs)

    @staticmethod
    def _drop_components(ids, *data):
        data__ = []

        for data_ in data:
            if isinstance(data_, np.ndarray):
                data_ = np.delete(data_, ids, axis=0)
            elif isinstance(data_, list):
                data_ = data_.copy()

                for j in sorted(ids, reverse=True):
                    del data_[j]
            else:
                raise NotImplementedError

            data__.append(data_ if len(data_) > 0 else None)

        data__ = tuple(data__)

        if len(data__) > 1:
            return data__

        return data__[0]

    def drop(self, ids):
        if not hasattr(ids, '__len__'):
            ids = [ids]

        # FIXME if n_components == 1 but n_appliances > 1
        v, i, appliances = self._drop_components(ids, *self.data,
                                                 self.appliances)

        if v is None:
            return self.empty()

        if self.has_locs():
            locs = self._drop_components(ids, self.locs)
        else:
            locs = None

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=appliances,
                        locs=locs)

    def __add__(self, vi):
        return self.add(vi)

    def __radd__(self, vi):
        return self.add(vi)

    def add(self, vi):
        if self.is_empty() and vi.is_empty():
            return self.empty()

        # FIXME
        if isinstance(vi, int) and vi == 0:
            return self

        if self.n_orthogonals != vi.n_orthogonals:
            raise OrthogonalsMismatch

        if self.n_samples != vi.n_samples:
            raise NSamplesMismatch

        if self.fs != vi.fs:
            raise SamplingRateMismatch

        if self.f0 != vi.f0:
            raise MainsFrequencyMismatch

        data1 = np.zeros_like(vi.data) if self.is_empty() else self.data
        data2 = np.zeros_like(self.data) if vi.is_empty() else vi.data
        v, i = np.concatenate((data1, data2), axis=1)
        appliances = self.appliances + vi.appliances

        assert len(v) == len(i) == len(appliances)

        if self._require_components and vi._require_components:
            locs = np.concatenate((self.locs, vi.locs))
            assert len(appliances) == len(locs)
        else:
            v = self.__vaggrule__(v, keepdims=True)
            i = self.__iaggrule__(i, keepdims=True)
            locs = None

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=appliances,
                        locs=locs)

    def has_appliances(self):
        return self._appliances is not None

    def has_locs(self):
        return self._locs is not None

    def resample(self, fs, **kwargs):
        # TODO critical sampling rate condition
        if self.is_empty():
            return self.empty()

        if fs == self.fs:
            return self.copy()

        locs = None
        Tn = math.ceil(fs / self.fs * self.cycle_size)

        if fs > self.fs:
            v, i = upsample(self.datafold, Tn, **kwargs)
        else:
            v, i = downsample(self.datafold, Tn)

        dims = self.n_components, self.n_orthogonals, -1
        v, i = v.reshape(*dims), i.reshape(*dims)
        assert v.shape[-1] % Tn == 0

        if self.has_locs():
            k = v.shape[-1] / self.dims[-1]
            locs = k * self.locs
            locs[:, 0], locs[:, 1] = np.floor(locs[:, 0]), np.ceil(locs[:, 1])
            locs = locs.astype(int)
            locs = np.clip(locs, a_min=0, a_max=v.shape[-1])

        return self.new(v,
                        i,
                        fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs)

    def roll(self, n):
        '''
        cycle-wise roll 
        '''
        if not hasattr(n, '__len__'):
            n = [n] * self.n_appliances

        n = np.asarray(n)

        if len(n) != self.n_appliances:
            raise ValueError

        if self.is_empty() or (n > self.n_samples).all():
            return self.empty()
        elif (n == 0).all():
            return self.new(self.vc,
                            self.ic,
                            self.fs,
                            self.f0,
                            appliances=self.appliances,
                            locs=self.locs if self.has_locs() else None)

        data = np.empty((2, 0, self.n_orthogonals, self.n_samples))

        for k, nk in zip(range(self.n_appliances), n):
            n_cycles = nk // self.cycle_size * self.cycle_size
            vk, ik = np.roll(self.data[:, k], n_cycles, axis=-1)
            mute = np.s_[nk:] if nk < 0 else np.s_[:nk]
            mute = np.s_[..., mute]
            ik[mute] = 0
            data_k = np.stack((vk, ik), axis=0)[:, None]
            data = np.concatenate((data, data_k), axis=1)

        appliances = self.appliances
        locs = np.clip(self.locs + n[:, None], a_min=0, a_max=self.n_samples)
        drop = np.argwhere(locs[:, 0] > self.n_samples).ravel()
        # FIXME if n_components == 1 but n_appliances > 1
        v, i, appliances, locs = self._drop_components(drop, *data, appliances,
                                                       locs)

        if v is None:
            return self.empty()

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=appliances,
                        locs=locs)

    def _adjust_by_cycle_size(self, n):
        dn = self.cycle_size - n % self.cycle_size
        n += dn if n % self.cycle_size > 0 else 0

        return n

    def _adjust_delta(self, n):
        if isinstance(n, tuple):
            a, b = n
        else:
            a, b = n - n // 2, n // 2

        if a != 0:
            a = self._adjust_by_cycle_size(a)

        if b != 0:
            b = self._adjust_by_cycle_size(b)

        return a, b

    def extrapolate(self, n):
        if n < 0:
            raise ValueError

        V, I = [], []
        n = self._adjust_delta(n)

        if self.is_empty():
            return self.empty()

        locs = self.locs if self.has_locs() else None
        dims = self.n_components, self.n_orthogonals, -1

        for k, (vo, io) in enumerate(zip(*self.data)):
            for v, i in zip(vo, io):
                v = extrapolate(v, n, fs=self.fs, f0=self.f0)

                try:
                    i = extrapolate(i, n, fs=self.fs, f0=self.f0)

                except MissingDataError:
                    i = pad(i, n)

                    if self.has_locs():
                        locs[k] += n[0]
                else:
                    if self.has_locs():
                        locs[k] += n[0]
                        locs[k][1] += n[1]

                V.append(v), I.append(i)

        v, i = np.stack(V), np.stack(I)
        v, i = v.reshape(*dims), i.reshape(*dims)

        if self.has_locs():
            locs = np.clip(locs, a_min=0, a_max=v.shape[-1])

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs)

    def pad(self, n):
        locs = None
        V, I = [], []
        n = self._adjust_delta(n)

        if self.is_empty():
            return self.empty()

        dims = self.n_components, self.n_orthogonals, -1

        for vo, io in zip(*self.data):
            for v, i in zip(vo, io):
                v = extrapolate(v, n, fs=self.fs, f0=self.f0)
                i = pad(i, n)
                V.append(v), I.append(i)

        v, i = np.stack(V), np.stack(I)
        v, i = v.reshape(*dims), i.reshape(*dims)

        if self.has_locs():
            locs = self.locs + n[0]
            locs = np.clip(locs, a_min=0, a_max=v.shape[-1])

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs)

    def cycle(self, mode='mean'):
        if self.is_empty():
            return self.empty()

        if mode == 'mean':
            v, i = np.mean(self.datafold, axis=-2)
        elif mode == 'median':
            v, i = np.median(self.datafold, axis=-2)
        else:
            raise ValueError

        locs = None

        if self.has_locs():
            locs = np.zeros((self.n_components, 2), dtype=int)
            locs[:, 1] = v.shape[-1]

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs)

    def unitscale(self, cycle_wise=False):
        if self.is_empty():
            return self.empty()

        if cycle_wise:
            raise NotImplementedError
        else:
            v = self.vc / abs(self.v).max()
            i = self.ic / abs(self.i).max()

        locs = self.locs if self.has_locs() else None

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs)

    def fryze(self):
        orthogonality = 'Fryze'
        v, i = self.data

        if self.is_empty():
            return self.empty()

        if self.n_orthogonals > 1:
            v, i = v[:, 0, None], i.sum(1, keepdims=True)

        i_a, i_r = fryze(v, i)
        i = np.concatenate((i_a, i_r), axis=1)
        v = np.repeat(v, 2, axis=1)
        locs = self.locs if self.has_locs() else None

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs,
                        orthogonality=orthogonality)

    def budeanu(self):
        orthogonality = 'Budeanu'
        v, i = self.data

        if self.is_empty():
            return self.empty()

        if self.n_orthogonals > 1:
            v, i = v[:, 0, None], i.sum(1, keepdims=True)

        i_a, i_q, i_d = budeanu(v, i)
        i = np.concatenate((i_a, i_q, i_d), axis=1)
        v = np.repeat(v, 3, axis=1)

        locs = self.locs if self.has_locs() else None

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        appliances=self.appliances,
                        locs=locs,
                        orthogonality=orthogonality)

    def features(self, features=None, format='list', **kwargs):
        if features is None:
            features = self.feature_names
        else:
            ufeatures = set(features)

            if len(ufeatures) != features:
                raise ValueError

        data = []
        f_kwargs = nested_dict()
        f_reps = {}

        for k, v in kwargs.items():
            ks = k.split('__')
            if len(ks) > 1 & len(ks[0]) > 0:
                f_kwargs[ks[0]].update({''.join(ks[1:]): v})

        for feature in features:
            if isinstance(feature, str):
                try:
                    feature_fn = getattr(self, feature)
                except AttributeError:
                    raise ValueError

                if not getattr(feature_fn, 'is_feature', False):
                    raise ValueError
            elif hasattr(feature, '__class__'):
                feature_fn = feature
                feature = feature_fn.__class__.__name__
            elif isfunction(feature):
                feature_fn = feature
                feature = feature_fn.__name__
            else:
                raise ValueError

            # TODO named output variables
            values = feature_fn(**f_kwargs[feature])

            if not isinstance(values, tuple):
                values = (values, )

            k = 0
            just_scalar = True

            for v in values:
                if isinstance(v, Number):
                    k += 1
                    data.append(v)
                elif hasattr(v, '__len__'):
                    just_scalar = False
                    k += len(v)
                    data.extend(v)
                else:
                    raise ValueError

            k = 0 if k == 1 and len(values) == 1 and just_scalar else k
            f_reps.update({feature: k})

        keys = lambda: list(
            it.chain(*[[
                f'{k}_{i}' if n > 0 else f'{k}'
                for i in range(1, n + 1 if n > 0 else 2)
            ] for k, n in f_reps.items()]))

        if format == 'numpy':
            data = np.asarray(data)
        elif format == 'pandas':
            data = pd.DataFrame([data], columns=keys())
        elif format == 'dict':
            data = dict(zip(keys(), data))
        elif format != 'list':
            raise ValueError

        return data

    def hash(self):
        return hash(self.data.tobytes()) & 0xFFFFFFFFFFFFFFFF

    def todict(self):
        comps = [(*self.ic[i], ) for i in range(self.n_components)]

        return {
            'v': self.v,
            'i': self.i,
            'fs': self.fs,
            'f0': self.f0,
            'components': comps,
            'locs': self.locs,
        }

    @staticmethod
    def _sine_waveform(amp, f0, fs, phase=0, dt=None, n=None):
        dt = 1 / f0 if dt is None else dt
        n = math.ceil(fs * dt) if n is None else n
        x = np.linspace(0, dt, n)
        y = amp * np.sin(2 * np.pi * f0 * x + phase)

        return y

    @classmethod
    def active_load(cls, v_amp, i_amp, fs, f0, dt=None, n=None):
        v = cls._sine_waveform(v_amp, f0, fs, dt=dt, n=n)
        i = cls._sine_waveform(i_amp, f0, fs, dt=dt, n=n)

        # TODO check dims are needed for all synced
        return cls(v, i, fs=fs, is_synced=False)

    @classmethod
    def reactive_load(
        cls,
        v_amp,
        i_amp,
        fs,
        f0,
        power_factor=1.0,
        dt=None,
        n=None,
    ):
        assert power_factor >= 0 and power_factor <= 1

        phase = np.arccos(power_factor)
        v = cls._sine_waveform(v_amp, f0, fs, dt=dt, n=n)
        i = cls._sine_waveform(i_amp, f0, fs, phase=phase, dt=dt, n=n)

        return cls(v, i, fs=fs, is_synced=False)

    def empty(self):
        return VIEmpty(self.fs, self.f0)


class VIEmpty(VI):
    '''
    A lazy and empty signal
    '''

    def __init__(self, fs, f0) -> None:
        v, i = np.zeros((2, 0, 0, 0))
        super().__init__(v, i, fs, f0)
        self._is_empty = True


class P(L):
    __recording_type__ = 'low_sampling_rate'

    @property
    def values(self):
        '''
        Source signal used for features calculation
        '''

        return self.p

    @staticmethod
    def __paggrule__(p, keepdims=False):
        return p.sum((0, 1), keepdims=keepdims)

    @property
    def p(self):
        return self.data[0].ravel()

    @feature
    def energy_bars(self, scale='hour', fmt='kWh'):
        if not self.has_timeline():
            raise AttributeError

        if scale == 'hour':
            bars = np.zeros(24)
            times = self._timeline.hour
            time_min, time_max = 0, 23
        elif scale == 'day':
            bars = np.zeros(31)
            times = self._timeline.day
            time_min, time_max = 1, 31
        elif scale == 'weekday':
            bars = np.zeros(7)
            times = self._timeline.weekday
            time_min, time_max = 1, 7
        else:
            raise ValueError

        p = self.p

        for time in range(time_min, time_max + 1):
            mask = times == time

            if mask.any():
                pt = p[mask]
                bar = pt.sum()
                bars[time] = bar

        bars = bars * self.fs / 3600

        if fmt == 'kWh':
            bars = bars / 1000
        elif fmt != 'Wh':
            raise ValueError

        return bars

    @feature
    def interevent_time(self):
        raise NotImplementedError

    def has_timeline(self):
        return self._timeline is not None

    def __init__(
        self,
        p,
        fs,
        timeline=None,
        appliances=None,
        locs=None,
        **kwargs,
    ) -> None:
        data = p[None, None, None]
        self._timeline = timeline
        super().__init__(data, fs, appliances=appliances, locs=locs)

    def isnan(self):
        return np.isnan(self.p).any()

    def resample(self, fs, window_size=None):
        if self.isnan():
            raise AttributeError

        if window_size is None:
            window_size = fs if fs > self.fs else fs * self.fs

        data = self.data
        data_min = data.min(-1, keepdims=True)
        n_samples = math.ceil(self.fs / fs * self.n_samples)
        data = resample(data, n_samples, axis=-1, window=window_size)
        data = np.clip(data, a_min=data_min, a_max=None)

        p = data[0]

        # return self.new(p, self.fs, appliances=self.appliances, locs=self.locs)
        return self.new(p, self.fs)

    def pad(self, n):
        if isinstance(n, int):
            a, b = n // 2, n - n // 2
        elif isinstance(n, tuple):
            a, b = n
        else:
            raise ValueError

        paddings = [(0, 0), (0, 0), (0, 0), (a, b)]
        data = np.pad(self.data, paddings)

        p = data[0]

        # return self.new(p, self.fs, appliances=self.appliances, locs=self.locs)
        return self.new(p, self.fs)

    def fillna(self):
        raise NotImplementedError


class DataSet:
    pass


class VISet(DataSet, BackupMixin):

    @property
    def n_appliances(self):
        return len(self.appliance_types)

    @property
    def n_signatures(self):
        return len(self)

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def size(self):
        return self.n_signatures, self.n_samples

    @property
    def data(self):
        data = np.stack([vi.data for vi in self.signatures])

        return data

    @property
    def labels(self):
        labels = [vi.labels for vi in self.signatures]

        return labels

    @property
    def appliance_types(self):
        return list(set(list(it.chain(*self.labels))))

    @property
    def targets(self):
        mlb = MultiLabelBinarizer()

        return mlb.fit_transform(self.labels)

    @property
    def fs(self):
        return self._fs

    @property
    def f0(self):
        return self._f0

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_array(
        cls,
        v: np.ndarray,
        i: np.ndarray,
        fs,
        f0,
        y=None,
        locs=None,
        safe_mode=True,
    ):
        assert v.shape == i.shape
        assert len(v.shape) == len(i.shape) == 2

        if y is None:
            y = [None] * len(v)
        else:
            assert len(y) == len(v)
            assert len(y.shape) == 1 or (len(y.shape) == 2
                                         and np.product(y.shape) == len(v))

        if locs is None:
            locs = [None] * len(v)

        data = []

        for vj, ij, yj, locsj in zip(v, i, y, locs):
            vi = VI(vj, ij, fs, f0, y=yj, locs=locsj)
            data.append(vi)

        return cls(data, safe_mode=safe_mode)

    def __init__(
        self,
        signatures: list[VI],
        safe_mode=True,
    ):
        n_samples = [vi.n_samples for vi in signatures]
        fs = [vi.fs for vi in signatures]
        f0 = [vi.f0 for vi in signatures]

        if len(set(n_samples)) > 1:
            raise ValueError

        if len(set(fs)) > 1:
            raise ValueError

        if len(set(f0)) > 1:
            raise ValueError

        self._hashes = set()

        if safe_mode:
            for vi in signatures:
                self._hashes.add(vi.hash())

        self.signatures = signatures.copy()
        self._n_samples = n_samples[0]
        self._fs = fs[0]
        self._f0 = f0[0]
        self._safe_mode = safe_mode

    def __len__(self):
        return len(self.signatures)

    def __getitem__(self, indexer):
        item = False

        if isinstance(indexer, slice):
            a = 0 if indexer.start is None else indexer.start
            b = len(self) if indexer.stop is None else indexer.stop
            indexer = list(range(a, b))
        elif isinstance(indexer, int):
            indexer = [indexer]
            item = True
        elif not isinstance(indexer, list):
            raise ValueError

        indexer = np.asarray(indexer)
        assert len(indexer.shape) == 1

        if indexer.dtype == bool:
            assert len(indexer) == len(self)
            indexer = np.argwhere(indexer).ravel()

        # TODO check for data mutability
        signatures = [self.signatures[i] for i in indexer]

        if len(signatures) == 0:
            # TODO return empty set
            return None

        if item:
            return signatures[0]

        return self.new(signatures)

    def is_multilabel(self):
        return (self.targets.sum(1) > 1).any()

    def appliances(self, names, exact_match=False):
        if not hasattr(names, '__len__'):
            names = [names]

        mlb = MultiLabelBinarizer()
        mlb.fit(self.labels)
        query = mlb.transform([names])

        if exact_match:
            mask = (self.targets == query).all(1)
        else:
            mask = (self.targets * query).sum(1) == len(names)

        ids = np.argwhere(mask).ravel()

        if len(ids) == 0:
            return None

        signatures = [self.signatures[i] for i in ids]

        return self.new(signatures)

    def append(self, signature):
        n_samples = signature.n_samples
        fs = signature.fs
        f0 = signature.f0

        if n_samples != self.n_samples:
            raise ValueError

        if fs != self.fs:
            raise ValueError

        if f0 != self.f0:
            raise ValueError

        self.signatures.append(signature)

    def extend(self, signatures):
        n_samples = [vi.n_samples for vi in signatures]
        fs = [vi.fs for vi in signatures]
        f0 = [vi.f0 for vi in signatures]

        if len(set(n_samples)) > 1:
            raise ValueError

        if len(set(fs)) > 1:
            raise ValueError

        if len(set(f0)) > 1:
            raise ValueError

        n_samples = n_samples[0]
        fs = fs[0]
        f0 = f0[0]

        if n_samples != self.n_samples:
            raise ValueError

        if fs != self.fs:
            raise ValueError

        if f0 != self.f0:
            raise ValueError

        self.signatures.extend(signatures)

    @safe_mode
    def features(self, features=None, format='list', **kwargs):
        X = []

        for vi in self.signatures:
            x = vi.features(features=features, format=format, **kwargs)
            X.append(x)

        if format == 'numpy':
            X = np.stack(X)
        elif format == 'pandas':
            X = pd.concat(X, axis=0, ignore_index=True)
        elif format == 'dict':
            X_dict = defaultdict(list)

            while len(X) > 0:
                x = X.pop(0)

                for k, v in x.items():
                    X_dict[k].append(v)

            X_dict = dict(X_dict)
            X = X_dict
        elif format != 'list':
            raise ValueError

        return X

    def stats(self, features=None, decimals=4, **kwargs):
        X = self.features(features=features, format='pandas', **kwargs)

        return X.describe().iloc[1:].round(decimals)

    def split(self, test_size=0.3, by_samples=True, random_state=None):
        if self.targets is not None:
            stratify = self.targets.sum(1)
        else:
            stratify = None

        if by_samples:
            train_idxs, test_idxs = train_test_split(range(len(self)),
                                                     test_size=test_size,
                                                     stratify=stratify,
                                                     random_state=random_state)
            train_samples = [self.signatures[i] for i in train_idxs]
            test_samples = [self.signatures[i] for i in test_idxs]
            train_set = self.new(train_samples)
            test_set = self.new(test_samples)
        else:
            # TODO by appliance type
            pass

        return train_set, test_set

    def shuffle(self, random_state=None):
        rng = np.random.RandomState(random_state)
        ordr = rng.choice(len(self), len(self), replace=False)
        samples = [self.signatures[i] for i in ordr]

        return self.new(samples)

    def random(self, random_state=None):
        rng = np.random.RandomState(random_state)
        idx = rng.randint(len(self))

        return self.signatures[idx]

    def to(self):
        raise NotImplementedError

    def hash(self):
        return hash(self.data.tobytes()) & 0xFFFFFFFFFFFFFFFF
