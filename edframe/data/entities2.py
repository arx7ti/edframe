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
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from datetime import datetime
from pickle import HIGHEST_PROTOCOL
from .decorators import feature, safe_mode
from ..features import fundamental, spectrum, thd, spectral_centroid, temporal_centroid, rms
from ..utils.exceptions import NotEnoughCycles, SingleCycleOnly, SamplingRateMismatch, MainsFrequencyMismatch
from ..signals import FITPS, downsample, upsample, roll, fryze, budeanu, extrapolate, pad
from ..utils.common import nested_dict


class Recording:

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
        if not hasattr(appliances, '__len__'):
            appliances = [appliances]

        if appliances is not None and locs is not None:
            assert len(appliances) == len(locs)

        self._data = data
        self._fs = fs
        self._appliances = appliances
        self._locs = locs
        self._require_components = True


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

    @staticmethod
    def __iaggrule__(i):
        return i.sum(0)

    @staticmethod
    def __vaggrule__(v):
        return v.mean(0)

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
    def vfold(self):
        return self.vc.reshape(self.n_components, self.n_cycles,
                               self.cycle_size)

    @property
    def ifold(self):
        return self.ic.reshape(self.n_components, self.n_cycles,
                               self.cycle_size)

    @property
    def labels(self):
        return self._appliances

    @property
    def n_channels(self):
        return self.data.shape[0]

    @property
    def n_components(self):
        return self.data.shape[1]

    @property
    def n_samples(self):
        return self.data.shape[2]

    @property
    def cycle_size(self):
        return math.ceil(self.fs / self.f0)

    @property
    def n_cycles(self):
        return self.n_samples // self.cycle_size

    @property
    def datafold(self):
        return self.data.reshape(self.n_channels, self.n_components,
                                 self.n_cycles, self.cycle_size)

    @property
    def components(self):
        return [
            self.new(self.vc[i], self.ic[i], self.fs, self.f0)
            for i in range(self.n_components)
        ]

    @property
    def locs(self):
        if self._locs is None and self._appliances is not None:
            return [[0, len(self)] * len(self._appliances)]
        elif self._locs is not None:
            return self._locs

        return None

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

    # @feature
    # def if0(self, mode='median'):
    #     return fundamental(self.i, self.fs, mode=mode)

    @feature
    def thd(self, **kwargs):
        return thd(self.i, self.fs, f0=self.f0(), **kwargs)

    @feature
    def power_factor(self, **kwargs):
        # TODO move to `..features`
        pf = np.cos(self.phase_shift()) / np.sqrt(1 + self.thd(**kwargs)**2)
        return pf

    @feature
    def spec(self, **kwargs):
        return spectrum(self.i, self.fs, f0=self.f0(), **kwargs)

    @feature
    def vspec(self, **kwargs):
        return spectrum(self.v, self.fs, **kwargs)

    @feature
    def spectral_centroid(self):
        return spectral_centroid(self.i)

    @feature
    def temporal_centroid(self):
        return temporal_centroid(self.i)

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
        if self.n_components == 1:
            return self.copy()

        v, i = self._data
        v = self.__vaggrule__(v)
        i = self.__iaggrule__(i)

        return self.new(v, i, self.fs, self.f0)

    def require_components(self, required=True):
        if required:
            vi = self.copy()
        else:
            vi = self.aggregate()

        vi._require_components = required

        return vi

    def __init__(
        self,
        v,
        i,
        fs,
        f0,
        appliances=None,
        locs=None,
    ) -> None:
        assert v.shape == i.shape

        T = math.ceil(fs / f0)
        with_components = len(v.shape) == 2

        if not with_components:
            v, i = v[None], i[None]

        # + VOLTAGE CHECK
        # Voltage should be synchronous and starting with positive semi-cycle
        if v.shape[1] // T == 0:
            raise ValueError

        if np.std(v[:, :T // 2]) <= 0:
            raise ValueError
        # - VOLTAGE CHECK

        data = np.stack((v, i))
        self._f0 = f0
        self._T = T

        super().__init__(data, fs, appliances=appliances, locs=locs)

    def __len__(self):
        return self.data.shape[-1]

    @property
    def dims(self):
        return self.n_samples // self.cycle_size, self.cycle_size

    @property
    def fulldims(self):
        return 2, self.n_components, *self.dims

    def __getitem__(self, indexer):
        if isinstance(indexer, tuple):
            assert len(indexer) == 2
            indexer, _ = indexer

            if not self.is_synced():
                raise ValueError

            if isinstance(_, slice):
                if not (_.start == _.stop == _.step == None):
                    raise ValueError
            elif _ != Ellipsis:
                raise ValueError

            data = self.data.reshape(2, *self._get_dims())
            keep_synced = True
        elif not isinstance(indexer, slice):
            raise ValueError
        else:
            data = self.data
            keep_synced = False

        if self.n_components > 1:
            data = data[:, :, indexer]
        else:
            data = data[:, indexer]

        if keep_synced:
            dims = data.shape[-2:]
            dims = (self.n_components, -1) if self.n_components > 1 else -1
            data = data.reshape(2, *dims)
        else:
            dims = None

        v, i = data

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        is_synced=keep_synced,
                        dims=dims)

    def __add__(self, vi):
        return self.add(vi)

    def __radd__(self, vi):
        return self.add(vi)

    def add(self, vi):
        # FIXME
        if isinstance(vi, int) and vi == 0:
            return self

        if self.fs != vi.fs:
            raise SamplingRateMismatch

        if self.f0 != vi.f0:
            raise MainsFrequencyMismatch

        data1, data2 = self.data, vi.data
        v, i = np.concatenate((data1, data2), axis=1)

        if not self._require_components or not vi._require_components:
            v = self.__vaggrule__(v)
            i = self.__iaggrule__(i)

        # TODO locs
        return self.new(v, i, self.fs, self.f0)

    def has_locs(self):
        return self.locs is not None

    def is_empty(self):
        return len(self) == 0

    # def sync(self):
    #     if self.n_components > 1:
    #         raise AttributeError

    #     if self.is_synced():
    #         return self.copy()

    #     fitps = FITPS()

    #     try:
    #         if self.has_locs():
    #             v, i, locs = fitps(self.v, self.i, fs=self.fs, locs=self.locs)
    #         else:
    #             v, i = fitps(self.v, self.i, fs=self.fs)
    #             locs = None

    #         dims = v.shape
    #         v, i = v.ravel(), i.ravel()
    #     except NotEnoughCycles:
    #         v, i = self.v, self.i
    #         dims = 1, len(v)

    #     return self.new(v,
    #                     i,
    #                     self.fs,
    #                     self.f0,
    #                     is_synced=True,
    #                     dims=dims,
    #                     locs=locs)

    def resample(self, fs, **kwargs):
        # TODO critical sampling rate condition
        Tn = math.ceil(fs / self.f0)
        n = self.n_cycles * Tn

        if fs > self.fs:
            v, i = upsample(self.data, n, **kwargs)
        elif fs < self.fs:
            v, i = downsample(self.data, n)
        else:
            v, i = self.data

        locs = None

        if self.has_locs():
            locs = np.asarray(self.locs)
            locs = np.ceil(locs * fs / self.fs).astype(int)
            locs = locs.tolist()

        return self.new(v, i, fs, self.f0, locs=locs)

    def cycle(self, mode='mean'):
        if not self.is_synced():
            self = self.sync()

        data = self.data.reshape(2, *self._get_dims())

        if mode == 'mean':
            data = np.mean(data, axis=-2)
        elif mode == 'median':
            data = np.median(data, axis=-2)
        else:
            raise ValueError

        v, i = data
        dims = 1, v.shape[-1]

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        is_synced=self.is_synced(),
                        dims=dims)

    def roll(self, n, outer=False):
        if n == 0:
            return self.new(self.__v,
                            self.__i,
                            self.fs,
                            self.f0,
                            is_synced=self.is_synced(),
                            dims=self._dims)

        n = len(self) if abs(n) > len(self) else n

        if outer and not self.is_synced():
            raise ValueError

        if outer:
            *_, p = self._get_dims()
            v, i = roll(self.data, abs(n) // p * p)
        else:
            v, i = self.data
            i = roll(i, n)

        mute = np.s_[n:] if n < 0 else np.s_[:n]
        mute = np.s_[:, mute] if self.n_components > 1 else np.s_[:n]
        i[mute] = 0

        locs = None

        if self.has_locs():
            locs = np.asarray(self.locs)
            locs = np.clip(locs + n, a_min=0, a_max=len(self))
            locs = locs.tolist()

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        is_synced=self.is_synced(),
                        dims=self._dims,
                        locs=locs)

    def extrapolate(self, n, **kwargs):
        if self.n_components > 1:
            raise NotImplementedError

        if self.is_synced():
            dims = self._get_dims()
            v, i = self.data.reshape(2, *dims)
            is_synced = n % dims[1] == 0
        else:
            v, i = self.data
            is_synced = False

        v, i = extrapolate(i, n, v=v, **kwargs)
        locs = None

        if self.has_locs():
            locs = np.asarray(self.locs)
            locs[:, 1] += n
            locs = locs.tolist()

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        is_synced=is_synced,
                        dims=self._dims,
                        locs=locs)

    def pad(self, n, **kwargs):
        if self.n_components > 1:
            raise NotImplementedError

        if isinstance(n, tuple):
            a, b = n
        else:
            a, b = n - n // 2, n

        if a != 0:
            a = a + self.cycle_size % a

        if b != 0:
            b = b + self.cycle_size % b

        V, I = [], []

        for v, i in zip(*self.data):
            v = extrapolate(v.ravel(), (a, b), fs=self.fs, f0=self.f0)
            i = pad(i.ravel(), (a, b), **kwargs)
            V.append(v), I.append(i)

        v, i = np.stack(V), np.stack(I)

        return self.new(v, i, self.fs, self.f0, locs=self.locs)

    def unitscale(self):
        v, i = self.data
        i = i / np.abs(self.i).max()

        return self.new(v,
                        i,
                        self.fs,
                        self.f0,
                        is_synced=self.is_synced(),
                        dims=self._dims)

    def steady_state(self):
        # vm = np.median(abs(self.v))
        # im = np.median(abs(self.i))
        raise NotImplementedError

    def fryze(self):
        i_a, i_r = fryze(*self.data)
        via = self.new(self.__v,
                       i_a,
                       self.fs,
                       self.f0,
                       is_synced=self.is_synced(),
                       dims=self._dims)
        vir = self.new(self.__v,
                       i_r,
                       self.fs,
                       self.f0,
                       is_synced=self.is_synced(),
                       dims=self._dims)

        return via, vir

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

        # TODO check dims are needed for all synced
        return cls(v, i, fs=fs, is_synced=False)

    def budeanu(self):
        v, i = self._data
        ia, iq, id = budeanu(v, i)

        a = self.new(v,
                     ia,
                     self.fs,
                     self.f0,
                     is_synced=self.is_synced(),
                     dims=self._get_dims())
        q = self.new(v,
                     iq,
                     self.fs,
                     self.f0,
                     is_synced=self.is_synced(),
                     dims=self._get_dims())
        d = self.new(v,
                     id,
                     self.fs,
                     self.f0,
                     is_synced=self.is_synced(),
                     dims=self._get_dims())

        return a, q, d

    def features(self, features=None, format='list', **kwargs):
        if features is None:
            features = [
                k for k, v in self.__class__.__dict__.items()
                if getattr(v, 'is_feature', False)
            ]
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
        return {'v': self.v, 'i': self.i, 'fs': self.fs, 'f0': self.f0}

    def todf(self):
        return pd.DataFrame(self.todict())

    def tolist(self):
        return [self.v.tolist(), self.i.tolist()]

    def toarray(self):
        return self.data


class P(L):

    def __init__(self, p, fs) -> None:
        super().__init__(p, fs)

    def resample(self, fs):
        raise NotImplementedError


class DataSet:
    pass


class VISet(DataSet, BackupMixin):

    @property
    def n_appliances(self):
        return len(self.appliances)

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
        data = []
        for vi in self.signatures:
            data.append(vi.data)

        data = np.asarray(data).transpose(1, 0, 2)

        return data

    @property
    def values(self):
        return self.s

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
        y=None,
        locs=None,
        **kwargs,
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
            vi = VI(v=vj, i=ij, fs=fs, y=yj, locs=locsj)
            data.append(vi)

        return cls(data, **kwargs)

    def __init__(
        self,
        signatures: list[VI],
        f0=None,
        adjust_len_by='median',
        safe_mode=True,
    ):
        ls = [len(vi) for vi in signatures]
        fs = [vi.fs for vi in signatures]
        # TODO if f0 is undefined
        # f0 = [vi.f0() for vi in signatures]

        if not all([fs[0] == f for f in fs[1:]]):
            raise ValueError

        # if not all([f0[0] == f for f in f0[1:]]):
        #     raise ValueError

        if adjust_len_by == 'median':
            n_samples = np.median(ls)
        elif adjust_len_by == 'mean':
            n_samples = np.mean(ls)
        else:
            raise ValueError

        self._hashes = set()
        new_signatures = []
        n_samples = int(math.ceil(n_samples))

        while len(signatures) > 0:
            vi = signatures.pop(0)
            dn = n_samples - len(vi)

            # TODO if not synced
            if dn > 0:
                vi = vi.extrapolate(dn)
            elif dn < 0:
                vi = vi[:n_samples]

            if safe_mode:
                self._hashes.add(vi.hash())

            new_signatures.append(vi)

        self.signatures = new_signatures
        self._safe_mode = safe_mode
        self._n_samples = n_samples
        self._fs = new_signatures[0].fs
        self._f0 = new_signatures[0].f0 if f0 is None else f0

    def __len__(self):
        return len(self.signatures)

    def __getitem__(self, indexer):
        just_signature = False

        if isinstance(indexer, slice):
            a = 0 if indexer.start is None else indexer.start
            b = len(self) if indexer.stop is None else indexer.stop
            indexer = list(range(a, b))
        elif isinstance(indexer, int):
            indexer = [indexer]
            just_signature = True

        indexer = np.asarray(indexer)
        assert len(indexer.shape) == 1

        if indexer.dtype == bool:
            assert len(indexer) == len(self)
            indexer = np.argwhere(indexer).ravel()

        signatures = [self.signatures[i] for i in indexer]

        if len(signatures) == 0:
            return None

        if just_signature:
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
