from __future__ import annotations

import math
import random
import numpy as np
import pandas as pd
import itertools as it
from numbers import Number
from inspect import isfunction
from collections import defaultdict
from sklearn.model_selection import train_test_split

from .decorators import feature
from ..features import fundamental, spectrum, thd
from ..signals.exceptions import NotEnoughPeriods
from ..signals import FITPS, downsample, upsample, roll, fryze, extrapolate, pad
from ..utils.common import nested_dict


class Gen:

    @property
    def fs(self):
        return self._fs

    @property
    def data(self):
        return self._data

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, data, fs, y=None, locs=None) -> None:
        if isinstance(y, str):
            y = [y]

        if y is not None and locs is not None:
            assert len(y) == len(locs)

        self._data = data
        self._fs = fs
        self._y = y
        self._locs = locs
        self._require_components = True


class L(Gen):
    pass


class VI(Gen):

    @property
    def v(self):
        v = self.data[0]

        if self.n_components > 1:
            v = v.mean(0)

        return v

    @property
    def i(self):
        i = self.data[1]

        if self.n_components > 1:
            i = i.sum(0)

        return i

    @property
    def s(self):
        return self.v * self.i

    @property
    # TODO rename
    def __v(self):
        return self.data[0]

    @property
    # TODO rename
    def __i(self):
        return self.data[1]

    @property
    def labels(self):
        return self._y

    @property
    def n_components(self):
        if len(self.data.shape) == 3:
            return self.data.shape[1]

        return 1

    @property
    def components(self):
        if self.n_components > 1:
            return [self.data[:, i] for i in range(self.n_components)]

        return []

    @property
    def locs(self):
        if self._locs is None and self._y is not None:
            return [[0, len(self)] * len(self._y)]
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

    @feature
    def f0(self, mode='median'):
        return fundamental(self.v, self.fs, mode=mode)

    @feature
    def if0(self, mode='median'):
        return fundamental(self.i, self.fs, mode=mode)

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
    def trajectory(self):
        raise NotImplementedError

    @feature
    def spectral_centroid(self):
        raise NotImplementedError

    @feature
    def temporal_centroid(self):
        raise NotImplementedError

    def components_required(self, required=True):
        self._require_components = required

    def __init__(self, v, i, fs, y=None, locs=None, **kwargs) -> None:
        assert v.shape == i.shape

        data = np.stack((v, i))
        self._is_aligned = kwargs.get('is_aligned', False)
        self._dims = kwargs.get('dims', None)
        super().__init__(data, fs, y=y, locs=locs)

    def __len__(self):
        return self.data.shape[-1]

    def _get_dims(self):
        dims = (-1, *self._dims) if self.n_components > 1 else self._dims
        return dims

    def __getitem__(self, indexer):
        if isinstance(indexer, tuple):
            assert len(indexer) == 2
            indexer, _ = indexer

            if not self.is_aligned():
                raise ValueError

            if isinstance(_, slice):
                if not (_.start == _.stop == _.step == None):
                    raise ValueError
            elif _ != Ellipsis:
                raise ValueError

            data = self.data.reshape(2, *self._get_dims())
            keep_aligned = True
        elif not isinstance(indexer, slice):
            raise ValueError
        else:
            data = self.data
            keep_aligned = False

        if self.n_components > 1:
            data = data[:, :, indexer]
        else:
            data = data[:, indexer]

        if keep_aligned:
            dims = data.shape[-2:]
            dims = (self.n_components, -1) if self.n_components > 1 else -1
            data = data.reshape(2, *dims)
        else:
            dims = None

        v, i = data

        return self.new(v, i, self.fs, is_aligned=keep_aligned, dims=dims)

    def __add__(self, vi):
        return self.add(vi)

    def __radd__(self, vi):
        return self.add(vi)

    def add(self, vi):
        if not (self.is_aligned() and vi.is_aligned()):
            raise ValueError

        if self.fs != vi.fs:
            raise ValueError

        data1, data2 = self.data, vi.data

        if self.n_components == 1:
            data1 = data1[:, None]

        if vi.n_components == 1:
            data2 = data2[:, None]

        v, i = np.concatenate((data1, data2), axis=1)

        return self.new(v,
                        i,
                        self.fs,
                        is_aligned=self.is_aligned(),
                        dims=self._dims)

    def is_aligned(self):
        return self._is_aligned

    def has_locs(self):
        return self.locs is not None

    def is_empty(self):
        return len(self) == 0

    # TODO rename to `sync`
    def align(self):
        # NOTE multi-component instance will be transformed into single-component
        fitps = FITPS()

        # try
        if self.has_locs():
            v, i, locs = fitps(self.v, self.i, fs=self.fs, locs=self.locs)
        else:
            v, i = fitps(self.v, self.i, fs=self.fs)
            locs = None

        dims = v.shape
        v, i = v.ravel(), i.ravel()
        # except NotEnoughPeriods:
        # v, i = self.v, self.i
        # dims = 1, len(v)

        return self.new(v, i, self.fs, is_aligned=True, dims=dims, locs=locs)

    def resample(self, fs, **kwargs):
        if fs > self.fs:
            v, i = upsample(self.data, self.fs, fs, **kwargs)
        elif fs < self.fs:
            v, i = downsample(self.data, self.fs, fs)
        else:
            v, i = self.data

        locs = None

        if self.has_locs():
            locs = np.asarray(self.locs)
            locs = np.round(locs * fs / self.fs).astype(int)
            locs = locs.tolist()

        return self.new(v,
                        i,
                        fs,
                        is_aligned=self.is_aligned(),
                        dims=self._dims,
                        locs=locs)

    def cycle(self, mode='mean'):
        if not self.is_aligned():
            self = self.align()

        data = self.data.reshape(2, *self._get_dims())

        if mode == 'mean':
            data = np.mean(data, axis=-2)
        elif mode == 'median':
            data = np.median(data, axis=-2)
        else:
            raise ValueError

        v, i = data
        dims = 1, v.shape[-1]

        return self.new(v, i, self.fs, is_aligned=self.is_aligned(), dims=dims)

    def roll(self, n, outer=False):
        if n == 0:
            return self.new(self.__v,
                            self.__i,
                            self.fs,
                            is_aligned=self.is_aligned(),
                            dims=self._dims)

        n = len(self) if abs(n) > len(self) else n

        if outer and not self.is_aligned():
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
                        is_aligned=self.is_aligned(),
                        dims=self._dims,
                        locs=locs)

    def extrapolate(self, n, **kwargs):
        if self.n_components > 1:
            raise NotImplementedError

        if self.is_aligned():
            dims = self._get_dims()
            v, i = self.data.reshape(2, *dims)
            is_aligned = n % dims[1] == 0
        else:
            v, i = self.data
            is_aligned = False

        v, i = extrapolate(i, n, v=v, **kwargs)
        locs = None

        if self.has_locs():
            locs = np.asarray(self.locs)
            locs[:, 1] += n
            locs = locs.tolist()

        return self.new(v,
                        i,
                        self.fs,
                        is_aligned=is_aligned,
                        dims=self._dims,
                        locs=locs)

    def pad(self, n, **kwargs):
        if self.n_components > 1:
            raise NotImplementedError

        if self.is_aligned():
            dims = self._get_dims()
            v, i = self.data.reshape(2, *dims)
            is_aligned = n % dims[1] == 0
        else:
            v, i = self.data
            is_aligned = False

        _, v = extrapolate(v, n, v=v, **kwargs)
        i = pad(i.ravel(), n, **kwargs)

        return self.new(v,
                        i,
                        self.fs,
                        is_aligned=is_aligned,
                        dims=self._dims,
                        locs=self.locs)

    def unitscale(self):
        v, i = self.data
        i = i / np.abs(self.i).max()

        return self.new(v,
                        i,
                        self.fs,
                        is_aligned=self.is_aligned(),
                        dims=self._dims)

    def steady_state(self):
        # vm = np.median(abs(self.v))
        # im = np.median(abs(self.i))
        raise NotImplementedError

    def fryze(self):
        # TODO component-wise
        i_a, i_r = fryze(*self.data)
        via = self.new(self.__v,
                       i_a,
                       fs=self.fs,
                       is_aligned=self.is_aligned(),
                       dims=self._dims)
        vir = self.new(self.__v,
                       i_r,
                       fs=self.fs,
                       is_aligned=self.is_aligned(),
                       dims=self._dims)

        return via, vir

    def features(self, features, format='list', **kwargs):
        data = []
        f_kwargs = nested_dict()
        f_reps = {}

        for k, v in kwargs.items():
            ks = k.split('__')
            if len(ks) > 1 & len(ks[0]) > 0:
                f_kwargs[ks[0]].update({''.join(ks[1:]): v})

        # TODO validate for dublicated names
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

            value = feature_fn(**f_kwargs[feature])

            if isinstance(value, Number):
                data.append(value)
                f_reps.update({feature: 0})
            elif hasattr(value, '__len__'):
                data.extend(value)
                f_reps.update({feature: len(value)})
            else:
                raise ValueError

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
        raise NotImplementedError

    def todict(self):
        data = {'v': self.v, 'i': self.i}
        return data

    def todf(self):
        df = pd.DataFrame(self.todict())
        return df

    def tolist(self):
        return [self.v.tolist(), self.i.tolist()]

    def toarray(self):
        return self.data


class P(L):

    def __init__(self, p, fs) -> None:
        super().__init__(p, fs)

    def resample(self, fs):
        raise NotImplementedError


class VISet:

    @property
    def n_appliances(self):
        raise NotImplementedError

    @property
    def n_signatures(self):
        return len(self)

    @property
    def size(self):
        return self.n_signatures

    @property
    def data(self):
        data = []
        for vi in self._data:
            data.append(vi.data)

        data = np.asarray(data).transpose(1, 0, 2)
        return data

    @property
    def targets(self):
        return None
        # raise NotImplementedError

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

    def __init__(self, samples: list[VI], adjust_len_by='median'):
        ls = [len(vi) for vi in samples]
        fs = [vi.fs for vi in samples]
        # TODO if f0 is undefined
        # f0 = [vi.f0() for vi in samples]

        if not all([fs[0] == f for f in fs[1:]]):
            raise ValueError

        # if not all([f0[0] == f for f in f0[1:]]):
        #     raise ValueError

        if adjust_len_by == 'median':
            lsm = np.median(ls)
        elif adjust_len_by == 'mean':
            lsm = np.mean(ls)
        else:
            raise ValueError

        lsm = int(round(lsm))
        new_samples = []

        for vi in samples:
            dn = lsm - len(vi)

            # TODO if not aligned
            if dn > 0:
                vi = vi.extrapolate(dn)
            elif dn < 0:
                vi = vi[:lsm]

            new_samples.append(vi)

        self._data = new_samples
        self._fs = vi.fs

    def __len__(self):
        return len(self._data)

    def features(self, features, format='list', **kwargs):
        X = []

        for vi in self._data:
            x = vi.features(features, format=format, **kwargs)
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

    def stats(self):
        raise NotImplementedError

    def split(self, test_size=0.3, by_samples=True, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        if self.targets is not None:
            stratify = self.targets.sum(1)
        else:
            stratify = None

        train_idxs, test_idxs = train_test_split(range(len(self)),
                                                 test_size=test_size,
                                                 stratify=stratify)
        train_samples = [self._data[i] for i in train_idxs]
        test_samples = [self._data[i] for i in test_idxs]

        if by_samples:
            train_set = self.new(train_samples)
            test_set = self.new(test_samples)
        else:
            # TODO by appliance type
            pass

        return train_set, test_set

    def shuffle(self, random_state=None):
        raise NotImplementedError

    def to(self):
        raise NotImplementedError

    def hash(self):
        raise NotImplementedError
