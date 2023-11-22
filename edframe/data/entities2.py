from __future__ import annotations

import math
import random
import numpy as np
import pandas as pd
import itertools as it
from numbers import Number
from inspect import isfunction

from ..signals.exceptions import NotEnoughPeriods
from ..signals import FITPS, downsample, upsample, roll
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

    def __init__(self, v, i, fs, y=None, locs=None, **kwargs) -> None:
        assert v.shape == i.shape

        data = np.stack((v, i))
        self._is_aligned = kwargs.get('is_aligned', False)
        self._dims = kwargs.get('dims', None)
        super().__init__(data, fs, y=y, locs=locs)

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, indexer):
        # FIXME if n_components > 1
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

            dims = (-1, *self._dims) if self.n_components > 1 else self._dims
            data = self.data.reshape(2, *dims)
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

    def is_empty(self):
        return len(self) == 0

    def align(self):
        # FIXME if n_components > 1
        fitps = FITPS()

        try:
            v, i = fitps(self.v, self.i, fs=self.fs)
            dims = v.shape
            v, i = v.ravel(), i.ravel()
        except NotEnoughPeriods:
            v, i = self.v, self.i
            dims = 1, len(v)

        return self.new(v, i, self.fs, is_aligned=True, dims=dims)

    def resample(self, fs, **kwargs):
        # FIXME if n_components > 1
        if fs > self.fs:
            v = upsample(self.v, self.fs, fs, **kwargs)
            i = upsample(self.i, self.fs, fs, **kwargs)
        elif fs < self.fs:
            v = downsample(self.v, self.fs, fs)
            i = downsample(self.i, self.fs, fs)
        else:
            v = self.v
            i = self.i

        return self.new(v,
                        i,
                        fs,
                        is_aligned=self.is_aligned(),
                        dims=self._dims)

    def cycle(self, mode='mean'):
        # FIXME if n_components > 1
        if not self.is_aligned():
            self = self.align()

        data = self.data.reshape(2, *self._dims)

        if mode == 'mean':
            data = np.mean(data, axis=1)
        elif mode == 'median':
            data = np.median(data, axis=1)
        else:
            raise ValueError

        v, i = data
        dims = 1, len(v)

        return self.new(v, i, self.fs, is_aligned=self.is_aligned(), dims=dims)

    def roll(self, n, outer=False):
        # FIXME if n_components > 1
        if n == 0:
            return self.new(self.v, self.i, self.fs)

        n = len(self) if abs(n) > len(self) else n

        if outer and not self.is_aligned():
            self = self.align()

        if outer:
            _, p = self._dims
            v, i = roll(self.data, abs(n) // p * p)
        else:
            i = roll(self.i, n)
            v = self.v

        if n < 0:
            i[n:] = 0
        else:
            i[:n] = 0

        return self.new(v,
                        i,
                        self.fs,
                        is_aligned=self.is_aligned(),
                        dims=self._dims)

    def unitscale(self):
        # FIXME if n_components > 1
        v = self.v / np.abs(self.v).max()
        i = self.i / np.abs(self.i).max()
        return self.new(v,
                        i,
                        self.fs,
                        is_aligned=self.is_aligned(),
                        dims=self._dims)

    def features(self, features, format='list', **kwargs):
        # FIXME if n_components > 1
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
                # TODO validate for existance
                feature_fn = getattr(self, feature)
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
            elif hasattr(value, '__len__'):
                data.extend(value)
                f_reps.update({feature: len(value)})
            else:
                raise ValueError

        keys = lambda: list(
            it.chain(*[[f'{k}_{i}' for i in range(1, n + 1)]
                       for k, n in f_reps.items()]))

        if format == 'numpy':
            data = np.asarray(data)
        elif format == 'pandas':
            data = pd.DataFrame([data], columns=keys())
        elif format == 'dict':
            data = dict(zip(keys(), data))
        elif format != 'list':
            raise ValueError

        return data

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
