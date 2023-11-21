from __future__ import annotations

import math
import random
import numpy as np
import pandas as pd
import itertools as it

from ..signals import FITPS, downsample, upsample
from ..signals.exceptions import NotEnoughPeriods


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

    def __init__(self, data, fs) -> None:
        self._data = data
        self._fs = fs


class H(Gen):

    @property
    def v(self):
        return self._data[0]

    @property
    def i(self):
        return self._data[1]


class L(Gen):
    pass


class VI(H):

    def __init__(self, v, i, fs, **kwargs) -> None:
        data = np.stack((v, i))
        self._is_aligned = kwargs.get('is_aligned', False)
        self._dims = kwargs.get('dims', None)
        super().__init__(data, fs)

    def is_aligned(self):
        return self._is_aligned

    def align(self, **kwargs):
        fitps = FITPS(**kwargs)

        try:
            v, i = fitps(self.v, self.i, fs=self.fs)
            dims = v.shape
            v, i = v.ravel(), i.ravel()
        except NotEnoughPeriods:
            v, i = self.v, self.i
            dims = 1, len(v)

        return self.new(v, i, self.fs, is_aligned=True, dims=dims)

    def resample(self, fs, **kwargs):
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

    def cycle(self, mode='mean', **kwargs):
        if not self.is_aligned():
            self = self.align(**kwargs)

        dims = self._dims
        data = self.data.reshape(2, *dims)

        if mode == 'mean':
            data = np.mean(data, axis=1)
        elif mode == 'median':
            data = np.median(data, axis=1)
        else:
            raise ValueError

        v, i = data
        dims = 1, len(v)

        return self.new(v, i, self.fs, is_aligned=self.is_aligned(), dims=dims)


class P(L):

    def __init__(self, p, fs) -> None:
        super().__init__(p, fs)

    def resample(self, fs):
        raise NotImplementedError
