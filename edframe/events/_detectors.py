from __future__ import annotations

import os
import math
import pickle
import random
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

from collections import defaultdict
from datetime import datetime
from pickle import HIGHEST_PROTOCOL


class EventDetector:
    __supported_recordings__ = []

    def __call__(self, *args, **kwargs):
        return self.oncall(*args, **kwargs)

    def oncall(self, *args, **kwargs):
        raise NotImplementedError


class WindowBasedDetector(EventDetector):

    def __init__(self, window, window_size, **kwargs) -> None:
        self._window = window
        self._window_size = window_size
        self._fill_nan = kwargs.get('fill_nan', None)

    def __call__(self, entity):
        return self.oncall(entity)

    def _sliding_window_view(self, x):
        x = np.pad(x, (self._window_size - 1, 0))
        x = np.lib.stride_tricks.sliding_window_view(x, self._window_size)

        return x

    def _striding_window_view(self, x, fill_values=0):
        axes = x.shape[:-1]
        rem = x.shape[-1] % self._window_size

        if rem > 0:
            # Padding with zeros
            n_pad = self._window_size - rem
            # TODO rewrite
            x_pad = fill_values * np.ones((*axes, n_pad), dtype=x.dtype)
            x = np.concatenate((x, x_pad), axis=-1)

        assert x.shape[-1] % self._window_size == 0
        n_windows = x.shape[-1] // self._window_size
        x = x.reshape(*axes, n_windows, self._window_size)

        return x

    def _apply_window(self, x):
        X = self._striding_window_view(x)
        x = np.apply_along_axis(self._window, -1, X)
        assert len(x.shape) == 1

        if self._fill_nan is not None:
            x = np.nan_to_num(x, nan=self._fill_nan)

        return x


class ThresholdBasedDetector(WindowBasedDetector):
    __supported_recordings__ = ['low_sampling_rate', 'high_sampling_rate']

    def __init__(
        self,
        window,
        window_size,
        thresh,
        greater=True,
        **kwargs,
    ) -> None:
        super().__init__(window, window_size, **kwargs)
        self._thresh = thresh
        self._greater = greater

    def oncall(self, entity):
        x = self._apply_window(entity.values)

        cond = x > self._thresh if self._greater else x < self._thresh
        x = np.where(cond, 1, 0)
        dx = np.diff(x, prepend=0, append=0)  # Prepend 0 due to assumption

        locs = np.argwhere(dx != 0)
        locs = locs.reshape(-1, 2)
        locs *= self._window_size
        locs = np.clip(locs, a_min=0, a_max=entity.n_samples)

        return locs


class DerivativeBasedDetector(WindowBasedDetector):
    __supported_recordings__ = ['high_sampling_rate', 'low_sampling_rate']

    def __init__(
        self,
        window,
        window_size,
        thresh,
        rel=False,
        **kwargs,
    ) -> None:
        super().__init__(window, window_size, **kwargs)
        self._thresh = thresh
        self._rel = rel

    def oncall(self, entity):
        x = self._apply_window(entity.values)
        dx = np.diff(x, prepend=0, append=0)

        if self._rel:
            dx[1:-1] /= x[:-1]
            dx = np.nan_to_num(dx, nan=0)

        cond = abs(dx) > self._thresh
        ids = np.argwhere(cond).ravel()
        locs = []

        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]

            if (x[a:b] != self._fill_nan).all():
                locs.append([a, b])

        locs = np.asarray(locs)
        locs *= self._window_size
        locs = np.clip(locs, a_min=0, a_max=entity.n_samples)

        return locs


class LLRDetector(WindowBasedDetector):
    '''
    Appliance event detector based on the log likelihood estimation. 
    '''
    __supported_recordings__ = ['low_sampling_rate']

    def __init__(
        self,
        window_size=20,
        thresh=5,
        patience=0,
        var_clip=1e-4,
        linear_factor=0.005,
        likelihood='Volker',
        precision=20,
    ) -> None:
        '''
        likelihood: 'Volker', 'Pereira'
        '''
        self._window_size = window_size
        self._thresh = thresh
        self._patience = patience
        self._var_clip = var_clip
        self._linear_factor = linear_factor
        self._likelihood = likelihood
        self._precision = precision

    def _volker_likelihood(self, xj, mi, mj, vi, vj):
        thresh = self._thresh + mi * self._linear_factor

        if abs(mj - mi) > thresh:
            likelihood = 0.5 * np.log(vi / vj)
            likelihood += (xj - mi)**2 / (2 * vi)
            likelihood -= (xj - mj)**2 / (2 * vj)

            return likelihood

        return 0

    def _pereira_likelihood(self, xj, mi, mj, v):
        if abs(mj - mi) > self._thresh:
            likelihood = (mj - mi) / v * abs(xj - 0.5 * (mj + mi))

            return likelihood

        return 0

    def _compute_likelihoods(self, x):
        likelihoods = np.zeros_like(x)
        x = np.pad(x, (self._window_size - 1, 0))

        for i in range(self._window_size, len(x) - 2 * self._window_size + 1):
            # Detection window
            j = i + self._window_size
            xij = x[i:j + self._window_size]

            # Pre-event window
            xi = xij[:self._window_size]
            mi = xi.mean()

            # Post-event window
            xj = xij[self._window_size:]
            mj = xj.mean()

            assert len(xi) == len(xj)

            # Change point
            x0 = xj[0]

            if self._likelihood == 'Volker':
                vi = np.clip(xi.std(), a_min=self._var_clip, a_max=None)
                vj = np.clip(xj.std(), a_min=self._var_clip, a_max=None)
                likelihood = self._volker_likelihood(x0, mi, mj, vi, vj)
            elif self._likelihood == 'Pereira':
                v = xij.var()
                likelihood = self._pereira_likelihood(x0, mi, mj, v)
            else:
                raise ValueError

            likelihoods[i] = likelihood

        return likelihoods

    def _get_change_points(self, likelihoods):
        k = 0
        x0 = []
        ws = 2 * self._precision + 1

        for i in range(len(likelihoods) - ws):
            # Mid point of activation window
            m = i + ws // 2

            if m - k < self._patience:
                continue

            lm = likelihoods[m]

            # Activation window
            vw = likelihoods[i:i + ws]
            vw = np.delete(vw, len(vw) // 2)

            if lm > vw.max() or lm < vw.min():
                x0.append(m)
                k = m

        x0 = np.asarray(x0)

        return x0

    def transform(self, entity):
        x = entity.values
        x = self._compute_likelihoods(x)
        x0 = self._get_change_points(x)

        return x0
