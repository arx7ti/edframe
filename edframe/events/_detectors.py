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

    def _striding_window_view(self, x, fill_values=0):
        axes = x.shape[:-1]
        rem = x.shape[-1] % self._window_size

        if rem > 0:
            # Padding with zeros
            n_pad = self._window_size - rem
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
    Appliance event detection based on the log likelihood ratio test (Völker et al.)
    This class relies on the code implementation proposed by @voelkerb.
    '''
    __supported_recordings__ = ['low_sampling_rate']

    def __init__(
        self,
        window_size=20,
        thresh=5,
        patience=0,
        std_clip=0.01,
        linear_factor=0.005,
    ) -> None:
        self._window_size = window_size
        self._thresh = thresh
        self._patience = patience
        self._std_clip = std_clip
        self._linear_factor = linear_factor

    def _sliding_window_view(self, x):
        x = np.pad(x, (self._window_size - 1, 0))
        x = np.lib.stride_tricks.sliding_window_view(x, self._window_size)

        return x

    def _compute_likelihoods(self, x):
        mu_pre = self._sliding_window_view(x).mean(1)
        mu_post = self._sliding_window_view(x).mean(1)
        std_pre = self._sliding_window_view(x).std(1)
        std_post = self._sliding_window_view(x).std(1)

        std_pre = np.clip(std_pre, a_min=self._std_clip, a_max=None)
        std_post = np.clip(std_post, a_min=self._std_clip, a_max=None)

        likelihoods = np.zeros_like(x)

        for i in range(self._window_size, len(x) - self._window_size):
            j = i + self._window_size
            thresh = self._thresh + mu_pre[i] * self._linear_factor

            if abs(mu_post[j] - mu_pre[i]) > thresh:
                likelihood = np.log(std_pre[i] / std_post[j])
                likelihood += (x[i] - mu_pre[i])**2 / (2 * std_pre[i]**2)
                likelihood -= (x[i] - mu_post[j])**2 / (2 * std_post[j]**2)
                likelihoods[i] = likelihood

        return likelihoods

    def _get_change_points(self, likelihoods):
        x0 = []
        alikelihoods = abs(likelihoods)
        ids = (alikelihoods > 0).nonzero()[0]

        if len(ids) > 0:
            k = 0
            ids = np.split(ids, (np.diff(ids) != 1).nonzero()[0] + 1)

            for ids_group in ids:
                if ids_group[0] < k:
                    continue

                i = np.argmax(alikelihoods[ids_group]) + ids_group[0]
                k = i + self._patience
                x0.append(i)

        x0 = np.asarray(x0)

        return x0

    def transform(self, entity):
        x = entity.values
        l = self._compute_likelihoods(x)
        x0 = self._get_change_points(l)

        return x0
