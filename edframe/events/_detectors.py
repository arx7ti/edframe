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