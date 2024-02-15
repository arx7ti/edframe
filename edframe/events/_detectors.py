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


class WindowBasedDetector(EventDetector):

    def __init__(self, window, window_size) -> None:
        self._window = window
        self._window_size = window_size

    def _striding_window_view(self, x, fill_values=0):
        axes = x.shape[:-1]
        n_pad = self._window_size - x.shape[-1] % self._window_size

        if n_pad > 0:
            # Padding with zeros
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

        return x


class ThresholdBasedDetector(WindowBasedDetector):
    __supported_recordings__ = ['low_sampling_rate', 'high_sampling_rate']

    def __init__(self, window, window_size, thresh, greater=True) -> None:
        super().__init__(window, window_size)
        self._thresh = thresh
        self._greater = greater

    def __call__(self, ):
        return self.detect()

    def callback(self, entity):
        x = self._apply_window(entity.values)

        cond = x > self._thresh if self._greater else x < self._thresh
        x = np.where(cond, 1, 0)
        dx = np.diff(x, prepend=0)  # Prepend 0 due to assumption
        locs = np.argwhere(dx != 0)

        if len(locs) % 2 != 0:
            locs = np.append(locs, [[len(x)]], axis=0)

        locs = locs.reshape(-1, 2)
        locs *= self._window_size
        locs = np.clip(locs, a_min=0, a_max=entity.n_samples)

        return locs
