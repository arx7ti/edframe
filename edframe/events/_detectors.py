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

    def _check_thresh_type(self, thresh_type):
        if thresh_type not in ['rel', 'abs']:
            raise ValueError

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

    def _get_change_points(self, scores):
        window_size = self._voting_window_size 
        patience = self._patience
        vote_thresh = self._vote_thresh
        score_thresh = self._score_thresh

        if self._score_thresh_type == 'rel':
            scores = scores / scores.max()

        k = 0
        x0 = []
        votes = np.zeros(len(scores) + window_size)

        for i in range(len(votes) - window_size):
            voting_window = scores[i:i + window_size]
            votes[i + np.argmax(voting_window)] += 1

        for i in range(len(votes)):
            if i - k < patience:
                continue

            if votes[i] > vote_thresh and scores[i] > score_thresh:
                k = i
                x0.append(i)

        x0 = np.asarray(x0)

        return x0


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
        power_thresh=5,
        likelihood='Volker',
        voting_window_size=20,
        patience=0,
        vote_thresh=10,
        likelihood_thresh=0.01,
        likelihood_thresh_type='rel',
        **kwargs,
    ) -> None:
        '''
        likelihood: 'Volker', 'Pereira'
        '''
        self._check_thresh_type(likelihood_thresh_type)

        self._window_size = window_size
        self._power_thresh = power_thresh
        self._linear_factor = kwargs.get('linear_factor', 0.005)
        self._likelihood = likelihood
        self._voting_window_size = voting_window_size
        self._patience = patience
        self._vote_thresh = vote_thresh
        self._score_thresh = likelihood_thresh
        self._score_thresh_type = likelihood_thresh_type
        self._var_clip = kwargs.get('var_clip', 1e-12)

    def _volker_likelihood(self, xj, mi, mj, vi, vj):
        thresh = self._power_thresh + mi * self._linear_factor

        if abs(mj - mi) > thresh:
            likelihood = 0.5 * np.log(vi / vj)
            likelihood += (xj - mi)**2 / (2 * vi)
            likelihood -= (xj - mj)**2 / (2 * vj)

            return likelihood

        return 0

    def _pereira_likelihood(self, xj, mi, mj, v):
        if abs(mj - mi) > self._power_thresh:
            likelihood = (mj - mi) / v * abs(xj - 0.5 * (mj + mi))

            return likelihood

        return 0

    def _compute_likelihoods(self, x):
        var_clip = self._var_clip
        window_size = self._window_size

        likelihoods = np.zeros_like(x)
        x = np.pad(x, (window_size - 1, window_size))

        for i in range(len(x) - 2 * window_size + 1):
            j = i + window_size
            dw = x[i:j + window_size]  # Detection window
            p = dw[:window_size]  # Pre-event window
            q = dw[window_size:]  # Post-event window

            assert len(p) == len(q) == window_size

            x0 = q[0]  # Change point
            mp = p.mean()
            mq = q.mean()

            assert len(p) == len(p)

            if self._likelihood == 'Volker':
                vp = np.clip(p.var(), a_min=var_clip, a_max=None)
                vq = np.clip(q.var(), a_min=var_clip, a_max=None)
                likelihood = self._volker_likelihood(x0, mp, mq, vp, vq)
            elif self._likelihood == 'Pereira':
                v = np.clip(dw.var(), a_min=var_clip, a_max=None)
                likelihood = self._pereira_likelihood(x0, mp, mq, v)
            else:
                raise ValueError

            likelihoods[i] = likelihood

        return likelihoods

    def transform(self, entity):
        x = entity.values
        x = self._compute_likelihoods(x)
        x0 = self._get_change_points(abs(x))

        return x0, x


class GOFDetector(WindowBasedDetector):
    '''
    Appliance event detector based on the "goodness of fit". 
    '''
    __supported_recordings__ = ['low_sampling_rate']

    def __init__(
        self,
        window_size=20,
        voting_window_size=20,
        patience=0,
        vote_thresh=10,
        gof_thresh=0.01,
        gof_thresh_type='rel',
        **kwargs,
    ) -> None:
        self._check_thresh_type(gof_thresh_type)

        self._window_size = window_size
        self._voting_window_size = voting_window_size
        self._patience = patience
        self._vote_thresh = vote_thresh
        self._score_thresh = gof_thresh
        self._score_thresh_type = gof_thresh_type
        self._power_clip = kwargs.get('power_clip', 1e-14)

    def _compute_gofs(self, x):
        power_clip = self._power_clip
        window_size = self._window_size

        gofs = np.zeros_like(x)
        x = np.pad(x, (window_size - 1, window_size))
        x = np.clip(x, a_min=power_clip, a_max=None)

        for i in range(len(x) - 2 * window_size + 1):
            j = i + window_size
            dw = x[i:j + window_size]  # Detection window
            p = dw[:window_size]  # Pre-event window
            q = dw[window_size:]  # Post-event window

            assert len(p) == len(q) == window_size

            gofs[i] = ((q - p)**2 / p).sum()

        return gofs

    def transform(self, entity):
        x = entity.values
        x = self._compute_gofs(x)
        x0 = self._get_change_points(x)

        return x0, x
