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


class DualWindowDetector(WindowBasedDetector):
    '''
    Appliance event detector based on the pre- and post-event windows. 
    '''
    __supported_recordings__ = ['low_sampling_rate']

    def __init__(
        self,
        window_size=20,
        power_thresh=5,
        scoring='Volker_likelihood',
        voting_window_size=20,
        patience=0,
        vote_thresh=10,
        score_thresh=0.01,
        score_thresh_type='rel',
        **kwargs,
    ) -> None:
        '''
        scoring: 'Volker_likelihood', 'Pereira_likelihood', 'Jin_GOF'
        power_thresh: ignored if scoring='Jin_GOF'
        '''
        self._check_thresh_type(score_thresh_type)

        self._window_size = window_size
        self._power_thresh = power_thresh
        self._scoring = scoring
        self._voting_window_size = voting_window_size
        self._patience = patience
        self._vote_thresh = vote_thresh
        self._score_thresh = score_thresh
        self._score_thresh_type = score_thresh_type
        self._var_clip = kwargs.get('var_clip', 1e-12)
        self._linear_factor = kwargs.get('linear_factor', 0.005)
        self._power_clip = kwargs.get('power_clip', 1e-12)

    def _volker_likelihood(self, xj, mi, mj, vi, vj):
        thresh = self._power_thresh + mi * self._linear_factor

        if abs(mj - mi) > thresh:
            likelihood = 0.5 * np.log(vi / vj)
            likelihood += (xj - mi)**2 / (2 * vi)
            likelihood -= (xj - mj)**2 / (2 * vj)

            return abs(likelihood)

        return 0

    def _pereira_likelihood(self, xj, mi, mj, v):
        if abs(mj - mi) > self._power_thresh:
            likelihood = (mj - mi) / v * abs(xj - 0.5 * (mj + mi))

            return abs(likelihood)

        return 0

    def _jin_gof_score(self, q, p):
        gof_score = ((q - p)**2 / p).sum()

        return gof_score

    def _compute_scores(self, x):
        var_clip = self._var_clip
        power_clip = self._power_clip
        window_size = self._window_size

        scores = np.zeros_like(x)
        x = np.pad(x, (window_size - 1, window_size))

        for i in range(len(x) - 2 * window_size + 1):
            j = i + window_size
            dw = x[i:j + window_size]  # Detection window
            p = dw[:window_size]  # Pre-event window
            q = dw[window_size:]  # Post-event window

            assert len(p) == len(q) == window_size

            if self._scoring in ['Volker_likelihood', 'Pereira_likelihood']:
                x0, mp, mq = q[0], p.mean(), q.mean()

            if self._scoring == 'Volker_likelihood':
                vp = np.clip(p.var(), a_min=var_clip, a_max=None)
                vq = np.clip(q.var(), a_min=var_clip, a_max=None)
                score = self._volker_likelihood(x0, mp, mq, vp, vq)
            elif self._scoring == 'Pereira_likelihood':
                v = np.clip(dw.var(), a_min=var_clip, a_max=None)
                score = self._pereira_likelihood(x0, mp, mq, v)
            elif self._scoring == 'Jin_GOF':
                p = np.clip(p, a_min=power_clip, a_max=None)
                q = np.clip(q, a_min=power_clip, a_max=None)
                score = self._jin_gof_score(q, p)
            else:
                raise ValueError

            scores[i] = score

        return scores

    def transform(self, entity):
        x = entity.values
        x = self._compute_scores(x)
        x0 = self._get_change_points(x)

        return x0
