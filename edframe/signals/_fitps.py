from __future__ import annotations

import math
import numpy as np

from numba import njit
from typing import Optional
from scipy.signal import butter, filtfilt

from edframe.utils.exceptions import NotEnoughCycles, OutliersDetected, NotFittedError


class FITPS:
    """
    Frequency Invariant Transformation of Periodic Signals.
    """

    def __init__(
        self,
        window_size=1,
        outlier_thresh=0.1,
        zero_thresh=1e-4,
    ) -> None:
        assert window_size > 0

        self._ws = window_size
        self._outlier_thresh = outlier_thresh
        self._zero_thresh = zero_thresh

        self._x0 = None
        self._Tm = None
        self._dx = None
        self._xf = None

    @property
    def zero_crossings(self):
        self._check_fit()

        return self._x0

    def fit(self, x):
        # TODO with and without filtering
        x0, xf = self._find_roots(x, self._ws, self._zero_thresh)

        if len(x0) < 2:
            raise NotEnoughCycles

        # Find outliers by cycle size
        T = np.diff(x0)
        Tm = np.median(T)
        T = T / np.median(T)
        outliers = T < min(0, 1 - self._outlier_thresh)
        outliers |= T > 1 + self._outlier_thresh

        if outliers.any():
            raise OutliersDetected

        dx = self._compute_shifts(xf, x0)

        if dx is None:
            raise OutliersDetected

        self._x0 = x0
        self._Tm = Tm
        self._dx = dx
        self._xf = xf

        return

    def transform(self, x, cycle_size=None, locs=None):
        self._check_fit()

        cycle_size = self._Tm if cycle_size is None else cycle_size
        x = self._allocate(x, self._x0, self._dx, cycle_size)

        if locs is not None and None not in locs:
            if not isinstance(locs, np.ndarray):
                locs = np.asarray(locs)

            locs -= self._x0[0]
            locs[locs >= self._x0[-1]] = np.product(x.shape)
            locs = np.clip(locs, a_min=0, a_max=None)

            return x, locs

        return x

    @staticmethod
    @njit
    def _find_roots(
        x: np.ndarray,
        window_size: int,
        zero_thresh: float,
    ) -> np.ndarray:
        # Padding signal before computing running mean
        padding = np.zeros(window_size, dtype=np.int32)
        left, right = padding[:window_size // 2], padding[window_size // 2:]
        x = np.concatenate((left, x, right))

        # Defining buffer for roots and running window
        xf = []
        x0 = []
        w = np.empty((2, window_size), dtype=np.float32)

        for j in np.arange(1, len(x), dtype=np.int32):
            # Windowindow_size at the current time and the next
            w[0, j % window_size], w[1, j % window_size] = x[j - 1], x[j]

            if j >= window_size:
                x1 = w[0].mean()
                x1 = abs(x1) if abs(x1) < zero_thresh else x1

                xf.append(x1)

            if j > window_size:
                x2 = w[1].mean()
                x2 = abs(x2) if abs(x2) < zero_thresh else x2

                s1 = math.copysign(1, x1) < 0
                s2 = math.copysign(1, x2) > 0

                if s1 & s2:
                    x0.append(j - window_size)

        x0 = np.asarray(x0)
        xf = np.asarray(xf)

        return x0, xf

    @staticmethod
    @njit
    def _compute_shifts(v, v0):
        dv = []

        for j in np.arange(len(v0) - 1, dtype=np.int32):
            k = v0[j]

            if v[k + 1] == v[k]:
                return
            else:
                dv.append(-v[k] / (v[k + 1] - v[k]))

        dv = np.asarray(dv)

        return dv

    @staticmethod
    @njit
    def _allocate(
        vec: np.ndarray,
        v0: np.ndarray,
        dv: np.ndarray,
        ns_int: int,
    ) -> np.ndarray:
        ns_float = vec.dtype.type(ns_int)
        n_periods = len(dv) - 1
        mat = np.zeros((n_periods, ns_int), dtype=vec.dtype)
        v0 = v0.astype(vec.dtype)

        for j in np.arange(n_periods, dtype=np.int32):
            length = (v0[j + 1] + dv[j + 1]) - (v0[j] + dv[j])
            dist = length / ns_float

            for k in np.arange(ns_int, dtype=np.int32):
                k1 = v0[j] + dv[j] + dist * k
                k2 = np.int32(np.floor(k1))
                k3 = np.int32(np.ceil(k1))
                mat[j, k] = vec[k2] + (vec[k3] - vec[k2]) * dv[j]

        return mat

    def _check_fit(self):
        fit_condition = self._x0 is not None
        fit_condition = self._Tm is not None
        fit_condition |= self._dx is not None
        fit_condition |= self._xf is not None

        if not fit_condition:
            raise NotFittedError
