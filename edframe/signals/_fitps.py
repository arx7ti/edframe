from __future__ import annotations

import numpy as np
from numba import njit
from typing import Optional
from scipy.signal import butter, filtfilt

from edframe.utils.exceptions import NotEnoughPeriods


class FITPSNotCalledError(Exception):
    pass


class FITPSCallingError(Exception):
    pass


class FITPS:
    """
    Frequency Invariant Transformation of Periodic Signals.
    """

    def __init__(self) -> None:
        self._zero_crossings = None

    @property
    def zero_crossings(self):
        return self._zero_crossings

    def __call__(
        self,
        v,
        i,
        locs=None,
        fs=None,
        f0=None,
    ) -> tuple[np.ndarray, ...]:

        v0 = self._compute_roots(v)
        self._zero_crossings = v0

        if len(v0) < 2:
            raise NotEnoughPeriods

        dv = self._compute_shifts(v, v0)

        if f0 is not None and fs is not None:
            ns = round(fs / f0)
        else:
            dv0 = np.diff(v0)
            ns = round(np.mean(dv0))

        v = self._allocate(v, v0, dv, ns)
        i = self._allocate(i, v0, dv, ns)

        if locs is not None:
            if not isinstance(locs, np.ndarray):
                locs = np.asarray(locs)

            locs -= v0[0]
            locs[locs >= v0[-1]] = np.product(v.shape)
            locs = np.clip(locs, a_min=0, a_max=None)

            return v, i, locs
        else:
            return v, i

    # @staticmethod
    # def _transform_locs(locs, v0):
    # shape = locs.shape

    # if len(shape) == 2:
    #     locs = locs.ravel()
    # elif len(shape) != 1:
    #     raise ValueError

    # ord = np.argsort(locs)

    # j = 0
    # plocs_sorted = []
    # n_periods = len(v0) - 2
    # for loc in locs[ord]:
    #     if loc < v0[0]:
    #         plocs_sorted.append(np.NINF)
    #     elif loc > v0[-2]:
    #         plocs_sorted.append(np.Inf)
    #     else:
    #         for k in range(j, n_periods):
    #             if (loc >= v0[k]) & (loc < v0[k + 1]):
    #                 plocs_sorted.append(k)
    #                 j = k
    #                 break

    # assert len(plocs_sorted) == len(locs)

    # plocs = np.empty(len(ord), dtype=float)
    # for i, j in enumerate(ord):
    #     plocs[j] = plocs_sorted[i]

    # # Calibration
    # if len(shape) == 2:
    #     plocs = plocs.reshape(*shape)
    #     q1 = (plocs[:, 0] == np.NINF) & (plocs[:, 1] == np.NINF)
    #     q2 = (plocs[:, 0] == np.Inf) & (plocs[:, 1] == np.Inf)
    #     q3 = (plocs[:, 0] == np.NINF) & (plocs[:, 1] != np.NINF)
    #     q4 = (plocs[:, 0] != np.Inf) & (plocs[:, 1] == np.Inf)
    #     plocs[q1 | q2] = -1
    #     plocs[q3, 0] = 0
    #     plocs[q4, 1] = n_periods - 1

    # plocs = plocs.astype(int)

    # return plocs

    # def _filter_v(self, v: np.ndarray) -> np.ndarray:
    #     f1, f2 = butter(self._n, self._cf)
    #     v = filtfilt(f1, f2, v).astype(v.dtype)
    #     return v

    @staticmethod
    @njit
    def _compute_roots(v: np.ndarray) -> np.ndarray:
        v0 = np.empty(0, dtype=np.int32)

        for j in np.arange(len(v) - 1, dtype=np.int32):
            if (v[j] < v.dtype.type(0.0)) & (v[j + 1] >= v.dtype.type(0.0)):
                v0 = np.append(v0, j)
            else:
                continue
        return v0

    @staticmethod
    @njit
    def _compute_shifts(v: np.ndarray, v0: np.ndarray) -> np.ndarray:
        dv = np.empty(0, dtype=v.dtype)

        for j in np.arange(len(v0) - 1, dtype=np.int32):
            k = v0[j]
            dv = np.append(dv, -v[k] / (v[k + 1] - v[k]))
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
                k1 = v0[j] + dv[j] + dist * vec.dtype.type(k)
                k2 = np.int32(np.floor(k1))
                k3 = np.int32(np.ceil(k1))
                mat[j, k] = vec[k2] + (vec[k3] - vec[k2]) * dv[j]
        return mat
