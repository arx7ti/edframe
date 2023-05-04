from __future__ import annotations

import numpy as np
from numba import njit
from typing import Optional
from scipy.signal import butter, filtfilt


class FITPSNotCalledError(Exception):
    pass


class FITPSCallingError(Exception):
    pass


class FITPS:
    """
    Frequency Invariant Transformation of Periodic Signals.
    """

    @property
    def roots(self) -> np.ndarray:
        if self._v0 is not None:
            return self._v0
        else:
            raise FITPSNotCalledError

    @property
    def n_samples(self) -> int:
        if self._ns is not None:
            return self._ns
        else:
            raise FITPSNotCalledError

    @property
    def n_periods(self) -> int:
        if self._np is not None:
            return self._np
        else:
            raise FITPSNotCalledError

    @property
    def f0(self) -> int:
        if self._f0 is not None:
            return self._f0
        else:
            raise FITPSNotCalledError

    @property
    def mains_frequency(self) -> int:
        return self.f0

    def __init__(self,
                 f0: Optional[float] = None,
                 n: int = 4,
                 cf: float = 0.01,
                 np_loss: int = 4,
                 v_quality: float = 0.9) -> None:
        self._n = n
        self._cf = cf
        self._f0 = f0
        self._np_loss = np_loss
        self._v_quality = v_quality
        self._v0 = None
        self._f0 = None
        self._ns = None
        self._np = None
        return None

    def __call__(self, v: np.ndarray, i: np.ndarray,
                 sr: int) -> tuple[np.ndarray, np.ndarray]:
        assert len(v.shape) == len(i.shape) == 1
        assert v.shape[0] == i.shape[0]
        vf = self._filter_v(v)
        v0 = self._compute_roots(vf)
        dv = self._compute_shifts(vf, v0)
        del vf
        dv0 = np.diff(v0)
        if self._f0 is not None:
            ns = round(sr / self._f0)
        else:
            ns = round(np.mean(dv0))
        v = self._allocate(v, v0, dv, ns)
        self._np = v.shape[0]
        i = self._allocate(i, v0, dv, ns)
        return v, i

    def _filter_v(self, v: np.ndarray) -> np.ndarray:
        f1, f2 = butter(self._n, self._cf)
        v = filtfilt(f1, f2, v).astype(np.float32)
        return v

    @staticmethod
    @njit
    def _compute_roots(v: np.ndarray) -> np.ndarray:
        v0 = np.empty(0, dtype=np.int32)
        for j in np.arange(len(v) - 1, dtype=np.int32):
            if (v[j] < np.float32(0.0)) & (v[j + 1] > np.float32(0.0)):
                v0 = np.append(v0, j)
            else:
                continue
        return v0

    @staticmethod
    @njit
    def _compute_shifts(v: np.ndarray, v0: np.ndarray) -> np.ndarray:
        dv = np.empty(0, dtype=np.float32)
        for j in np.arange(len(v0) - 1, dtype=np.int32):
            k = v0[j]
            dv = np.append(dv, -v[k] / (v[k + 1] - v[k]))
        return dv

    @staticmethod
    @njit
    def _allocate(vec: np.ndarray, v0: np.ndarray, dv: np.ndarray,
                  ns: int) -> np.ndarray:
        n_periods = len(dv) - 1
        mat = np.zeros((n_periods, ns), dtype=np.float32)
        v0 = v0.astype(np.float32)
        for j in np.arange(n_periods, dtype=np.int32):
            length = (v0[j + 1] + dv[j + 1]) - (v0[j] + dv[j])
            dist = length / np.float32(ns)
            for k0 in np.arange(1, ns + 1, dtype=np.int32):
                k1 = v0[j] + dv[j] + dist * np.float32(k0 - 1)
                k2 = np.int32(np.floor(k1))
                k3 = np.int32(np.ceil(k1))
                mat[j, k0] = vec[k2] + (vec[k3] - vec[k2]) * dv[j]
        return mat