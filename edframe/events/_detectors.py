from __future__ import annotations

from typing import Optional, Callable, Iterable
from sklearn.cluster import AgglomerativeClustering
from beartype import abby

import numpy as np
import itertools as it

from ..data.entities import PowerSample, DataSet


class EventDetector:
    __low__: bool = False
    __high__: bool = False
    __continuous__: bool = False
    __verbose_name__: str | None = None

    @property
    def verbose_name(self):
        if self.__verbose_name__ is None:
            return self.__class__.__name__
        else:
            return self.__verbose_name__

    def is_continuous(self):
        return self.__continuous__

    def is_compatible(self, obj: PowerSample | DataSet | np.ndarray) -> bool:
        if not (hasattr(obj, "__low__") or isinstance(obj, np.ndarray))\
            and not (hasattr(obj, "__high__") or isinstance(obj, np.ndarray)):
            return False

        if not (self.__low__ == obj.__low__ == True
                or self.__high__ == obj.__high__ == True):
            return False

        return True

    def __init__(
        self,
        window: Callable,
        window_size: int,
        verbose_name: Optional[str] = None,
    ) -> None:
        if not isinstance(window, Callable):
            raise ValueError

        if not self.__low__ and not self.__high__:
            raise ValueError

        if verbose_name is not None:
            self.__verbose_name__ = verbose_name

        self._window = window
        self._window_size = window_size

    def __call__(self, x: PowerSample | DataSet | np.ndarray):
        return self.detect(x)

    def _striding_window_view(self, x: np.ndarray) -> np.ndarray:
        axes = x.shape[:-1]
        n_pad = self._window_size - (x.shape[-1] % self._window_size)

        if n_pad > 0:
            # Padding with zeros
            x_pad = np.zeros((*axes, n_pad), dtype=x.dtype)
            x = np.concatenate((x, x_pad), axis=-1)

        n_windows = x.shape[-1] // self._window_size
        x = x.reshape(*axes, n_windows, self._window_size)

        return x

    def _detect_from_dataset(self, x: DataSet, **kwargs):
        v = []
        for _x in x:
            v.append(self.detect(_x, **kwargs))
        return v

    def detect(self, x: PowerSample | DataSet | np.ndarray):
        raise NotImplementedError

    def check_fn(self, x, **kwargs):
        return None


class ThresholdEvent(EventDetector):
    __low__ = True
    __high__ = True
    __continuous__ = False

    def __init__(
        self,
        window: Callable,
        alpha: float = 0.05,
        above: bool = True,
        window_size: int = 80,
        verbose_name: Optional[str] = None,
    ) -> None:
        super().__init__(window=window,
                         window_size=window_size,
                         verbose_name=verbose_name)
        self._alpha = alpha
        self._window_size = window_size
        self._above = above

    def detect(self, x: PowerSample | DataSet | np.ndarray, **kwargs):
        if isinstance(x, PowerSample | DataSet):
            x = x.values
        elif not isinstance(x, np.ndarray):
            raise ValueError

        if abby.is_bearable(x, list[PowerSample]):
            return self._detect_from_dataset(x, **kwargs)

        is_dataset = kwargs.pop("is_dataset", len(x.shape) > 1)

        if not isinstance(x, np.ndarray):
            raise ValueError

        if len(x.shape) > 1:
            raise ValueError("Only 1D signals are supported")

        self.check_fn(x, is_dataset=is_dataset, **kwargs)

        n = len(x)
        x = self._striding_window_view(x)
        x = np.apply_along_axis(self._window, axis=1, arr=x)
        dx = np.diff(x > self._alpha, prepend=False).astype(int)
        signs = np.sign(dx)
        locs0 = np.argwhere(signs > 0).ravel()
        locs1 = np.argwhere(signs < 0).ravel()
        # locs1 -= 1
        # locs = np.sort(np.concatenate((locs0, locs1)))
        locs = np.concatenate((locs0, locs1))
        # locs1[signs[locs] < 0] -= 1

        # Calibration
        if len(locs) % 2 > 0:
            locs = np.append(locs, n)

        signs = signs[locs]
        signs = np.where(signs < 0, 0, 1)
        locs *= self._window_size
        locs[len(locs0):] -= 1
        events = list(zip(locs, signs))

        return events


class DerivativeEvent(EventDetector):
    __high__ = True
    __continuous__ = True

    def __init__(
        self,
        window: Callable,
        alpha: float = 0.05,
        beta: int = 10,
        interia: bool = True,
        relative: bool = True,
        window_size: int = 80,
        verbose_name: Optional[str] = None,
    ) -> None:
        super().__init__(window=window,
                         window_size=window_size,
                         verbose_name=verbose_name)
        self._alpha = alpha
        self._beta = beta
        self._interia = interia
        self._relative = relative

    def detect(self, x: np.ndarray, **kwargs):
        if isinstance(x, PowerSample | DataSet):
            x = x.values
        elif not isinstance(x, np.ndarray):
            raise ValueError

        if abby.is_bearable(x, list[PowerSample]):
            return self._detect_from_dataset(x, **kwargs)

        is_dataset = kwargs.pop("is_dataset", len(x.shape) > 1)

        if not isinstance(x, np.ndarray):
            raise ValueError

        self.check_fn(x, is_dataset=is_dataset, **kwargs)

        x = self._striding_window_view(x)
        x = np.apply_along_axis(self._window, axis=1, arr=x)
        dx = np.diff(x, prepend=np.NINF)

        if self._relative:
            dx[1:] /= x[:-1]

        locs = np.argwhere(np.abs(dx) > self._alpha).ravel()

        if len(locs) > 1 and self._interia:
            locs_upd = []
            cl = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=self._beta)
            clusters = cl.fit_predict(locs.reshape(-1, 1))

            for cluster in np.unique(clusters):
                cluster_locs = locs[clusters == cluster]
                locs_upd.append(cluster_locs[0])

            locs = np.sort(locs_upd)

        # Calibration
        if locs[-1] < len(x) - 1:
            locs = np.append(locs, len(x) - 1)

        # Signs
        signs = np.sign(dx[locs])
        signs = np.where(signs < 0, 0, 1)
        locs *= self._window_size
        print(locs)
        assert locs[-1] < len(x) * self._window_size

        # Events
        events = list(zip(locs, signs))

        return events


class ROI:
    def __init__(self, detectors: EventDetector | Iterable["EventDetector"]):
        if isinstance(detectors, EventDetector):
            detectors = [detectors]
        elif not isinstance(detectors, Iterable):
            raise ValueError

        self._detectors = detectors
        self._k = 0

    def __call__(self, x) -> PowerSample | DataSet | np.ndarray:
        x = self.crop(x)
        return x

    def crop(
        self,
        x: PowerSample | DataSet | np.ndarray,
    ) -> PowerSample | DataSet | np.ndarray:
        n = len(x)

        def _crop(x: np.ndarray, detectors):
            detector0 = detectors[0]
            events0 = detector0(x)
            locs0, _ = zip(*events0)

            if detector0.is_continuous():
                locs2d0 = []
                for i in range(1, len(locs0)):
                    a = locs0[i - 1]
                    b = locs0[i]

                    # if i == len(locs0) - 1:
                    #     b = locs0[i]
                    # else:
                    #     b = locs0[i] - 1
                    locs2d0.append((a, b))

                # if b < n:
                #     locs2d0.append((b, n))
            else:
                locs2d0 = [locs0[i:i + 2] for i in range(0, len(locs0), 2)]

            del locs0

            roi = []

            for a0, b0 in locs2d0:
                print("-> %s" % detector0._window, a0, b0)
                xab0 = x[a0:b0]

                if len(detectors) > 1 and len(xab0) > 1:
                    roi.extend(_crop(xab0, detectors[1:]))
                else:
                    roi.append(xab0)

            return roi

        return _crop(x, self._detectors)
