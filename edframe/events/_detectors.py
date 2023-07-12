from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering
from typing import Optional, Union, Any, Callable, Iterable

import numpy as np
import itertools as it

from ..data.entities import PowerSample, DataSet


class EventDetector:
    __window__: Callable | None = None
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

    def __init__(self, verbose_name: Optional[str] = None) -> None:
        if not self.__low__ and not self.__high__:
            raise ValueError

        if self.__window__ is None:
            raise ValueError

        if verbose_name is not None:
            self.__verbose_name__ = verbose_name

    def __call__(self, x: np.ndarray) -> list[tuple[str, int, Any]]:
        return self.detect(x)

    def _striding_window_view(self, x, window_size, drop_last: bool = True):
        n = x.shape[-1]
        axes = x.shape[:-1]
        rem = n % window_size

        if rem > 0:
            if drop_last:
                # In case of dropping last not-full window:
                x = x[..., :n - rem]
            else:
                # In case of padding last not-full window:
                n_pad = window_size - (n % window_size)

                if n_pad > 0:
                    # Padding with zeros
                    x_pad = np.zeros((*axes, n_pad), dtype=x.dtype)
                    x = np.concatenate((x, x_pad), axis=-1)

        x = x.reshape(*axes, x.shape[-1] // window_size, window_size)

        return x

    def detect(self, x):
        raise NotImplementedError


class ThresholdEvent(EventDetector):
    __continuous__ = False

    def __init__(self,
                 alpha: float = 0.05,
                 objective_fn: str = None,
                 above: bool = True,
                 verbose_name: Optional[str] = None,
                 window_size: int = 80,
                 drop_last: bool = True) -> None:
        super().__init__(verbose_name=verbose_name)
        self._alpha = alpha
        self._window_size = window_size
        self._drop_last = drop_last

        if objective_fn is None:
            self._objective_fn = self.defaults['objective_fn']
        else:
            self._objective_fn = objective_fn

        self._above = above

    # @property
    # def verbose_name(self):
    #     if self._verbose_name is None:
    #         return "%s threshold of %s" % ("Above" if self._above else "Below",
    #                                        self._objective_fn.__name__)
    #     else:
    #         return self._verbose_name

    def detect(self, x):

        #################
        x = self._striding_window_view(x, self._window_size, self._drop_last)
        #################

        y = np.apply_along_axis(self._objective_fn, axis=1, arr=x)
        f = (y > self._alpha).astype(int)
        df = np.diff(f, prepend=False)
        signs = np.sign(df)
        locs0 = np.argwhere(signs > 0).ravel()
        locs1 = np.argwhere(signs < 0).ravel()
        locs1 -= 1
        locs = np.sort(np.concatenate((locs0, locs1)))
        # locs1[signs[locs] < 0] -= 1

        # Calibration
        if len(locs) % 2 != 0:
            locs = np.append(locs, len(x) - 1)

        signs = signs[locs]
        signs = np.where(signs < 0, 0, 1)

        events = list(zip(locs, signs))

        return events


class DerivativeEvent(EventDetector):
    __continuous__ = True

    def __init__(
        self,
        alpha: float = 0.05,
        beta: int = 10,
        interia: bool = True,
        relative: bool = True,
        objective_fn: str = None,
        verbose_name: Optional[str] = None,
        window_size: int = 80,
        drop_last: bool = True,
    ) -> None:
        super().__init__(verbose_name=verbose_name)
        self._alpha = alpha
        self._beta = beta
        self._interia = interia
        self._relative = relative
        self._window_size = window_size
        self._drop_last = drop_last

        if objective_fn is None:
            self._objective_fn = self.defaults['objective_fn']
        else:
            self._objective_fn = objective_fn

    @property
    def verbose_name(self):
        return "Derivative of %s" % self._objective_fn.__name__

    def detect(self, x: np.ndarray) -> list[tuple[int, int]]:
        # TODO Ivan :: striding window
        # Now x -> 1d
        # need to split x into windows
        # IMPORTANT ASSUMPTION: drop last window
        # CODE HERE

        ######################
        x = self._striding_window_view(x, self._window_size, self._drop_last)
        ######################

        y = np.apply_along_axis(self._objective_fn, axis=1, arr=x)
        dy = np.diff(y, prepend=np.NINF)

        if self._relative:
            dy[1:] /= y[:-1]

        f = np.abs(dy) > self._alpha
        locs = np.argwhere(f).ravel()

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
        assert locs[-1] < len(x)

        # Signs
        signs = np.sign(dy[locs])
        signs = np.where(signs < 0, 0, 1)

        # Events
        events = list(zip(locs * self._window_size, signs))

        return events


class ROI:

    def __init__(self, detectors: Union[EventDetector, list[EventDetector]]):
        if not isinstance(detectors, Iterable):
            detectors = [detectors]

        self._detectors = detectors

    def __call__(
        self,
        x: Union[PowerSample, np.ndarray],
    ) -> Union[PowerSample, np.ndarray]:
        return self._get(x, self._detectors)

    def _get(
        self,
        x: Union[PowerSample, np.ndarray],
        detectors: list[EventDetector],
    ) -> list[Union[PowerSample, np.ndarray]]:
        detector0 = detectors[0]
        events0 = detector0(x)
        locs0, _ = zip(*events0)

        if detector0.is_continuous():
            locs2d0 = []
            for i in range(1, len(locs0)):
                a = locs0[i - 1]
                if i == len(locs0) - 1:
                    b = locs0[i]
                else:
                    b = locs0[i] - 1
                locs2d0.append([a, b])
        else:
            locs2d0 = [locs0[i:i + 2] for i in range(0, len(locs0), 2)]

        del locs0

        roi = []

        for a0, b0 in locs2d0:
            xab0 = x[a0:b0 + 1]

            if len(detectors) > 1 and len(xab0) > 1:
                roi.extend(self._get(xab0, detectors[1:]))
            else:
                roi.append(xab0)

        return roi