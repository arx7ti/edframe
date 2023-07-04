from __future__ import annotations
from sklearn.cluster import AgglomerativeClustering
from typing import Optional, Union, Any, Callable
from ..signals import rms

import numpy as np


# TODO striding window
class EventDetector:
    @property
    def defaults(self):
        return {'objective_fn': rms}

    @property
    def event_name(self):
        if self._event_name is None:
            return "Unnamed Event of %s" % self.__class__.__name__
        else:
            return self._event_name

    def __init__(self, event_name: Optional[str] = None) -> None:
        self._event_name = event_name

    def __call__(self, x: np.ndarray) -> list[tuple[str, int, Any]]:
        return self.detect(x)

    def detect(self, x):
        raise NotImplementedError

    @staticmethod
    def _get_paired_locs(locs, successive: bool = False):
        if successive:
            pairs = []
            for i in range(1, len(locs)):
                a = locs[i - 1]
                if i == len(locs) - 1:
                    b = locs[i]
                else:
                    b = locs[i] - 1
                pairs.append((a, b))
            return pairs
        else:
            return list(map(tuple, locs.reshape(-1, 2)))

    @staticmethod
    def _get_unpaired_locs(locs2d):
        # TODO successive
        locs = list(map(lambda x: x[0], locs2d))
        locs += [locs2d[-1][1] + 1]
        return locs


class ThresholdEvent(EventDetector):
    def __init__(
        self,
        alpha: float = 0.05,
        objective_fn: str = None,
        above: bool = True,
        event_name: Optional[str] = None,
    ) -> None:
        super().__init__(event_name=event_name)
        self._alpha = alpha

        if objective_fn is None:
            self._objective_fn = self.defaults['objective_fn']
        else:
            self._objective_fn = objective_fn

        self._above = above

    @property
    def event_name(self):
        if self._event_name is None:
            return "%s threshold of %s" % ("Above" if self._above else "Below",
                                           self._objective_fn.__name__)
        else:
            return self._event_name

    def detect(self, x):
        if len(x.shape) != 2:
            raise NotImplementedError

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
        # if locs_type == '1d':
        #     return locs
        # else:
        #     return self._get_paired_locs(locs, successive=False)


class DerivativeEvent(EventDetector):
    def __init__(
        self,
        alpha: float = 0.05,
        beta: int = 10,
        interia: bool = True,
        relative: bool = True,
        objective_fn: str = None,
    ) -> None:
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._interia = interia
        self._relative = relative
        if objective_fn is None:
            self._objective_fn = self.defaults['objective_fn']
        else:
            self._objective_fn = objective_fn

    @property
    def event_name(self):
        return "Derivative of %s" % self._objective_fn.__name__

    def detect(
        self,
        x: np.ndarray,
        # locs_type: str = '2d',
    ) -> list[tuple[int, int]]:
        if len(x.shape) != 2:
            raise NotImplementedError

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
        events = list(zip(locs, signs))

        return events

        # if locs_type == '1d':
        #     return locs
        # else:
        #     return self._get_paired_locs(locs, successive=True)


class SequentialEvent(EventDetector):
    def __init__(self, detectors: list[EventDetector]) -> None:
        super().__init__()
        self._detectors = detectors

    @classmethod
    def _apply_detectors(
        cls,
        detectors: list[EventDetector],
        x: np.ndarray,
    ) -> np.ndarray:
        locs0 = detectors[0].detect(x, locs_type='2d')

        if len(detectors) > 1:
            locs_prime = []

            for a0, b0 in locs0:
                xab0 = x[a0:b0 + 1]

                if len(xab0) > 1:
                    locs = cls._apply_detectors(detectors[1:], xab0)
                    locs = [(a + a0, b + a0) for a, b in locs]
                else:
                    locs = [(a0, b0)]

                locs_prime.extend(locs)
            return locs_prime
        else:
            return locs0

    def detect(self, x: np.ndarray, locs_type: str = '2d') -> np.ndarray:
        locs = self._apply_detectors(self._detectors, x)

        if locs_type == '1d':
            return self._get_unpaired_locs(locs)
        else:
            return locs
