from __future__ import annotations
from tqdm import tqdm
# from beartype import abby
from typing import Optional, Callable, Iterable
from sklearn.cluster import AgglomerativeClustering
from ..data.entities import PowerSample, DataSet

import numpy as np
# import itertools as it


class EventDetector:
    __low__: bool = False
    __high__: bool = False
    __continuous__: bool = False
    __verbose_name__: str | None = None

    @property
    def verbose_name(self):
        if self.__verbose_name__ is None:
            return self.__class__.__name__

        return self.__verbose_name__

    @property
    def source_name(self) -> str:
        return self._source_name

    @source_name.setter
    def source_name(self, source_name: str) -> None:
        self._source_name = source_name

    def is_continuous(self):
        return self.__continuous__

    def is_compatible(self, obj: PowerSample | DataSet | np.ndarray) -> bool:
        if not (hasattr(obj, "__low__") or isinstance(obj, np.ndarray))\
            and not (hasattr(obj, "__high__") or isinstance(obj, np.ndarray)):
            return False

        if not isinstance(obj, np.ndarray):
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

    def __call__(self, x: PowerSample | DataSet | np.ndarray, **kwargs):
        return self.detect(x, **kwargs)

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
        e = []

        for _x in x:
            e.append(self.detect(_x, **kwargs))

        return e

    def _check_compatibility(
        self,
        x: PowerSample | DataSet | np.ndarray,
    ) -> None:
        if not self.is_compatible(x):
            raise ValueError

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
        alpha: float = 0.005,
        above: bool = True,
        window_size: int = 80,
        verbose_name: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> None:
        super().__init__(window=window,
                         window_size=window_size,
                         verbose_name=verbose_name)
        self._alpha = alpha
        self._window_size = window_size
        self._above = above
        self._source_name = "values" if source_name is None else source_name

    def detect(self, x: PowerSample | DataSet | np.ndarray, **kwargs):
        self._check_compatibility(x)

        if isinstance(x, DataSet):
            return self._detect_from_dataset(x, **kwargs)

        if isinstance(x, PowerSample):
            xlocs = x.locs
            x = x.source(self.source_name)
        else:
            xlocs = None
        # elif not isinstance(x, np.ndarray):
        #     raise ValueError

        if not isinstance(x, np.ndarray):
            raise ValueError

        if len(x.shape) > 1:
            raise ValueError("Only 1D signals are supported")

        self.check_fn(x, **kwargs)

        x = self._striding_window_view(x)
        x = np.apply_along_axis(self._window, axis=1, arr=x)
        x = np.where(x > self._alpha, 1, 0)
        dx = np.diff(x > self._alpha, prepend=False).astype(int)
        signs = np.sign(dx)
        locs0 = np.argwhere(signs > 0).ravel()
        locs1 = np.argwhere(signs < 0).ravel()
        locs = np.concatenate((locs0, locs1))

        # Calibration
        if len(locs) % 2 > 0:
            locs = np.append(locs, len(x) - 1)

        signs = signs[locs]
        locs *= self._window_size
        locs[len(locs0):] -= 1

        # if xlocs is not None:
        #     # locs are not binded to labels
        #     # what to do if we have labels?
        #     # locs_upd = []
        #     locs = np.clip(locs,
        #                    a_min=xlocs[:, 0].min(),
        #                    a_max=xlocs[:, 1].max())
        if xlocs is not None:
            # N -> m
            # N: [0 40] [50 70] [80 110] [120 150] labels domain
            # m: [15 100], [30 90] events in signal
            # -> [15 40] [50 70] [80 100]
            # -> [30 40] [50 70] [80 90]
            # --> [15 40] [50 70] [80 90]
            ### ALG:
            ### define labelled intervals for xlocs
            ### extract locs from labelled intervals
            ### for each set of labels define its own locs

            i = np.arange(max(xlocs.max(), locs.max()))
            D = xlocs[:, 1] - xlocs[:, 0]
            i0 = xlocs[:, 0]

            mask = (i[:, None] >= i0) & (i[:, None] < (i0 + D))
            acts = mask.sum(1)
            i = np.argwhere(acts > 0).ravel()

            di = np.diff(i, append=i[-1] + 1)

            # Jumps in activations
            jacts = np.argwhere(di != 1).ravel()

            J = [[_j[0], _j[-1] + 1] for _j in np.split(i, jacts + 1)]
            J = np.asarray(J)

            # Jumps in original locs
            jlocs = np.unique(xlocs.ravel()).reshape(-1, 2)

            # Corrected locs
            clocs = np.empty((0, 2), dtype=int)
            Jmax = J.max()

            locs2d = locs.reshape(-1, 2)
            locs2d[:, 1] += 1

            for x0, x1 in locs2d:
                _jacts = jlocs[(jlocs > x0) & (jlocs < x1)]

                if np.any(x0 >= J[:, 0]):
                    _jacts = np.insert(_jacts, 0, x0)

                if np.all(x1 <= J[:, 1]) or x1 == Jmax:
                    _jacts = np.append(_jacts, x1)

                _clocs = np.repeat(_jacts, 2)[1:-1].reshape(-1, 2)
                clocs = np.concatenate((clocs, _clocs))

            # FIXME end-index is exclusive
            # FIXME 0-length intervals
            # clocs[:, 1] -= 1
            # locs = clocs.ravel()
            locs = np.unique(clocs.ravel())

            # for ax,bx in xlocs:
            #     ilocs = np.clip(locs, a_min=ax, a_max=bx)

        # events = list(zip(locs, signs))
        events = locs

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
        source_name: Optional[str] = None,
    ) -> None:
        super().__init__(window=window,
                         window_size=window_size,
                         verbose_name=verbose_name)
        self._alpha = alpha
        self._beta = beta
        self._interia = interia
        self._relative = relative
        self._source_name = "values" if source_name is None else source_name

    def detect(self, x: np.ndarray, **kwargs):
        self._check_compatibility(x)

        if isinstance(x, DataSet):
            return self._detect_from_dataset(x, **kwargs)

        if isinstance(x, PowerSample):
            xlocs = x.locs
            x = x.source(self.source_name)
        else:
            xlocs = None

        if not isinstance(x, np.ndarray):
            raise ValueError

        self.check_fn(x, **kwargs)

        x = self._striding_window_view(x)
        x = np.apply_along_axis(self._window, axis=1, arr=x)
        x = np.nan_to_num(x, nan=np.Inf)
        dx = np.diff(x, prepend=np.NINF)

        if self._relative:
            dx[1:] /= x[:-1]

        dx = np.nan_to_num(dx, nan=np.Inf)

        locs = np.argwhere(np.abs(dx) > self._alpha).ravel()
        # TODO if no locs found

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
        # signs = np.where(signs < 0, 0, 1)
        locs *= self._window_size
        assert locs[-1] < len(x) * self._window_size

        if xlocs is not None:
            # N -> m
            # N: [0 40] [50 70] [80 110] [120 150] labels domain
            # m: [15 100], [30 90] events in signal
            # -> [15 40] [50 70] [80 100]
            # -> [30 40] [50 70] [80 90]
            # --> [15 40] [50 70] [80 90]
            ### ALG:
            ### define labelled intervals for xlocs
            ### extract locs from labelled intervals
            ### for each set of labels define its own locs

            i = np.arange(max(xlocs.max(), locs.max()))
            D = xlocs[:, 1] - xlocs[:, 0]
            i0 = xlocs[:, 0]

            mask = (i[:, None] >= i0) & (i[:, None] < (i0 + D))
            acts = mask.sum(1)

            i = np.argwhere(acts > 0).ravel()
            di = np.diff(i, append=i[-1] + 1)
            # Jumps in activations
            jacts = np.argwhere(di != 1).ravel()
            J = [[_j[0], _j[-1] + 1] for _j in np.split(i, jacts + 1)]
            J = np.asarray(J)

            # Jumps in original locs
            jlocs = np.unique(xlocs.ravel()).reshape(-1, 2)

            # Corrected locs
            clocs = np.empty((0, 2), dtype=int)
            Jmax = J.max()

            locs = np.repeat(locs, 2)[1:-1]

            locs2d = locs.reshape(-1, 2)
            # locs2d[:, 1] += 1

            for x0, x1 in locs2d:
                _jacts = jlocs[(jlocs > x0) & (jlocs < x1)]

                if np.any(x0 >= J[:, 0]):
                    _jacts = np.insert(_jacts, 0, x0)

                if np.all(x1 <= J[:, 1]) or x1 == Jmax:
                    _jacts = np.append(_jacts, x1)

                _clocs = np.repeat(_jacts, 2)[1:-1].reshape(-1, 2)
                clocs = np.concatenate((clocs, _clocs))

            # clocs[:, 1] -= 1
            locs = np.unique(clocs.ravel())

        # Events
        # events = list(zip(locs, signs))
        events = locs

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

        def _crop(x: PowerSample | np.ndarray, detectors):
            # if x.isfullyfit():

            detector0 = detectors[0]
            # FIXME if not isfullyfit
            locs0 = detector0(x)
            # locs0, _ = zip(*events0)

            if detector0.is_continuous():
                locs2d0 = []

                for i in range(1, len(locs0)):
                    a = locs0[i - 1]
                    b = locs0[i]
                    locs2d0.append((a, b))
            else:
                locs2d0 = [locs0[i:i + 2] for i in range(0, len(locs0), 2)]

            del locs0

            roi = []

            for a0, b0 in locs2d0:
                xab0 = x[a0:b0]

                if len(detectors) > 1 and len(xab0) > 1:
                    roi.extend(_crop(xab0, detectors[1:]))
                else:
                    roi.append(xab0)

            return roi

            # roi = []

            # for a, b in x.locs:
            #     # print(a, b)
            #     # print(x[a:b].locs, x[a:b].isfullyfit())
            #     roi.extend(_crop(x[a:b], detectors))

            # return roi

        if isinstance(x, DataSet):
            # TODO the same style for event detectors
            # TODO the same style for feature extraction
            roi = []

            for _x in tqdm(x.data):
                roi.extend(_crop(_x, self._detectors))

            dataset = x.new(roi)

            return dataset

        roi = _crop(x, self._detectors)

        return roi