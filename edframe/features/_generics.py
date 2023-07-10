from __future__ import annotations

from beartype import abby
from numbers import Number
from functools import partial
from typing import Callable, Union
from beartype.typing import Iterable

import inspect
import numpy as np

from ..data.entities import PowerSample


class Feature:
    # TODO divide into different classes e.g. EstimatorFeature, Feature

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def estimator(self):
        if self.is_estimator():
            return self._fn
        else:
            raise AttributeError

    @property
    def transform(self):
        if self.is_estimator():
            return self._fn.transform
        else:
            return self._fn

    def __init__(
        self,
        fn,
        verbose_name: str = None,
        numerical=True,
        vector=False,
        check_fn: Callable = None,
    ):
        self._fn = fn

        if verbose_name is None:
            if inspect.isclass(fn):
                verbose_name = fn.__name__
            else:
                verbose_name = fn.__class__.__name__

        self._verbose_name = verbose_name
        self._numerical = numerical
        self._vector = vector
        self._check_fn = check_fn

        if self.is_estimator():
            self._fitted = False

        if self.is_vector():
            self._numel = None

    def __str__(self):
        return self.verbose_name

    def __repr__(self):
        return str(self)

    def is_numerical(self):
        return self._numerical

    def is_vector(self):
        return self._vector

    def is_estimator(self):
        return hasattr(self._fn, "fit") and hasattr(self._fn, "transform")

    def is_fitted(self):
        if self.is_estimator():
            return self._fitted
        else:
            raise AttributeError

    def fit(self, x, **kwargs):
        if self.is_fitted():
            self.reset()

        self._fn = self._fn(**kwargs)
        self._fn.fit(x)
        self._fitted = True

    def reset(self):
        if not self.is_estimator():
            raise AttributeError
        elif not inspect.isclass(self._fn):
            self._fn = self._fn.__class__
        self._fitted = False

    def numel(self):
        if self.is_vector():
            if self._numel is None:
                raise AttributeError("Not fitted")
            else:
                return self._numel
        else:
            raise AttributeError

    def __call__(
        self,
        x: PowerSample | np.ndarray,
        *args,
        **kwargs,
    ) -> Union[int, float, str, np.ndarray]:

        if self._check_fn is not None:
            self._check_fn(x, *args, **kwargs)

        if isinstance(x, PowerSample):
            x = x.values

        single = len(x.shape) == 1

        if self.is_estimator():

            if not self.is_fitted():
                if single:
                    raise ValueError("The feature estimator was not fitted. "
                                     "Call this feature on a dataset first")
                self.fit(x, **kwargs)

            x = self.transform(x)
        else:
            if single:
                x = self.transform(x, **kwargs)
            else:
                x = np.apply_along_axis(partial(self.transform, **kwargs),
                                        axis=-1,
                                        arr=x)

        if single:
            if self.is_estimator():
                x = x[0]

            if self.is_vector() and not isinstance(x, Iterable):
                raise ValueError

            if self.is_vector():
                self._numel = len(x)

            x = list(x)
        else:
            if isinstance(x, np.ndarray):
                x = x.tolist()

            if self.is_vector() and not isinstance(x[0], Iterable):
                raise ValueError

            if self.is_vector():
                self._numel = len(x[0])

        if self.is_numerical() and\
            not abby.is_bearable(x, list[list[Number]]|list[Number]| Number):
            raise ValueError

        return x
