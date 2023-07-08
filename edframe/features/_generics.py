from __future__ import annotations

from numbers import Number
from beartype import abby
from typing import Callable
from sklearn.exceptions import NotFittedError

import inspect
import numpy as np

from ..data.entities import PowerSample


class Feature:
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
            raise AttributeError
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
                verbose_name = fn.__class__.__name__
            else:
                verbose_name = fn.__name__

        self._verbose_name = verbose_name
        self._numerical = numerical
        self._vector = vector
        self._check_fn = check_fn

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

    def __call__(self, x: PowerSample | list[PowerSample] | np.ndarray, *args,
                 **kwargs):

        if self._check_fn is not None:
            self._check_fn(x)

        if self.is_estimator():
            if abby.is_bearable(x, list[PowerSample]):
                self.estimator.fit(x, *args, **kwargs)
            else:
                x = [x]
            transform = self.estimator.transform
        else:
            transform = self.transform

        try:
            x = transform(x, *args, **kwargs)
        except NotFittedError:
            raise ValueError("The feature estimator was not fitted. "
                             "Call this feature on a dataset first")

        if self.is_estimator():
            x = x[0]

        if self.is_numerical() and not isinstance(x, Number):
            raise AttributeError

        if self.is_vector() and not isinstance(x, Iterable):
            raise AttributeError

        if self.is_vector():
            x = [x for x in x]

        return x
