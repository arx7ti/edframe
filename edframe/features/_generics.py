from __future__ import annotations

from typing import Callable, Union, Self, Any
from numbers import Number
from beartype.typing import Iterable
from sklearn.base import BaseEstimator
from beartype import abby

import numpy as np

from ..data.entities import PowerSample, DataSet


class Feature:
    transform: Callable | None = None
    estimator: BaseEstimator | None = None
    __numerical__: bool = True
    __vector__: bool = False
    __verbose_name__: str | None = None

    @property
    def size(self):
        if self.is_vector():
            if self._size is None:
                raise AttributeError("Not called")
            return self._size
        raise AttributeError

    @property
    def verbose_name(self: Self) -> str:
        if self.__verbose_name__ is None and self.transform is not None:
            name = self.transform.__name__
        elif self.__verbose_name__ is None and self.estimator is not None:
            name = self.estimator.__class__.__name__
        else:
            name = self.__verbose_name__

        # In case if imported from the submodule we keep only final name
        # e.g. import <package>.<extractor_name> as <extractor_name>
        name = name.split('.')[-1]

        return name

    @classmethod
    def _to_array(cls, x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x

        if isinstance(x, Iterable) and not isinstance(x, (list, tuple)):
            x = list(x)

        if isinstance(x, Iterable):
            xlist = []
            for _x in x:
                if isinstance(_x, (list, tuple)):
                    xlist.append(_x)
                else:
                    _xlist = [_x for _x in _x]

                    if any(isinstance(__x, Iterable) for __x in _xlist):
                        raise ValueError

                    xlist.append(_xlist)
        else:
            xlist = x

        return np.asarray(xlist)

    def __init__(self, **kwargs):

        if self.is_estimator() and self.is_transform():
            raise ValueError

        if not (self.is_estimator() or self.is_transform()):
            raise ValueError

        if self.is_estimator():
            self.estimator = self.estimator(**kwargs)
            self._fitted = False
        else:
            self.transform = self.__class__.transform

        if self.is_vector():
            self._size = None

    def __call__(
        self,
        x: PowerSample | DataSet | np.ndarray,
        *args,
        **kwargs,
    ) -> Union[int, float, str, np.ndarray]:
        if isinstance(x, PowerSample | DataSet):
            x = x.values
        elif not isinstance(x, np.ndarray):
            raise ValueError

        if abby.is_bearable(x, list[PowerSample]):
            return self._extract_from_dataset(x, *args, **kwargs)

        is_dataset = kwargs.get("is_dataset", len(x.shape) > 1)

        if not isinstance(x, np.ndarray):
            raise ValueError

        self.check_fn(x, *args, **kwargs)

        if self.is_estimator():
            # TODO check if fitted estimator called over single sample
            if not self.is_fitted() and not is_dataset:
                raise ValueError("The feature estimator was not fitted. "
                                 "Call this feature on a dataset first")

            refit = kwargs.pop("refit", False)

            if not self.is_fitted() or refit:
                self.fit(x, **kwargs)
                self._fitted = True

            x = self.estimator.transform(x if is_dataset else [x], *args,
                                         **kwargs)

            x = x if is_dataset else x[0]
        elif is_dataset:
            shape = list(x.shape)
            shape[-1] = 1
            x = np.apply_along_axis(
                self.transform,
                axis=-1,  # TODO axis support
                arr=x,
                *args,
                **kwargs)
            x = x.reshape(*shape)
        else:
            x = self.transform(x, *args, **kwargs)

        # TODO to np.ndarray
        x = self._to_array(x)

        if self.is_vector():
            if (is_dataset and len(x.shape) != 2)\
                or (not is_dataset and len(x.shape) != 1):
                raise ValueError

            self._size = x.shape[1] if is_dataset else x.shape[0]

        if self.is_numerical() and not np.issubdtype(x.dtype, np.number):
            raise ValueError

        return x

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)

    def _extract_from_dataset(self, x, *args, **kwargs):
        values = []

        for sample in x:
            values.append(self(sample, *args, **kwargs))

        return values

    def is_numerical(self):
        return self.__numerical__

    def is_vector(self):
        return self.__vector__

    def is_estimator(self):
        return self.estimator is not None

    def is_transform(self):
        return self.transform is not None

    def is_fitted(self):
        if self.is_estimator():
            return self._fitted
        else:
            raise AttributeError

    def fit(self, x, **kwargs):
        if self.is_estimator():
            self.estimator.fit(x, **kwargs)
            self._fitted = True
        else:
            raise AttributeError

    def check_fn(self, x: np.ndarray, *args, **kwargs):
        return None