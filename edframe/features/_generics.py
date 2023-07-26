from __future__ import annotations
from sklearn.base import BaseEstimator
from types import FunctionType
from typing import Optional, Any
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA
from ..data.entities import DataSet
from inspect import isclass

import numpy as np
import pandas as pd


class Feature:

    @property
    def verbose_name(self):
        if self._verbose_name is None:
            return self.__class__.__name__

        return self._verbose_name

    @property
    def source_name(self) -> str:
        if self._source_name is None:
            return "values"

        return self._source_name

    @source_name.setter
    def source_name(self, source_name: str) -> None:
        self._source_name = source_name

    def __init__(
        self,
        source_name: Optional[str] = None,
        verbose_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._source_name = source_name
        self._verbose_name = verbose_name
        self._kwargs = kwargs

    def __call__(self, x) -> Any:
        return self.compute(x)

    def compute(self, x) -> pd.DataFrame:
        is_dataset = isinstance(x, DataSet)
        x = x.source(self.source_name)
        is_array = isinstance(x, np.ndarray)

        if is_array and is_dataset:
            # TODO axis support
            x = np.apply_along_axis(self.feature, axis=1, arr=x)
            do_iters = False
        elif is_array:
            x = [x]
            do_iters = True
        elif is_dataset:
            do_iters = True
        else:
            raise ValueError

        try:
            x = np.asarray([self.feature(x) for x in x]) if do_iters else x
        except ValueError:
            raise ValueError

        x = x[:, None] if len(x.shape) == 1 else x

        if len(x.shape) != 2:
            raise ValueError

        x = self._wrap(x)

        return x

    def _wrap(self, x) -> pd.DataFrame:

        if x.shape[1] < 2:
            columns = [self.verbose_name]
        else:
            columns = [f"{self.verbose_name}{i}" for i in range(x.shape[1])]

        df = pd.DataFrame(x, columns=columns)

        return df

    def feature(
        self,
        x,
    ) -> float | int | str | list[float | int
                                  | str] | np.ndarray:  # TODO to FeatureType
        raise NotImplementedError


class FeatureEstimator(Feature):

    def __init__(self,
                 source_name: str | None = None,
                 verbose_name: str | None = None,
                 **kwargs: Any) -> None:
        super().__init__(source_name, verbose_name, **kwargs)
        self.estimator = self.feature(**self._kwargs)

    def compute(self, x) -> np.ndarray:
        is_dataset = isinstance(x, DataSet)
        x = x.source(self.source_name)

        if not isinstance(x, np.ndarray):
            raise ValueError("Only array-like values are supported")

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:

            if is_dataset:
                self.estimator.fit(x)
            else:
                raise ValueError("The feature estimator was not fitted. "
                                 "Call this feature on a dataset first")

        x = self.estimator.transform(x if is_dataset else x[None])

        if len(x.shape) != 2:
            raise ValueError

        x = self._wrap(x)

        return x

    def feature(self, **kwargs) -> BaseEstimator:
        raise NotImplementedError


class CustomFeature:

    def __new__(
        cls,
        feature: callable | BaseEstimator,
        source_name: Optional[str] = None,
        verbose_name: Optional[str] = None,
        **kwargs,
    ) -> Feature | FeatureEstimator:
        estimator = {}
        msg = "Argument `feature` takes only function or BaseEstimator class"

        if isinstance(feature, BaseEstimator):
            # TODO make dynamic handling
            raise ValueError(msg)

        if isinstance(feature, FunctionType):

            def _feature(_, x):
                # TODO check arguments
                return feature(x)

            f = Feature.__new__(Feature)
            bound_method = _feature.__get__(f, Feature)

        elif isclass(feature):
            if not issubclass(feature, BaseEstimator):
                raise ValueError(msg)

            def _feature(_, **kwargs):
                return feature(**kwargs)

            f = FeatureEstimator.__new__(FeatureEstimator)
            bound_method = _feature.__get__(f, FeatureEstimator)
            estimator.update(estimator=_feature(feature, **kwargs))
        else:
            raise ValueError(msg)

        if verbose_name is None:
            verbose_name = feature.__name__

        f.__dict__.update(**estimator,
                          _verbose_name=verbose_name,
                          _source_name=source_name)
        setattr(f, "feature", bound_method)

        return f


class PrincipalComponents(FeatureEstimator):

    def feature(self, **kwargs) -> PCA:
        return PCA(**kwargs)