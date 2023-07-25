from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from ..data.entities import DataSet

import numpy as np


class Disaggregator:

    def __init__(self, problem_type: str = "classification", **kwargs) -> None:
        self._problem_type = problem_type
        self.model = self.disaggregator(**kwargs)
        self._from_features = None

    def disaggregator(self, **kwargs):
        raise NotImplementedError

    def fit(self, d: DataSet, from_features: bool = True) -> None:
        # TODO if torch
        if not hasattr(self.model, "fit"):
            raise AttributeError("Disaggregator must have a fit attribute")

        self._from_features = from_features

        if from_features:
            x = d.features.values
        else:
            x = d.values

        y = d.labels

        if self._problem_type in ["classification", "ranking"] and\
            not np.issubdtype(y.dtype, int):

            raise ValueError

        if self._problem_type == "regression" and\
            not np.issubdtype(y.dtype, float):

            raise ValueError

        self.model.fit(x, y)

    def disaggregate(self, d: DataSet):
        # TODO if single sample
        if self._from_features:
            x = d.features.values
        else:
            x = d.values

        # TODO validation
        y = self.model.transform(x)

        return y


class RFAppIdentifier(Disaggregator):

    def disaggregator(self, **kwargs):
        return RandomForestClassifier(**kwargs)