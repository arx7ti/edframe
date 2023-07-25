from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from ..data.entities import DataSet

import numpy as np


class Disaggregator:

    def __init__(self) -> None:
        self.model = None
        self._from_features = None

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def tune(self):
        raise NotImplementedError

    def disaggregate(self, d: DataSet):
        # TODO if single sample
        if self._from_features:
            x = d.features.values
        else:
            x = d.values

        # TODO validation
        y = self.model.transform(x)

        return y


class Classifier(Disaggregator):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = self.classifier(**kwargs)

    def classifier(self, **kwargs):
        raise NotImplementedError


class Regressor(Disaggregator):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = self.regressor(**kwargs)

    def regressor(self, **kwargs):
        raise NotImplementedError


class Ranker(Disaggregator):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = self.ranker(**kwargs)

    def ranker(self, **kwargs):
        raise NotImplementedError


class RFAppIdentifier(Classifier):

    def classifier(self, **kwargs):
        return RandomForestClassifier(**kwargs)