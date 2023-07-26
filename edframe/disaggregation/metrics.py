from __future__ import annotations

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from typing import Any

import numpy as np


class Metric:
    @property
    def verbose_name(self):
        if self._verbose_name is None:
            return self.__class__.__name__

        return self._verbose_name

    def __init__(
        self,
        componentwise: bool = False,
        verbose_name: str = None,
        **kwargs: Any,
    ) -> None:
        self._componentwise = componentwise
        self._verbose_name = verbose_name
        self._kwargs = kwargs

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        return self.compute(y_true, y_pred)

    def is_componentwise(self):
        return self._componentwise

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray):
        if self.is_componentwise():
            scores = {}
            n_true_comps = np.unique(y_true.sum(1))
            n_pred_comps = np.unique(y_pred.sum(1))
            n_comps = np.unique(np.concatenate((n_true_comps, n_pred_comps)))

            for n in n_comps:
                mask = y_true.sum(1) == n

                if np.any(mask):
                    score = self.metric(y_true[mask], y_pred[mask])
                    scores.update({n: score})

            return scores

        score = self.metric(y_true, y_pred)

        return score

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        # TODO the same style for features, events
        raise NotImplementedError


class CustomMetric(Metric):
    @property
    def verbose_name(self):
        if self._verbose_name is None:
            try:
                name = self._metric.__name__
            except AttributeError:
                name = self._metric.__class__.__name__

            return name

        return self._verbose_name

    def __init__(
        self,
        metric: callable,
        y_true_arg=None,
        y_pred_arg=None,
        componentwise: bool = False,
        verbose_name: str = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(componentwise=componentwise,
                         verbose_name=verbose_name,
                         **kwargs)
        self._metric = metric
        self._y_true_arg = y_true_arg
        self._y_pred_arg = y_pred_arg

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        if self._y_true_arg is not None and self._y_pred_arg is not None:
            args = ()
            kwargs = {self._y_true_arg: y_true, self._y_pred_arg: y_pred}
        else:
            args = (y_true, y_pred)
            kwargs = {}

        score = self._metric(*args, **kwargs, **self._kwargs)

        return score


class F1(Metric):
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        return f1_score(y_true, y_pred, **self._kwargs)


class Accuracy(Metric):
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        return accuracy_score(y_true, y_pred, **self._kwargs)


class Precision(Metric):
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        return precision_score(y_true, y_pred, **self._kwargs)


class Recall(Metric):
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        return recall_score(y_true, y_pred, **self._kwargs)
