from __future__ import annotations
from typing import Iterable, Callable

import inspect
import numpy as np
import pandas as pd

from .metrics import Metric


class Report:

    def __init__(self, metrics: Iterable['Metric' | 'Callable']) -> None:
        self._metrics = metrics
        self._columns = ["Model"]

        for metric in metrics:
            if isinstance(metric, Metric):
                column = metric.verbose_name
            elif inspect.isclass(metric):
                column = metric.__name__
            else:
                column = metric.__class__.__name__

            self._columns.append(column)

    def __call__(
        self,
        models: Iterable,
        X: np.ndarray,
        y: np.ndarray,
    ) -> pd.DataFrame:
        metrics = []

        for model in models:
            model_name = model.__class__.__name__
            model_metrics = [model_name]

            if hasattr(model, "predict"):
                y_pred = model.predict(X)
            elif hasattr(model, "transform"):
                y_pred = model.transform(X)
            elif callable(model):
                y_pred = model(X)
            else:
                raise ValueError("Model should be callable or have one of the methods "\
                                    "`predict` or `transform`")

            for metric in self._metrics:
                # TODO wrapper for **kwargs
                metric = metric(y, y_pred)
                model_metrics.append(metric)

            metrics.append(model_metrics)

        return pd.DataFrame(metrics, columns=self._columns)