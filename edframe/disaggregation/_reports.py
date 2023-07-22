from __future__ import annotations
from typing import Iterable, Callable

import inspect
import numpy as np
import pandas as pd

from ..data.entities import DataSet
from ._generics import Disaggregator
from ._metrics import Metric


class Report:

    def __init__(self, metrics: Iterable['Metric' | 'Callable']) -> None:
        self._metrics = metrics
        self._columns = []

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
        models: Iterable['Disaggregator'],
        dataset: DataSet,
    ) -> pd.DataFrame:
        metrics = []
        y_true = dataset.labels

        for model in models:
            model_metrics = []

            for metric in self._metrics:
                y_pred = model.disaggregate(dataset)
                # TODO wrapper for **kwargs
                metric = metric(y_true, y_pred)
                model_metrics.append(metric)

            metrics.append(model_metrics)

        return pd.DataFrame(metrics, columns=self._columns)
