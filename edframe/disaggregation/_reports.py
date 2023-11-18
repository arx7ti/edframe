from __future__ import annotations
from typing import Iterable, Callable

import torch
import inspect
import numpy as np
import pandas as pd

from .metrics import Metric


class Report:

    def __init__(self, metrics: Iterable['Metric' | 'Callable']) -> None:
        self._metrics = metrics
        # self._columns = ["Model"]

        # for metric in metrics:

        # self._columns.append(column)

    @staticmethod
    def _get_column_name(metric):
        if isinstance(metric, Metric):
            column = metric.verbose_name
        elif inspect.isclass(metric):
            column = metric.__name__
        else:
            column = metric.__class__.__name__

        return column

    def __call__(
        self,
        models: Iterable,
        X: np.ndarray,
        y: np.ndarray,
    ) -> pd.DataFrame:
        scores = []
        columns = ["Model"]

        for model in models:
            model_name = model.__class__.__name__
            model_scores = [model_name]

            if hasattr(model, "predict"):
                y_pred = model.predict(X)
            elif hasattr(model, "transform"):
                y_pred = model.transform(X)
            elif callable(model):
                if isinstance(model, torch.nn.Module):
                    model.eval()
                    with torch.no_grad():
                        # NOTE temporal
                        y_pred = torch.nn.functional.sigmoid(
                            model(X)).cpu().numpy()
                        y_pred = np.where(y_pred > 0.5, 1, 0)
                        y = y.cpu().numpy()
                else:
                    y_pred = model(X)
            else:
                raise ValueError("Model should be callable or have one of the methods "\
                                    "`predict` or `transform`")

            fetch_columns = len(columns) == 1

            for metric in self._metrics:
                # TODO wrapper for **kwargs
                score = metric(y, y_pred)

                if isinstance(score, dict):
                    model_scores.extend(list(score.values()))
                else:
                    model_scores.append(score)

                if fetch_columns:
                    column = self._get_column_name(metric)

                    if metric.is_componentwise():
                        for n in score.keys():
                            columns.append(f"{column}@{n}")
                    else:
                        columns.append(column)

            scores.append(model_scores)

        return pd.DataFrame(scores, columns=columns)