from __future__ import annotations

import random
from typing import Any
import numpy as np
from ..data.entities import VI, I


class Augmentation:
    __compatibility__ = []

    def __init__(self, p: float = 0.5) -> None:
        self._proba = p

    def __call__(self, x):
        self._check_compatibility(x)

        if self.toss():
            x = self.augment()

        return x

    def _check_compatibility(self, x):
        if type(x) not in self.__compatibility__:
            raise ValueError

    def toss(self):
        probas = [1. - self._proba, self._proba]
        do = np.random.choice([False, True], p=probas)
        return do


class RandomPhaseShift(Augmentation):
    __compatibility__ = [I, VI]

    def __init__(self, a=0, b=1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._a = a
        self._b = b

    def augment(self, x):
        self._check_compatibility(x)

        if self.toss():
            n = np.random.choice(range(self._a, self._b)).item()
            x = x.roll(n)

        return x


class RandomNoise(Augmentation):

    def __init__(
        self,
        std: float = 0.01,
        rel: bool = True,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self._std = std
        self._rel = rel

    def augment(self, x):
        if self._rel:
            x = x * (1 + self._std * np.random.randn(*x.shape))
        else:
            x = x + self._std * np.random.randn(*x.shape)

        return x