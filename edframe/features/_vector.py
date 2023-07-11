from ._generics import Feature
from ..signals._vector import fft_amplitudes
from sklearn.decomposition import PCA, FastICA

import numpy as np


class FourierAmplitudes(Feature):
    transform = fft_amplitudes
    __vector__ = True

    def check_fn(self, x: np.ndarray, *args, **kwargs):
        if len(x.shape) > 1:
            raise ValueError


class PrincipalComponents(Feature):
    estimator = PCA
    __vector__ = True
    __verbose_name__ = "PC"


class IndependentComponents(Feature):
    estimator = FastICA
    __vector__ = True
    __verbose_name__ = "IC"

    @property
    def mixing(self) -> np.ndarray:
        return self.estimator.mixing_

    @property
    def mean(self) -> np.ndarray | None:
        if self.estimator.whiten:
            return self.estimator.mean_
        else:
            return None
