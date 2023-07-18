from __future__ import annotations

from typing import Optional
from functools import reduce
from collections import defaultdict

import os
import sys
import math
import warnings
import numpy as np

from ..entities import DataSet


class Composer:

    def __init__(
        self,
        dataset: DataSet,
        random_state: Optional[int] = None,
    ) -> None:
        self._dataset = dataset 
        # self._classes = np.unique(y)
        self._domains = {}
        for l in self._classes:
            domain = np.argwhere(y == l).ravel().tolist()
            self._domains[l] = domain
        self._rng_state = random_state
        if random_state is not None:
            seed_shift = round(np.sum(np.std(X, axis=1)))
            modified_seed = random_state + seed_shift
        else:
            modified_seed = random_state
        self._rng = np.random.RandomState(modified_seed)

    @property
    def classes(self):
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def domains(self):
        return self._domains

    def _save_sample(
        self,
        i: int,
        y: np.ndarray,
        x: np.ndarray,
    ) -> None:
        data = defaultdict()
        data['y'] = y
        data['x'] = x
        file_name = '%d-%d' % (y.shape[0], i + 1)
        file_path = os.path.join(self.dir_path, file_name)
        np.save(file_path, data)

    def make_index_set(
        self,
        n_samples: int = 100,
        n_classes: int = 2,
        min_freqs: np.ndarray = None,
        max_freqs: np.ndarray = None,
    ):
        n_combs_max = math.comb(self.n_classes, n_classes)
        n_combs = min(n_samples, n_combs_max)
        Y = set()
        while len(Y) < n_combs:
            comb = self._rng.choice(self.classes, n_classes, replace=False)
            comb = tuple(sorted(comb))
            if comb not in Y:
                Y.add(comb)
        Y = list(map(list, Y))

        if min_freqs is None:
            min_freqs = np.ones(n_classes)
        if max_freqs is None:
            max_freqs = np.ones(n_classes)

        F = self._rng.randint(min_freqs,
                              max_freqs + 1,
                              size=(n_samples, n_classes))

        n_reps = n_samples // len(Y)
        n_rem = n_samples % len(Y)
        rep_distr = np.asarray([n_reps] * len(Y))
        to_rep = self._rng.choice(range(len(rep_distr)),
                                  size=n_rem,
                                  replace=False)
        rep_distr[to_rep] += 1

        I = set()
        for i, n_rep in enumerate(rep_distr):
            y = Y[i]
            f = F[i]
            D = [self.domains[l] for l in y]
            n_max = reduce(
                lambda x, y: x * y,
                [math.comb(len(D[i]) + f[i] - 1, f[i]) for i in range(len(D))])

            Ii = set()
            while len(Ii) < min(n_rep, min(n_max, sys.maxsize)):
                sample = []
                for domain, freq in zip(D, f):
                    sample += self._rng.choice(domain, size=freq, replace=True).\
                                tolist()
                sample = tuple(sorted(sample))
                if sample not in Ii:
                    Ii.add(sample)
            I = I.union(Ii)

        dn = n_samples - len(I)
        if dn > 0:
            warnings.warn('%d samples were not obtained due to '
                          'combinatorial limit.' % dn)

        I = list(map(list, I))

        return I

    def compose_single(self, Ii, keep_components: bool = False):
        Ii = np.asarray(Ii)
        x = self._X[Ii]
        y = self.get_labels(Ii)
        if not keep_components:
            x = np.sum(x, axis=0)
            y = np.unique(y)
        return x, y

    def get_labels(self, Ii):
        y = self._y[np.asarray(Ii)]
        return y

    def make_samples(
        self,
        n_samples: int = 100,
        n_classes: int = 2,
        min_freqs: np.ndarray = None,
        max_freqs: np.ndarray = None,
        dir_path: Optional[str] = None,
        keep_components: bool = False,
        n_jobs: int = 1,
    ):
        I = self.make_index_set(n_samples=n_samples,
                                n_classes=n_classes,
                                min_freqs=min_freqs,
                                max_freqs=max_freqs)
        if dir_path is None:
            X = []
            Y = []
            for Ii in I:
                x, y = self.compose_single(Ii, keep_components=keep_components)
                X.append(x)
                Y.append(y)
        else:
            # TODO parallel
            pass
        return X, Y