from __future__ import annotations

from tqdm import tqdm
from operator import add
from functools import reduce
from collections import defaultdict
from typing import Optional, Iterable

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
        # TODO concat datasets
        self._dataset = dataset
        self._domains = []
        labels = dataset.labels

        if np.any(labels.sum(1) > 1):
            raise ValueError(
                "Datasets of stand-alone appliances only are supported")

        for j in range(len(dataset.class_names)):
            domain = np.argwhere(labels[:, j] == 1).ravel().tolist()
            self._domains.append(domain)

        self._rng_state = random_state

        if random_state is not None:
            seed_shift = dataset.hash(int)
            modified_seed = random_state + seed_shift
        else:
            modified_seed = random_state

        self._rng = np.random.RandomState(modified_seed)

    @property
    def dataset(self):
        return self._dataset

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

    def sample(
        self,
        n_samples: int = 100,
        n_classes: int = 2,
        n_reps: int | tuple[int, int] | Iterable = None,
    ):
        if n_reps is None:
            n_reps_min, n_reps_max = 1, 1
        elif isinstance(n_reps, int):
            n_reps_min, n_reps_max = n_reps, n_reps
        elif isinstance(n_reps, tuple):
            n_reps_min, n_reps_max = n_reps
        elif isinstance(n_reps, Iterable):
            n_reps_min, n_reps_max = [], []
            assert len(n_reps) == self.dataset.n_classes

            for n in n_reps:
                if isinstance(n, int):
                    n_min, n_max = n, n
                elif isinstance(n, tuple):
                    n_min, n_max = n
                else:
                    raise ValueError

                n_reps_min.append(n_min)
                n_reps_max.append(n_max)

            n_reps_min = np.asarray(n_reps_min)
            n_reps_max = np.asarray(n_reps_max)
        else:
            raise ValueError

        Y = set()
        n_combs_max = math.comb(self.dataset.n_classes, n_classes)
        n_combs = min(n_samples, n_combs_max)
        class_indices = list(range(self.dataset.n_classes))

        while len(Y) < n_combs:
            comb = self._rng.choice(class_indices, n_classes, replace=False)
            comb = tuple(sorted(comb))

            if comb not in Y:
                Y.add(comb)

        Y = list(map(list, Y))
        # Repetitions of each appliance
        R = self._rng.randint(n_reps_min,
                              n_reps_max + 1,
                              size=(n_samples, self.dataset.n_classes))
        # Distribution of samples per combination
        p = n_samples // len(Y)
        p = np.asarray([p] * len(Y))
        c_size = n_samples % len(Y)
        # Correction for `p`
        c = self._rng.choice(range(len(p)), size=c_size, replace=False)
        p[c] += 1
        # Final set of indices
        I = set()

        for y, r, pi in zip(Y, R, p):
            dj = [(self.domains[j], j) for j in y]
            n_max = reduce(
                lambda x, y: x * y,
                # TODO check if r[j] is correct sampling
                [math.comb(len(djk) + r[j] - 1, r[j]) for djk, j in dj])
            Ii = set()

            while len(Ii) < min(pi, n_max, sys.maxsize):
                sample = []

                for djk, j in dj:
                    sample.extend(
                        self._rng.choice(djk, size=r[j], replace=True))

                sample = tuple(sorted(sample))

                if sample not in Ii:
                    Ii.add(sample)

            I |= Ii

        loss = n_samples - len(I)

        if loss > 0:
            warnings.warn('%d samples were not obtained due to '
                          'combinatorial limit.' % loss)

        I = list(map(list, I))

        return I

    def schedule(
        self,
        n_samples: int,
        n_classes: int,
        n_rolls: int = 1,
    ) -> np.ndarray:
        window_size = self.dataset.values.shape[1]  # TODO if 2d

        if n_rolls > 0:
            rolls = self._rng.randint(-window_size + 1,
                                      window_size,
                                      size=(n_samples, n_rolls, n_classes - 1))
        else:
            rolls = np.zeros((n_samples, 1, n_classes - 1))

        return rolls

    def compose(self, idxs, rolls, keep_components: bool = False):
        raise NotImplementedError

    def make_samples(
        self,
        n_samples: int = 100,
        n_classes: int = 2,
        n_reps: np.ndarray = None,
        n_rolls: int = 1,
        keep_components: bool = False,
    ):
        samples = []
        I = self.sample(n_samples=n_samples,
                        n_classes=n_classes,
                        n_reps=n_reps)

        R = self.schedule(n_samples=len(I),
                          n_classes=n_classes,
                          n_rolls=n_rolls)

        for i, r in tqdm(zip(I, R), total=len(I)):
            samples.extend(self.compose(i, r, keep_components=keep_components))

        dataset = self.dataset.new(samples)

        return dataset


class HComposer(Composer):

    def compose(self, idxs, rolls, keep_components: bool = False):
        samples = []
        components = [self.dataset[i] for i in idxs]
        sample0 = components.pop(0)

        for r in rolls:
            sample = reduce(add, [x.roll(rx) for x, rx in zip(components, r)])
            samples.append(sample0 + sample)

        return samples