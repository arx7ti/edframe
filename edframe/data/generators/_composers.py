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


def _distribute_samples(n_samples, n_appliances, n_modes_per_appliance):
    if not isinstance(n_modes_per_appliance, list | np.ndarray):
        n_modes_per_appliance = np.asarray([n_modes_per_appliance] *
                                           n_appliances)

    n_modes_per_appliance = np.asarray(n_modes_per_appliance)
    assert n_appliances == len(n_modes_per_appliance)
    assert np.all(n_modes_per_appliance > 0)
    n_clusters = n_modes_per_appliance.sum()

    if not isinstance(n_samples, list | np.ndarray):
        n_spc = n_samples // n_clusters
        n_spc = n_spc * np.ones(n_clusters, dtype=int)
        n_spc[np.arange(n_samples - n_spc.sum()) % n_clusters] += 1
        assert n_spc.sum() == n_samples
    else:
        n_samples = np.asarray(n_samples)
        n_spc = n_samples // n_modes_per_appliance
        n_spc = np.repeat(n_spc, n_modes_per_appliance)
        n_spc[np.arange(n_samples.sum() - n_spc.sum()) % n_clusters] += 1
        assert len(n_spc) == n_clusters

    # Class indices with regards to the clusters
    class_for_cluster = np.repeat(np.arange(n_appliances),
                                  n_modes_per_appliance)

    return n_spc, class_for_cluster


class Composer:

    def __init__(self, dataset, random_state: Optional[int] = None) -> None:
        if dataset.is_multilabel():
            raise ValueError(
                "Only dataset of individual appliances is supported")

        self._dataset = dataset
        self.D = []
        Y = dataset.targets

        for k in range(dataset.n_appliances):
            domain = np.argwhere(Y[:, k] == 1).ravel()
            self.D.append(domain)

        self._rng = np.random.RandomState(random_state)

    # def _save_sample(
    #     self,
    #     i: int,
    #     y: np.ndarray,
    #     x: np.ndarray,
    # ) -> None:
    #     data = defaultdict()
    #     data['y'] = y
    #     data['x'] = x
    #     file_name = '%d-%d' % (y.shape[0], i + 1)
    #     file_path = os.path.join(self.dir_path, file_name)
    #     np.save(file_path, data)

    def _find_appliances(self, n_signatures, n_appliances, replace=False):
        Y = set()
        n_combs_max = math.comb(len(self.D), n_appliances)
        n_combs = min(n_signatures, n_combs_max)
        class_indices = np.arange(len(self.D))

        while len(Y) < n_combs:
            comb = self._rng.choice(class_indices, n_appliances, replace)
            comb = tuple(sorted(comb))

            if comb not in Y:
                Y.add(comb)

        return list(Y)

    def _find_signatures(self, n_signatures, Y, replace):
        S = set()
        N, _ = _distribute_samples(n_signatures, len(Y), 1)

        for y, n in zip(Y, N):
            Si = set()
            Di = [(self.D[j], j) for j in y]
            r = dict(zip(*np.unique(y, return_counts=True)))

            if replace:
                n_max = sys.maxsize
            else:
                n_max = reduce(
                    lambda x, y: x * y,
                    [math.comb(len(d) + r[k] - 1, r[k]) for d, k in Di])

            while len(Si) < min(n, n_max, sys.maxsize):
                sample = []

                for d, k in Di:
                    repeat = r[k] > len(d) or replace
                    sample.extend(d[self._rng.choice(len(d), r[k], repeat)])

                sample = tuple(sorted(sample))

                if sample not in Si:
                    Si.add(sample)

            S |= Si

        return list(map(list, S))

    def _compose(self, S):
        Snew = []

        for s in S:
            s = sum(self._dataset[s].signatures)
            Snew.append(s)

        return Snew

    def _random_roll(self, S, max_roll=0.9):
        assert max_roll != 1.0

        Snew = []
        max_roll = int(max_roll * (self._dataset.n_samples - 1))
        a, b = -max_roll, max_roll

        for s in S:
            n = self._rng.randint(a, b + 1, size=s.n_appliances)
            s = s.roll(n)
            Snew.append(s)

        return Snew

    def make_signatures(
        self,
        n_signatures: int = 100,
        n_appliances: int = 2,
        replace: bool = False,
        random_roll: bool = False,
        **kwargs,
    ):
        if len(self.D) < n_appliances and not replace:
            raise ValueError

        Y = self._find_appliances(n_signatures, n_appliances, replace)
        S = self._find_signatures(n_signatures, Y, random_roll)
        S = self._compose(S)

        if random_roll:
            S = self._random_roll(S, kwargs.get('max_roll', .9))

        S = self._dataset.new(S, safe_mode=self._dataset._safe_mode)

        dn = n_signatures - len(S)

        if dn > 0:
            # FIXME not correct
            warnings.warn(f'{dn} samples were not obtained due to '
                          'combinatorial limit.')

        return S
