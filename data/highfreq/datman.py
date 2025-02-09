import math
import numpy as np
import types
import os
import torch
import pandas as pd
import random

from scipy.signal import resample
from tqdm import tqdm
import warnings
from ...fitps import FITPS

from .entity import HighFreqSample


class HighFreqDataset:

    @property
    def fs(self):
        return sorted(set(sample.fs for sample in self.data))

    @property
    def f0(self):
        return sorted(set(sample.f0 for sample in self.data))

    @property
    def devices(self):
        return sorted(set(sample.devices for sample in self.data))

    @property
    def brands(self):
        pass

    def is_homogeneous(self):
        return len(set([x.i.shape for x in self.data])) == 1

    def v(self):
        return self.data[0].v

    def i(self):
        if self.is_homogeneous():
            return np.concatenate([x.i for x in self.data])

        raise AttributeError

    def __len__(self):
        return len(self.data)

    def map(self, fn):
        data = list(map(fn, self.data))

        return HighFreqDataset(data)

    def _check_if_invariant(self):
        if not all([sample.is_invariant() for sample in self.data]):
            raise ValueError

    def split_by_cycles(self, n_cycles):
        self._check_if_invariant()

        data = []

        for sample in self.data:
            agg_fmt = True

            if isinstance(app, str):
                app = [app]
                agg_fmt = False

            for j in range(0, sample.n_cycles, n_cycles):
                vj = sample.v[..., j:j + n_cycles, :]
                ij = sample.i[..., j:j + n_cycles, :]

                if len(vj) == n_cycles:
                    if sample.locs is not None:
                        _locs = []
                        _devices = []

                        for device, (on, off) in zip(app, sample.locs):
                            on = max(on - j * n_cycles * sample.i.shape[-1], 0)
                            off = max(
                                min(
                                    off - on // sample.i.shape[-1] *
                                    sample.i.shape[-1],
                                    n_cycles * sample.i.shape[-1] - 1), 0)

                            if off == on:
                                continue

                            _devices.append(device)
                            _locs.append([on, off])

                        if not agg_fmt:
                            _devices = _devices[0]

                        data.append(
                            HighFreqSample(vj, ij, sample.fs, sample.f0,
                                           _devices, _locs))
                    else:
                        if not agg_fmt:
                            app = app[0]

                        data.append(
                            HighFreqSample(vj, ij, sample.fs, sample.f0, app))

        return HighFreqDataset(data)

    def split_by_ncomponents(self):
        data = []

        for sample in tqdm(self.data):
            Q = np.zeros((len(sample.devices), sample.i.shape[1]), dtype=int)

            for j, (a, b) in enumerate(sample.locs):
                a = a // sample.i.shape[1]
                b = b // sample.i.shape[1] + 1
                Q[j, a:b] += 1

            n_components = Q.sum(0)
            dn = np.diff(n_components, prepend=0, append=0)
            chpts = (dn != 0).nonzero()[0].ravel()

            a = chpts[0]

            for b in chpts[1:]:
                ids = Q[:, a:b].sum(1).nonzero()[0]
                assert len(ids) > 0

                _locs = []
                _devices = []

                for j in ids:
                    device = sample.devices[j]
                    on, off = sample.locs[j]

                    on = max(0, on - a * sample.v.shape[-1])
                    off = max(
                        0,
                        min((b - a) * sample.v.shape[-1] - 1,
                            off - a * sample.shape[-1]))
                    assert off >= on

                    if off == on:
                        continue

                    _devices.append(device)
                    _locs.append([on, off])

                data.append(
                    HighFreqSample(sample.v[a:b], sample.i[a:b], sample.fs,
                                   sample.f0, _devices, _locs))
                a = b

        return data

    def to_invariant(
        self,
        tol=20,
        buff_size=1.2,
        progress_bar=True,
    ):
        data = []

        for sample in tqdm(self.data, disable=not progress_bar):
            cycle_size = int(sample.fs / sample.f0)
            fitps = FITPS(cycle_size, int(buff_size * cycle_size), tol)
            v, i = fitps.transform(sample.v, sample.i)
            sample = HighFreqSample(v, i, sample.fs, sample.f0, sample.devices,
                                    sample.locs)
            data.append(sample)

        return HighFreqDataset(data)

    def submetered(self):
        data = list(filter(lambda sample: sample.n_components == 1, self.data))

        return HighFreqDataset(data)

    def aggregated(self):
        data = list(filter(lambda sample: sample.n_components > 1, self.data))

        return HighFreqDataset(data)

    def drop_low_power(self, thresh=10):
        pass

    def drop_correlated(self, thresh=0.001, metric='cosine'):
        pass

    def drop_rare(self, thresh=0.01):
        pass

    def train_test(self, test_size=0.3, groupby=None):
        pass

    def rename(self, naming):
        pass

    def random(self, random_seed=None):
        random.seed(random_seed)
        idx = random.sample(range(len(self)))

        return self.data[idx]

    def similarity(self, dataset, metric='cosine'):
        I1, I2 = self.i, dataset.i

        if metric == 'cosine':
            I1n = I1 / np.linalg.norm(I1, axis=1, keepdims=True)
            I2n = I2 / np.linalg.norm(I2, axis=1, keepdims=True)
            scores = I1n @ I2n.T
        else:
            raise ValueError

        return scores

    def filter(self, **kwargs):
        data = self.data

        for k, v in kwargs.items():
            if k == 'devices':
                data = list(filter(lambda sample: sample.devices == v, data))
            elif k == 'devices__in':
                v = set(v)
                data = list(
                    filter(lambda sample: v.issubset(set(sample.devices)),
                           data))
            elif k == 'brands':
                data = list(filter(lambda sample: sample.brands == v, data))
            elif k == 'brands__in':
                v = set(v)
                data = list(
                    filter(lambda sample: v.issubset(set(sample.brands)),
                           data))
            elif k == 'power__leq':
                data = list(
                    filter(lambda sample: sample.active_power <= v, data))
            elif k == 'power__le':
                data = list(
                    filter(lambda sample: sample.active_power < v, data))
            elif k == 'power__geq':
                data = list(
                    filter(lambda sample: sample.active_power >= v, data))
            elif k == 'power__ge':
                data = list(
                    filter(lambda sample: sample.active_power > v, data))

        return HighFreqDataset(data)

    def count_components(self):
        return [sample.n_components for sample in self.data]

    def groupby(self, method='devices'):
        data = dict()

        for id, sample in enumerate(self.data):
            if method in ['devices', 'devices__id']:
                key = sample.devices
            elif method in ['brands', 'brands__id']:
                key = sample.brands
            else:
                raise ValueError

            key = tuple(sorted(key))

            if key not in data:
                data[key] = []

            if 'id' in method:
                data[key].append(id)
            else:
                data[key].append(sample)

        return data

    def transients(self, thresh=1e-4):
        pass

    def steady_states(self, thresh=1e-4):
        pass

    def union(self, dataset):
        pass

    def __getitem__(self, idx):
        pass

    def _check_if_read(self):
        if self.data is None:
            raise ValueError
