import math
import numpy as np
import pandas as pd
import itertools as it
import unittest as test
from functools import reduce

from edframe.data.entities import VI
from edframe.data.generators import make_hf_cycles, make_oscillations
from edframe.events._detectors import ThresholdBasedDetector

# Experiment setup
F0 = [40, 45, 49.8, 50, 51.123, 55, 59.45646, 60, 60.8123, 65]
FS = [1111, 2132, 4558, 5000, 4000, 9999, 10001]
# N_CYCLES = [1, 2, 3, 4, 5, 10, 11, 20, 23, 50, 57]
N_CYCLES = [10, 11, 15]
N_COMPONENTS = [1, 2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
N_CHOICES = 5
V_DC_OFFSET = 0
N_SIGNATURES = 10
N_SIGNATURES_PER_ITER = 1
ITERGRID = list(
    it.islice(it.product(F0, FS, N_CYCLES),
              N_SIGNATURES // N_SIGNATURES_PER_ITER))

rng = np.random.RandomState(RANDOM_STATE)

# rng = np.random.RandomState(None)


def init_signatures(with_components=False, noclip=False):
    signatures = []

    for f0, fs, n_cycles in ITERGRID:
        dt = n_cycles / f0
        output_size = math.ceil(fs / f0)
        h_loc = output_size // 2

        if with_components:
            n_signatures = max(N_COMPONENTS) * N_SIGNATURES_PER_ITER
        else:
            n_signatures = N_SIGNATURES // len(ITERGRID)

        I, y = make_oscillations(n_signatures,
                                 n_signatures,
                                 fs=fs,
                                 f0=f0,
                                 dt=dt,
                                 h_loc=h_loc)
        y = y.tolist()

        t = np.linspace(0, dt, I.shape[1])
        v = np.sin(2 * np.pi * f0 * t) + V_DC_OFFSET
        V = np.stack(np.repeat(v[None], len(I), axis=0))

        if len(I) // 2 == 0:
            dropsize = rng.choice(2)
        else:
            dropsize = len(I) // 2

        for j in rng.choice(len(y), size=dropsize, replace=False):
            y[j] = None

        if noclip:
            locs = [None] * len(y)
        else:
            a = np.random.randint(0, I.shape[1] - 1, size=len(I))
            b = np.random.randint(a + 1, I.shape[1], size=len(I))
            locs = np.stack((a, b), axis=1).tolist()

            for j in rng.choice(len(locs), size=dropsize, replace=False):
                locs[j] = None

            for j, locs_ in enumerate(locs):
                if locs_ is not None:
                    a, b = locs_
                    I[j][:a], I[j][b:] = 0, 0

        signatures_ = [
            VI(v,
               i,
               fs,
               f0,
               appliances=y_,
               locs=None if locs_ is None else [locs_])
            for v, i, y_, locs_ in zip(V, I, y, locs)
        ]

        if with_components:
            replace = N_SIGNATURES_PER_ITER > len(N_COMPONENTS)
            n_components = rng.choice(N_COMPONENTS,
                                      N_SIGNATURES_PER_ITER,
                                      replace=replace)

            for n_components_ in n_components:
                replace = n_components_ > len(signatures_)
                combs_ = rng.choice(len(signatures_),
                                    n_components_,
                                    replace=replace)
                rolls = rng.choice(I.shape[1], len(combs_))
                combs_ = [
                    signatures_[i].unitscale() for i, n in zip(combs_, rolls)
                ]
                signatures.append(sum(combs_))
        else:
            signatures.extend([
                VI(v,
                   i,
                   fs,
                   f0,
                   appliances=y_,
                   locs=None if locs_ is None else [locs_])
                for v, i, y_, locs_ in zip(V, I, y, locs)
            ])

    return signatures


class TestThresholdBasedDetector(test.TestCase):
    signatures = init_signatures(with_components=True, noclip=True)

    def test_threshold_based_detector(self):
        amp = lambda x: abs(x).max()
        tbd = ThresholdBasedDetector(amp, 1, 0)

        for vi in self.signatures:
            a = np.random.randint(0, vi.n_samples - 1)
            b = np.random.randint(a + 1, vi.n_samples)
            vi = vi[a:b]

            if vi.is_empty():
                continue

            xlocs = vi.split_locs(mode='running')
            locs = tbd.callback(vi)
            self.assertTrue(np.allclose(locs, xlocs),
                            msg=f'{a} {b}\n{xlocs}\n{locs}\n{vi.locs}')
