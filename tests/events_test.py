import math
import numpy as np
import pandas as pd
import itertools as it
import unittest as test
from functools import reduce

from edframe.data.entities import VI
from edframe.data.generators import make_hf_cycles, make_oscillations
from edframe.features import spectral_centroid
from edframe.events._detectors import ThresholdBasedDetector, DerivativeBasedDetector

# Experiment setup
F0 = [40, 45, 49.8, 50, 51.123, 55, 59.45646, 60, 60.8123, 65]
FS = [1111, 2132, 4558, 5000, 4000, 9999, 10001]
N_CYCLES = [1, 2, 3, 4, 5, 10, 11, 20, 23, 50, 57]
N_COMPONENTS = [1, 2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
N_CHOICES = 5
V_DC_OFFSET = 0
N_SIGNATURES = 100
N_SIGNATURES_PER_ITER = 1
ITERGRID = list(
    it.islice(it.product(F0, FS, N_CYCLES),
              N_SIGNATURES // N_SIGNATURES_PER_ITER))

rng = np.random.RandomState(RANDOM_STATE)

# rng = np.random.RandomState(None)


def make_vi_test_sample(
    with_components=False,
    random_crop=False,
    random_nolabel=True,
    constant_spectrum=False,
    unitscale=False,
    **kwargs,
):
    sample = []

    for f0, fs, n_cycles in ITERGRID:
        output_size = math.ceil(fs / f0)
        h_loc = output_size // 2
        dt = 1 / f0 if constant_spectrum else n_cycles / f0

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

        if unitscale:
            I /= abs(I).max(1, keepdims=True)

        if constant_spectrum:
            I = np.repeat(I[:, None], n_cycles, axis=1)
            I = I.reshape(len(I), -1)
            dt *= n_cycles

        t = np.linspace(kwargs.get('eps', 1e-12), 1 / f0, output_size)
        v = np.sin(2 * np.pi * f0 * t) + V_DC_OFFSET
        v = np.repeat(v[None], n_cycles, axis=0).ravel()
        V = np.stack(np.repeat(v[None], len(I), axis=0))

        if len(I) // 2 == 0:
            dropsize = rng.choice(2)
        else:
            dropsize = len(I) // 2

        if random_nolabel:
            for j in rng.choice(len(y), size=dropsize, replace=False):
                y[j] = None

        if random_crop:
            a = np.random.randint(0, I.shape[1] - 1, size=len(I))
            b = np.random.randint(a + 1, I.shape[1], size=len(I))
            locs = np.stack((a, b), axis=1).tolist()

            for j in rng.choice(len(locs), size=dropsize, replace=False):
                locs[j] = None

            for j, locs_ in enumerate(locs):
                if locs_ is not None:
                    a, b = locs_
                    I[j][:a], I[j][b:] = 0, 0
        else:
            locs = [None] * len(y)

        signatures = [
            VI(v,
               i,
               fs,
               f0,
               appliances=appliances,
               locs=None if locs_ is None else [locs_])
            for v, i, appliances, locs_ in zip(V, I, y, locs)
        ]

        if with_components:
            replace = N_SIGNATURES_PER_ITER > len(N_COMPONENTS)
            n_components = rng.choice(N_COMPONENTS,
                                      N_SIGNATURES_PER_ITER,
                                      replace=replace)

            for n in n_components:
                replace = n > len(signatures)
                combs = rng.choice(len(signatures), n, replace=replace)
                combs = [signatures[i] for i in combs]
                sample.append(combs)
        else:
            sample.extend(signatures)

    return sample


class TestThresholdBasedDetector(test.TestCase):
    signatures = make_vi_test_sample(with_components=True, random_crop=True)

    def test_threshold_based_detector(self):
        k = 0
        tbd = ThresholdBasedDetector(lambda x: abs(x).max(), 1, 0)

        for vi in self.signatures:
            vi = sum(vi)

            a = np.random.randint(0, vi.n_samples - 1)
            b = np.random.randint(a + 1, vi.n_samples)
            vi = vi[a:b]

            if vi.is_empty():
                continue

            xlocs = vi.split_locs(mode='running')
            locs = tbd(vi)
            self.assertTrue(np.allclose(locs, xlocs))

            k += 1

        self.assertGreaterEqual(k, len(self.signatures) // 2)


class TestDerivativeBasedDetector(test.TestCase):
    '''
    Testing derivative based event detector with the window size of a cycle size
    '''
    signatures = make_vi_test_sample(with_components=True,
                                     random_crop=False,
                                     constant_spectrum=True)

    def test_derivative_based_detector(self):
        k = 0

        for vi in self.signatures:
            vi = sum(vi)
            dbd = DerivativeBasedDetector(spectral_centroid,
                                          vi.cycle_size,
                                          1e-8,
                                          rel=False,
                                          fill_nan=0)

            if vi.is_empty():
                continue

            xlocs = vi.split_locs(mode='onchange')
            locs = dbd(vi)
            self.assertTrue(np.allclose(locs, xlocs))

            k += 1

        self.assertGreaterEqual(k, len(self.signatures) // 2)
