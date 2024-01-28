import math
import numpy as np
import itertools as it
import unittest as test

from edframe.data.entities2 import VI
from edframe.data.generators import make_hf_cycles, make_oscillations
from edframe.utils.exceptions import NotEnoughCycles, SingleCycleOnly, SamplingRateMismatch, MainsFrequencyMismatch

# Experiment setup
F0 = [40, 45, 50, 55, 60]
FS = [1000, 2000, 4000]
N_CYCLES = [1, 2, 3, 4, 5, 10, 20, 50]
N_COMPONENTS = [2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
N_CHOICES = 1
N_SIGNATURES = 2
V_DC_OFFSET = 0
rng = np.random.RandomState(RANDOM_STATE)


class TestVI(test.TestCase):

    def init_signatures(self):
        signatures = []
        combs = it.product(F0, FS, N_CYCLES)

        for f0, fs, n_cycles in combs:
            dt = n_cycles / f0
            output_size = math.ceil(fs / f0)
            h_loc = output_size // 2

            I, _ = make_oscillations(N_SIGNATURES,
                                     N_SIGNATURES,
                                     fs=fs,
                                     f0=f0,
                                     dt=dt,
                                     h_loc=h_loc)
            # I /= abs(I).max(-1, keepdims=True)

            t = np.linspace(0, dt, I.shape[1])
            v = np.sin(2 * np.pi * f0 * t) + V_DC_OFFSET
            V = np.stack(np.repeat(v[None], len(I), axis=0))

            signatures.extend([VI(v, i, fs, f0) for v, i in zip(V, I)])

        return signatures

    def test_components(self):
        signatures = self.init_signatures()

        for n_components in N_COMPONENTS:
            replace = n_components > len(signatures)
            combs = rng.choice(len(signatures), n_components, replace=replace)
            combs = [signatures[i] for i in combs]

            with self.assertRaises(
                (SamplingRateMismatch, MainsFrequencyMismatch)):
                vi_1 = sum(combs)
                vi_2 = combs[0]

                for vi in combs[1:]:
                    vi_2 += vi

                self.assertEqual(vi_1.hash(), vi_2.hash())
                self.assertEqual(vi_1.n_components, len(combs))

                vi_3 = vi_1.require_components(False)
                vi_3 = vi_1 + vi_3

                self.assertEqual(vi_3.n_components, 1)

                I = np.asarray([vi.i for vi in combs])
                V = np.asarray([vi.v for vi in combs])

                self.assertTrue(np.allclose(vi_1.v, VI.__vaggrule__(V)))
                self.assertTrue(np.allclose(vi_1.i, VI.__iaggrule__(I)))

    def test_resample(self):
        for vi in self.init_signatures():
            for fs in rng.choice(range(1000, 10000), N_CHOICES, replace=False):
                vi_ = vi.resample(fs)
                self.assertEqual(vi_.fs, fs)

                if fs < vi.fs:
                    self.assertLessEqual(np.product(vi_.i.shape),
                                         np.product(vi.i.shape))
                elif fs > vi.fs:
                    self.assertGreaterEqual(np.product(vi_.i.shape),
                                            np.product(vi.i.shape))
                else:
                    self.assertEqual(vi_.i.shape, vi.i.shape)

    def test_pad(self):
        for vi in self.init_signatures():
            for n in rng.choice(100, N_CHOICES, replace=False):
                vi_ = vi.pad(n)

                if isinstance(n, tuple):
                    a, b = n
                else:
                    a, b = n - n // 2, n

                if a != 0:
                    a = a + vi.cycle_size % a

                if b != 0:
                    b = b + vi.cycle_size % b

                self.assertEqual(len(vi_), len(vi) + a + b)

    def generic_test_cycle(self, signatures):
        for vi in signatures:
            pass


if __name__ == '__main__':
    test.main()
