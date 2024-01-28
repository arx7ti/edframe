import numpy as np
import itertools as it
import unittest as test

from edframe.data.entities2 import VI
from edframe.data.generators import make_hf_cycles, make_oscillations

# Experiment setup
F0 = [40, 45, 50, 55, 60]
FS = [1000, 2000, 4000]
N_COMPONENTS = [2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
N_CHOICES = 10
rng = np.random.RandomState(RANDOM_STATE)


def init_signatures(n, f0, fs, dt=None, v_dc_offset=0):
    is_synced = False

    if dt is None:
        dt = 1 / f0
        is_synced = False

    output_size = round(fs / f0)
    h_loc = output_size // 2
    I, _ = make_oscillations(n, n, fs=fs, f0=f0, dt=dt, h_loc=h_loc)
    I /= abs(I).max(-1, keepdims=True)
    t = np.linspace(0, dt, I.shape[1])
    v = np.sin(2 * np.pi * f0 * t) + v_dc_offset
    X = [VI(v, i, fs, f0=f0, is_synced=is_synced, dims=(1, len(i))) for i in I]

    return X


class TestVI(test.TestCase):

    def test_all(self):
        combs = it.product(F0, FS, N_COMPONENTS)

        for f0, fs, n in combs:
            signatures = init_signatures(n, f0, fs)
            # self.generic_test_components(signatures)
            # self.generic_test_resample(signatures)
            self.generic_test_pad(signatures)

    def generic_test_components(self, signatures):
        vi_1 = sum(signatures)
        vi_2 = signatures[0]

        for vi in signatures[1:]:
            vi_2 += vi

        self.assertEqual(vi_1.hash(), vi_2.hash())

        with self.assertRaises(ValueError):
            vi = signatures[0].copy()
            vi._is_synced = False
            vi += sum(signatures[0])

        self.assertEqual(vi_1.n_components, len(signatures))

        vi_3 = vi_1.require_components(False)
        vi_3 = vi_1 + vi_3
        self.assertEqual(vi_3.n_components, 1)

        I = np.asarray([vi.i for vi in signatures])
        V = np.asarray([vi.v for vi in signatures])

        self.assertTrue(np.allclose(vi_1.v, VI.__vaggrule__(V)))
        self.assertTrue(np.allclose(vi_1.i, VI.__iaggrule__(I)))
        self.assertTrue(vi_1.is_synced())

    def generic_test_resample(self, signatures):
        for vi in signatures:
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

    def generic_test_pad(self, signatures):
        for vi in signatures:
            for n in rng.choice(10000, N_CHOICES, replace=False):
                print(n)
                vi_ = vi.pad(n)
                vi_left = vi.pad((n, 0))
                vi_right = vi.pad((0, n))

                self.assertEqual(len(vi_), len(vi) + n)
                self.assertEqual(len(vi_left), len(vi) + n)
                self.assertEqual(len(vi_right), len(vi) + n)

    def generic_test_cycle(self, signatures):
        for vi in signatures:
            pass


if __name__ == '__main__':
    test.main()
