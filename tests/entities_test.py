import numpy as np
import itertools as it
import unittest as test

from edframe.data.entities2 import VI
from edframe.data.generators import make_hf_cycles

# Experiment setup
F0 = [40, 45, 50, 55, 60]
FS = [1000, 2000, 4000]
N_COMPONENTS = [2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)


def init_signatures(n, f0, fs, dt=None, v_dc_offset=0):
    if dt is None:
        dt = 1 / f0
        is_synced = True
    else:
        is_synced = False

    I, _ = make_hf_cycles(n, n, fs=fs, f0=f0, dt=dt)
    I /= abs(I).max(-1, keepdims=True)
    t = np.linspace(0, dt, I.shape[1])
    v = np.sin(2 * np.pi * f0 * t) + v_dc_offset
    X = [VI(v, i, fs, f0=f0, is_synced=is_synced, dims=(1, len(i))) for i in I]

    return X


class TestVI(test.TestCase):

    def generic_test_components(self, signatures):
        vi_1 = sum(signatures)
        vi_2 = signatures[0]

        for vi in signatures[1:]:
            vi_2 += vi

        self.assertEqual(vi_1.hash(), vi_2.hash())
        self.assertEqual(vi_1.n_components, len(signatures))

        vi_3 = vi_1.require_components(False)
        vi_3 = vi_1 + vi_3
        self.assertEqual(vi_3.n_components, 1)

        I = np.asarray([vi.i for vi in signatures])
        V = np.asarray([vi.v for vi in signatures])
        self.assertTrue(np.allclose(vi_1.v, VI.__vaggrule__(V)))
        self.assertTrue(np.allclose(vi_1.i, VI.__iaggrule__(I)))

    def test_components(self):
        combs = it.product(F0, FS, N_COMPONENTS)

        for f0, fs, n in combs:
            signatures = init_signatures(n, f0, fs)
            self.generic_test_components(signatures)
            self.generic_test_resample(signatures)

    def generic_test_resample(self, signatures):
        for vi in signatures:
            # for fs in [1000, 1001, 1111, 1321, 9999, 10232, 2301022]:
            for fs in [1000, 1001, 1111, 1321, 9999]:
                vi_1 = vi.resample(fs)
                self.assertEqual(vi_1.fs, fs)

                if fs < vi.fs:
                    self.assertLessEqual(np.product(vi_1.i.shape),
                                         np.product(vi.i.shape))
                elif fs > vi.fs:
                    self.assertGreaterEqual(np.product(vi_1.i.shape),
                                            np.product(vi.i.shape))
                else:
                    self.assertEqual(vi_1.i.shape, vi.i.shape)


if __name__ == '__main__':
    test.main()
