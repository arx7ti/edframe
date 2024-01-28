import math
import numpy as np
import itertools as it
import unittest as test

from edframe.data.entities2 import VI
from edframe.data.generators import make_hf_cycles, make_oscillations

# Experiment setup
F0 = [40, 45, 50, 55, 60]
FS = [1000, 2000, 4000]
N_CYCLES = [1, 2, 3, 4, 5, 10, 20, 50]
N_COMPONENTS = [2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
N_CHOICES = 1
N_SIGNATURES = 2
rng = np.random.RandomState(RANDOM_STATE)


class TestVI(test.TestCase):

    def init_signatures(
        self,
        n,
        f0,
        fs,
        n_cycles,
        v_dc_offset=0,
    ):
        # if dt is None:
        #     dt = 2 / f0
        dt = n_cycles / f0

        output_size = math.ceil(fs / f0)
        h_loc = output_size // 2
        I, _ = make_oscillations(n, n, fs=fs, f0=f0, dt=dt, h_loc=h_loc)
        I /= abs(I).max(-1, keepdims=True)
        t = np.linspace(0, dt, I.shape[1])
        v = np.sin(2 * np.pi * f0 * t) + v_dc_offset
        V = np.stack(np.repeat(v[None], len(I), axis=0))
        # print('A', V.shape, output_size, fs, f0)

        # if not is_synced:
        #     A, B = rng.randint(0, output_size, size=(2, len(I)))
        #     B = I.shape[1] - B
        #     I = [i[a:b] for i, a, b in zip(I, A, B)]
        #     V = [v[a:b] for v, a, b in zip(V, A, B)]

        return [VI(v, i, fs, f0) for v, i in zip(V, I)]

        # for v, i in zip(V, I):
        #     vi = lambda: VI(v, i, fs, f0)

        #     if len(v) < output_size:
        #         self.assertRaises(ValueError, vi)
        #     else:
        #         print(len(v), output_size)
        #         X.append(vi())
        #         # self.assertEqual(np.product(X[-1].i.shape),
        #         #                  X[-1].cycle_size * 2)

        # return X

    def test_all(self):
        combs = it.product(F0, FS, N_CYCLES)

        for f0, fs, n_cycles in combs:
            signatures = self.init_signatures(N_SIGNATURES, f0, fs, n_cycles)
            self.generic_test_components(signatures)
            self.generic_test_resample(signatures)
            self.generic_test_pad(signatures)

    def generic_test_components(self, signatures):
        for n_components in N_COMPONENTS:
            replace = n_components > len(signatures)
            combs = rng.choice(len(signatures), n_components, replace=replace)
            combs = [signatures[i] for i in combs]

            vi_1 = sum(combs)
            vi_2 = combs[0]

            for vi in combs[1:]:
                vi_2 += vi

            self.assertEqual(vi_1.hash(), vi_2.hash())

            with self.assertRaises(ValueError):
                vi = combs[0].copy()
                vi._is_synced = False
                vi += sum(combs[0])

            self.assertEqual(vi_1.n_components, len(combs))

            vi_3 = vi_1.require_components(False)
            vi_3 = vi_1 + vi_3
            self.assertEqual(vi_3.n_components, 1)

            I = np.asarray([vi.i for vi in combs])
            V = np.asarray([vi.v for vi in combs])

            self.assertTrue(np.allclose(vi_1.v, VI.__vaggrule__(V)))
            self.assertTrue(np.allclose(vi_1.i, VI.__iaggrule__(I)))

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
