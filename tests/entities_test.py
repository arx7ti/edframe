import math
import numpy as np
import itertools as it
import unittest as test
from functools import reduce

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

    def init_signatures(self, include_locs=False, with_components=False):
        # TODO half with locs
        # TODO with components
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

            t = np.linspace(0, dt, I.shape[1])
            v = np.sin(2 * np.pi * f0 * t) + V_DC_OFFSET
            V = np.stack(np.repeat(v[None], len(I), axis=0))

            if include_locs:
                a = np.random.randint(0, I.shape[1] - 1)
                b = np.random.randint(a + 1, I.shape[1])
                locs = [[a, b]]
            else:
                locs = None

            signatures_ = [VI(v, i, fs, f0, locs=locs) for v, i in zip(V, I)]

            if with_components:
                for n_components in N_COMPONENTS:
                    replace = n_components > len(signatures_)
                    n_signatures = len(signatures_) / len(N_COMPONENTS)

                    for _ in range(math.ceil(n_signatures)):
                        combs_ = rng.choice(len(signatures_),
                                            n_components,
                                            replace=replace)
                        combs_ = [signatures_[i] for i in combs_]
                        signatures.append(combs_)
            else:
                signatures.extend(
                    [VI(v, i, fs, f0, locs=locs) for v, i in zip(V, I)])

        return signatures

    def test_getitem(self):
        for vi in self.init_signatures(include_locs=True):
            n = np.random.randint(0, len(vi), size=2 * N_CHOICES)

            for a, b in np.sort(n).reshape(-1, 2):
                vi_ = lambda: vi[a:b]

                if b <= a:
                    self.assertRaises(ValueError, vi_)
                else:
                    vi_ = vi_()

                    if (a >= vi.locs.T[1]).any() or (b <= vi.locs.T[0]).any():
                        self.assertLess(len(vi_.locs), len(vi.locs))

                    if vi_.n_components > 0:
                        self.assertGreaterEqual(vi_.locs.min(), 0)
                        self.assertLessEqual(vi_.locs.max(), vi_.n_samples)
                        self.assertEqual(len(vi_) % vi.cycle_size, 0)
                        self.assertGreaterEqual(len(vi_), b - a)
                        self.assertTrue(len(vi_) % vi.cycle_size == 0)
                        self.assertAlmostEqual(vi_.i.sum(), vi.i[a:b].sum())

    def test_components(self):
        signatures = self.init_signatures(include_locs=True)

        for n_components in N_COMPONENTS:
            replace = n_components > len(signatures)
            combs = rng.choice(len(signatures), n_components, replace=replace)
            combs = [signatures[i] for i in combs]

            if len(set([x.fs for x in combs
                        ])) > 1 | len(set([x.f0 for x in combs])):
                self.assertRaises(
                    (SamplingRateMismatch, MainsFrequencyMismatch),
                    lambda: sum(combs))

        signatures = self.init_signatures(include_locs=True,
                                          with_components=True)

        for combs in signatures:
            vi_ = sum(combs)

            self.assertEqual(vi_.hash(),
                             reduce(lambda a, b: a + b, combs).hash())
            self.assertEqual(vi_.n_components, len(combs))

            if vi_.has_locs():
                self.assertEqual(len(vi_.locs), len(combs))

            vi_copy = vi_.copy()
            vi_copy = vi_copy.require_components(False)

            self.assertEqual(vi_copy.n_components, 1)
            self.assertEqual((vi_copy + vi_).n_components, 1)

            I = np.asarray([vi.i for vi in combs])[:, None]
            V = np.asarray([vi.v for vi in combs])[:, None]

            self.assertTrue(np.allclose(vi_.v, VI.__vaggrule__(V)))
            self.assertTrue(np.allclose(vi_.i, VI.__iaggrule__(I)))

            locs = np.concatenate([x.locs for x in combs], axis=0)
            self.assertTrue(np.allclose(locs, vi_.locs))

    def test_resample(self):
        for vi in self.init_signatures(include_locs=True):
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

                self.assertGreaterEqual(vi_.locs.min(), 0)
                self.assertLessEqual(vi_.locs.max(), vi_.n_samples)
                self.assertTrue((vi_.locs[:, 1] > vi_.locs[:, 0]).all())
                self.assertFalse(np.any(vi_.locs[:, 1] == vi_.locs[:, 0]))

    def test_roll(self):
        for vi in self.init_signatures(include_locs=True):
            for n in rng.choice(len(vi) + 1, N_CHOICES, replace=False):
                vi_ = vi.roll(n)

                if n > 0:
                    mute = np.s_[n:] if n < 0 else np.s_[:n]
                    mute = np.s_[..., mute]

                    self.assertEqual(vi_.ic[mute].std(), 0)
                else:
                    self.assertTrue(np.allclose(vi_.data, vi.data))

                self.assertEqual(len(vi_), len(vi))
                self.assertEqual(vi_.cycle_size, vi.cycle_size)
                self.assertEqual(vi_.fs, vi.fs)
                self.assertEqual(vi_.f0, vi.f0)

                if n > 1:
                    self.assertNotEqual(vi_.vc[mute].std(), 0)

                self.assertGreaterEqual(vi_.locs.min(), 0)
                self.assertLessEqual(vi_.locs.max(), vi_.n_samples)

                dt0 = np.diff(vi.locs, axis=1)
                dt1 = np.diff(vi_.locs, axis=1)
                ids = np.argwhere(dt0 == dt1).ravel()

                for i in ids:
                    self.assertEqual(vi_.locs[i][0] - vi.locs[i][0], n)
                    self.assertEqual(vi_.locs[i][1] - vi.locs[i][1], n)

    def test_pad(self):
        for vi in self.init_signatures(include_locs=True):
            for n in rng.choice(100, N_CHOICES, replace=False):
                vi_ = vi.pad(n)
                a, b = vi._adjust_delta(n)

                self.assertEqual(len(vi_), len(vi) + a + b)
                self.assertEqual(vi_.cycle_size, vi.cycle_size)
                self.assertEqual(vi_.fs, vi.fs)
                self.assertEqual(vi_.f0, vi.f0)

                self.assertGreaterEqual(vi_.n_components, 1)
                self.assertEqual(len(vi_), len(vi) + a + b)
                self.assertEqual(a % vi_.cycle_size, 0)
                self.assertEqual(b % vi_.cycle_size, 0)
                self.assertEqual(len(vi_) % vi.cycle_size, 0)
                self.assertAlmostEqual(vi_.i.sum(), vi.i.sum())
                self.assertGreaterEqual(vi_.locs.min(), 0)
                self.assertLessEqual(vi_.locs.max(), vi_.n_samples)
                self.assertEqual(vi_.locs.min() - vi.locs.min(), a)
                self.assertEqual(vi_.locs.max() - vi.locs.max(), a)

    def test_extrapolate(self):
        for vi in self.init_signatures(include_locs=True):
            for n in rng.choice(100, N_CHOICES, replace=False):
                vi_ = vi.extrapolate(n)
                a, b = vi._adjust_delta(n)

                self.assertEqual(len(vi_), len(vi) + a + b)
                self.assertEqual(vi_.cycle_size, vi.cycle_size)
                self.assertEqual(vi_.fs, vi.fs)
                self.assertEqual(vi_.f0, vi.f0)

                self.assertGreaterEqual(vi_.n_components, 1)
                self.assertEqual(len(vi_), len(vi) + a + b)
                self.assertEqual(a % vi_.cycle_size, 0)
                self.assertEqual(b % vi_.cycle_size, 0)
                self.assertEqual(len(vi_) % vi.cycle_size, 0)

                dt0 = np.diff(vi.locs, axis=1)
                dt1 = np.diff(vi_.locs, axis=1)
                ids = np.argwhere(dt1 != dt0).ravel()

                self.assertGreaterEqual(vi_.locs.min(), 0)
                self.assertLessEqual(vi_.locs.max(), vi_.n_samples)
                self.assertTrue(all([d1 >= d0 for d0, d1 in zip(dt0, dt1)]))

                for i in ids:
                    self.assertEqual(vi_.locs[i][0] - vi.locs[i][0], a)
                    self.assertEqual(vi_.locs[i][1] - vi.locs[i][1], b + a)

    def test_cycle(self):
        for vi in self.init_signatures():
            vi_ = vi.cycle('mean')

            self.assertEqual(len(vi_), vi_.cycle_size)

            vi_ = vi.cycle('median')

            self.assertEqual(len(vi_), vi_.cycle_size)
            self.assertEqual(vi_.cycle_size, vi.cycle_size)
            self.assertEqual(vi_.fs, vi.fs)
            self.assertEqual(vi_.f0, vi.f0)
            # TODO locs

    def test_unitscale(self):
        for vi in self.init_signatures():
            vi_ = vi.unitscale()
            V, I = vi_.datafold

            self.assertNotEqual(V.mean(), 1)
            self.assertNotEqual(I.mean(), 1)
            self.assertLessEqual(abs(vi_.i).max(), 1)
            self.assertLessEqual(abs(vi_.v).max(), 1)
            # TODO locs

    def test_fryze(self):
        for vi in self.init_signatures():
            vi_ = vi.fryze()

            self.assertTrue(np.allclose(vi_.i, vi.i))
            self.assertTrue(np.allclose(vi_.v, vi.v))
            # TODO locs

    def test_budeanu(self):
        for vi in self.init_signatures():
            vi_ = vi.budeanu()

            self.assertTrue(np.allclose(vi_.i, vi.i))
            self.assertTrue(np.allclose(vi_.v, vi.v))
            # TODO locs


if __name__ == '__main__':
    test.main()
