import math
import numpy as np
import pandas as pd
import itertools as it
import unittest as test
from functools import reduce

from edframe.data.entities2 import VI, VISet
from edframe.data.generators import make_hf_cycles, make_oscillations
from edframe.utils.exceptions import NotEnoughCycles, SingleCycleOnly, NSamplesMismatch, SamplingRateMismatch, MainsFrequencyMismatch

# Experiment setup
F0 = [40, 45, 49.8, 50, 51.123, 55, 59.45646, 60, 60.8123, 65]
FS = [1111, 2132, 4558, 5000, 4000, 9999, 10001]
N_CYCLES = [1, 2, 3, 4, 5, 10, 11, 20, 23, 50, 57]
N_COMPONENTS = [1, 2, 3, 4, 5, 10, 20]
RANDOM_STATE = 42
N_CHOICES = 10
V_DC_OFFSET = 0
N_SIGNATURES = 1000
N_SIGNATURES_PER_ITER = 20
ITERGRID = list(
    it.islice(it.product(F0, FS, N_CYCLES),
              N_SIGNATURES // N_SIGNATURES_PER_ITER))

rng = np.random.RandomState(RANDOM_STATE)
# rng = np.random.RandomState(None)


def is_immutable(x):
    return isinstance(x, int | float | str | tuple | frozenset)


def init_signatures(with_components=False):
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

        a = np.random.randint(0, I.shape[1] - 1, size=len(I))
        b = np.random.randint(a + 1, I.shape[1], size=len(I))
        locs = np.stack((a, b), axis=1).tolist()

        if len(I) // 2 == 0:
            dropsize = rng.choice(2)
        else:
            dropsize = len(I) // 2

        for j in rng.choice(len(locs), size=dropsize, replace=False):
            y[j] = None

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
                combs_ = [signatures_[i] for i in combs_]
                signatures.append(combs_)
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
        # break

    print('[!]', len(ITERGRID), len(signatures), '\n')

    return signatures


class TestVI(test.TestCase):
    signatures = init_signatures(with_components=True)

    # TODO deeper tests of data leakage between newly created instances
    def test_getitem(self):
        for vi in self.signatures:
            vi = sum(vi)
            n = np.random.randint(0, len(vi), size=2 * N_CHOICES)

            for a, b in np.sort(n).reshape(-1, 2):
                if b <= a:
                    self.assertRaises(ValueError, lambda: vi[a:b])
                else:
                    vi_ = vi[a:b]
                    dnc = vi.n_components - vi_.n_components

                    if dnc == vi.n_components:
                        self.assertTrue(vi_.is_empty())

                    if (a >= vi.locs.T[1]).any() or (b <= vi.locs.T[0]).any():
                        if not vi_.is_empty():
                            self.assertLess(len(vi_.locs), len(vi.locs))

                        self.assertLess(vi_.n_components, vi.n_components)

                    if vi_.n_components > 0:
                        self.assertGreaterEqual(vi_.locs.min(), 0)
                        self.assertLessEqual(vi_.locs.max(), vi_.n_samples)
                        self.assertEqual(len(vi_) % vi.cycle_size, 0)
                        self.assertGreaterEqual(len(vi_), b - a)
                        self.assertTrue(len(vi_) % vi.cycle_size == 0)
                        self.assertAlmostEqual(vi_.i.sum(), vi.i[a:b].sum())
                        self.assertEqual(len(vi_.appliances), vi_.n_components)

                    if dnc > 0:
                        self.assertIsNot(vi_.appliances, vi.appliances)

                    if a == 0 and b == vi.n_samples:
                        self.assertEqual(vi_._data.shape, vi._data.shape)
                        self.assertTrue(np.allclose(vi_._data, vi._data))
                        # Otherwise many exceptions to handle to test.
                        # Was asserted in the method instead.

                    self.assertIsNot(id(vi_._data), id(vi._data))
                    self.assertFalse(np.may_share_memory(vi_._data, vi._data))
                    self.assertIsNot(id(vi_._f0), id(vi._f0))
                    self.assertIsNot(id(vi_._T), id(vi._T))
                    self.assertIsNot(id(vi_._appliances), id(vi._appliances))
                    self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_components(self):
        signatures = init_signatures()

        for n_components in N_COMPONENTS:
            replace = n_components > len(signatures)
            combs = rng.choice(len(signatures), n_components, replace=replace)
            combs = [signatures[i] for i in combs]

            if len(set([x.fs for x in combs
                        ])) > 1 | len(set([x.f0 for x in combs])) | len(
                            set([x.n_samples for x in combs])):
                self.assertRaises((NSamplesMismatch, SamplingRateMismatch,
                                   MainsFrequencyMismatch), lambda: sum(combs))

        for combs in self.signatures:
            vi_ = sum(combs)

            self.assertEqual(vi_.hash(),
                             reduce(lambda a, b: a + b, combs).hash())
            self.assertEqual(vi_.n_components, len(combs))

            if vi_.has_locs():
                self.assertEqual(len(vi_.locs), len(combs))

            vi_copy = vi_.copy()
            vi_copy = vi_copy.require_components(False)

            self.assertEqual(vi_copy.n_components, 1)
            self.assertIsNone(vi_copy._locs)
            # self.assertIsNone((vi_copy + vi_)._locs)
            self.assertEqual((vi_copy + vi_).n_components, 1)

            I = np.asarray([vi.i for vi in combs])[:, None]
            V = np.asarray([vi.v for vi in combs])[:, None]

            self.assertTrue(np.allclose(vi_.v, VI.__vaggrule__(V)))
            self.assertTrue(np.allclose(vi_.i, VI.__iaggrule__(I)))

            locs = np.concatenate([x.locs for x in combs], axis=0)
            self.assertTrue(np.allclose(locs, vi_.locs))

            self.assertEqual(len(vi_.appliances), vi_.n_components)
            apps = list(it.chain(*[vi.appliances for vi in combs]))
            self.assertEqual(set(vi_.appliances), set(apps))

    def test_resample(self):
        for vi in self.signatures:
            vi = sum(vi)

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

                self.assertEqual(len(vi_.appliances), vi_.n_components)
                self.assertEqual(vi_.appliances, vi.appliances)

                self.assertIsNot(id(vi_._data), id(vi._data))
                self.assertFalse(np.may_share_memory(vi_._data, vi._data))
                self.assertIsNot(id(vi_._f0), id(vi._f0))
                self.assertIsNot(id(vi_._T), id(vi._T))
                self.assertIsNot(id(vi_._appliances), id(vi._appliances))
                self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_roll(self):
        for vi in self.signatures:
            vi = sum(vi)

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

                if vi_.is_empty():
                    self.assertIsNone(vi_.locs)
                else:
                    self.assertGreaterEqual(vi_.locs.min(), 0)
                    self.assertLessEqual(vi_.locs.max(), vi_.n_samples)

                if vi_.n_components == vi.n_components:
                    for i in range(vi_.n_components):
                        a = min(vi.n_samples, max(vi.locs[i][0] + n, 0))
                        b = min(vi.n_samples, max(vi.locs[i][1] + n, 0))
                        self.assertEqual(vi_.locs[i][0], a)
                        self.assertEqual(vi_.locs[i][1], b)

                self.assertEqual(len(vi_.appliances), vi_.n_components)
                self.assertEqual(vi_.appliances, vi.appliances)

                self.assertIsNot(id(vi_._data), id(vi._data))
                self.assertFalse(np.may_share_memory(vi_._data, vi._data))
                self.assertIsNot(id(vi_._f0), id(vi._f0))
                self.assertIsNot(id(vi_._T), id(vi._T))
                self.assertIsNot(id(vi_._appliances), id(vi._appliances))
                self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_pad(self):
        for vi in self.signatures:
            vi = sum(vi)

            for n in rng.choice(100, N_CHOICES, replace=False):
                vi_ = vi.pad(n)
                a, b = vi._adjust_delta(n)

                if n // 2 % vi_.cycle_size == 0:
                    self.assertEqual(n // 2, b)

                if (n - n // 2) % vi_.cycle_size == 0:
                    self.assertEqual(n - n // 2, a)

                self.assertEqual(a % vi_.cycle_size, 0)
                self.assertEqual(b % vi_.cycle_size, 0)

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

                if vi.has_locs():
                    self.assertEqual(vi_.locs.min() - vi.locs.min(), a)
                    self.assertEqual(vi_.locs.max() - vi.locs.max(), a)
                else:
                    self.assertEqual(vi_.locs.min(), 0)
                    self.assertEqual(vi_.locs.max(), vi_.n_samples)

                self.assertEqual(len(vi_.appliances), vi_.n_components)
                self.assertEqual(vi_.appliances, vi.appliances)

                self.assertIsNot(id(vi_._data), id(vi._data))
                self.assertFalse(np.may_share_memory(vi_._data, vi._data))
                self.assertIsNot(id(vi_._f0), id(vi._f0))
                self.assertIsNot(id(vi_._T), id(vi._T))
                self.assertIsNot(id(vi_._appliances), id(vi._appliances))
                self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_extrapolate(self):
        for vi in self.signatures:
            vi = sum(vi)

            for n in rng.choice(100, N_CHOICES, replace=False):
                vi_ = vi.extrapolate(n)
                a, b = vi._adjust_delta(n)

                if n // 2 % vi_.cycle_size == 0:
                    self.assertEqual(n // 2, b)

                if (n - n // 2) % vi_.cycle_size == 0:
                    self.assertEqual(n - n // 2, a)

                self.assertEqual(a % vi_.cycle_size, 0)
                self.assertEqual(b % vi_.cycle_size, 0)

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

                if vi.has_locs():
                    for i in ids:
                        self.assertEqual(vi_.locs[i][0] - vi.locs[i][0], a)
                        self.assertGreaterEqual(vi_.locs[i][1] - vi.locs[i][1],
                                                a)
                        self.assertLessEqual(vi_.locs[i][1] - vi.locs[i][1],
                                             a + b)
                else:
                    self.assertEqual(vi_.locs.min(), 0)
                    self.assertEqual(vi_.locs.max(), vi_.n_samples)

                self.assertEqual(len(vi_.appliances), vi_.n_components)
                self.assertEqual(vi_.appliances, vi.appliances)

                self.assertIsNot(id(vi_._data), id(vi._data))
                self.assertFalse(np.may_share_memory(vi_._data, vi._data))
                self.assertIsNot(id(vi_._f0), id(vi._f0))
                self.assertIsNot(id(vi_._T), id(vi._T))
                self.assertIsNot(id(vi_._appliances), id(vi._appliances))
                self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_cycle(self):
        for comps in self.signatures:
            vi = sum(comps)
            vi_ = vi.cycle('mean')

            self.assertEqual(len(vi_), vi_.cycle_size)

            vi_ = vi.cycle('median')

            self.assertEqual(len(vi_), vi_.cycle_size)
            self.assertEqual(vi_.cycle_size, vi.cycle_size)
            self.assertEqual(vi_.fs, vi.fs)
            self.assertEqual(vi_.f0, vi.f0)

            self.assertEqual(len(vi_.locs), len(vi.locs))
            self.assertTrue((vi_.locs[:, 0] == 0).all())
            self.assertTrue((vi_.locs[:, 1] == vi_.n_samples).all())

            self.assertEqual(len(vi_.appliances), vi_.n_components)
            self.assertEqual(vi_.appliances, vi.appliances)

            self.assertIsNot(id(vi_._data), id(vi._data))
            self.assertFalse(np.may_share_memory(vi_._data, vi._data))
            self.assertIsNot(id(vi_._f0), id(vi._f0))
            self.assertIsNot(id(vi_._T), id(vi._T))
            self.assertIsNot(id(vi_._appliances), id(vi._appliances))
            self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_unitscale(self):
        for comps in self.signatures:
            vi = sum(comps)
            vi_ = vi.unitscale()
            V, I = vi_.datafold

            self.assertNotEqual(V.mean(), 1)
            self.assertNotEqual(I.mean(), 1)
            self.assertAlmostEqual(abs(vi_.v).max(), 1)
            self.assertAlmostEqual(abs(vi_.i).max(), 1)

            self.assertEqual(len(vi_.locs), len(vi.locs))
            self.assertTrue(np.allclose(vi_.locs, vi.locs))

            self.assertEqual(len(vi_.appliances), vi_.n_components)
            self.assertEqual(vi_.appliances, vi.appliances)

            self.assertIsNot(id(vi_._data), id(vi._data))
            self.assertFalse(np.may_share_memory(vi_._data, vi._data))
            self.assertIsNot(id(vi_._f0), id(vi._f0))
            self.assertIsNot(id(vi_._T), id(vi._T))
            self.assertIsNot(id(vi_._appliances), id(vi._appliances))
            self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_fryze(self):
        for vi in self.signatures:
            vi = sum(vi)
            vi_ = vi.fryze()

            self.assertIsNone(vi.orthogonality)
            self.assertEqual(vi_.orthogonality, 'Fryze')
            self.assertTrue(np.allclose(vi_.i, vi.i))
            self.assertTrue(np.allclose(vi_.v, vi.v))
            self.assertTrue(np.allclose(vi_.locs, vi.locs))

            self.assertEqual(len(vi_.appliances), vi_.n_components)
            self.assertEqual(vi_.appliances, vi.appliances)

            self.assertIsNot(id(vi_._data), id(vi._data))
            self.assertFalse(np.may_share_memory(vi_._data, vi._data))
            self.assertIsNot(id(vi_._f0), id(vi._f0))
            self.assertIsNot(id(vi_._T), id(vi._T))
            self.assertIsNot(id(vi_._appliances), id(vi._appliances))
            self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_budeanu(self):
        for vi in self.signatures:
            vi = sum(vi)
            vi_ = vi.budeanu()

            self.assertIsNone(vi.orthogonality)
            self.assertEqual(vi_.orthogonality, 'Budeanu')
            self.assertTrue(np.allclose(vi_.i, vi.i))
            self.assertTrue(np.allclose(vi_.v, vi.v))
            self.assertTrue(np.allclose(vi_.locs, vi.locs))

            self.assertEqual(len(vi_.appliances), vi_.n_components)
            self.assertEqual(vi_.appliances, vi.appliances)

            self.assertIsNot(id(vi_._data), id(vi._data))
            self.assertFalse(np.may_share_memory(vi_._data, vi._data))
            self.assertIsNot(id(vi_._f0), id(vi._f0))
            self.assertIsNot(id(vi_._T), id(vi._T))
            self.assertIsNot(id(vi_._appliances), id(vi._appliances))
            self.assertIsNot(id(vi_._locs), id(vi._locs))

    def test_features(self):
        for vi in self.signatures:
            vi = sum(vi)
            n_features = vi.n_features

            fs_numpy = vi.features(format="numpy")
            self.assertIsInstance(fs_numpy, np.ndarray)
            self.assertGreaterEqual(len(fs_numpy), n_features)

            fs_pandas = vi.features(format="pandas")
            self.assertIsInstance(fs_pandas, pd.DataFrame)
            self.assertGreaterEqual(fs_pandas.shape[1], n_features)

            fs_dict = vi.features(format="dict")
            self.assertIsInstance(fs_dict, dict)
            self.assertGreaterEqual(len(fs_dict), n_features)

            fs_list = vi.features(format="list")
            self.assertIsInstance(fs_list, list)
            self.assertGreaterEqual(len(fs_list), n_features)

    def test_hash(self):
        for vi in self.signatures:
            vi = sum(vi)
            h = vi.hash()
            self.assertIsInstance(h, int)
            self.assertGreaterEqual(h, 0)

    def test_todict(self):
        for vi in self.signatures:
            vi = sum(vi)
            vi = vi.fryze()
            d = vi.todict()

            self.assertIn('v', d)
            self.assertIn('i', d)
            self.assertIn('fs', d)
            self.assertIn('f0', d)
            self.assertIn('components', d)
            self.assertIn('locs', d)

            self.assertEqual(d['v'].tolist(), vi.v.tolist())
            self.assertEqual(d['i'].tolist(), vi.i.tolist())
            self.assertEqual(d['fs'], vi.fs)
            self.assertEqual(d['f0'], vi.f0)
            self.assertEqual(len(d['components']), vi.n_components)
            self.assertEqual(len(d['components'][0]), vi.n_orthogonals)
            self.assertEqual(d['locs'].tolist(), vi.locs.tolist())

            self.assertNotEqual(id(d['v']), id(vi.v))
            self.assertNotEqual(id(d['i']), id(vi.i))
            self.assertNotEqual(id(d['locs']), id(vi.locs))


class TestVISet(test.TestCase):

    @staticmethod
    def init_datasets(n_datasets=1, with_components=False):
        datasets = []

        for f0, fs, n_cycles in ITERGRID:
            signatures = []
            dt = n_cycles / f0
            output_size = math.ceil(fs / f0)
            h_loc = output_size // 2

            I, _ = make_oscillations(N_SIGNATURES // n_datasets,
                                     N_SIGNATURES // n_datasets,
                                     fs=fs,
                                     f0=f0,
                                     dt=dt,
                                     h_loc=h_loc)

            t = np.linspace(0, dt, I.shape[1])
            v = np.sin(2 * np.pi * f0 * t) + V_DC_OFFSET
            V = np.stack(np.repeat(v[None], len(I), axis=0))

            a = np.random.randint(0, I.shape[1] - 1, size=len(I))
            b = np.random.randint(a + 1, I.shape[1], size=len(I))
            locs = np.stack((a, b), axis=1)[:, None].tolist()

            if len(I) // 2 == 0:
                nolocs_size = rng.choice(2)
            else:
                nolocs_size = len(I) // 2

            for j in rng.choice(len(locs), size=nolocs_size, replace=False):
                locs[j] = None

            signatures_ = [
                VI(v, i, fs, f0, locs=locs_)
                for v, i, locs_ in zip(V, I, locs)
            ]

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
                signatures = [
                    VI(v, i, fs, f0, locs=locs_)
                    for v, i, locs_ in zip(V, I, locs)
                ]

            datasets.append(VISet(signatures))

        return datasets

    def test_getitem(self):
        viset = self.init_datasets()[0]
        vi = viset[rng.randint(len(viset))]

        self.assertIsInstance(vi, VI)

        n = np.random.randint(0, len(vi), size=2 * N_CHOICES)

        for a, b in np.sort(n).reshape(-1, 2):
            viset_ = viset[a:b]

            if b == a:
                self.assertIsNone(viset_)
            else:
                self.assertTrue(len(viset_), b - a + 1)

    def test_is_multilabel(self):
        pass

    def test_features(self):
        viset = self.init_datasets()[0]

        fs_numpy = viset.features(format='numpy')
        self.assertIsInstance(fs_numpy, np.ndarray)

        fs_pandas = viset.features(format='pandas')
        self.assertIsInstance(fs_pandas, pd.DataFrame)

        fs_dict = viset.features(format='dict')
        self.assertIsInstance(fs_dict, dict)

        fs_list = viset.features(format='list')
        self.assertIsInstance(fs_list, list)

    def test_data(self):
        viset = self.init_datasets()[0]
        data = viset.data
        self.assertEqual(len(data.shape), 5)
        self.assertEqual(len(data), len(viset))


if __name__ == '__main__':
    test.main()
