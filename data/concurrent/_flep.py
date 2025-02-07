from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

import h5py
import numpy as np
import itertools as it

from sklearn.neighbors import NearestNeighbors


def active_power(v, i):
    return np.round((v * i).mean())


def reactive_power(v, i):
    S = np.sqrt(np.power(v * i, 2).mean())
    P = active_power(v, i)

    return np.sqrt(S**2 - P**2)


def adj16(components, total):
    assert components.dtype == np.float16
    assert total.dtype == np.float16

    difference = total - components.sum()
    idx = np.argmax(abs(components))
    components[idx] += difference

    return components


def filter_with_devices(D, devices, strict=False):
    if strict:
        return [data for data in D if set(data[4]) == set(devices)]

    return [data for data in D if data[4] in set(devices)]


def get_device_names(D, mask=None):
    labels = []

    if mask is None:
        mask = np.ones(len(D), dtype=bool)
    else:
        assert len(D) == len(mask)

    for (_, _, _, _, devices, *_), take in zip(D, mask):
        # assert not isinstance(devices, str | np.str_)

        if take:
            labels.append(devices)

    return labels


def get_group_ids(D):
    I = defaultdict(list)

    for i, v in enumerate(D):
        v = tuple(sorted(v))
        I[v].append(i)

    return list(I.items())


def filter_with_ids(D, ids):
    return [data for i, data in enumerate(D) if i in ids]


def get_total_current(D, mask=None, normalize=False):
    I = []

    for _, i, _, _, _, *_ in D:
        I.append(i)

    I = np.asarray(I)

    if mask is not None:
        assert len(D) == len(mask)
        I = I[mask]

    if normalize:
        I = I / abs(I).max((1, 2), keepdims=True)

    return I


def get_similar_devices(Xa, Xb, metric='cosine'):
    knn = NearestNeighbors(metric=metric)
    knn.fit(Xb)
    d, ids = knn.kneighbors(Xa, n_neighbors=1)

    return d.ravel(), ids.ravel()


class FLEPGen:

    def __init__(
        self,
        background,
        activated=None,
        alwaysmix=False,
        filepath=None,
        random_state=None,
    ):
        self._alwaysmix = alwaysmix

        if filepath is not None:
            self.filepath = filepath if '.h5' in filepath else f'{filepath}.h5'
        else:
            self.filepath = None

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

        self.n_samples = background[0][0].size
        self._make_space(background, activated)

    @staticmethod
    def _get_device_types(samples):
        devices = set(get_device_names(samples))
        devices = sorted(devices)

        return list(devices)

    def _make_space(self, bg, on=None):
        _, _, self.fs, self.f0, *_ = bg[0]

        self.space = {
            'bg': defaultdict(list),
            'on': defaultdict(list),
        }
        self.bg_devices = self._get_device_types(bg)

        if on:
            self.on_devices = self._get_device_types(on)
        else:
            self.on_devices = []

        devices = self.devices()

        for device in self.bg_devices:
            if device in devices:
                elems = filter_with_devices(bg, [device])
            else:
                elems = []

            self.space['bg'][device] = elems
            assert len(self.space['bg'][device]) > 0

        for device in self.on_devices:
            if device in devices:
                elems = filter_with_devices(on, [device])
            else:
                elems = []

            self.space['on'][device] = elems
            assert len(self.space['on'][device]) > 0

        self.history = []

    def withon(self):
        return len(self.space['on']) > 0

    def alwaysmix(self):
        return self._alwaysmix

    def ondisk(self):
        return self.filepath is not None

    def flatten_space(self, space_name):
        return list(it.chain.from_iterable(self.space[space_name].values()))

    def space_size(self, space_name):
        return len(self.flatten_space(space_name))

    def devices(self):
        devices = set(self.bg_devices + self.on_devices)
        devices = sorted(devices)

        return list(devices)

    def get_n_signatures(self, n, on=False):
        devices = self.get_n_devices(n, on)
        signatures = self.get_signatures_for(devices)

        return signatures

    def get_n_devices(self, n, on=False):
        devices = {'bg': [], 'on': []}
        devices['bg'] = self.rng.choice(self.bg_devices,
                                        n - 1 if on else n,
                                        replace=n > len(self.bg_devices))

        if on:
            assert self.withon(), "On-device data is required."
            devices['on'] = self.rng.choice(self.on_devices, 1)

        return devices

    def get_signatures_for(self, devices):
        signatures = []
        # Track codes for each combination as list of tuples (device_id, index)
        # This will be used to prevent duplicate samples
        code = []

        for space_name, query in devices.items():
            for device in query:
                # Sample a signature from the chosen device's subspace
                subspace = self.space[space_name][device]
                idx = self.rng.choice(len(subspace))
                signature = subspace[idx]

                signatures.append(signature)

                # Generate code
                dev_id = list(self.space[space_name].keys()).index(device)

                if space_name == 'on':
                    dev_id += len(self.bg_devices)

                code.append((dev_id, idx))

        code = tuple(sorted(code))

        # Return nothing if we've already generated this signature
        if code in self.history:
            return None

        self.history.append(code)

        return signatures

    @staticmethod
    def aggregate_signatures(signatures, qtol=1e-4):
        _fs, _f0 = None, None
        i, devices, locs, P, Q = np.float16(0), [], [], [], []

        for v, ik, fs, f0, device, *other in signatures:
            ik = ik.copy()

            if _fs is None:
                _fs = fs

            if _f0 is None:
                _f0 = f0

            assert fs == _fs, 'Sampling frequencies must be the same for all devices'
            assert f0 == _f0, 'Fundamental frequencies must be the same for all devices'

            Pk = active_power(v, ik)
            Qk = reactive_power(v, ik)

            i += ik
            devices.append(device), P.append(Pk), Q.append(Qk)

            if len(other) > 0:
                # TODO can one device have multiple locations?
                locs.append(other[0])

        return v, i, fs, f0, devices, locs, P, Q

    def _sub2agg(self, signature):
        v, i, fs, f0, device, *other = signature
        # v, i = v.astype(np.float16), i.astype(np.float16)
        assert isinstance(device, str | np.str_)

        devices = [device]

        if len(other) > 0:
            locs = [other[0]]
        else:
            locs = None

        P = [active_power(v, i)]
        Q = [reactive_power(v, i)]

        return v, i, fs, f0, devices, locs, P, Q

    def _yield_submetered(self, n_samples):
        # FIXME
        if n_samples >= self.space_size('on') + self.space_size('bg'):
            bg_samples = self.space_size('bg')
        else:
            bg_samples = n_samples // 2 if self.withon() else n_samples

        bg_space = self.flatten_space('bg')
        ids = self.rng.choice(len(bg_space),
                              min(len(bg_space), bg_samples),
                              replace=False)

        yield from ((self._sub2agg(bg_space[i]), False) for i in ids)

        if self.withon():
            on_space = self.flatten_space('on')
            ids = self.rng.choice(len(on_space),
                                  min(len(on_space), n_samples - bg_samples),
                                  replace=False)

            yield from ((self._sub2agg(on_space[i]), True) for i in ids)

    def _yield_aggregated(self, n_components, n_samples, qtol=1e-4):
        for j in range(n_samples):
            with_activation = self.withon() & ((j % 2) | self.alwaysmix())
            signatures = self.get_n_signatures(n_components, with_activation)

            if signatures is None:
                continue

            signature = FLEPGen.aggregate_signatures(signatures, qtol=qtol)

            yield signature, with_activation

    def _yield_signatures(self, n_components, d, qtol=1e-4):
        for n, m in zip(n_components, d):
            if n == 1:
                yield from self._yield_submetered(m)
            else:
                yield from self._yield_aggregated(n, m, qtol=qtol)

    def __call__(
            self,
            n_samples=1000,
            n_components=(2, 10),
            progress_bar=True,
            qtol=1e-4,
    ):
        if isinstance(n_components, int):
            n_components = (n_components, n_components + 1)
        else:
            n_components = (n_components[0], n_components[1] + 1)

        n_components = range(*n_components)
        d = n_samples // len(n_components)
        d = d * np.ones(len(n_components), dtype=int)
        d[:int(n_samples - d.sum())] += 1
        print(d)
        print(0.95 * d[1:].sum())
        # raise
        assert d.sum() == n_samples

        # bar = tqdm(
        #     total=n_samples,
        #     desc=
        #     f"Saving whole-house power data {'onto disk' if self.ondisk() else 'into RAM'}",
        #     unit="Signatures",
        #     disable=not progress_bar)

        # I, P, Q = [], [], []
        # Devices, Locs = [], []
        signatures = self._yield_signatures(n_components, d, qtol=qtol)

        enc = LabelEncoder()
        enc.fit(self.devices())

        # if self.ondisk():
        import gc
        for obj in gc.get_objects():  # Browse through ALL objects
            if isinstance(obj, h5py.File):  # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass  # Was already closed

        h5file = h5py.File(self.filepath, 'w')
        agg = h5file.create_group('aggregated')
        sub = h5file.create_group('submetered')

        v_ds = agg.create_dataset(
            "v",
            maxshape=(None, self.n_samples),
            chunks=True,
            #   compression='lzf',
            dtype="float16",
            data=np.empty((0, self.n_samples), dtype=np.float16))
        i_ds = agg.create_dataset(
            "i",
            maxshape=(None, self.n_samples),
            chunks=True,
            #   compression='lzf',
            dtype="float16",
            data=np.empty((0, self.n_samples), dtype=np.float16))
        targets_ds = sub.create_dataset(
            "power_shares",
            maxshape=(None, len(self.devices())),
            chunks=True,
            # compression='lzf',
            dtype="float32",
            data=np.empty((0, len(self.devices())), dtype=np.float32))
        # else:
        # data = []
        err = 0

        for j, (signature, with_activation) in enumerate(signatures):
            v, i, _, _, devices, locs, Pj, Qj = signature

            P = active_power(v, i)
            other = self.devices().index('other')
            devices = enc.transform(devices)
            targets = np.zeros(len(self.devices()), dtype=np.float32)
            targets[other] += np.maximum(0, P - sum(Pj))
            np.add.at(targets, devices, Pj)
            targets = np.maximum(0, targets)
            e = abs(P - targets.sum())

            if e > err:
                err = e

            # + Half-precision conversion
            v, i = v.astype(np.float16), i.astype(np.float16)
            # Pj = np.round(np.asarray(Pj)).astype(np.float16)
            # Qj = np.round(np.asarray(Qj)).astype(np.float16)

            # P_total = np.round(active_power(v, i)).astype(np.float16)
            # Q_total = np.round(reactive_power(v, i)).astype(np.float16)
            # Pj, Qj = adj16(Pj, P_total), adj16(Qj, Q_total)
            # - Half-precision conversion

            # try:
            #     assert np.allclose(Pj.sum(), P_total)
            #     # assert np.allclose(Qj.sum(), Q_total)
            # except:
            #     print(Pj.sum() - P_total)

            # I.append(i), P.append(Pj), Q.append(Qj)
            # targets[devices] = Pj
            # Devices.append(devices)
            # Locs.append(locs)

            # pointer.append((j, xstart, len(devices)))
            # xstart += len(devices)

            if j % 10000 == 0 and j > 0:
                print(j)

            if self.ondisk():
                v_ds.resize(v_ds.shape[0] + 1, axis=0)
                v_ds[-1:] = v.reshape(1, -1)

                i_ds.resize(i_ds.shape[0] + 1, axis=0)
                i_ds[-1:] = i.reshape(1, -1)

                targets_ds.resize(targets_ds.shape[0] + 1, axis=0)
                targets_ds[-1:] = targets.reshape(1, -1)
            # else:
            # data.append((v, i, self.fs, self.f0, devices, locs, Pj, Qj))

        print('ERR', err)
        agg.attrs['fs'] = self.fs
        agg.attrs['f0'] = self.f0
        sub.attrs['devices'] = self.devices()

        h5file.attrs['Count'] = len(v_ds)
        h5file.attrs['Author'] = 'arx7ti (Ilia Kamyshev)'
        h5file.attrs['Original dataset'] = 'PLAID'

        # if not self.ondisk():
        # return data

    def asfor(
        self,
        aggregated,
        trials_per_comb=100,
        metric='cosine',
        return_ground_truth=False,
        return_scores=False,
        qtol=1e-4,
    ):
        ground_truth, synthetic, dist = [], [], []
        groups = get_group_ids(get_device_names(aggregated))

        for devices, ids in tqdm(groups):
            gt = filter_with_ids(aggregated, ids)
            snt = [
                self.get_signatures_for({'bg': devices})
                for _ in range(trials_per_comb)
            ]
            snt = [
                self.aggregate_signatures(s, qtol=qtol) for s in snt
                if s is not None
            ]

            I_gt = get_total_current(gt, normalize=True)
            I_snt = get_total_current(snt, normalize=True)
            I_gt = I_gt.reshape(len(I_gt), -1)
            I_snt = I_snt.reshape(len(I_snt), -1)
            d, ids = get_similar_devices(I_gt, I_snt, metric=metric)
            idx_gt = d.argsort()
            idx_snt = ids[idx_gt]

            used_snt_indices = set()

            for i, j in zip(idx_gt, idx_snt):
                if j not in used_snt_indices:
                    ground_truth.append(filter_with_ids(gt, [i])[0])
                    synthetic.append(filter_with_ids(snt, [j])[0])
                    dist.append(d[i])
                    used_snt_indices.add(j)

        if return_ground_truth and return_scores:
            return ground_truth, synthetic, dist

        if return_ground_truth:
            return ground_truth, synthetic

        if return_scores:
            return synthetic, dist

        return synthetic
