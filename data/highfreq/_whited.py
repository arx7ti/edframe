from __future__ import annotations
from collections.abc import Sequence

import os
import numpy as np

import soundfile as sf


def fundamental(x, fs):
    amps = abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    f0 = freqs[np.argmax(amps)]

    return f0


class WHITED(Sequence):
    MK1 = {"volt": 1033.64, "amp": 61.4835}
    MK2 = {"volt": 861.15, "amp": 60.200}
    MK3 = {"volt": 988.926, "amp": 60.9562}

    def __init__(self, dirpath: str, f0_decimals=2):
        filenames = os.listdir(dirpath)
        filenames = list(filter(lambda x: x.endswith('.flac'), filenames))

        self.metadata = []

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            app_type, model, _, mk_type, _ = filename.split('_')
            # self.metadata.append({'app_type': app_type, 'model': model, 'mk_type': mk_type, 'filepath': filepath})
            self.metadata.append((app_type, model, mk_type, filepath))

        self._f0_decimals = f0_decimals

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, indexer: slice | int):
        item = False

        if isinstance(indexer, slice):
            iterator = self.metadata[indexer]
        elif isinstance(indexer, list):
            iterator = [self.metadata[idx] for idx in indexer]
        elif isinstance(indexer, int):
            iterator = [self.metadata[indexer]]
            item = True
        else:
            raise ValueError

        recordings = []

        for app_type, _, mk_type, filepath in iterator:
            app_type = self.default_label(app_type)
            data, fs = sf.read(filepath)

            coefs = getattr(self, mk_type)

            v, i = data[:, 0], data[:, 1]
            v = coefs['volt'] * v
            i = coefs['amp'] * i

            f0 = round(fundamental(v, fs), self._f0_decimals)

            recordings.append((v, i, fs, f0, app_type))

        if item:
            return recordings[0]

        return recordings

    def default_label(self, label: str) -> str:
        """
        Format an appliance's label by default

        Arguments:
            label: str
        Returns:
            str
        """
        label = label.lower().replace(' ', '_')

        return label

    def random(self, random_state=None):
        rng = np.random.RandomState(random_state)
        idx = rng.randint(len(self))

        return self[idx]

    @property
    def devices(self):
        return sorted(
            set([
                self.default_label(app_type) for app_type, *_ in self.metadata
            ]))
