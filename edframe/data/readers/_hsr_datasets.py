from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd

from ._generics import Reader
from ...features import fundamental


class PLAID(Reader):

    def __init__(
        self,
        dirpath: str,
        metadata: dict | str,
        f0_decimals=2,
        # TODO
        # n_jobs=0,
    ):
        self._dirpath = dirpath

        if isinstance(metadata, str):
            with open(metadata) as jf:
                metadata = json.load(jf)
        elif not isinstance(metadata, dict):
            raise ValueError

        self.metadata = list(sorted(metadata.items(), key=lambda x: int(x[0])))
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

        for idx, metadata in iterator:
            fs = metadata['header']['sampling_frequency']
            fs = int(fs.replace('Hz', ''))
            filename = f'{idx}.csv'
            filepath = os.path.join(self._dirpath, filename)

            # Read the waveforms
            waveforms = pd.read_csv(filepath, names=['current', 'voltage'])
            v = waveforms.voltage.to_numpy()
            i = waveforms.current.to_numpy()

            # Mains frequency estimation
            f0 = round(fundamental(v, fs), self._f0_decimals)

            # Read the meta information about an appliance/appliances
            if 'appliance' in metadata:
                appliances = self.default_label(metadata['appliance']['type'])
                locs = None
            elif 'appliances' in metadata:
                appliances, locs = self._parse_agg_data(
                    metadata['appliances'], len(i))

            recordings.append((v, i, fs, f0, appliances, locs))

        if item:
            return recordings[0]

        return recordings

    def _parse_agg_data(self, apps_data, n_samples):
        appliances = []
        locs = []

        for app_data in apps_data:
            app_label = self.default_label(app_data['type'])
            app_locs = self._parse_locs(app_data, n_samples)

            appliances.extend([app_label] * len(app_locs))
            locs.extend(app_locs)

        if len(appliances) > 1:
            ord = sorted(range(len(appliances)),
                         key=lambda idx: appliances[idx])
            appliances = [appliances[idx] for idx in ord]
            locs = [locs[idx] for idx in ord]

        return appliances, locs

    def _parse_locs(self, app_data, n_samples):
        parse_fn = lambda x: re.findall("\d+", x)
        locs_on = list(map(int, parse_fn(app_data["on"])))
        locs_off = list(map(int, parse_fn(app_data["off"])))
        dn = len(locs_on) - len(locs_off)

        assert dn >= 0

        if dn > 0:
            locs_off.extend([n_samples] * dn)

        assert len(locs_on) == len(locs_off)

        locs = list(zip(locs_on, locs_off))

        return locs

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
