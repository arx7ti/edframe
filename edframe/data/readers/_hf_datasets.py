from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd

from typing import Optional
from collections.abc import Sequence


class ExhaustiveArgumentError(Exception):

    def __init__(self, arg_name, *args: object) -> None:
        message = 'Argument "{}" is exhaustive'.format(arg_name)
        self.message = message
        return super().__init__(message, *args[1:])


class Reader(Sequence):
    pass


class PLAID(Reader):

    def __init__(self, dirpath: str, metadata: dict | str):
        self._dirpath = dirpath

        if isinstance(metadata, str):
            with open(metadata) as jf:
                metadata = json.load(jf)
        elif not isinstance(metadata, dict):
            raise ValueError

        self.metadata = list(sorted(metadata.items(), key=lambda x: int(x[0])))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, indexer: slice | int):
        if isinstance(indexer, slice):
            iterator = self.metadata[indexer]
        elif isinstance(indexer, list):
            iterator = [self.metadata[idx] for idx in indexer]
        elif isinstance(indexer, int):
            iterator = [self.metadata[indexer]]
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

            # Read the meta information about an appliance/appliances
            if 'appliance' in metadata:
                apps_data = [metadata['appliance']]
            elif 'appliances' in metadata:
                apps_data = metadata['appliances']

            appliances, locs = self._parse_appliances(apps_data, len(i))
            recordings.append((v, i, fs, appliances, locs))

        return recordings

    def _parse_appliances(self, apps_data, n_samples: int = None):
        appliances = []
        locs = []

        for app_data in apps_data:
            labels, app_locs = self._parse_appliance(app_data, n_samples)
            appliances.extend(labels)
            locs.extend(app_locs)

        if len(appliances) > 1:
            ord = sorted(range(len(appliances)),
                         key=lambda idx: appliances[idx])
            appliances = [appliances[idx] for idx in ord]
            locs = [locs[idx] for idx in ord]

        return appliances, locs

    def _parse_appliance(self, app_data, n_samples=None):
        app_label = self.default_label(app_data['type'])

        if app_data.get('on') and app_data.get('off'):
            assert n_samples is not None

            parse_fn = lambda x: re.findall("\d+", x)
            locs_on = list(map(int, parse_fn(app_data["on"])))
            locs_off = list(map(int, parse_fn(app_data["off"])))
            dn = len(locs_on) - len(locs_off)

            assert dn >= 0

            if dn > 0:
                locs_off.extend([n_samples] * dn)

            assert len(locs_on) == len(locs_off)

            locs = list(zip(locs_on, locs_off))
        else:
            locs = [None]

        labels = [app_label] * len(locs)

        return labels, locs

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
