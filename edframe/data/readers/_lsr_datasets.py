from __future__ import annotations

import os
import re
import json
import yaml
import numpy as np
import pandas as pd

from ._generics import Reader


class UKDALE(Reader):

    def __init__(
        self,
        dirpath: str,
        metadata_path: dict | str,
        default_threshold=5,
    ) -> None:
        with open(metadata_path) as yf:
            metadata = yaml.safe_load(yf)

        metadata_dirpath = os.path.dirname(metadata_path)
        meters_path = os.path.join(metadata_dirpath, 'meter_devices.yaml')

        with open(meters_path) as yf:
            meter_devices = yaml.safe_load(yf)

        filenames = {}

        for filename in os.listdir(dirpath):
            regexp = re.match(r'^channel_(\d+)\.dat$', filename)

            if regexp:
                channel = int(regexp.group(1))
                filenames[channel] = filename

        filenames = dict(sorted(filenames.items(), key=lambda x: x[0]))

        labelspath = os.path.join(dirpath, 'labels.dat')
        labels = pd.read_csv(labelspath,
                             sep=' ',
                             header=None,
                             index_col=0,
                             names=['labels'])
        labels = labels.to_dict()['labels']

        assert len(filenames) == len(labels)

        meters = {}

        for channel, meter in metadata['elec_meters'].items():
            meters[channel] = meter_devices[meter['device_model']]

        appliances = {}

        for app_data in metadata['appliances']:
            app_name = app_data.pop('original_name', None)

            if app_name is not None:
                appliances[app_name] = app_data

        self._dirpath = dirpath
        self._filenames = list(filenames.items())
        self._labels = labels
        self._meters = meters
        self._appliances = appliances
        self._default_threshold = default_threshold

        self.metadata = metadata

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, indexer: slice | int):
        item = False

        if isinstance(indexer, slice):
            iterator = self._filenames[indexer]
        elif isinstance(indexer, list):
            iterator = [self._filenames[idx] for idx in indexer]
        elif isinstance(indexer, int):
            iterator = [self._filenames[indexer]]
            item = True
        else:
            raise ValueError

        recordings = []

        for channel, filename in iterator:
            filepath = os.path.join(self._dirpath, filename)
            data = pd.read_csv(filepath,
                               sep=' ',
                               header=None,
                               index_col=0,
                               names=['p'])
            timeline = pd.to_datetime(data.index * 10**9, unit='ns')
            p = data['p'].values.astype(float)

            appliance = self._labels[channel]
            fs = self._meters[channel]['sample_period']
            app_data = self._appliances.get(appliance, {})
            on_power = app_data.get('on_power_threshold',
                                    self._default_threshold)

            recordings.append((p, fs, timeline, on_power, appliance))

        if item:
            return recordings[0]

        return recordings
