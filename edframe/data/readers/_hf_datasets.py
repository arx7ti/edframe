from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd

from collections.abc import Sequence
from typing import Optional, Callable, Union


class ExhaustiveArgumentError(Exception):

    def __init__(self, arg_name, *args: object) -> None:
        message = 'Argument "{}" is exhaustive'.format(arg_name)
        self.message = message
        return super().__init__(message, *args[1:])


class Reader(Sequence):
    pass


def default_label(label: str) -> str:
    """
    Format a label by default

    Arguments:
        label: str
    Returns:
        str
    """
    label = label.lower().replace(' ', '_')
    return label


class PLAID(Reader):

    def __init__(
        self,
        dataset_path: str,
        metadata: Optional[dict] = None,
        metadata_path: Optional[str] = None,
        dtype=None,
        label_fn: Optional[Callable] = None,
    ):
        self._dataset_path = dataset_path
        if metadata is not None:
            if metadata_path is not None:
                raise ExhaustiveArgumentError('metadata_path')
            self._metadata = metadata
        elif metadata_path is not None:
            if metadata is not None:
                raise ExhaustiveArgumentError('metadata')
            with open(metadata_path) as j:
                self._metadata = json.load(j)
        else:
            raise ValueError('Metadata is required')
        self._metadata = list(
            sorted(self._metadata.items(), key=lambda x: int(x[0])))
        self._dtype = np.float32 if dtype is None else dtype
        self._label_fn = label_fn
        return None

    def __len__(self):
        return len(self._metadata)

    def __getitem__(
        self,
        idxs: Union[slice, int],
    ) -> tuple[str, np.ndarray, np.ndarray, int]:
        if isinstance(idxs, slice):
            iterator = self._metadata[idxs]
        elif isinstance(idxs, list):
            iterator = [self._metadata[idx] for idx in idxs]
        elif isinstance(idxs, int):
            iterator = []
        else:
            raise ValueError

        if len(iterator) > 0:
            samples = []
            for sample_idx, meta in iterator:
                sample = self._get(sample_idx, meta)
                samples.append(sample)
            return samples
        else:
            idx = idxs
            sample_idx, meta = self._metadata[idx]
            sample = self._get(sample_idx, meta)
            return sample

    def _get_label(self, app_info: dict) -> str:
        label = app_info['type']
        if self._label_fn is not None:
            label = self._label_fn(label)
        return label

    def _get_target(
        self,
        app_meta: dict,
        maxlen: Optional[int] = None,
    ) -> Union[str, tuple[str, list[tuple[int, int]]]]:
        label = self._get_label(app_meta)
        if app_meta.get('on') and app_meta.get('off'):
            assert maxlen is not None
            parse_fn = lambda x: re.findall("\d+", x)
            locs_on = list(map(int, parse_fn(app_meta["on"])))
            locs_off = list(map(int, parse_fn(app_meta["off"])))
            if len(locs_on) - len(locs_off) == 1:
                locs_off.append(maxlen - 1)
            elif len(locs_on) - len(locs_off) > 1:
                print(locs_on)
                print(locs_off)
                raise NotImplementedError
            locs = list(zip(locs_on, locs_off))
            return label, locs
        else:
            return label

    def _get_targets(
        self,
        apps_meta,
        maxlen: int,
    ) -> tuple[list[str], list[int]]:
        labels = []
        locs = []
        for app_meta in apps_meta:
            label, app_locs = self._get_target(app_meta, maxlen=maxlen)
            labels += [label] * len(app_locs)
            locs += app_locs
        if len(labels) > 1:
            ord = sorted(range(len(labels)), key=lambda idx: labels[idx])
            labels = [labels[idx] for idx in ord]
            locs = [locs[idx] for idx in ord]
        return labels, locs

    def _get(
        self,
        sample_idx: int,
        meta: dict,
    ) -> tuple[str, np.ndarray, np.ndarray]:

        # Read the waveforms
        fs = int(meta['header']['sampling_frequency'].replace('Hz', ''))
        filename = '{idx}.{ext}'.format(idx=sample_idx, ext='csv')
        filepath = os.path.join(self._dataset_path, filename)
        waveforms = pd.read_csv(filepath,
                                names=['current', 'voltage'],
                                dtype=self._dtype)
        v = waveforms.voltage.to_numpy()
        i = waveforms.current.to_numpy()

        # Read the meta information about an appliance/appliances
        keys = filter(lambda key: key.startswith('appliance'), meta.keys())
        keys = list(keys)

        if len(keys) != 1:
            raise ValueError('Given meta-data is not supported')

        key = keys[0]
        apps_meta = meta[key]

        if key.endswith('s'):
            y, locs = self._get_targets(apps_meta, len(i))
            return v, i, fs, y, locs
        else:
            app_meta = apps_meta
            y = self._get_target(app_meta)
            return v, i, fs, y

        # power_sample = PowerSample(y, fs, fs_type="high", locs=locs, v=v, i=i)
