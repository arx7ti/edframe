from pathlib import Path

import os
import json
import pandas as pd
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .utils import fundamental


class PLAID:
    """PLAID dataset reader."""
    # __dataset_type__: object = HighFreqDataset
    __f0_decimals__: int = 2  # Number of decimal places for fundamental frequency rounding.

    def __init__(
        self,
        dirpath: str | Path,
        metadata: dict | str | Path,
    ):
        """
        Initializes the dataset reader.

        Args:
            dirpath (str | Path): Directory path containing the dataset files.
            metadata (dict | str | Path): Path to metadata file or dictionary containing metadata.
        """
        if not os.path.exists(dirpath):
            raise ValueError(f"Directory {dirpath} does not exist.")

        self._dirpath = Path(dirpath)

        if isinstance(metadata, (str, Path)):
            with open(metadata, 'r') as jf:
                metadata = json.load(jf)
        elif not isinstance(metadata, dict):
            raise ValueError(
                "Metadata should be a dictionary or a valid file path.")

        # Sort metadata keys numerically
        self.metadata = sorted(metadata.items(), key=lambda x: int(x[0]))
        self.devices = sorted(
            set((self.format_label(x['appliance']['type'])
                 for _, x in self.metadata)))

    def __len__(self) -> int:
        """Returns the number of available recordings."""

        return len(self.metadata)

    def __getitem__(self, indexer: slice | int | list[int]):
        """
        Retrieves dataset items by index, slice, or list of indices.

        Args:
            indexer (slice | int | list[int]): Index, slice, or list of indices.

        Returns:
            A tuple (voltage, current, fs, f0, appliances, locs) or a list of such tuples.
        """
        if isinstance(indexer, slice):
            iterator = self.metadata[indexer]
        elif isinstance(indexer, list):
            iterator = [self.metadata[idx] for idx in indexer]
        elif isinstance(indexer, int):
            iterator = [self.metadata[indexer]]
        else:
            raise ValueError(
                "Invalid index type. Must be int, slice, or list of int.")

        recordings = [
            self._read_data(idx, metadata) for idx, metadata in iterator
        ]

        return recordings[0] if isinstance(indexer, int) else recordings

    def _read_data(self, idx: str, metadata: dict):
        """
        Processes metadata and loads the corresponding dataset.

        Args:
            idx (str): Index of the recording.
            metadata (dict): Metadata for the recording.

        Returns:
            tuple: (voltage, current, fs, f0, appliances, locs)
        """
        fs = int(metadata['header']['sampling_frequency'].replace('Hz', ''))
        filepath = self._dirpath / f"{idx}.csv"

        # Read waveform data
        waveforms = pd.read_csv(filepath, names=['current', 'voltage'])
        v, i = waveforms.voltage.to_numpy(), waveforms.current.to_numpy()

        # Estimate mains frequency
        f0 = round(fundamental(v, fs), self.__f0_decimals__)

        # Parse appliance information
        if 'appliance' in metadata:
            appliances = self.format_label(metadata['appliance']['type'])
            locs = None
        elif 'appliances' in metadata:
            appliances, locs = self._parse_agg_data(metadata['appliances'],
                                                    len(i))
        else:
            appliances, locs = None, None

        return v, i, fs, f0, appliances, locs

    def _parse_agg_data(
        self,
        apps_data: list[dict],
        n_samples: int,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """
        Parses aggregated appliance data.

        Args:
            apps_data (list[dict]): List of appliance metadata.
            n_samples (int): Number of samples in the recording.

        Returns:
            tuple: (appliance names, activation periods)
        """
        appliances, locs = [], []

        for app_data in apps_data:
            label = self.format_label(app_data['type'])
            locations = self._parse_locs(app_data, n_samples)

            appliances.extend([label] * len(locations))
            locs.extend(locations)

        # Sort appliances and locs based on appliance names
        if len(appliances) > 1:
            sorted_indices = sorted(range(len(appliances)),
                                    key=lambda i: appliances[i])
            appliances = [appliances[i] for i in sorted_indices]
            locs = [locs[i] for i in sorted_indices]

        return appliances, locs

    def _parse_locs(self, app_data: dict,
                    n_samples: int) -> list[tuple[int, int]]:
        """
        Parses activation periods of appliances.

        Args:
            app_data (dict): Appliance data.
            n_samples (int): Number of samples.

        Returns:
            list of tuples: Activation (on, off) periods.
        """
        extract_ints = lambda x: list(map(int, re.findall(r"\d+", x)))

        locs_on = extract_ints(app_data.get("on", ""))
        locs_off = extract_ints(app_data.get("off", ""))

        if len(locs_on) > len(locs_off):
            locs_off.extend([n_samples] * (len(locs_on) - len(locs_off)))

        assert len(locs_on) == len(locs_off), "Mismatched on/off locations"

        return list(zip(locs_on, locs_off))

    def format_label(self, label: str) -> str:
        """
        Formats an appliance's label.

        Args:
            label (str): Original appliance label.

        Returns:
            str: Standardized appliance label.
        """

        return label.lower().replace(' ', '_')

    def random(self, random_state: int | None = None):
        """
        Returns a random dataset sample.

        Args:
            random_state (int | None): Random seed for reproducibility.

        Returns:
            tuple: A single dataset sample.
        """
        rng = np.random.default_rng(random_state)
        idx = rng.integers(len(self))

        return self[idx]
