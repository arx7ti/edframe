from pathlib import Path
from collections.abc import Sequence

import os
import numpy as np
import soundfile as sf
import logging
import audioread

from .utils import fundamental

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WHITED:
    """WHITED dataset reader"""
    __f0_decimals__: int = 2  # Number of decimal places for fundamental frequency rounding.
    # Scaling coefficients for different measurement kits
    __factors__ = {
        "MK1": {
            "volt": 1033.64,
            "amp": 61.4835
        },
        "MK2": {
            "volt": 861.15,
            "amp": 60.200
        },
        "MK3": {
            "volt": 988.926,
            "amp": 60.9562
        },
    }

    def __init__(self, dirpath: str | Path):
        """
        Initializes the WHITED dataset reader.

        Args:
            dirpath (str | Path): Directory path containing the FLAC files.
        """
        if not os.path.exists(dirpath):
            raise ValueError(f"Directory {dirpath} does not exist.")

        self._dirpath = Path(dirpath)

        # Collect metadata from file names
        self.metadata = self._parse_metadata()

        # Extract unique appliance labels
        self.devices = sorted({
            self.format_label(app_type)
            for app_type, _, _, _ in self.metadata
        })

    def __len__(self) -> int:
        """Returns the number of available recordings."""

        return len(self.metadata)

    def __getitem__(self, indexer: slice | int | list[int]):
        """
        Retrieves dataset items by index, slice, or list of indices.

        Args:
            indexer (slice | int | list[int]): Index, slice, or list of indices.

        Returns:
            A tuple (voltage, current, fs, f0, appliance_type) or a list of such tuples.
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
            self._read_data(app_type, mk_type, filepath)
            for app_type, _, mk_type, filepath in iterator
        ]

        return recordings[0] if isinstance(indexer, int) else recordings

    def _parse_metadata(self) -> list[tuple[str, str, str, Path]]:
        """
        Parses metadata from file names in the dataset directory.

        Returns:
            list of tuples: (appliance_type, model, mk_type, filepath)
        """
        metadata = []

        for filepath in self._dirpath.glob("*.flac"):
            # Extract filename without extension
            parts = filepath.stem.split("_")

            if len(parts) < 4:
                logger.warning(f"Skipping invalid file name: {filepath.name}")
                continue

            app_type, model, _, mk_type = parts[:4]
            metadata.append((app_type, model, mk_type, filepath))

        return metadata

    def _read_data(self, app_type: str, mk_type: str, filepath: Path):
        """
        Reads waveform data from a FLAC file and applies calibration.

        Args:
            app_type (str): Type of appliance.
            mk_type (str): Measurement kit type.
            filepath (Path): Path to the FLAC file.

        Returns:
            tuple: (voltage, current, fs, f0, appliance_type)
        """
        app_type = self.format_label(app_type)

        with audioread.audio_open(filepath) as f:
            fs = f.samplerate
            num_channels = f.channels
            data = np.frombuffer(b''.join(f), dtype=np.int16)
            data = data.reshape(-1, num_channels).astype(np.float32) / 32768.0

        if mk_type not in self.__factors__:
            raise ValueError(f"Unknown measurement kit type: {mk_type}")

        coefs = self.__factors__[mk_type]

        v = coefs['volt'] * data[:, 0]
        i = coefs['amp'] * data[:, 1]

        # Estimate mains frequency
        f0 = round(fundamental(v, fs), self.__f0_decimals__)

        return v, i, fs, f0, app_type

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
