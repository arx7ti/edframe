import math
import numpy as np
import types
import os
import torch
import pandas as pd
import random

from scipy.signal import resample
from tqdm import tqdm
import warnings
import fitps


class Datman:

    @classmethod
    def read(cls, reader):
        sampling_range = reader.__sampling_range__
        data = list(reader)

        return cls(data, sampling_range)

    def __init__(self, data=None, sampling_range=None):
        self.__sampling_range__ = sampling_range
        self.data = data

        for name, method in self.__sampling_range__.__dict__.items():
            if callable(method):
                setattr(self, name, types.MethodType(method, self))

    def is_read(self):
        return self._data__ is not None