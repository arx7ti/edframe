from __future__ import annotations

import os
import math
import pickle
import random
import numpy as np
import pandas as pd
import itertools as it
from copy import deepcopy
from numbers import Number
from inspect import isfunction
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.tools.sm_exceptions import MissingDataError

from ._fitps import FITPS


def list_f0(recordings):
    return list({f0 for _, _, _, f0, *_ in recordings})


def nearest_f0(f0_list):
    f0_list = [round(f0) for f0 in f0_list]

    if len(set(f0_list)) > 1:
        raise ValueError

    return f0_list[0]


def sync_recordings(recordings, f0_ref):
    fitps = FITPS()
    sync_recordings = []

    for v, i, fs, f0, appliances, locs in recordings:
        assert abs(f0 - f0_ref) < 1

        v, i = fitps(v, i, locs=locs, fs=fs, f0=f0_ref)
        v, i = v.ravel(), i.ravel()
        sync_recordings.append((v, i, fs, f0, appliances, locs))

    return sync_recordings
