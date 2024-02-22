from __future__ import annotations

import os
import math
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import itertools as it

from tqdm import tqdm
from copy import deepcopy
from numbers import Number
from inspect import isfunction
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.tools.sm_exceptions import MissingDataError

from ._fitps import FITPS
from ..utils.exceptions import OutliersDetected


def nearest_f0(recordings):
    f0_list = list({f0 for _, _, _, f0, *_ in recordings})
    f0 = int(round(np.mean(f0_list)))

    return f0


def sync_recordings(
    recordings,
    window_size=1,
    outlier_thresh=0.1,
    zero_thresh=1e-4,
    f0_ref=None,
):
    fitps = FITPS(window_size=window_size,
                  outlier_thresh=outlier_thresh,
                  zero_thresh=zero_thresh)
    sync_recordings = []

    if f0_ref is None:
        f0_ref = nearest_f0(recordings)

    for v, i, fs, _, appliances, locs in tqdm(recordings):
        try:
            fitps.fit(v)
        except OutliersDetected:
            continue
        else:
            T = math.ceil(fs / f0_ref)
            v = fitps.transform(v, cycle_size=T)
            i = fitps.transform(i, cycle_size=T, locs=locs)

            if isinstance(i, tuple):
                i, locs = i

            v, i = v.ravel(), i.ravel()
            sync_recordings.append((v, i, fs, f0_ref, appliances, locs))

    dn = len(recordings) - len(sync_recordings)

    if dn > 0:
        warnings.warn(f'{dn} outliers were omitted.')

    return sync_recordings


def align_signatures(
    signatures,
    extrapolate_if_short=True,
    split_if_long=True,
    progress_bar=True,
):
    asignatures = []
    n_samples = [vi.n_samples for vi in signatures]
    n_samples = int(np.median(n_samples))

    for vi in tqdm(signatures, disable=not progress_bar):
        k = len(vi) // n_samples

        if k > 1 and split_if_long:
            for j in range(k - 1):
                a, b = j * n_samples, (j + 1) * n_samples
                vij = vi[a:b]

                if j == k - 1:
                    vij = vi[(j + 1) * n_samples:]

                    if len(vij) != n_samples and extrapolate_if_short:
                        vij = vij.extrapolate(n_samples - len(vij))

                asignatures.append(vij)

        if k == 0 and extrapolate_if_short:
            vi = vi.extrapolate(n_samples - len(vi))

    return asignatures
