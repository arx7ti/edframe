from __future__ import annotations

import numpy as np
from numba import njit


@njit
def zero_crossings(x):
    x0 = np.empty(0, dtype=np.int32)

    for j in np.arange(len(x) - 1, dtype=np.int32):
        if ((x[j] > x.dtype.type(0.0)) & (x[j + 1] <= x.dtype.type(0.0))) or (
            (x[j] < x.dtype.type(0.0)) & (x[j + 1] >= x.dtype.type(0.0))):
            x0 = np.append(x0, j)
        else:
            continue

    if len(x0) % 2 == 0:
        x0 = x0[:-1]

    return x0

@njit
def zero_crossings_pos(x):
    x0 = np.empty(0, dtype=np.int32)

    for j in np.arange(len(x) - 1, dtype=np.int32):
        if (x[j] < x.dtype.type(0.0)) & (x[j + 1] >= x.dtype.type(0.0)):
            x0 = np.append(x0, j)
        else:
            continue

    if len(x0) % 2 == 0:
        x0 = x0[:-1]

    return x0
