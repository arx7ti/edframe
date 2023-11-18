from __future__ import annotations

from typing import Optional

import math
import numpy as np
from scipy.stats import truncnorm 

def randmask_2d(
    n: int,
    m: int,
    p: Optional[np.ndarray] = None,
    axis: int = 0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    if axis == 1:
        n, m = m, n
    mask = set()
    n_max_combs = 2**m
    choices = [False, True]
    if p is not None:
        if len(p) != m or not math.isclose(p.sum(), 1.0):
            raise ValueError
        p = np.stack((p, 1 - p), axis=1 - axis)
    while True:
        if len(mask) == n or len(mask) == n_max_combs:
            break
        if p is None:
            bvs = rng.choice(choices, size=m)
        else:
            choice = lambda p: rng.choice(choices, p=p)
            bvs = np.apply_along_axis(choice, 1 - axis, p)
        bvs = tuple(bvs)
        if bvs not in mask:
            mask.add(bvs)
    mask = np.stack(tuple(mask), axis=axis)
    if mask.shape[axis] < n:
        repeats = rng.choice(range(mask.shape[axis]),
                             size=n - mask.shape[axis],
                             replace=True)
        if axis == 1:
            repeats = np.s_[:, repeats]
        mask = np.concatenate((mask, mask[repeats]), axis=axis)
    return mask


def tnormal(a=None, b=None, loc=0, scale=1, size=0):
    if a is None:
        a = np.NINF

    if b is None:
        b = np.Inf

    tn = truncnorm((a - loc) / scale, (b - loc) / scale, loc=loc, scale=scale)

    return tn.rvs(size=size)
