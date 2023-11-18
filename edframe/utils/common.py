from __future__ import annotations

from collections import defaultdict


def nested_dict():
    build = lambda: defaultdict(build)
    ndict = build()
    return ndict


def to_regular_dict(x):
    if isinstance(x, defaultdict):
        x = {k: to_regular_dict(v) for k, v in x.items()}

    return x
