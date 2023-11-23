from __future__ import annotations


def feature(feature_fn):
    setattr(feature_fn, 'is_feature', True)
    return feature_fn
