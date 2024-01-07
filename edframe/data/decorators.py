from __future__ import annotations


def feature(feature_fn):
    setattr(feature_fn, 'is_feature', True)
    return feature_fn


def safe_mode(method):

    def fn(self, *args, **kwargs):
        if self._safe_mode:
            for i, vi in enumerate(self.signatures):
                if vi.hash() not in self._hashes:
                    # Check f0
                    if vi.f0 != self.f0:
                        raise ValueError

                    # Check fs
                    if vi.fs != self.fs:
                        raise ValueError

                    # Check shape
                    if len(vi) != self.n_samples:
                        raise ValueError

        result = method(self, *args, **kwargs)

        return result

    return fn
