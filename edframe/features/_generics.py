from __future__ import annotations

from numbers import Number


class Feature:

    def __init__(self, fn, verbose_name: str = None, numerical=True, multioutput=False, check_fn: Callable = None):
        self._fn = fn
        self._verbose_name = fn.__name__ if verbose_name is None else verbose_name
        self._numerical = numerical
        self._multioutput = multioutput
        self._check_fn = check_fn

    @property
    def verbose_name(self):
        return self._verbose_name

    def __str__(self):
        return self.verbose_name

    def __repr__(self):
        return str(self)

    def isnumerical(self):
        return self._numerical

    def ismultioutput(self):
        return self._multioutput

    def __call__(self, x: PowerSample | np.ndarray, *args, **kwargs):

        if self._check_fn is not None:
            self._check_fn(x)

        x = self._fn(x, *args, **kwargs)

        if self.isnumerical() and not isinstance(x, Number):
            raise AttributeError

        if self.ismultioutput() and not isinstance(x, Iterable):
            raise AttributeError

        if self.ismultioutput():
            x = [x for x in x]

        return x
