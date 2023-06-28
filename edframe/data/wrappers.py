from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
from beartype import abby
from copy import deepcopy
from beartype.typing import Iterable
from typing import Optional, Callable, Union, Any, Iterable


def crop(x, a, b):
    pass


class EventDetector:
    pass


class FeatureExtractor:
    pass


class Generic:

    # __verbose__ = "{cls}()"

    # def __str__(self):
    #     return self.__verbose__.format(cls=self.__class__.__name__)

    # def __repr__(self):
    #     return str(self)

    def __backref__(self, **kwargs):
        for k, v in inspect.getmembers(self):
            if inspect.isclass(v):
                if issubclass(v, Backref):
                    data = kwargs.get(k, None)
                    setattr(self, k, v(backref=self, data=data))

    def properties(self):
        properties = [
            p for p in dir(self.__class__)
            if isinstance(getattr(self.__class__, p), property)
        ]
        return properties

    def update(self, **kwargs):

        cls = self.__class__
        inst = cls.__new__(cls)
        state_dict = {}

        for p, v in self.__dict__.items():
            if p in kwargs:
                v = kwargs[p]
                kwargs.pop(p)
            else:
                v = deepcopy(v)
            state_dict.update({p: v})

        inst.__dict__.update(state_dict)

        properties = self.properties()
        for p, v in kwargs.items():
            if p in properties:
                setattr(inst, p, v)

        return inst

    def withbackrefs(self):

        kk = []

        for k, v in self.__dict__.items():
            if issubclass(v, Backref):
                kk.append(k)

        return kk


class Backref(Generic):

    __default_data__ = None

    # TODO get source attr by method
    # ___attr__ = None

    def __before__(self):
        pass

    def __after__(self):
        pass

    def __init__(
        self,
        backref: Optional[PowerSample],
        data: Optional[Any] = None,
    ) -> None:

        self.__before__()

        self._backref = backref

        if data is None:
            self._data = self.__default_data__
        else:
            self._data = data

        self.__after__()

    @property
    def backref(self):
        return self._backref

    @backref.setter
    def backref(self, backref):
        self._backref = backref

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def related_name(self) -> str:
        for k, v in self.backref.__dict__.items():
            if isinstance(v, self.__class__):
                return k

    @property
    def values(self) -> np.ndarray:
        raise NotImplementedError

    def new(self):
        return self.__class__(backref=self.backref, data=None)

    def update(self, **kwargs):
        x = super().update(backref=self.backref, **kwargs)
        return x

    def to_numpy(self):
        return np.asarray(self.values)

    def to_torch(self):
        return torch.Tensor(self.values)

    def clear(self):
        return self.update(data=self.__default_data__)


class BackrefDataFrame(Backref):

    __default_data__ = pd.DataFrame()

    # __verbose__ = "{cls}({values})"

    # def __str__(self):
    #     return self.__verbose__.format(cls=self.__class__.__name__,
    #                                    values=self.values)

    def __getitem__(
        self,
        *indexer: Iterable[int | str],
    ) -> Backref:

        if len(indexer) == 2:
            rows, cols = indexer
        elif len(indexer) == 1:
            rows = slice(None, None, None)
            cols = indexer
        else:
            raise ValueError

        # TODO fix Iterable[...]
        if abby.is_bearable(rows, Iterable[int]) and abby.is_bearable(
                cols, Iterable[str]):
            values = self.data.loc[rows, cols]
        elif abby.is_bearable(rows, Iterable[int]) and abby.is_bearable(
                cols, Iterable[int]):
            values = self.data.iloc[rows, cols]
        else:
            raise ValueError

        return self.update(data=values)

    def __len__(self):
        return len(self.data)

    @property
    def values(self):
        return self.data.values

    @property
    def names(self):
        """
        Alias for self.keys
        """
        return self.data.columns.values

    def _check_callables(self, fs):
        pass

    def count(self):
        return len(self.columns)

    def get(
        self,
        fs: Callable | Iterable[Callable],
        variable: str = None,
    ) -> BackrefDataFrame:

        self._check_callables(fs)

        if issubclass(self.backref.__class__, PowerSet):

            df = pd.DataFrame()
            # features = self.new()

            for sample in self.backref.data:
                features = sample.features.get(fs, variable=variable)
                df = pd.concat((df, features.data), axis=0)

            df = df.reset_index(drop=True)

            return self.update(data=df)

        else:
            names = []
            values = []

            for f in fs:
                name = str(f)

                if variable is None:
                    source = self.backref.values
                else:
                    name = "%s_%s" % (variable, name)
                    source = getattr(self.backref, variable)

                names.append(name)
                values.append(f(source))

            df = pd.DataFrame([values], columns=names)

            return self.update(data=df)

    def pop(self, name: str) -> pd.Series:
        item = self.data.pop(name)
        return item

    def drop(self, names: Iterable[str]) -> BackrefDataFrame:

        values = self.data.drop(names, axis=1)

        return self.update(data=values)

    def stack(self, inst: BackrefDataFrame) -> BackrefDataFrame:

        df = pd.concat((self.data, inst.data), axis=1)

        return self.update(data=df)

    # def (self, inst: BackrefDataFrame) -> BackrefDataFrame:

    # df = pd.concat((self.data, inst.data), axis=1)

    # return self.update(data=df)

    def to_dataframe(self):
        return self.data


class Events(BackrefDataFrame):

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def to_features(self) -> Features:
        pass


class Features(BackrefDataFrame):

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, values: pd.DataFrame):
        if isinstance(values, pd.DataFrame):
            self._data = values
        else:
            raise ValueError

    def add(self, features: Union[Events, Features]) -> Features:
        # TODO events
        # TODO categorical
        # TODO onchange

        if self.backref == features.backref:
            if isinstance(features, Events):
                values = self.concat(features.to_features())
            elif isinstance(features, Features):
                values = self.concat(features)
            else:
                raise ValueError
        else:
            raise ValueError

        return self.update(data=values)


class Components(Backref):
    """
    All methods except pop and setitem produce new copies with inherited legacy from the previous object
    """

    __default_data__ = np.empty((0, ))
    __verbose__ = "{cls}(N={nc})"

    # def _check_values(values):
    #     if not abby.is_bearable(values, (list[np.ndarray],\
    #             tuple[np.ndarray, ...], np.ndarray, Components, PowerSample)):
    #         raise ValueError

    def __str__(self):
        return self.__verbose__.format(cls=self.__class__.__name__,
                                       nc=self.count())

    # def __init__(self, backref: PowerSample) -> None:
    #     # TODO why not backref? what will happen if components from different power samples?
    #     super(Components, self).__init__(backref)
    #     # TODO Can't add components of different types
    #     # TODO Can't add components of different sampling rates
    #     # self._check_values(values)

    def __after__(self):
        self._is_allowed_transform = False

    def __getitem__(
        self, indexer: np.ndarray | list[int] | set[int] | tuple[int, ...]
    ) -> Backref:
        return self.update(data=self.data[indexer])

    def __len__(self):
        return self.data.shape[0]

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(
        self,
        values: Union[np.ndarray, list[np.ndarray], set[np.ndarray],
                      tuple[np.ndarray, ...]],
    ) -> None:

        if abby.is_bearable(values, (list[np.ndarray],\
                                      set[np.ndarray], tuple[np.ndarray, ...])):
            values = np.stack(values, axis=0)
        elif not isinstance(values, np.ndarray):
            raise ValueError
        # TODO all relations to the labels in backref.labels

        self._data = values

    @property
    def values(self):
        return self.data

    def count(self):
        return len(self)

    def allow_transform(self, status: bool = True):
        self._is_allowed_transform = status

    def is_allowed_transform(self):
        return self._is_allowed_transform

    def add(self, components: Components):
        values = {}

        if isinstance(components, Components):
            values = {**self.data, **components.data}
        else:
            raise ValueError

        return self.update(data=values)

    def apply(
        self,
        fs: Callable | Iterable[Callable | tuple[int, Callable] | None],
    ) -> Components:
        if self.is_allowed_transform():
            if isinstance(fs, Callable):
                values = np.apply_along_axis(fs, axis=0, arr=self.data)
            elif abby.is_bearable(
                    fs, Iterable[Callable | tuple[int, Callable] | None]):
                if abby.is_bearable(fs, Iterable[Callable | None]):
                    fs = enumerate(fs)
                    fs = [None if f is None else (i, f) for i, f in fs]
                    if len(fs) != self.count():
                        raise ValueError
                elif not abby.is_bearable(fs, Iterable[tuple[int, Callable]]):
                    raise ValueError
                values = np.empty_like(self.data)
                mask = np.ones(self.count(), dtype=bool)
                for _fs in fs:
                    if _fs is None:
                        continue
                    else:
                        i, f = _fs
                        values[i] = f(self.data[i])
                        mask[i] = False
                values[mask] = self.data[mask]
            else:
                raise ValueError
        else:
            raise AttributeError("Transformation is not allowed")

        return self.update(data=values)

    def map(
        self,
        f: Callable,
        *args,
        **kwargs,
    ) -> list[Any]:
        values = []
        for v in self.data:
            values.append(f(v, *args, **kwargs))
        return values

    def crop(self, a, b):
        return self.transform(crop, a, b)

    def sum(self, rule: str = "+"):
        if rule == "+":
            values = np.sum(self.values, axis=0)
        else:
            raise ValueError
        return self.backref.update(values=values, clear_components=True)


class PowerSample(Generic):
    # TODO about named vars, v,i, p etc. to be used further e.g. from Features

    events = Events
    features = Features
    components = Components
    data_attr = "values"

    def __init__(
        self,
        data: Optional[Union[np.ndarray, dict[str, np.ndarray]]] = None,
        fs: Optional[int] = None,
        fs_type: str = "high",
        f0: float = None,
        labels: Optional[Union[list[str], dict[str, float]]] = None,
        appliances: Optional[list[str]] = None,
        locs: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
        metadata: Optional[list[dict[str, Union[str, int, float]]]] = None,
        aggregation: Optional[str] = '+',
        **kwargs: Any,
    ) -> None:
        """
            y: can be either appliance(-s), or state(-s) of appliance(-s), or share(-s) of appliance(-s)
            components: stand for appliance power profile
            locs: stand for events locations
        """

        # if kwargs.get("sort", True):
        #     # TODO metadata
        #     order = np.argsort(appliances)
        #     y = y[order]
        #     appliances = appliances[order]
        #     locs = locs[order]
        #     components = components[order]

        # self.check_lengths(x, y, appliances, locs, components, metadata)
        # self.check_laziness(x, components)
        # self.check_components(components)
        # self.check_locs(locs)
        # self.check_y(y)

        self._data = data
        self._fs = fs
        self._fs_type = fs_type
        self._f0 = f0
        self._labels = labels
        self._appliances = appliances
        self._locs = locs
        if abby.is_bearable(components, (list[np.ndarray],\
                                      set[np.ndarray], tuple[np.ndarray, ...])):
            components = np.stack(components, axis=0)
        # self._components = components
        self._metadata = metadata
        self._aggregation = aggregation
        # self._events = pd.DataFrame()
        # self._features = pd.DataFrame()

        self.__backref__(components=components)

    def is_lazy(self):
        return self._data is None

    @property
    def data(self):
        # TODO think which are most important and will be used!
        return self._data

    @property
    def values(self):
        return self.data

    @property
    def f0(self):
        return self._f0

    @property
    def locs(self):
        return self._locs

    @property
    def appliances(self):
        return self._appliances

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels) -> PowerSample:
        if self.components is not None:
            if len(labels) != self.components.count():
                raise ValueError
        if self.appliances is not None:
            if len(labels) != len(self.appliances):
                raise ValueError
        return self.update(_labels=labels)
        # if abby.is_bearable(labels, (list[str],tuple[str,...], set[str]))\
        #     and self.components is not None:

    # def labels(self):
    #   if abby.is_bearable(self.labels, list[str]):
    #     return self.labels
    # def new_labels(self):
    # TODO 1. if no labels but appliances
    # TODO 2. if no labels, but appliances and components
    # def update_labels(self, labels=None, comap: Callable=None, **kwargs):
    # TODO check arguments
    #   if comap is not None and labels is None:
    #     labels = self.components.map(comap, **kwargs)
    #   return self.update(labels=labels)
    # TODO 3. if labels
    # TODO 4. if labels and components

    @property
    def appliances(self):
        return self._appliances

    def __getitem__(self, locs):
        # ps = self.clear()
        data = self.data[locs]
        # TODO
        # ps._data = ps._data[locs]
        if self.components.count() > 0:
            components = self.components[:, locs]
        if self._locs is not None:
            locs = np.clips(self._locs, a_min=locs[:, 0], a_max=locs[:, 1])
            # TODO locs?
        return self.update(data=data, _locs=locs, _components=components)

    def apply(self, fs, variable=None):

        if not isinstance(fs, Iterable):
            fs = [fs]

        if variable is None:
            data = self.values
        else:
            data = getattr(self, variable)

        for f in fs:
            if issubclass(f, PowerSampleTransform):
                data, variables = f(data, *f.fetch_args(self),
                                    **f.fetch_kwargs(self))
            elif isinstance(f, Callable):
                data = f(data)
                variables = {}
            else:
                raise ValueError

        return self.update(data=data, **variables)

class PowerSet(Generic):
    events = Events
    features = Features

    def __init__(self, data) -> None:
        self._data = data
        self.__backref__()

    def __len__(self):
      return len(self.data)

    def count(self):
      return len(self)

    def labels(self):
        pass

    @property
    def data(self):
        return self._data

    def map(self, fs):
        data = []
        for sample in self.data:
            data.append(sample.apply(fs))
        return self.update(data=data)