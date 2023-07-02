from __future__ import annotations

import torch
import inspect
import numpy as np
import pandas as pd

from beartype import abby
from copy import deepcopy
from beartype.typing import Iterable
from typing import Optional, Callable, Union, Any, Iterable

from edframe.signals import F


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

    def copy(self):
        return self.update()

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

    # def to_dataframe(self):
    #     return self.data


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

    __default_data__ = []
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
        return len(self.data)

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

    # def crop(self, a, b):
    #     return self.transform(crop, a, b)

    def sum(self, rule: str = "+"):
        if rule == "+":
            values = np.sum(self.values, axis=0)
        else:
            raise ValueError
        return self.backref.update(values=values, clear_components=True)


class LockedError(Exception):
    pass


class GenericState:

    def __init__(self) -> None:
        self._locked = False
        self._msg = ""

    def is_locked(self):
        return self._locked

    def raise_error(self):
        raise LockedError(self._msg)

    def lock(self, msg=None):
        msg = "" if msg is None else msg
        self._locked = True
        self._msg = msg
        self.raise_error()

    def unlock(self):
        self._locked = False
        self._msg = ""

    def verify(self):
        if self.is_locked():
            self.raise_error()


class PowerSample(Generic):
    # TODO about named vars, v,i, p etc. to be used further e.g. from Features

    events = Events
    features = Features
    components = Components

    class Meta:
        values_attr = None

    class State(GenericState):

        @classmethod
        def check(cls, method: Callable):

            def wrapper(self, *args, **kwargs):
                if not issubclass(self.__class__, PowerSample):
                    raise ValueError("Argument \"self\" is required")

                self.state.verify()

                labels = self.labels
                appliances = self.appliances
                locs = self.locs
                components = self.components

                msg = cls._get_msg_lengths(labels=labels,
                                           appliances=appliances,
                                           locs=locs,
                                           components=components)
                if msg is not None:
                    self.state.lock("Parameter(-s) %s" % msg)

                return method(self, *args, **kwargs)

            return wrapper

        @classmethod
        def check_init_args(cls, method):

            def wrapper(*args, **kwargs):
                labels = kwargs.get("labels", None)
                appliances = kwargs.get("appliances", None)
                locs = kwargs.get("locs", None)
                components = kwargs.get("components", None)

                msg = cls._get_msg_lengths(labels=labels,
                                           appliances=appliances,
                                           locs=locs,
                                           components=components)
                if msg is not None:
                    raise ValueError("Argument(-s) %s" % msg)

                return method(*args, **kwargs)

            return wrapper

        @staticmethod
        def _get_msg_lengths(
            labels=None,
            appliances=None,
            locs=None,
            components=None,
        ) -> Union[str, None]:
            lengths = {}

            if labels is not None:
                lengths.update(labels=len(labels))

            if appliances is not None:
                lengths.update(appliances=len(appliances))

            if locs is not None:
                lengths.update(locs=len(locs))

            if components is not None:
                if len(components) > 0:
                    lengths.update(components=len(components))

            if len(lengths) > 1:
                vs = list(lengths.values())

                if not np.all(vs[1:] == vs[0]):
                    msg = "%s must have the same length" % (", ".join(
                        map(lambda x: "\"%s\"" % x, lengths.keys())))

                    return msg

    @State.check_init_args
    def __init__(
        self,
        data,
        fs: Optional[int] = None,
        fs_type: str = "high",
        f0: float = None,
        labels: Optional[Union[list[str], dict[str, float],
                               list[int, float]]] = None,
        appliances: Optional[list[str]] = None,
        locs: Optional[np.ndarray] = None,
        components: Optional[list[PowerSample]] = None,
        aggregation: Optional[str] = '+',
        sort: bool = False,
    ) -> None:
        """
            y: can be either appliance(-s), or state(-s) of appliance(-s), or share(-s) of appliance(-s)
            components: stand for appliance power profile
            locs: stand for events locations
        """

        if sort and appliances is not None:
            order = np.argsort(appliances)
            appliances = appliances[order]
            if labels is not None:
                labels = labels[order]
            if locs is not None:
                locs = locs[order]
            if components is not None:
                components = components[order]

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
        self._aggregation = aggregation

        self.__backref__(components=components)

        self.state = self.State()

    def is_lazy(self):
        return self.data is None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def f0(self):
        return self._f0

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, fs):
        self._fs = fs

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
    def labels(self, labels) -> None:
        self._labels = labels

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

    @State.check
    def __getitem__(self, locs):
        data = self.data[locs]
        # TODO
        # ps._data = ps._data[locs]
        if self.components.count() > 0:
            components = self.components[:, locs]
        if self.locs is not None:
            locs = np.clips(self.locs, a_min=locs[:, 0], a_max=locs[:, 1])
            # TODO locs?
        return self.update(data=data, _locs=locs, _components=components)

    @State.check
    def apply(
        self,
        fns: Union[Callable, Iterable[Callable, F]],
    ) -> PowerSample:
        if not isinstance(fns, Iterable):
            fns = [fns]

        if len(fns) == 0:
            raise ValueError

        ps = self.copy()

        for fn in fns:
            if not isinstance(fn, F):
                arg = tuple(inspect.signature(fn).parameters)[0]
                # TODO values to data
                fn = F(fn, ("values", ), **{arg: "values"})

            ps = fn(ps)

        return ps

    @property
    def values(self):
        return getattr(self, self.Meta.values_attr)

    @values.setter
    def values(self, values):
        # TODO check setters in inherited classes
        return setattr(self, self.Meta.values_attr, values)


class VISample(PowerSample):

    class Meta:
        values_attr = "i"

    def __init__(
        self,
        v,
        i,
        fs,
        f0: float = None,
        labels: Optional[Union[list[str], dict[str, float]]] = None,
        appliances: Optional[list[str]] = None,
        locs: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
        aggregation: Optional[str] = '+',
        **kwargs: Any,
    ):
        super().__init__(data=[v, i],
                         fs=fs,
                         fs_type="high",
                         f0=f0,
                         labels=labels,
                         appliances=appliances,
                         locs=locs,
                         components=components,
                         aggregation=aggregation,
                         **kwargs)

    @property
    def v(self):
        return self.data[0]

    @v.setter
    def v(self, v):
        # if v.shape != self.v.shape:
        #     raise ValueError
        self.data[0] = v

    @property
    def i(self):
        return self.data[1]

    @i.setter
    def i(self, i):
        # if i.shape != self.i.shape:
        #     raise ValueError
        self.data[1] = i

    @property
    def s(self):
        return self.v * self.i


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

    def map(self, fs):
        data = []
        for sample in self.data:
            data.append(sample.apply(fs))
        return self.update(data=data)
