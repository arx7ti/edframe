from __future__ import annotations
from tqdm import tqdm
from beartype import abby
from ..signals import FITPS
from copy import copy, deepcopy
from collections import defaultdict
from beartype.typing import Iterable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional, Callable, Union, Any, Iterable
from edframe.signals import F, pad, roll, extrapolate, replicate, enhance, downsample, crop

# import torch
import random
import inspect
import numpy as np
import pandas as pd
import itertools as it


class Generic:

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

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
                # TODO time & memory tests
                v = copy(v)
                # v = deepcopy(v)

            if isinstance(v, Backref):
                # TODO check
                v.backref = inst

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

    def __before__(self, backref, data=None):
        pass

    def __after__(self, backref, data=None):
        pass

    def __init__(
        self,
        backref: Optional[PowerSample],
        data: Optional[Any] = None,
    ) -> None:

        self.__before__(backref, data=data)

        self._backref = backref

        if data is None:
            self._data = self.__default_data__
        else:
            self._data = data

        self.__after__(backref, data=data)

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

    # def new(self):
    #     return self.__class__(backref=self.backref, data=None)

    def update(self, **kwargs):
        x = super().update(backref=self.backref, **kwargs)
        return x

    def to_numpy(self):
        return np.asarray(self.values)

    def to_torch(self):
        return torch.Tensor(self.values)

    def clear(self):
        return self.update(data=self.__default_data__)


class AttributeExtractors(Backref):

    @property
    def values(self):
        values = [getattr(self, n) for n in self.names]
        return values

    def __after__(self, _, data):
        names = []

        for extractor in data:
            if inspect.isclass(extractor):
                attr_name = extractor.__name__
            else:
                attr_name = extractor.__class__.__name__

            attr_name = attr_name.split('.')[-1]

            # if extractor.is_estimator():
            #     attr_value = extractor.estimator
            # else:
            #     attr_value = extractor.transform
            # setattr(self, attr_name, attr_value)

            setattr(self, attr_name, extractor)
            names.append(attr_name)

        self.names = names

    def __str__(self) -> str:
        ann = "Extractors(%s)" % ", ".join(self.names)
        return ann

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, indexer):
        attr_names = self._get_attr_names(indexer)
        extractors = [getattr(self, n) for n in attr_names]

        if isinstance(indexer, Iterable):
            return self.update(data=extractors)
        else:
            extractor = getattr(self, attr_names[0])
            return extractor

    def _get_attr_name(self, indexer):
        if isinstance(indexer, int):
            if indexer > 0 and indexer < len(self.names):
                attr_name = self.names[indexer]
            else:
                raise ValueError
        elif isinstance(indexer, str):
            if indexer in self.names:
                attr_name = indexer
            else:
                raise ValueError

        return attr_name

    def _get_attr_names(self, indexer):
        attr_names = []

        if isinstance(indexer, Iterable):
            for i in indexer:
                attr_names.append(self._get_attr_name(i))
        else:
            attr_names.append(self._get_attr_name(indexer))

        return attr_names

    def pop(self, indexer):
        attr_name = self._get_attr_name(indexer)
        e = getattr(self, attr_name)
        delattr(self, attr_name)
        self.names.pop(self.names.index(attr_name))
        return e

    def drop(self, indexer):
        attr_names = self._get_attr_names(indexer)
        extractors = [getattr(self, n) for n in attr_names]
        return self.update(data=extractors)


class BackrefDataFrame(Backref):

    __default_data__ = pd.DataFrame()

    # __verbose__ = "{cls}({values})"

    # def __str__(self):
    #     return self.__verbose__.format(cls=self.__class__.__name__,
    #                                    values=self.values)
    # def __init__(self, backref: PowerSample | None, data: Any | None = None) -> None:
    #     super().__init__(backref, data)

    def __after__(self, *args, **kwargs):
        # TODO handle extractors in the old methods
        self._extractors = []

    def __getitem__(
        self,
        *indexer: Iterable[int | str],
    ) -> Backref:
        # TODO if not Iterable then item only

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

    @property
    def extractors(self):
        return AttributeExtractors(self, data=self._extractors)

    def _check_callables(self, fs):
        pass

    def count(self):
        return len(self.names)

    def pop(self, name: str) -> pd.Series:
        item = self.data.pop(name)
        idx = self.names.index(name)
        self.extractors.pop(idx)
        return item

    def drop(self, names: Iterable[str]) -> BackrefDataFrame:
        values = self.data.drop(names, axis=1)
        extractors = self.extractors.drop(names).values
        return self.update(data=values, _extractors=extractors)

    def stack(self, inst: BackrefDataFrame) -> BackrefDataFrame:
        df = pd.concat((self.data, inst.data), axis=1)
        extractors = self.extractors.values + inst.extractors.values
        return self.update(data=df, _extractors=extractors)

    # def (self, inst: BackrefDataFrame) -> BackrefDataFrame:

    # df = pd.concat((self.data, inst.data), axis=1)

    # return self.update(data=df)

    # def to_dataframe(self):
    #     return self.data


# TODO not DataFrame, but dict[timestamp, list[event.verbose_name]]
class Events(Backref):

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def extractors(self):
        return AttributeExtractors(self, data=self._extractors)

    def __call__(self, *args, **kwargs) -> BackrefDataFrame:
        return self.detect(*args, **kwargs)

    def __after__(self, *args, **kwargs):
        self._extractors = []

    def detect(
        self,
        fns: Callable | Iterable[Callable],
        source_name: Optional[str] = None,
    ) -> BackrefDataFrame:
        # self._check_callables(fns)

        data = defaultdict(list)

        for fn in fns:
            if source_name is not None:
                fn.source_name = source_name

            events = fn(self.backref)

            for loc, sign in events:
                data[loc].append((fn.verbose_name, sign))

        data = dict(data)

        return self.update(data=data, _extractors=list(fns))

    def to_features(self) -> Features:
        raise NotImplementedError


class Features(BackrefDataFrame):

    # TODO onchange
    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, values: pd.DataFrame):
        if isinstance(values, pd.DataFrame):
            self._data = values
        else:
            raise ValueError

    @property
    def values(self) -> np.ndarray:
        return self.data.values

    def __call__(self, fns: Callable | Iterable[Callable]) -> BackrefDataFrame:
        return self.extract(fns)

    def add(self, features: Events | Features) -> Features:

        if self.backref == features.backref:
            if isinstance(features, Events):
                values = self.stack(features.to_features())
            elif isinstance(features, Features):
                values = self.stack(features)
            else:
                raise ValueError
        else:
            raise ValueError

        return self.update(data=values)

    def extract(self, fns: Callable | Iterable[Callable]) -> BackrefDataFrame:
        if not isinstance(fns, Iterable):
            fns = [fns]

        dfs = []

        for fn in fns:
            # TODO Must be feature
            df = fn(self.backref)
            dfs.append(df)
            del df

        df = pd.concat(dfs, axis=1)

        d = self.update(data=df, _extractors=list(fns))
        return d
        # TODO prop estimators


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

    def __after__(self, *args, **kwargs):
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

        if abby.is_bearable(
                values,
            (list[np.ndarray], set[np.ndarray], tuple[np.ndarray, ...])):
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

    # def apply(
    #     self,
    #     fs: Callable | Iterable[Callable | tuple[int, Callable] | None],
    # ) -> Components:
    #     if self.is_allowed_transform():
    #         if isinstance(fs, Callable):
    #             data = np.apply_along_axis(fs, axis=0, arr=self.data)
    #         elif abby.is_bearable(
    #                 fs, Iterable[Callable | tuple[int, Callable] | None]):
    #             if abby.is_bearable(fs, Iterable[Callable | None]):
    #                 fs = enumerate(fs)
    #                 fs = [None if f is None else (i, f) for i, f in fs]
    #                 if len(fs) != self.count():
    #                     raise ValueError
    #             elif not abby.is_bearable(fs, Iterable[tuple[int, Callable]]):
    #                 raise ValueError

    #             data = np.empty_like(self.data)
    #             mask = np.ones(self.count(), dtype=bool)

    #             for _fs in fs:
    #                 if _fs is not None:
    #                     i, f = _fs
    #                     data[i] = f(self.data[i])
    #                     mask[i] = False

    #             data[mask] = self.data[mask]
    #         else:
    #             raise ValueError
    #     else:
    #         raise AttributeError("Transformation is not allowed")

    #     return self.update(data=data)

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

    events: Events = Events
    features: Features = Features
    components: Components = Components
    __low__: bool = False  # TODO add check
    __high__: bool = False

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
                vs = np.asarray(list(lengths.values()))

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

    def __len__(self):
        return len(self.values)

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
        if self._locs is None:
            n_labels = self.n_labels

            if n_labels == 0:
                n_labels = 1

            return np.asarray([[0, len(self.values)]] * n_labels)

        return self._locs

    @property
    def n_labels(self):
        if self.labels is None:
            return 0

        return len(self.labels)

    @property
    def appliances(self):
        return self._appliances

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels) -> None:
        self._labels = labels

    @property
    def label(self):
        if len(self.labels) > 1:
            raise AttributeError

        return self._labels[0]

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

    @State.check
    def __getitem__(self, ab: slice):
        if not isinstance(ab, slice):
            raise ValueError

        data = self.data[..., ab]
        components = self.components

        # TODO
        if components.count() > 0:
            components = components[..., ab]

        # FIXME
        if not self.isfullyfit():
            a = 0 if ab.start is None else ab.start
            b = data.shape[-1] - a if ab.stop is None else ab.stop
            locs = np.clip(self.locs, a_min=a, a_max=b)
        else:
            locs = None

        return self.update(data=data, _locs=locs, _components=components)

    def isfullyfit(self):
        return np.all((self.locs[:, 0] == 0) & (self.locs[:, 1] == len(self)))

    # @State.check
    # def apply(
    #     self,
    #     fns: Union[Callable, Iterable[Callable, F]],
    #     # source_name: Optional[str]=None,
    # ) -> PowerSample:
    #     if not isinstance(fns, Iterable):
    #         fns = [fns]

    #     if len(fns) == 0:
    #         raise ValueError

    #     ps = self.copy()

    #     for fn in fns:
    #         if not isinstance(fn, F):
    #             # TODO generalize F to features and events?
    #             arg = tuple(inspect.signature(fn).parameters)[0]
    #             # TODO values to data
    #             fn = F(fn, ("values", ), **{arg: "values"})

    #         # TODO must return signal
    #         ps = fn(ps)

    #         # if not is_signal(ps.)
    #         # if not isinstance(ps.source(), np.ndarray):
    #         #     raise ValueError
    #         # elif len(ps)

    #     return ps

    def map(self, fns):
        # TODO signal to custom value
        pass

    @property
    def values(self) -> Any:
        raise NotImplementedError

    @values.setter
    def values(self, values: Any) -> None:
        raise NotImplementedError

    def source(self, source_name: str):
        return getattr(self, source_name)

    def set_source(self, source_name: str, source_values: Any):
        setattr(self, source_name, source_values)

    def roi(self, a=None, b=None, roi=None) -> list[PowerSample]:
        if a is not None and b is not None:
            return self[a:b]

        if roi is not None:
            return roi.crop(self)

    def crop(self, a: int, b: int) -> PowerSample:
        # TODO generalize for ndarray data
        data = crop(self.data, a, b)
        return self.update(data=data)

    def random_crop(self, dn: int) -> PowerSample:
        if dn < 0:
            raise ValueError

        a = random.randint(0, dn - 1)
        b = a + dn
        return self.crop(a, b)

    def pad(self, n: int) -> PowerSample:
        data = pad(self.data, n)
        return self.update(data=data)

    def roll(self, n: int) -> PowerSample:
        data = roll(self.data, n)
        locs = np.clip(self.locs + n, a_min=0, a_max=len(self))
        # TODO roll components
        return self.update(data=data, _locs=locs)

    def enhance(self, fs: int, kind: str = "linear") -> PowerSample:
        if fs < self.fs:
            raise ValueError

        data = enhance(self.data, self.fs, fs, kind=kind)

        return self.update(data=data, fs=fs)

    def downsample(self, fs: int) -> PowerSample:
        data = downsample(self.data, self.fs, fs)
        return self.update(data=data, fs=fs)

    def replicate(self, n: int) -> PowerSample:
        data = replicate(self.data, n)
        return self.update(data=data)

    def extrapolate(self, n: int, lags: int = 10) -> PowerSample:
        data = extrapolate(self.data, n, lags=lags)
        return self.update(data=data)


class VI(PowerSample):
    __high__ = True

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
        if len(v) != len(i):
            raise ValueError

        data = np.stack((v, i), axis=0)
        super().__init__(data=data,
                         fs=fs,
                         fs_type="high",
                         f0=f0,
                         labels=labels,
                         appliances=appliances,
                         locs=locs,
                         components=components,
                         aggregation=aggregation,
                         **kwargs)
        self._sync = False or len(data.shape) == 3

    def __add__(self, sample):
        return self.agg(sample)

    def __radd__(self, sample):
        return self.agg(sample)

    def agg(self, sample: VI) -> VI:
        # TODO fix v update as well during transform
        v = np.mean((self.v, sample.v), axis=0)
        i = self.i + sample.i
        labels = self.labels + sample.labels
        locs = np.concatenate((self.locs, sample.locs))
        # TODO assure that n locs == n labels
        return self.update(v=v, i=i, labels=labels, _locs=locs)

    def is_sync(self):
        return self._sync

    @property
    def v(self):
        return self.data[0]

    @v.setter
    def v(self, v):
        self.data[0] = v

    @property
    def i(self):
        return self.data[1]

    @i.setter
    def i(self, i):
        self.data[1] = i

    @property
    def s(self):
        return self.v * self.i

    @property
    def values(self):
        return self.i

    @values.setter
    def values(self, i: np.ndarray):
        self.i = i

    def mean_cycle(self):
        if not self.is_sync():
            raise ValueError

        if len(self.data.shape) < 3:
            axes = self.data.shape
            data = self.data.reshape(*axes[:-1], -1, round(self.fs / self.f0))
        else:
            data = self.data

        data = np.mean(data, axis=1)

        return self.update(data=data)

    def sync(self):
        fitps = FITPS()
        v, i = fitps(self.v, self.i, fs=self.fs)
        v, i = v.ravel(), i.ravel()
        data = np.stack((v, i), axis=0)

        if self.f0 is None:
            return self.update(data=data,
                               _sync=True,
                               _f0=round(self.fs / v.shape[1]))

        return self.update(data=data, _sync=True)

    def roll(self, n: int) -> PowerSample:
        if abs(n) >= len(self):
            raise ValueError

        if n == 0:
            return self.update()

        period = round(self.fs / self.f0)
        data = roll(self.data, abs(n) // period * period)

        if n < 0:
            data[1, n:] = 0
        else:
            data[1, :n] = 0

        locs = np.clip(self.locs + n, a_min=0, a_max=len(self))

        return self.update(data=data, _locs=locs)


class I(PowerSample):
    __high__ = True

    def roll(self, n: int) -> PowerSample:
        if abs(n) >= len(self):
            raise ValueError

        if n == 0:
            return self.update()

        period = round(self.fs / self.f0)
        data = roll(self.data, n // period * period)

        if n < 0:
            data[n:] = 0
        else:
            data[:n] = 0

        locs = np.clip(self.locs + n, a_min=0, a_max=len(self))

        return self.update(data=data, _locs=locs)

    def __init__(
        self,
        i,
        fs,
        f0: float = None,
        labels: Optional[Union[list[str], dict[str, float]]] = None,
        components: Optional[np.ndarray] = None,
        aggregation: Optional[str] = '+',
        **kwargs: Any,
    ):
        super().__init__(data=i,
                         fs=fs,
                         fs_type="high",
                         f0=f0,
                         labels=labels,
                         components=components,
                         aggregation=aggregation,
                         **kwargs)
        self._sync = len(i.shape) == 2

    def __add__(self, sample):
        return self.agg(sample)

    def __radd__(self, sample):
        return self.agg(sample)

    def agg(self, sample: I) -> I:
        i = self.i + sample.i
        labels = self.labels + sample.labels
        locs = np.concatenate((self.locs, sample.locs))
        return self.update(i=i, labels=labels, _locs=locs)

    def is_sync(self):
        return self._sync

    @property
    def i(self):
        return self.data

    @i.setter
    def i(self, i):
        self.data = i

    @property
    def values(self):
        return self.i

    @values.setter
    def values(self, i: np.ndarray):
        self.i = i


class DataSet(Generic):
    # TODO for each subclass its own implementation
    __low__ = False
    __high__ = False
    events = Events
    features = Features

    # sample = PowerSample

    @property
    def class_names(self):
        return self._class_names

    @property
    def n_classes(self):
        return len(self._class_names)

    @property
    def labels(self):
        return self._labels

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, list):
            self._data = data
        else:
            raise ValueError

    @property
    def fs(self):
        return self.data[0].fs

    @property
    def values(self):
        return self.source("values")

    def __init__(self, data, random_seed: Optional[int] = None) -> None:
        # self._check_fs()

        self._data = data

        dtypes = set()
        class_labels = []
        labels = []
        self._n_components = 0

        for s in data:
            # TODO if no labels
            # TODO if identical labels
            if isinstance(s.labels, dict):
                _class_labels = list(s.labels.keys())
                _labels = list(s.labels.values())
            else:
                _class_labels = s.labels
                _labels = s.labels

            class_labels.append(_class_labels)
            labels.append(_labels)
            dtypes |= {
                type(v.item()) if hasattr(v, "item") else type(v)
                for v in _labels
            }
            self._n_components = max(self._n_components, len(_labels))

        self._class_names = set(it.chain(*class_labels))
        cn_dtypes = {type(cn) for cn in self._class_names}

        # TODO move labels type check inside sample's class
        if len(dtypes) != 1:
            raise TypeError("All class labels must have the same type")

        if len(cn_dtypes) != 1:
            raise TypeError("All labels must have the same type")

        cn_dtype = cn_dtypes.pop()

        if cn_dtype is not str:
            raise TypeError("Class names must be str")

        mlbin = MultiLabelBinarizer(classes=list(self._class_names))
        dtype = dtypes.pop()

        if dtype is str:
            problem_type = "classification"
        elif dtype is float:
            problem_type = "regression"
        elif dtype is int:
            problem_type = "ranking"
        else:
            raise ValueError

        y = np.zeros((len(data), len(self._class_names)),
                     dtype=float if problem_type == "regression" else int)
        mask = np.nonzero(mlbin.fit_transform(labels) > 0)

        if problem_type == "classification":
            y[mask] = 1
        else:
            # TODO not tested at all
            for maski, yi in zip(mask, labels):
                y[maski] = yi

        self._labels = y
        self._rng = np.random.RandomState(random_seed)

        self.__backref__()

    def __getitem__(self, indexer):
        # TODO everywhere: if len == 1 then just item
        if abby.is_bearable(indexer, Iterable[int]):
            data = [self.data[i] for i in indexer]
        else:
            data = self.data[indexer]

        if isinstance(data, Iterable):
            return self.update(data=data)
        else:
            return data

    def __len__(self):
        return len(self.data)

    def create(self, *args: Any, **kwargs: Any) -> PowerSample:
        sample_cls = self.data[0].__class__
        sample = sample_cls(*args, **kwargs)
        return sample

    def count(self):
        return len(self)

    def random(self):
        idx = self._rng.randint(0, self.count())
        return self[idx]

    # def apply(self, fs):
    #     data = []

    #     for sample in self.data:
    #         data.append(sample.apply(fs))

    #     return self.update(data=data)

    def is_aligned(self, source_name: Optional[str] = None):
        if source_name is None:
            source_name = "values"

        lens = [len(s.source(source_name)) for s in self.data]
        lens = np.asarray(lens)

        return all(lens[0] == lens[1:])

    def source(self, source_name: str):
        values = [s.source(source_name) for s in self.data]

        if self.is_aligned(source_name=source_name):
            values = np.asarray(values)

        return values

    def is_standalone(self):
        return np.all(self.labels.sum(0) / self.n_classes == 1)

    def train_test_split(self, test_size: float = 0.3):
        if self.is_standalone():
            stratify = np.nonzero(self.labels > 0)[1]
        else:
            stratify = self.labels.sum(1)

        train, test = train_test_split(range(self.count()),
                                       test_size=test_size,
                                       stratify=stratify)
        # Two lines below extremely slow
        X_train = [self.data[i] for i in train]
        X_test = [self.data[i] for i in test]
        Y_train = self._labels[train]
        Y_test = self._labels[test]

        # TODO to .new but deal with inconsistent labels
        train = self.update(data=X_train, _labels=Y_train)
        test = self.update(data=X_test, _labels=Y_test)

        return train, test

    def align(
        self,
        n: int = None,
        if_less: str = "pad",
    ) -> DataSet:
        # TODO overlapped cropping
        if if_less not in ["pad", "extrapolate", "drop"]:
            raise ValueError

        if n is None:
            ls = [len(s) for s in self.data]
            ls, cs = np.unique(ls, return_counts=True)
            n = ls[np.argmax(cs)]

        samples = []

        for sample in tqdm(self.data):
            if n > len(sample):
                raise ValueError(
                    "Argument `n` cannot be more than the length of a sample")

            for i in range(0, len(sample), n):
                subsample = sample[i:i + n]
                dn = n - len(subsample)

                if dn > 0 and if_less == "pad":
                    subsample = subsample.pad((0, dn))
                elif dn > 0 and if_less == "extrapolate":
                    subsample = subsample.extrapolate((0, dn))
                elif dn > 0 and if_less == "drop":
                    break

                samples.append(subsample)

        return self.new(samples)

    # TODO
    # def map(self, fs):
    #     data = []
    #     return data


class HIDataSet(DataSet):
    __high__ = True

    def sync(self):
        data = []

        for sample in tqdm(self.data):
            if hasattr(sample, "sync"):
                data.append(sample.sync())
            else:
                raise AttributeError

        return self.new(data)
