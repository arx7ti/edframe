from __future__ import annotations

# import torch
import inspect
from multiprocessing import Value
import numpy as np
import pandas as pd

from beartype import abby
from copy import deepcopy
from collections import defaultdict
from beartype.typing import Iterable
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
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


class Linkage: pass
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

    def __call__(self, fns: Callable | Iterable[Callable]) -> BackrefDataFrame:
        return self.extract(fns)

    def add(self, features: Union[Events, Features]) -> Features:

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
        columns = []
        values = []

        for fn in fns:
            if isinstance(fn, Linkage):
                X = self.backref.source(fn.source_name)
                col = fn.verbose_name
                args = fn.args
                kwargs = fn.kwargs
            else:
                X = self.backref.values

                try:
                    col = fn.__name__
                except AttributeError:
                    col = fn.__class__.__name__

                col = col.split('.')[-1]
                args = ()
                kwargs = {}

            is_estimator = isinstance(fn, BaseEstimator) or hasattr(fn, "fit")\
                            or hasattr(fn, "transform") or hasattr(fn, "fit_transform")
            is_dataset = isinstance(self.backref, DataSet)
            is_array = isinstance(X, np.ndarray)

            if is_estimator and is_array:
                try:
                    check_is_fitted(fn)
                except NotFittedError:

                    if is_dataset:
                        fn.fit(X)
                    else:
                        raise ValueError(
                            "The feature estimator was not fitted. "
                            "Call this feature on a dataset first")

                X = fn.transform(X)
                do_iters = False
            elif is_array and is_dataset:
                shape = list(X.shape)
                shape[1] = 1
                # TODO axis support
                X = np.apply_along_axis(fn, axis=1, arr=X, *args, **kwargs)
                X = X.reshape(*shape)
                do_iters = False
            elif is_array:
                X = [X]
                do_iters = True
            elif is_dataset:
                do_iters = True
            else:
                raise ValueError

            if do_iters:
                X = self._to_array([fn(x) for x in X])

            if len(X.shape) != 2:
                raise ValueError("2D")

            values.append(X)
            
            if X.shape[1]:
                columns.append(col)
            else:
                columns.extend([f"{col}{i}" for i in range(X.shape[1])])

        values = np.concatenate(values, axis=1)
        df = pd.DataFrame(values, columns=columns)

        return self.update(data=df, _extractors=list(fns))
    
    @classmethod
    def _to_array(cls, x: Any) -> np.ndarray:
        if isinstance(x, Iterable) and not isinstance(x, (list, tuple)):
            x = list(x)

        if isinstance(x, Iterable):
            l = None 
            xlist = []

            for _x in x:
                if isinstance(_x, (list, tuple, np.ndarray)):
                    xlist.append(_x)
                else:
                    _xlist = [_x for _x in _x]

                    if any(isinstance(__x, Iterable) for __x in _xlist):
                        raise ValueError

                    xlist.append(_xlist)

                if l is None:
                    l = len(xlist[-1])
                elif len(xlist[-1]) != l:
                    raise ValueError
        else:
            xlist = x
        
        x = np.asarray(xlist)

        if len(x.shape) == 1:
            x = x[:, None]

        return x


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

    def apply(
        self,
        fs: Callable | Iterable[Callable | tuple[int, Callable] | None],
    ) -> Components:
        if self.is_allowed_transform():
            if isinstance(fs, Callable):
                data = np.apply_along_axis(fs, axis=0, arr=self.data)
            elif abby.is_bearable(
                    fs, Iterable[Callable | tuple[int, Callable] | None]):
                if abby.is_bearable(fs, Iterable[Callable | None]):
                    fs = enumerate(fs)
                    fs = [None if f is None else (i, f) for i, f in fs]
                    if len(fs) != self.count():
                        raise ValueError
                elif not abby.is_bearable(fs, Iterable[tuple[int, Callable]]):
                    raise ValueError

                data = np.empty_like(self.data)
                mask = np.ones(self.count(), dtype=bool)

                for _fs in fs:
                    if _fs is not None:
                        i, f = _fs
                        data[i] = f(self.data[i])
                        mask[i] = False

                data[mask] = self.data[mask]
            else:
                raise ValueError
        else:
            raise AttributeError("Transformation is not allowed")

        return self.update(data=data)

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

    @State.check
    def __getitem__(self, ab: slice):
        if not isinstance(ab, slice):
            raise ValueError

        values = self.values[ab]
        components = self.components
        locs = self.locs
        # TODO
        if components.count() > 0:
            components = components[:, ab]
        if locs is not None:
            a = 0 if ab.start is None else ab.start
            b = len(values) - a if ab.stop is None else ab.stop
            locs = np.clips(locs, a_min=a, a_max=b - 1)
        return self.update(values=values, _locs=locs, _components=components)

    @State.check
    def apply(
        self,
        fns: Union[Callable, Iterable[Callable, F]],
        # source_name: Optional[str]=None,
    ) -> PowerSample:
        if not isinstance(fns, Iterable):
            fns = [fns]

        if len(fns) == 0:
            raise ValueError

        ps = self.copy()

        for fn in fns:
            if not isinstance(fn, F):
                # TODO generalize F to features and events?
                arg = tuple(inspect.signature(fn).parameters)[0]
                # TODO values to data
                fn = F(fn, ("values", ), **{arg: "values"})

            # TODO must return signal
            ps = fn(ps)

            # if not is_signal(ps.)
            # if not isinstance(ps.source(), np.ndarray):
            #     raise ValueError
            # elif len(ps)

        return ps

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

    def crop(self, a=None, b=None, roi=None) -> list[PowerSample]:
        if a is not None and b is not None:
            return self[a:b]

        if roi is not None:
            return roi.crop(self)


class VISample(PowerSample):
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
        self._2d = len(v.shape) == len(i.shape) == 2

    def is_2d(self):
        return self._2d

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


class DataSet(Generic):
    # TODO for each task its own implementation
    events = Events
    features = Features

    def __init__(self, data) -> None:
        self._data = data
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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, list):
            self._data = data
        else:
            raise ValueError

    def count(self):
        return len(self)

    def labels(self):
        pass

    def apply(self, fs):
        data = []

        for sample in self.data:
            data.append(sample.apply(fs))

        return self.update(data=data)

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

    @property
    def values(self):
        return self.source("values")

    # TODO
    # def map(self, fs):
    #     data = []
    #     return data


class VIDataSet(DataSet):
    pass