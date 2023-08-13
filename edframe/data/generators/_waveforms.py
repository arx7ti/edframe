from __future__ import annotations

# External packages
from scipy.stats import truncnorm
from typing import Any, Union, Optional

import numpy as np

# Internal packages
from ... import utils
from ... import signals
from ..entities import I, HIDataSet


class WaveformModel:

    def __init__(self, **kwargs: Any) -> None:
        self._rng_state = kwargs.pop('random_state', None)
        self._rng = np.random.RandomState(self._rng_state)

        params = kwargs.pop('params', {})
        assert params
        repeats = params.pop('repeats', {})

        self._bounds = {}
        self._groups_pos = {}
        prev_pos = 0
        for param_name, bounds in params.items():
            count = repeats.get(param_name, 1)
            if count == 1:
                self._bounds.update({param_name: bounds})
            else:
                for i in range(1, count + 1):
                    self._bounds.update({'%s%d' % (param_name, i): bounds})
            self._groups_pos[param_name] = (prev_pos, count)  # idx, len
            prev_pos += count
        self._bounds = dict(self._bounds)

        self.clear()

    def _take_group(self, P: np.ndarray, group_name: str) -> np.ndarray:
        if group_name not in self._groups_pos:
            raise ValueError
        if len(P.shape) == 1:
            P = P[None]
            squeeze = True
        else:
            squeeze = False
        idx, count = self._groups_pos[group_name]
        P = P[:, idx:idx + count]
        if squeeze:
            P = P[0]
        return P

    def _take_groups(
            self,
            P: np.ndarray,
            group_names: Optional[list[str]] = None) -> tuple[np.ndarray]:
        if group_names is None:
            group_names = self._groups_pos.keys()
        groups = []
        for group_name in group_names:
            groups.append(self._take_group(P, group_name))
        return groups

    def __call__(self, P: np.ndarray) -> np.ndarray:
        return self.transform(P)

    @property
    def bounds(self):
        return self._bounds

    @property
    def params(self):
        return tuple(self.bounds.keys())

    @property
    def n_params(self) -> int:
        return len(self.bounds)

    @property
    def param_groups(self):
        return self._param_groups

    @property
    def centers(self):
        self.check_has_centers()
        return self._centers.get('data', None)

    @property
    def labels(self):
        self.check_has_centers()
        return self._centers.get('labels', None)

    @property
    def classes(self):
        return np.unique(self.labels)

    @property
    def mask(self):
        self.check_has_centers()
        return self._centers.get('mask', None)

    def clear(self) -> None:
        self._centers = {'data': None, 'labels': None, 'mask': None}

    def update(self, **kwargs):
        self._centers.update(**kwargs)

    def check_has_centers(self):
        centers = self._centers.get('data', None)
        labels = self._centers.get('labels', None)
        if centers is None and labels is None:
            raise ValueError

    def transform(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def centers_from(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Optional[str] = None,
        median: bool = True,
        **kwargs: Any,
    ) -> None:
        self.clear()
        P = self.inverse_transform(X, optimizer=optimizer, **kwargs)
        C, y = utils.clusters.centroids(P, y, median=median)
        self.update(data=C, labels=y)

    def init_centers(
        self,
        n_classes: int = 2,
        n_clusters_per_class: int = 1,
        kind: str = 'linear',
        sep: float = 1.0,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.clear()
        adjust_kwargs = kwargs.pop('adjust_kwargs', {})
        C, y = self._make_centers(n_classes=n_classes,
                                  n_clusters_per_class=n_clusters_per_class,
                                  sep=sep,
                                  kind=kind)
        try:
            C, mask = self._adjust_centers(C, **adjust_kwargs)
        except NotImplementedError:
            mask = None
        self.update(data=C, labels=y, mask=mask)

    def make_samples(
        self,
        n_samples: int = 100,
        imbalanced_clusters: bool = False,
        class_weights: Optional[Union[list[float], np.ndarray]] = None,
        std: float = 1.0,
        shuffle: bool = False,
        noise: bool = False,
        normalize: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.check_has_centers()
        P, y = self._make_blobs(self.centers,
                                self.labels,
                                n_samples=n_samples,
                                class_weights=class_weights,
                                imbalanced_clusters=imbalanced_clusters,
                                std=std,
                                mask=self.mask,
                                shuffle=shuffle)

        X = self.transform(P)

        if noise:
            noise_params = kwargs.pop('noise_params', (0, 1e-3))
            noise_kind = kwargs.pop('noise_kind', 'multiplicative')
            X = signals.gaussian_noise(X, *noise_params, kind=noise_kind)

        if normalize:
            X /= np.abs(X).max(axis=1, keepdims=True)

        return X, y

    def _make_centers(
        self,
        n_classes: int = 2,
        n_clusters_per_class: int = 1,
        kind: str = 'linear',
        sep: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_clusters = n_classes * n_clusters_per_class
        lower_bounds, upper_bounds = zip(*self.bounds.values())

        if kind == 'linear':
            if sep < 1.0:
                raise ValueError('\'sep\' should be no less than 1.0')
            n_inits = int(sep * 2 * n_clusters)
            n_inits += n_inits % (2 * n_clusters)
            boxes = np.linspace(lower_bounds, upper_bounds, n_inits)
            boxes = boxes.reshape(n_inits // 2, 2, self.n_params)
            rnd = self._rng.randn(len(boxes)).argsort()
            boxes = boxes[rnd[:n_clusters]]
            C = self._rng.uniform(*boxes.transpose(1, 0, 2))
        elif kind == 'random':
            C = self._rng.uniform(lower_bounds,
                                  upper_bounds,
                                  size=(n_clusters, self.n_params))
            C *= sep
        else:
            raise ValueError

        y = np.arange(n_classes)
        y = np.repeat(y, n_clusters_per_class)

        return C, y

    def _make_blobs(
        self,
        centers: np.ndarray,
        y: np.ndarray,
        n_samples: int = 100,
        class_weights: Optional[Union[list[float], np.ndarral]] = None,
        imbalanced_clusters: bool = False,
        std: float = 1.0,
        mask: Optional[np.ndarray] = None,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = kwargs.pop('alpha', 10)
        classes, n_cpc_distr = np.unique(y, return_counts=True)
        n_classes = len(classes)
        n_clusters = len(centers)
        assert n_cpc_distr.sum() == n_clusters

        if class_weights is None:
            class_weights = np.ones(n_classes, dtype=float) / n_classes
        else:
            assert len(class_weights) == n_classes
            class_weights = np.asarray(class_weights)

        n_spc_distr = []

        for n_cpc, w in zip(n_cpc_distr, class_weights):
            if imbalanced_clusters:
                shares = self._rng.dirichlet(alpha * np.ones(n_cpc), size=1)
                counts = np.floor(w * shares * n_samples).ravel()
            else:
                count = np.floor(w / n_cpc * n_samples)
                counts = np.asarray([count] * n_cpc)

            counts = counts.astype(int)

            if np.any(counts == 0):
                # TODO msg
                raise ValueError('Try to increase number of samples.')

            n_spc_distr.append(counts)

        n_spc_distr = np.concatenate(n_spc_distr)
        n_spc_distr[np.arange(n_samples - n_spc_distr.sum()) % n_clusters] += 1
        assert n_spc_distr.sum() == n_samples

        P = []
        l, u = zip(*self.bounds.values())
        l, u = np.asarray(l), np.asarray(u)

        for n_spc, locs in zip(n_spc_distr, centers):
            a, b = (l - locs) / std, (u - locs) / std
            rvs = truncnorm.rvs(a,
                                b,
                                locs,
                                std,
                                size=(n_spc, len(locs)),
                                random_state=self._rng_state)
            P.append(rvs)

        P = np.concatenate(P)
        y = np.repeat(np.repeat(classes, n_cpc_distr), n_spc_distr)

        if mask is not None:
            mask = np.repeat(mask, n_spc_distr, axis=0)
            P[~mask] = 0.0

        if shuffle:
            rnd = self._rng.randn(n_samples).argsort()
            P, y = P[rnd], y[rnd]

        return P, y


class FourierModel(WaveformModel):
    EPS = 1e-9

    def __init__(
        self,
        dt: float = 0.1,
        fs: int = 30000,
        f0: float = 60.0,
        n_harmonics: int = 10,
        a: tuple[float, float] = (0, 1),
        # phi: tuple[float, float] = (-np.pi / 4, np.pi / 4),
        theta: tuple[float, float] = (-np.pi, np.pi),
        gamma: tuple[float, float] = (0, 100),
        random_state: Optional[int] = None,
    ) -> None:
        self._fs = fs
        self._f0 = f0
        self._t_axis = np.linspace(self.EPS,
                                   dt,
                                   round(dt * fs),
                                   endpoint=False)
        self._n_harmonics = n_harmonics
        params = {
            'a': a,
            # 'phi': phi,
            'theta': theta,
            'gamma': gamma,
            'repeats': {
                'a': n_harmonics,
                'theta': n_harmonics
            },
        }
        super().__init__(params=params, random_state=random_state)

    @property
    def n_harmonics(self) -> int:
        return self._n_harmonics

    def init_centers(
        self,
        n_classes: int = 2,
        n_clusters_per_class: int = 1,
        kind: str = 'linear',
        sep: float = 1.0,
        f0_bias: float = 1.0,
        mask_probas: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        adjust_kwargs = {'f0_bias': f0_bias, 'mask_probas': mask_probas}
        super().init_centers(n_classes=n_classes,
                             n_clusters_per_class=n_clusters_per_class,
                             kind=kind,
                             sep=sep,
                             adjust_kwargs=adjust_kwargs)

    def _adjust_centers(
        self,
        centers: np.ndarray,
        f0_bias: float = 1.0,
        mask_probas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # a, phi, *param_groups = self._take_groups(centers)
        a, *param_groups = self._take_groups(centers)
        h_decay = f0_bias / np.log(len(a))**2
        h_axis = np.arange(self.n_harmonics)
        h_decay = np.exp(-h_decay * h_axis[None])
        a *= h_decay
        a[:, 0] += a.max(axis=1) - a[:, 0]
        a[:, 0] *= self.n_harmonics**0.5

        # centers = np.concatenate((a, phi, *param_groups), axis=1)

        mask = np.ones(centers.shape, dtype=bool)

        a_pos, a_count = self._groups_pos['a']
        submask = utils.random.randmask_2d(len(mask),
                                           a_count - 1,
                                           axis=0,
                                           p=mask_probas,
                                           random_state=self._rng_state)
        mask[:, a_pos + 1:a_pos + a_count] = submask

        theta_pos, _ = self._groups_pos['theta']
        mask[:, theta_pos:theta_pos + 1] = np.zeros((len(mask), 1), dtype=bool)

        centers = np.concatenate((a, *param_groups), axis=1)
        # centers *= mask

        return centers, mask

    def transform(self, P: np.ndarray) -> np.ndarray:
        if len(P.shape) == 1:
            P = P[None]
            squeeze = True
        else:
            squeeze = False
        wt = 2 * np.pi * self._f0 * self._t_axis[None]
        h = np.arange(1, self.n_harmonics + 1)
        # a = P[:, :self.n_harmonics]
        a = self._take_group(P, 'a')
        # phi = self._take_group(P, 'phi')
        theta = self._take_group(P, 'theta')
        gamma = self._take_group(P, 'gamma')
        # print(gamma.min(), gamma.max())
        # theta = P[:, self.n_harmonics:2 * self.n_harmonics]
        # gamma = P[:, -1, None]
        # print('theta', theta[:, 0].min(), theta[:, 0].max())
        # X = a[..., None] * np.sin(h[..., None] * (wt + phi[..., None]) +\
        #                             theta[..., None])
        X = a[..., None] * np.sin(h[..., None] * wt + theta[..., None])
        X = np.sum(X, axis=1)
        # print((1 + np.exp(-gamma * self._t_axis[None])).min(),
        #       (1 + np.exp(-gamma * self._t_axis[None]).max()))
        X *= 1 + np.exp(-gamma * self._t_axis[None])
        if squeeze:
            X = X[0]
        return X

    def make_samples(self,
                     n_samples: int = 100,
                     imbalanced_clusters: bool = False,
                     class_weights: list[float] | np.ndarray | None = None,
                     std: float = 1,
                     shuffle: bool = False,
                     noise: bool = False,
                     normalize: bool = False,
                     **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        X, y = super().make_samples(n_samples, imbalanced_clusters,
                                    class_weights, std, shuffle, noise,
                                    normalize, **kwargs)
        samples = []

        for i, label in zip(X, y):
            sample = I(i=i, fs=self._fs, f0=self._f0, labels=[str(label)])
            # TODO remove underscore from _{name} arguments in _update() method
            sample = sample.update(_sync=True, _period=len(self._t_axis))
            samples.append(sample)

        dataset = HIDataSet(samples)

        return dataset