from __future__ import annotations

# External packages
import math
import numpy as np
from scipy.stats import lognorm 

from ...utils.random import tnormal


def make_periods(
        n_samples=100,
        n_classes=2,
        n_clusters_per_class=1,
        output_size=100,
        h_loc=30,
        a_loc=5,
        centers=(-10, 10, 20),
        decay_range=(10, 30),
        z0_range=(0, 1e-1),
        cluster_std=1,
        noise_std=1e-4,
):
    n_clusters = n_classes * n_clusters_per_class

    assert h_loc > 1 and h_loc <= output_size // 2
    assert n_samples >= n_clusters

    re0, re1, imw = centers
    X = np.empty((0, output_size), dtype=float)
    y = np.empty((0, ), dtype=int)

    n_spc = n_samples // n_clusters
    n_spc = n_spc * np.ones(n_clusters, dtype=int)
    n_spc[np.arange(n_samples - n_spc.sum()) % n_clusters] += 1
    assert n_spc.sum() == n_samples

    for n, k in zip(n_spc, range(n_clusters)):
        h = np.random.poisson(h_loc)  # max harmonics
        h = np.clip(h, a_min=2, a_max=output_size // 2)
        a = np.random.poisson(a_loc)

        # Basis for random variables
        theta = np.random.uniform(-np.pi, np.pi, (n, h + 1))
        r = cluster_std * np.abs(np.random.randn(n, h + 1))

        r_max = r.max(0, keepdims=True)

        # Vertices of classes
        re = np.random.uniform(re0, re1, (1, h + 1))
        im = np.random.uniform(-r_max - imw, -r_max, (1, h + 1))
        decay = np.random.uniform(*decay_range)
        z0 = np.random.uniform(*z0_range)
        dropout_shift = np.random.randint(0, 2)

        # Random complex variables
        real = re + r * np.cos(theta)
        imag = im + r * np.sin(theta)
        Z = real + 1j * imag

        # Physics-informed model
        Z[:, 0].real = z0
        Z[:, 0].imag = 0
        x = np.arange(h + 1)
        L = lognorm.pdf(x=x, scale=1, s=np.random.uniform(0.5, 2))
        L /= np.max(L)
        Z *= L
        Z[:, 2:] *= np.fmod(np.arange(h - 1) + dropout_shift, 2)
        Z += noise_std * np.random.randn(n, h + 1)
        assert np.all(Z.imag <= 0)

        # Transform signals into time-domain
        Xk = np.fft.irfft(Z, output_size, axis=1)
        Xk = Xk / np.abs(Xk).max(1, keepdims=True)

        # Scale
        ak = a + cluster_std * np.abs(np.random.randn(n, 1))
        Xk *= ak

        # Create labels
        yk = np.ones(n, dtype=int) * k // n_clusters_per_class

        X = np.concatenate((X, Xk))
        y = np.concatenate((y, yk))

    return X, y


def make_oscillations(
        n_samples=100,
        diversity=1,
        n_classes=1,
        n_clusters_per_class=1,
        cluster_std=1,
        spectral_std=1,
        psr_range=(1, 5),
        decay_range=(5, 50),
        dt=1.0,
        fs=5000,
        f0=50.,
        **periods_kwargs,
):
    n_reps = math.ceil(dt * f0)
    period_size = round(fs / f0)
    time = np.linspace(0, dt, round(dt * f0 * period_size))

    if diversity == 0:
        divs = np.ones(n_classes) / n_classes
    else:
        divs = np.random.poisson(diversity, n_classes)
        divs = np.clip(divs, a_min=1, a_max=None)

    psr_centers = np.random.uniform(*psr_range, n_classes)
    decay_centers = np.random.uniform(*decay_range, n_classes)

    X = np.empty((0, len(time)), dtype=float)
    y = np.empty(0, dtype=int)

    for k in range(n_classes):
        Xk, _ = make_periods(
            n_samples=n_samples * divs[k],  # FIXME n_spc
            n_classes=1,
            n_clusters_per_class=n_clusters_per_class,
            cluster_std=spectral_std,
            output_size=period_size,
            **periods_kwargs)
        Xk = Xk.reshape(n_samples, divs[k], -1)
        Xk = np.repeat(Xk[:, None], n_reps, axis=1)
        Xk = Xk.reshape(n_samples, -1)
        Xk = Xk[:, :len(time)]

        decay = cluster_std * np.abs(np.random.randn(n_samples, 1))
        decay += decay_centers[k]
        psr = tnormal(a=-1,
                      b=None,
                      loc=psr_centers[k],
                      scale=cluster_std,
                      size=(n_samples, 1))

        Xk *= 1 + (psr * np.exp(-decay * time))
        yk = np.ones(n_samples) * k

        X = np.concatenate((X, Xk))
        y = np.concatenate((y, yk))

    return X, y
