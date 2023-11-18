from __future__ import annotations

# External packages
import numpy as np
from scipy.stats import lognorm


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
