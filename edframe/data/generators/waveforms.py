from __future__ import annotations

# External packages
import math
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.signal import impulse
from scipy.linalg import toeplitz
from datetime import datetime, timedelta

# Internal packages
from ...utils.random import tnormal, gaussian_mixture
from ...utils.common import nested_dict, to_regular_dict


def _distribute_samples(n_samples, n_appliances, n_modes_per_appliance):
    if not isinstance(n_modes_per_appliance, list | np.ndarray):
        n_modes_per_appliance = np.asarray([n_modes_per_appliance] *
                                           n_appliances)

    n_modes_per_appliance = np.asarray(n_modes_per_appliance)
    assert n_appliances == len(n_modes_per_appliance)
    assert np.all(n_modes_per_appliance > 0)
    n_clusters = n_modes_per_appliance.sum()

    if not isinstance(n_samples, list | np.ndarray):
        n_spc = n_samples // n_clusters
        n_spc = n_spc * np.ones(n_clusters, dtype=int)
        n_spc[np.arange(n_samples - n_spc.sum()) % n_clusters] += 1
        assert n_spc.sum() == n_samples
    else:
        n_samples = np.asarray(n_samples)
        n_spc = n_samples // n_modes_per_appliance
        n_spc = np.repeat(n_spc, n_modes_per_appliance)
        n_spc[np.arange(n_samples.sum() - n_spc.sum()) % n_clusters] += 1
        assert len(n_spc) == n_clusters

    # Class indices with regards to the clusters
    class_for_cluster = np.repeat(np.arange(n_appliances),
                                  n_modes_per_appliance)

    return n_spc, class_for_cluster


def make_periods_from(X, n_samples=100, reg=1e-12):
    output_size = X.shape[1]
    X = X / np.abs(X).max(1, keepdims=True)
    Z = np.fft.rfft(X, axis=-1)

    # Sample statistics
    m = np.mean(Z, axis=0, keepdims=True)
    S = np.cov(Z, rowvar=False)
    S += reg * np.eye(len(S))
    L = np.linalg.cholesky(S)
    a = np.abs(X).max(1)
    ma = a.mean()
    sa = a.std()

    # Random variables
    real, imag = np.random.randn(2, n_samples, len(L))
    Zn = real + 1j * imag

    # Synthetic periods based on statistics
    Zn = m + np.dot(Zn, L.T)
    Xn = np.fft.irfft(Zn, output_size, axis=1)
    Xn = Xn / np.abs(Xn).max(1, keepdims=True)
    an = ma + sa * np.abs(np.random.randn(n_samples, 1))
    Xn *= an

    return Xn


def make_periods(
        n_samples=100,
        n_appliances=2,
        n_modes_per_appliance=1,
        output_size=100,
        h_loc=30,
        a_loc=5,
        centers=(-10, 10, 20),
        s_range=(0.5, 2),
        ac_range=(0, 100),
        z0_range=(0, 1e-1),
        cluster_std=1,
        noise_std=1e-4,
        **kwargs,
):
    assert h_loc > 1 and h_loc <= output_size // 2
    # assert n_samples >= n_modes_per_appliance # FIXME array support

    eps = kwargs.get('eps', 1e-12)
    re0, re1, imw = centers
    X = np.empty((0, output_size), dtype=float)
    y = np.empty((0, ), dtype=int)

    n_dist, app4mode = _distribute_samples(n_samples, n_appliances,
                                           n_modes_per_appliance)
    n_modes = len(n_dist)

    for n, m in zip(n_dist, range(n_modes)):
        app = app4mode[m]
        h = np.random.poisson(h_loc)  # max harmonics
        h = np.clip(h, a_min=2, a_max=output_size // 2)
        ac_coef = np.random.uniform(*ac_range)
        a = np.random.poisson(a_loc)
        s = np.random.uniform(*s_range)
        m_theta = np.random.uniform(-np.pi, np.pi)
        m_theta = [m_theta] * n
        m_r = np.zeros(n)

        # Toeplitz matrix for time-correlation
        ac = np.exp(-np.linspace(0, 1, n) / (ac_coef + eps))
        # TODO check `cluster_std *`
        tpl = cluster_std * toeplitz(ac)

        # Basis for time-correlated random variables
        theta = np.random.multivariate_normal(m_theta, tpl, size=h + 1).T
        r = np.abs(np.random.multivariate_normal(m_r, tpl, size=h + 1)).T
        r_max = r.max(0, keepdims=True)

        # Vertices of appliances
        re = np.random.uniform(re0, re1, (1, h + 1))
        im = np.random.uniform(-r_max - imw, -r_max, (1, h + 1))
        z0 = np.random.uniform(*z0_range)
        dropout_shift = np.random.randint(0, 2)

        # Random complex variables
        real = re + r * np.cos(theta)
        imag = im + r * np.sin(theta)
        Z = real + 1j * imag

        # Physics-informed model
        Z[:, 0].real = z0
        Z[:, 0].imag = 0
        L = lognorm.pdf(x=np.arange(h + 1), scale=cluster_std, s=s)
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
        yk = np.ones(n, dtype=int) * app

        X = np.concatenate((X, Xk))
        y = np.concatenate((y, yk))

    return X, y


def make_oscillations(
        n_samples=100,
        n_appliances=1,
        n_modes_per_appliance=1,
        cluster_std=1,
        ac_range=(0, 100),
        psr_range=(1, 5),
        decay_range=(5, 50),
        dt=1.0,
        fs=5000,
        f0=50.,
        **periods_kwargs,
):
    period_size = round(fs / f0)
    time = np.linspace(0, dt, round(dt * f0 * period_size))
    n_cycles_per_signature = math.ceil(len(time) / period_size)
    # FIXME not working as expected
    n_samples = n_samples * n_cycles_per_signature

    psr_centers = np.random.uniform(*psr_range, n_appliances)
    decay_centers = np.random.uniform(*decay_range, n_appliances)

    X, y = make_periods(n_samples=n_samples,
                        n_appliances=n_appliances,
                        n_modes_per_appliance=n_modes_per_appliance,
                        cluster_std=cluster_std,
                        output_size=period_size,
                        ac_range=ac_range,
                        **periods_kwargs)

    signatures = []

    for k in np.unique(y):
        Xk = X[y == k]
        Xk = Xk.reshape(-1, n_cycles_per_signature * period_size)
        Xk = Xk[:, :len(time)]
        n = len(Xk)

        decay = cluster_std * abs(np.random.randn(n, 1))
        decay += decay_centers[k]
        psr = tnormal(a=-1,
                      b=None,
                      loc=psr_centers[k],
                      scale=cluster_std,
                      size=(n, 1))

        Xk *= 1 + (psr * np.exp(-decay * time[None]))

        signatures.append(Xk)

    signatures = np.concatenate(signatures)
    y = y[::n_cycles_per_signature]

    return signatures, y


def make_rms_cycle(
    dt=300,
    fs=1,
    level=1000,
    overshoot=2,
    decay=1,
    damping=1,
    freq=1,
    beta_noise=False,
    a=0.1,
    b=0.01,
    std=1e-2,
    **kwargs,
):
    n = np.round(fs * dt).astype(int)
    t_exp = np.linspace(0, dt, n)

    cycle = level * np.ones(n)
    cycle *= 1 + (overshoot - 1) * np.exp(-decay * t_exp)
    cycle *= 1 + impulse([(damping), (freq, 1, 1)], N=n)[1]
    cycle *= 1 + std * np.random.randn(n)

    if beta_noise:
        eps = kwargs.get('eps', 1e-12)
        a = a + eps if a == 0 else a
        b = b + eps if b == 0 else b
        cycle *= np.random.beta(a, b, n)

    cycle = np.clip(cycle, a_min=0, a_max=None)

    return cycle


def pad_cycle(cycle, pad_width, std=0.01):
    padding = np.ones(pad_width)
    padding *= 1 + std * np.random.randn(pad_width)
    padding = np.clip(padding, a_min=0, a_max=None)

    return np.concatenate((cycle, padding))


def make_rms_signature(
    n_cycles=5,
    dt=60,
    pad_width=15,
    fs=1,
    level=1000,
    overshoot=1.,
    decay=1.,
    freq=2,
    damping=0.1,
    p_beta=0.5,
    a=0.1,
    b=0.01,
    std=0.5,
    noise_std=1e-2,
    return_events=False,
):
    # Multiplicative scale
    dt = tnormal(1 / fs, loc=dt, scale=dt * std, size=n_cycles)
    pad_width = tnormal(0, loc=pad_width, scale=pad_width * std, size=n_cycles)
    pad_width = np.round(fs * pad_width).astype(int)
    level = tnormal(a=0, loc=level, scale=level * std, size=n_cycles)
    decay = tnormal(a=0, loc=decay, scale=decay * std, size=n_cycles)
    a = tnormal(0, loc=a, scale=a * std, size=n_cycles)
    b = tnormal(0, loc=b, scale=b * std, size=n_cycles)

    # Standard scale
    overshoot = tnormal(a=0, loc=overshoot, scale=std, size=n_cycles)
    freq = tnormal(a=0., loc=freq, scale=std, size=n_cycles)
    damping = damping + std * np.random.randn(n_cycles)

    # Beta Noise Conditioning
    beta = np.random.choice([1, 0], p=[p_beta, 1 - p_beta], size=n_cycles)
    beta *= np.where(level <= level.mean(), 1, 0)
    beta *= np.where(dt > dt.mean(), 1, 0)

    # Fluctuation Conditioning
    damping *= 1 - beta

    cycles = []

    if return_events:
        events = []
        on = 0

    for k in range(n_cycles):
        cycle = make_rms_cycle(dt=dt[k],
                               fs=fs,
                               level=level[k],
                               overshoot=overshoot[k],
                               decay=decay[k],
                               freq=freq[k],
                               damping=damping[k],
                               beta_noise=beta[k],
                               a=a[k],
                               b=b[k],
                               std=noise_std)

        if return_events:
            off = on + len(cycle)
            event = on, off
            events.append(event)
            on = off + pad_width[k]

        if k < n_cycles - 1:
            cycle = pad_cycle(cycle, pad_width[k], std=noise_std)

        cycles.append(cycle)

    if return_events:
        return np.concatenate(cycles), events

    return np.concatenate(cycles)


def make_rms_signatures(
        n_signatures=100,
        n_appliances=2,
        n_modes_per_appliance=1,
        fs=1,
        n_cycles_range=(1, 10),
        dt_range=(1, 1800),
        pad_width_range=(0, 120),
        level_range=(10, 1500),
        decay_range=(0.5, 10),
        a_range=(0.01, 0.1),
        b_range=(0.01, 0.1),
        overshoot_range=(1, 10),
        freq_range=(0.5, 3),
        damping_range=(-0.1, 0.1),
        p_beta_range=(0.1, 0.6),
        std=0.5,
        noise_std=0.01,
        return_events=False,
):
    n_dist, app4mode = _distribute_samples(n_signatures, n_appliances,
                                           n_modes_per_appliance)
    n_modes = len(n_dist)

    n_cycles = np.random.randint(*n_cycles_range, n_modes)
    dt = np.random.uniform(*dt_range, n_modes)
    pad_width = np.random.uniform(*pad_width_range, n_modes)
    level = np.random.uniform(*level_range, n_modes)
    decay = np.random.uniform(*decay_range, n_modes)
    a = np.random.uniform(*a_range, n_modes)
    b = np.random.uniform(*b_range, n_modes)
    overshoot = np.random.uniform(*overshoot_range, n_modes)
    freq = np.random.uniform(*freq_range, n_modes)
    damping = np.random.uniform(*damping_range, n_modes)
    p_beta = np.random.uniform(*p_beta_range, n_modes)

    signatures = []
    labels = []

    if return_events:
        events = []

    for m in range(n_modes):
        app = app4mode[m]
        n_signatures = n_dist[m]

        for _ in range(n_signatures):
            signature = make_rms_signature(n_cycles=n_cycles[m],
                                           dt=dt[m],
                                           pad_width=pad_width[m],
                                           fs=fs,
                                           level=level[m],
                                           overshoot=overshoot[m],
                                           decay=decay[m],
                                           freq=freq[m],
                                           damping=damping[m],
                                           p_beta=p_beta[m],
                                           a=a[m],
                                           b=b[m],
                                           noise_std=noise_std,
                                           std=std,
                                           return_events=return_events)

            if return_events:
                signature, evs = signature
                events.append(evs)

            signatures.append(signature)

        labels.extend([app] * n_signatures)

    if return_events:
        return signatures, labels, events

    return signatures, labels


def make_households(
        n_households=3,
        n_days=7,
        n_appliances=2,
        n_modes_range=(1, 5),
        n_activations_range=(1, 10),
        start_date=None,
        datetimefmt=False,
        peak_time=[7, 19],
        peak_weights=[0.3, 0.7],
        fs=1,
        n_cycles_range=(1, 10),
        identical_signatures=0.7,
        dt_cycle_range=(1, 1800),
        pad_width_range=(0, 120),
        level_range=(10, 1500),
        decay_range=(0.5, 10),
        a_range=(0.01, 0.1),
        b_range=(0.01, 0.1),
        overshoot_range=(1, 10),
        freq_range=(0.5, 3),
        damping_range=(-0.1, 0.1),
        p_beta_range=(0.1, 0.6),
        std=0.5,
        noise_std=0.01,
        **kwargs,
):
    assert identical_signatures >= 0 and identical_signatures <= 1

    tday = np.linspace(0, 24, round(86400 * fs))
    n_activations = np.random.randint(*n_activations_range, n_appliances)
    n_activations = np.random.poisson(n_activations,
                                      (n_households * n_days, n_appliances))
    n_modes_per_appliance = np.random.randint(*n_modes_range, n_appliances)
    activations = gaussian_mixture(loc=peak_time,
                                   scale=[std] * len(peak_time),
                                   weights=peak_weights,
                                   size=n_activations.sum(),
                                   a=tday.min(),
                                   b=tday.max())
    activations = np.abs(activations[None] - tday[:, None]).argmin(0)

    a = 0
    activations_tmp = []
    for app in range(n_appliances):
        b = n_activations[:, app].sum()
        activations_tmp.append(activations[a:a + b])
        a = b

    activations = activations_tmp
    n_signatures = n_activations.sum(0)
    assert all(n_signatures == list(map(len, activations)))

    n_unique = np.round((1 - identical_signatures) * n_signatures).astype(int)
    n0 = (n_unique == 0) & (n_signatures > 0)
    n_unique[n0] = 1
    n_identical = n_signatures - n_unique
    assert np.all(n_identical + n_unique == n_signatures)
    signatures, labels, events = make_rms_signatures(
        n_signatures=n_unique,
        n_appliances=n_appliances,
        n_modes_per_appliance=n_modes_per_appliance,
        return_events=True)
    app_cols = [f'appliance_{app}' for app in range(1, n_appliances + 1)]
    app_cols_on = [f'{app_col}_on' for app_col in app_cols]
    X = pd.DataFrame(columns=[
        'timestamp',
        'total',
        *app_cols,
        *app_cols_on,
    ])
    meta_data = nested_dict()

    if start_date is None:
        start_date = datetime.now()

    end_date = start_date + timedelta(seconds=n_days * 86400)
    timestamp = pd.date_range(start_date, end_date, periods=n_days * len(tday))

    if not datetimefmt:
        timestamp = list(map(lambda t: int(t.timestamp()), timestamp))

    X.loc[:, 'timestamp'] = timestamp
    X.loc[:, app_cols + app_cols_on] = 0
    X.loc[:, app_cols_on] = X.loc[:, app_cols_on].astype(int)
    X = X.set_index('timestamp')

    for app in range(n_appliances):
        app_signatures = [s for s, l in zip(signatures, labels) if l == app]

        if len(app_signatures) == 0:
            continue

        app_events = [e for e, l in zip(events, labels) if l == app]
        idn = np.random.randint(0, len(app_signatures), n_identical[app])
        app_signatures.extend([app_signatures[i] for i in idn])
        app_events.extend([app_events[i] for i in idn])
        n_signatures_per_house, _ = _distribute_samples(
            n_signatures[app], n_households, 1)
        app_activations = activations[app]

        a = 0
        for house in range(n_households):
            b = a + n_signatures_per_house[house]
            house_signatures = app_signatures[a:b]
            house_events = app_events[a:b]
            house_activations = app_activations[a:b]
            days = np.repeat(np.arange(n_days), len(house_activations))
            off_max = n_days * len(tday)
            meta_data[f'House_{house}'][f'appliance_{app+1}'][
                'activations'] = []
            meta_data[f'House_{house}'][f'appliance_{app+1}']['events'] = []

            for sgn, evs, act, day in zip(house_signatures, house_events,
                                          app_activations, days):
                on = day * len(tday) + act
                off = on + len(sgn)

                if off > off_max:
                    off = off_max
                    sgn = sgn[:off_max - on]

                evs = np.asarray(evs) + on
                mask = (evs >= off_max).any(1)
                evs = np.delete(evs, mask.nonzero()[0], axis=0).tolist()
                evs = list(
                    map(lambda e: (timestamp[e[0]], timestamp[e[1] - 1]), evs))

                on, off = timestamp[on], timestamp[off - 1]
                X.loc[on:off, f'appliance_{app+1}'] = sgn
                X.loc[on:off, f'appliance_{app+1}_on'] = 1
                meta_data[f'House_{house}'][f'appliance_{app+1}'][
                    'activations'].append((on, off))
                meta_data[f'House_{house}'][f'appliance_{app+1}'][
                    'events'].extend(evs)

    X.loc[:, 'total'] = X.loc[:, app_cols].sum(1)
    meta_data = to_regular_dict(meta_data)

    return X, meta_data
