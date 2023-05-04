from __future__ import annotations

import numpy as np


def gaussian_noise(
    x: np.ndarray,
    mean: float = 0.0,
    std: float = 0.01,
    kind: str = 'multiplicative',
) -> np.ndarray:
    noise = mean + std * np.random.randn(*x.shape)
    if kind == 'additive':
        x = x + noise
    elif kind == 'multiplicative':
        x = x * noise
    else:
        raise ValueError
    return x