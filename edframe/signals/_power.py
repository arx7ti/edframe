import numpy as np
# from edframe.features._scalar import rms


def fryze(v: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = v * i
    ia = s.mean() * v / rms(v)**2
    ir = i - ia
    return ia, ir


def budeanu(
    v: np.ndarray,
    i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass
