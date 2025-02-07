import numpy as np


def fundamental(x: list | np.ndarray, fs: int):
    amps = abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    f0 = freqs[np.argmax(amps)]

    return f0
