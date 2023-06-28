import numpy as np
from ._generics import Feature


def spectral_centroid(x):
    x = np.abs(np.fft.rfft(x))[1:]
    x = (x * np.arange(1, len(x) + 1)).sum() / x.sum()
    return x


def temporal_centroid(x):
    x = np.abs(np.fft.rfft(x))[2:]
    x = (x * 60 * np.arange(1, len(x) + 1)).sum() / x.sum()
    return x


def spf(x):
    x = np.abs(np.fft.rfft(x))[2:]
    x = np.power(np.prod(x), 1 / len(x)) / np.mean(x)
    return x


def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def fft_amplitudes(x):
    x = np.fft.rfft(x) * 2 / len(x)
    x = np.abs(x)
    return x

def fft_amplitudes_check_fn(x):
    if len(x.shape) > 1:
        raise ValueError

SpectralCentroid = Feature(spectral_centroid)
TemporalCentroid = Feature(temporal_centroid)
RMS = Feature(rms)
SpectralFlatness = Feature(spf)
FourierAmplitudes = Feature(fft_amplitudes, multioutput=True,check_fn=fft_amplitudes_check_fn)
