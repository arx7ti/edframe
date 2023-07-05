import numpy as np
from ._generics import Feature
from ..signals._scalar import spectral_centroid, temporal_centroid, spf, rms
from ..signals._vector import fft_amplitudes, fft_amplitudes_check_fn

SpectralCentroid = Feature(spectral_centroid)
TemporalCentroid = Feature(temporal_centroid)
RMS = Feature(rms)
SpectralFlatness = Feature(spf)
FourierAmplitudes = Feature(fft_amplitudes,
                            multioutput=True,
                            check_fn=fft_amplitudes_check_fn)
