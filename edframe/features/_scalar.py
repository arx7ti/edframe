from ._generics import Feature
from ..signals._scalar import spectral_centroid, temporal_centroid, spectral_flatness, rms


class SpectralCentroid(Feature):
    transform = spectral_centroid


class TemporalCentroid(Feature):
    transform = temporal_centroid


class SpectralFlatness(Feature):
    transform = spectral_flatness


class RMS(Feature):
    transform = rms
