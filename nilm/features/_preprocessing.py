import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numba import njit


class StandardScale(nn.Module):

    def __init__(self, mean, std) -> None:
        """
        Initialize StandardScale with mean and standard deviation.
        
        Args:
            mean: float or list/torch.Tensor
                The mean value(s) for normalization. Can be a single value or a list/torch.Tensor for per-channel normalization.
            std: float or list/torch.Tensor
                The standard deviation value(s) for normalization. Can be a single value or a list/torch.Tensor for per-channel normalization.
        """
        super().__init__()

        # Convert mean and std to tensors if they are lists
        if isinstance(mean, list):
            mean = torch.tensor(mean, dtype=torch.float32)
        if isinstance(std, list):
            std = torch.tensor(std, dtype=torch.float32)

        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Normalize the input tensor.
        
        Args:
            x: torch.Tensor
                The input tensor of shape [batch_size, channels, height, width].
        
        Returns:
            torch.Tensor
                The normalized tensor.
        """
        if isinstance(self.mean, torch.Tensor) and self.mean.ndim == 1:
            # Per-channel normalization
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)  # Reshape to [1, channels, 1, 1]
            x = (x - mean) / std
        else:
            # Global normalization
            x = (x - self.mean) / self.std

        return x


class DownSample(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        x = self.pool(x)

        return x


class DropVoltage(nn.Module):

    def __init__(self, squeeze=True):
        super().__init__()
        self.squeeze = squeeze

    def forward(self, x):
        x = x[1:]

        if self.squeeze:
            x = x.squeeze(0)

        return x


class Squeeze(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze(self.dim)

        return x


class Unsqueeze(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.unsqueeze(self.dim)

        return x


@torch.jit.script
def medfilt(v: torch.Tensor, window_size: int) -> torch.Tensor:
    assert window_size % 2 != 0, '`window_size` must be odd number.'

    pad_size = window_size // 2
    v = torch.nn.functional.pad(v[None], (0, pad_size), mode="reflect")
    v = v.squeeze(0)
    v_clean = torch.empty_like(v)

    for i in range(v.size(0) - pad_size):
        window = v[i:i + window_size]
        v_clean[i] = torch.median(window)

    return v_clean


class PAA(nn.Module):

    def __init__(self, target_size: int):
        super().__init__()
        self.target_size = target_size

    @torch.jit.export
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply Piecewise Aggregate Approximation (PAA) to reduce dimensionality of a signal along the last axis.

        Args:
            signal (torch.Tensor): Input signal of shape (..., N).

        Returns:
            torch.Tensor: Reduced signal of shape (..., target_size).
        """
        n = signal.size(-1)
        segment_size = n // self.target_size
        reduced_signal = torch.empty(*signal.shape[:-1],
                                     self.target_size,
                                     dtype=signal.dtype,
                                     device=signal.device)

        for i in range(self.target_size):
            start = i * segment_size
            end = start + segment_size
            reduced_signal[..., i] = signal[..., start:end].mean(dim=-1)

        return reduced_signal


class Mean(nn.Module):

    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(self.dim)
