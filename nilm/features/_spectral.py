import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numba import njit


class Spectrogram(nn.Module):
    """
    Computes the Short-Time Fourier Transform of a given mini-batch of waveforms
    """

    def __init__(
        self,
        window_size: int,
        hop_size: int,
        n_fft: int,
        power: bool = True,
        **kwargs,
    ) -> None:
        """
        Arguments:
            window_size: int
            hop_size: int - distance between the overlapped windows
        Returns:
            None
        """
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.power = power
        self.eps = kwargs.get('eps', 1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor - mini-batch of waveforms
        Returns:
            torch.Tensor
        """
        x = torch.stft(
            x,
            self.window_size,
            hop_length=self.hop_size,
            window=torch.hann_window(self.window_size, device=x.device),
            return_complex=True,
            normalized=True,
        )
        x = torch.abs(x)

        if self.power:
            x = 10 * torch.log10(torch.clamp(x, min=self.eps))

        x = x.T[:, :self.n_fft]

        return x


class DFIA(nn.Module):

    def __init__(self, frame_length=320, n_fft=None, mode='pad'):
        super().__init__()
        self.frame_length = frame_length
        self.n_fft = n_fft
        self.mode = mode

    def forward(self, x) -> torch.Tensor:
        """
        Perform Double Fourier Integral Analysis (DFIA) for 6-second data.
        """
        v, i = x
        total_samples = v.size(0)
        num_frames = total_samples // self.frame_length
        remainder = total_samples % self.frame_length

        if remainder > 0:
            if self.mode == "pad":
                pad_size = self.frame_length - remainder
                v = F.pad(v, (0, pad_size))
                i = F.pad(i, (0, pad_size))
                num_frames += 1
            elif self.mode == "omit":
                v = v[:, :num_frames * self.frame_length]
                i = i[:, :num_frames * self.frame_length]
            else:
                raise ValueError

        # Reshape into frames
        vframes = v.view(num_frames, self.frame_length)
        iframes = i.view(num_frames, self.frame_length)

        # Compute instantaneous power matrix for each frame
        P = vframes.unsqueeze(2) * iframes.unsqueeze(1)

        # Perform 2D Fourier Transform on each frame
        Z2 = torch.fft.fft2(P, norm='forward')

        if self.n_fft is not None:
            Z2 = Z2[..., :self.n_fft[0], :self.n_fft[1]]

        H = torch.abs(Z2)
        Phi = torch.angle(Z2)

        x = torch.stack([H, Phi], dim=1)

        return x
