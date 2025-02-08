import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numba import njit


@njit
def vicoding(v, i, image_size=128):
    max_v = np.max(np.abs(v))
    max_i = np.max(np.abs(i))

    norm_v = v / max_v
    norm_i = i / max_i

    canvas_red = np.zeros((image_size, image_size), dtype=np.uint8)
    canvas_green = np.zeros((image_size, image_size), dtype=np.uint8)
    canvas_blue = np.zeros((image_size, image_size), dtype=np.uint8)

    for v, i in zip(norm_v, norm_i):
        x = int((v + 1) / 2 * (image_size - 1))
        y = int((1 - (i + 1) / 2) * (image_size - 1))
        canvas_red[y, x] = 255

    slopes = []

    for idx in range(len(norm_v) - 1):
        dv = norm_v[idx + 1] - norm_v[idx]
        di = norm_i[idx + 1] - norm_i[idx]
        if dv != 0:
            slope = np.arctan(di / dv) / np.pi + 0.5
        else:
            slope = 0.5
        slopes.append(slope)
    slopes.append(slopes[-1])

    for (v, i, s) in zip(norm_v, norm_i, slopes):
        x = int((v + 1) / 2 * (image_size - 1))
        y = int((1 - (i + 1) / 2) * (image_size - 1))
        canvas_green[y, x] = int(s * 255)

    powers = norm_v * norm_i
    powers = (powers - np.min(powers)) / (np.max(powers) - np.min(powers))

    for (v, i, p) in zip(norm_v, norm_i, powers):
        x = int((v + 1) / 2 * (image_size - 1))
        y = int((1 - (i + 1) / 2) * (image_size - 1))
        canvas_blue[y, x] = int(p * 255)

    rgb_image = np.stack((canvas_red, canvas_green, canvas_blue), axis=-1)

    return rgb_image


class VITrajectory(nn.Module):

    def __init__(self, image_size=32):
        super().__init__()
        self.image_size = image_size

    def __call__(self, x):
        is_torch = False

        if isinstance(x, torch.Tensor):
            x = x.numpy()
            is_torch = True

        v, i = x

        x = vicoding(v, i, self.image_size)

        if is_torch:
            x = torch.from_numpy(x)

        return x


class DistanceMatrix(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.jit.export
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute the Euclidean distance similarity matrix for a signal.

        Args:
            signal (torch.Tensor): Input signal of shape (..., w).

        Returns:
            torch.Tensor: Distance similarity matrix of shape (..., w, w).
        """
        *batch_dims, w = signal.shape
        distance_matrix = torch.empty(*batch_dims,
                                      w,
                                      w,
                                      dtype=signal.dtype,
                                      device=signal.device)

        distance_matrix = abs(signal.unsqueeze(-1) - signal.unsqueeze(-2))

        return distance_matrix
