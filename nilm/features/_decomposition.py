import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from numba import njit


class Fryze(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        v, i = x
        del x
        p = (v * i).mean()
        vrms = torch.sqrt((v**2).mean())
        ia = p / (vrms**2) * v
        ina = i - ia

        x = torch.stack((v, ia, ina))

        return x
