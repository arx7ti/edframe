import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import TransformerEncoderLayer

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ...metrics import f1_score, teca_score, modified_f1_score, jaccard_score, modified_jaccard_score
import torchvision.models as models
from ..base import BaseModel


class FaustineCNN(BaseModel):

    def __init__(
        self,
        n_devices: int,
        thresh=10,
    ):
        super(FaustineCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 2 * n_devices),
        )

        self.thresh = thresh

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 2)
        x = F.softmax(x, dim=-1)[..., 0]

        return x

    def loss(self, outputs, labels):
        labels = labels / labels.sum(1, keepdims=True)
        loss_fn = nn.BCELoss()

        return loss_fn(outputs, labels)

    def scores(self, Y_true, Y_pred):
        P = Y_true.sum(1, keepdims=True)
        Y_pred = np.where(P * Y_pred > self.thresh, Y_pred, 0)
        Y_pred = Y_pred / (Y_pred.sum(1, keepdims=True) + 1e-10)
        Y_pred = P * Y_pred

        F1 = f1_score(Y_true, Y_pred)
        MF = modified_f1_score(Y_true, Y_pred)
        TECA = teca_score(Y_true, Y_pred)
        J = jaccard_score(Y_true, Y_pred)
        MJ = modified_jaccard_score(Y_true, Y_pred)
        scores = dict(F1=F1, MF=MF, J=J, MJ=MJ, TECA=TECA)

        return scores
