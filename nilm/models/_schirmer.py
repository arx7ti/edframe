import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import TransformerEncoderLayer

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ...metrics import f1_score, teca_score, modified_f1_score, jaccard_score, modified_jaccard_score
import torchvision.models as models
from ..disaggregator import Disaggregator


class SchirmerCNN(Disaggregator):

    def __init__(
        self,
        num_appliances: int,
        thresh=10,
    ):
        super(SchirmerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, num_appliances),
        )

        self.thresh = thresh

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = F.softmax(x)

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
