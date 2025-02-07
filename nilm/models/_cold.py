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


class COLD(Disaggregator):

    def __init__(
        self,
        n_devices,
        input_size=50,
        seq_size=60,
        hidden_size=512,
        n_head=8,
        activation='relu',
        pools=[30, 10, 1],
        dropout=0.2,
        thresh=10,
    ):
        super().__init__()

        self.back = []

        for pool_size in pools:
            if input_size != hidden_size:
                self.back.append(nn.Linear(input_size, hidden_size,
                                           bias=False))

            self.back.append(
                TransformerEncoderLayer(hidden_size,
                                        dropout=dropout,
                                        batch_first=True,
                                        dim_feedforward=4 * hidden_size,
                                        activation=activation,
                                        nhead=n_head))

            if pool_size != seq_size:
                self.back.append(AvgSeqAdaptivePool(pool_size))
                seq_size = pool_size

            input_size = hidden_size

        self.back = nn.Sequential(*self.back, nn.Flatten(1))
        self.head = nn.Linear(seq_size * hidden_size, n_devices)

        self.thresh = thresh

    def forward(self, x):
        x = self.back(x)
        x = self.head(x)
        x = torch.softmax(x, -1)

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
