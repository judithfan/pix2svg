from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class PredictorCONV42(nn.Module):
    def __init__(self):
        super(PredictorCONV42, self).__init__()
        # self.photo_sampler = DownsamplerCONV42()
        self.sketch_adaptor = AdaptorNetCONV42()
        self.classifier = FusePredictor()

    def forward(self, photo, sketch):
        # photo = swish(self.photo_sampler(photo))
        photo = swish(torch.mean(photo, dim=1).view(-1, 784))
        sketch = swish(self.sketch_adaptor(sketch))
        pred = self.classifier(photo, sketch)
        return pred


class DownsamplerCONV42(nn.Module):
    def __init__(self):
        super(DownsamplerCONV42, self).__init__()
        self.attention = Parameter(torch.normal(torch.zeros(512), 1))
    
    def forward(self, x):
        W = F.softplus(self.attention)
        W = W / torch.sum(W)
        W = W.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        h = W * x
        h = torch.sum(h, dim=1)
        return h.view(-1, 28 * 28)


class AdaptorNetCONV42(nn.Module):
    def __init__(self):
        super(AdaptorNetCONV42, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Swish(),
            nn.MaxPool2d(2, stride=2))
        self.net = nn.Sequential(
            nn.Linear(64 * 14 * 14, 2048),
            nn.BatchNorm1d(2048),
            Swish(),
            nn.Dropout(0.5),
            nn.Linear(2048, 784))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class FusePredictor(nn.Module):
    def __init__(self):
        super(FusePredictor, self).__init__()
        self.fc1 = nn.Linear(784 * 2, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, e1, e2):
        h = torch.cat((e1, e2), dim=1)
        h = swish(self.fc1(h))
        return self.fc2(h)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)
