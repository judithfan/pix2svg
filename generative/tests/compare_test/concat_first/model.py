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
        self.adaptor = AdaptorNetCONV42()
        self.classifier = FusePredictor()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, photo, sketch):
        h = torch.cat((photo, sketch), dim=1)
        h = swish(self.adaptor(h))
        pred = self.classifier(h)
        return pred


class AdaptorNetCONV42(nn.Module):
    def __init__(self):
            super(AdaptorNetCONV42, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(512 * 2, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                Swish(),
                nn.MaxPool2d(2, stride=2))
            self.net = nn.Sequential(
                nn.Linear(64 * 14 * 14, 2048),
                nn.BatchNorm1d(2048),
                Swish(),
                nn.Dropout(0.5),
                nn.Linear(2048, 1000))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)
