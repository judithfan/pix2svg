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
        self.sketch_adaptor = AdaptorNetCONV42()
        self.classifier = FusePredictor()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, photo, sketch):
        sketch = swish(self.sketch_adaptor(sketch))
        pred = self.classifier(photo, sketch)
        return pred


class AdaptorNetCONV42(nn.Module):
    def __init__(self):
        super(AdaptorNetCONV42, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            Swish(),
            nn.MaxPool2d(2, stride=2))
        self.net = nn.Linear(64 * 14 * 14, 4096)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class FusePredictor(nn.Module):
    def __init__(self):
        super(FusePredictor, self).__init__()
        self.fc1 = nn.Linear(4096, 256)
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
