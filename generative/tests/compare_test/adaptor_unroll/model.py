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


class PredictorPOOL1(nn.Module):
    def __init__(self):
        super(PredictorPOOL1, self).__init__()
        self.photo_adaptor = AdaptorNetPOOL1()
        self.sketch_adaptor = AdaptorNetPOOL1()
        self.classifier = FusePredictor()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, photo, sketch):
        photo = swish(self.photo_adaptor(photo))
        sketch = swish(self.sketch_adaptor(sketch))
        pred = self.classifier(photo, sketch)
        return pred


class PredictorCONV42(nn.Module):
    def __init__(self):
        super(PredictorCONV42, self).__init__()
        self.photo_adaptor = AdaptorNetCONV42()
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
        photo = swish(self.photo_adaptor(photo))
        sketch = swish(self.sketch_adaptor(sketch))
        pred = self.classifier(photo, sketch)
        return pred


class PredictorFC6(nn.Module):
    def __init__(self):
        super(PredictorFC6, self).__init__()
        self.photo_adaptor = AdaptorNetFC6()
        self.sketch_adaptor = AdaptorNetFC6()
        self.classifier = FusePredictor()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, photo, sketch):
        photo = swish(self.photo_adaptor(photo))
        sketch = swish(self.sketch_adaptor(sketch))
        pred = self.classifier(photo, sketch)
        return pred


class AdaptorNetPOOL1(nn.Module):
    def __init__(self):
        super(AdaptorNetPOOL1, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.attention = Parameter(torch.normal(torch.zeros(64), 1))
        self.net = nn.Linear(56 * 56, 1000)

    def forward(self, x):
        h = self.pool(x)
        W = F.softplus(self.attention)
        W = W / torch.sum(W)
        W = W.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        h = W * h
        h = torch.sum(h, dim=1)
        h = h.view(-1, 56 * 56)
        h = swish(h)
        return self.net(h)


class AdaptorNetCONV42(nn.Module):
    def __init__(self):
        super(AdaptorNetCONV42, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            Swish(),
            nn.MaxPool2d(2, stride=2))
        self.net = nn.Linear(64 * 14 * 14, 1000)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class AdaptorNetFC6(nn.Module):
    def __init__(self):
        super(AdaptorNetFC6, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096, 2048),
            Swish(),
            nn.Linear(2048, 1000))

    def forward(self, x):
        return self.net(x)


class FusePredictor(nn.Module):
    def __init__(self):
        super(FusePredictor, self).__init__()
        self.fc1 = nn.Linear(2000, 256)
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
