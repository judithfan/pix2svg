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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, photo, sketch):
        h = torch.cat((photo, sketch), dim=1)
        return self.adaptor(swish(h))


class FilterCollapseCONV42(nn.Module):
    def __init__(self):
        super(FilterCollapseCONV42, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784 * 2, 512),
            Swish(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(256, 1))

    def forward(self, photo, sketch):
        photo = torch.mean(photo, dim=1).view(-1, 28 * 28)
        sketch = torch.mean(sketch, dim=1).view(-1, 28 * 28)
        hiddens = swish(torch.cat((photo, sketch), dim=1))
        return self.net(hiddens)


class SpatialCollapseCONV42(nn.Module):
    def __init__(self):
        super(SpatialCollapseCONV42, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            Swish(),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # Swish(),
            nn.Dropout(),
            # nn.Linear(256, 1))
            nn.Linear(512, 1))

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        filter_size = photo.size(1)
        photo = photo.view(batch_size, filter_size, 28 * 28)
        sketch = sketch.view(batch_size, filter_size, 28 * 28)
        photo = torch.mean(photo, dim=2)
        sketch = torch.mean(sketch, dim=2)
        hiddens = swish(torch.cat((photo, sketch), dim=1))
        return self.net(hiddens)


class AdaptorNetCONV42(nn.Module):
    def __init__(self):
            super(AdaptorNetCONV42, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(512 * 2, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                Swish(),
                nn.MaxPool2d(2, stride=2))
            self.net = nn.Sequential(
                nn.Linear(64 * 14 * 14, 512),
                nn.BatchNorm1d(512),
                Swish(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                Swish(),
                nn.Dropout(0.1),
                nn.Linear(256, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)
