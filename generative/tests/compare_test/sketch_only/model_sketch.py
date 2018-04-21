from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchOnlyPredictor(nn.Module):
    def __init__(self):
        super(SketchOnlyPredictor, self).__init__()
        self.adaptor = AdaptorNetCONV42()
        self.classifier = CategoryClassifier()

    def forward(self, sketch):
        sketch = swish(self.adaptor(sketch))
        return self.classifier(sketch)


class AdaptorNetCONV42(nn.Module):
    def __init__(self):
        super(AdaptorNetCONV42, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Swish(),
            nn.MaxPool2d(2, stride=2, dilation=1),
        )
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


class CategoryClassifier(nn.Module):
    def __init__(self):
        super(CategoryClassifier, self).__init__()
        self.fc = nn.Linear(1000, 32)

    def forward(self, x):
        return self.fc(x)


def swish(x):
    return x * F.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

