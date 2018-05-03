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

    def forward(self, photo, sketch):
        h = torch.cat((photo, sketch), dim=1)
        return self.adaptor(swish(h))


class PredictorFC6(nn.Module):
    # there isn't any spatial pooling to be done here
    def __init__(self):
        super(PredictorFC6, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096 * 2, 512),
            Swish(),
            nn.Dropout(),
            nn.Linear(512, 1))

    def forward(self, photo, sketch):
        h = torch.cat((photo, sketch), dim=1)
        return self.net(swish(h))


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


class AttendedSpatialCollapseCONV42(nn.Module):
    def __init__(self):
        super(AttendedSpatialCollapseCONV42, self).__init__()
        self.photo_attn = Parameter(torch.normal(torch.zeros(28 * 28), 1))
        self.sketch_attn = Parameter(torch.normal(torch.zeros(28 * 28), 1))
        self.net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            Swish(),
            nn.Dropout(),
            nn.Linear(512, 1))

    def normalize_attention(self, attn):
        W = F.softplus(attn)
        W = W / torch.sum(W)
        return W.unsqueeze(0).unsqueeze(0)

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        filter_size = photo.size(1)
        photo = photo.view(batch_size, filter_size, 28 * 28)
        photo_attn = self.normalize_attention(self.photo_attn)
        photo = torch.sum(photo_attn * photo, dim=2)
        sketch = sketch.view(batch_size, filter_size, 28 * 28)
        sketch_attn = self.normalize_attention(self.sketch_attn)
        sketch = torch.sum(sketch_attn * sketch, dim=2)        
        hiddens = swish(torch.cat((photo, sketch), dim=1))
        return self.net(hiddens)


class AttendedSpatialCollapsePOOL1(nn.Module):
    def __init__(self):
        super(AttendedSpatialCollapsePOOL1, self).__init__()
        self.photo_attn = Parameter(torch.normal(torch.zeros(112 * 112), 1))
        self.sketch_attn = Parameter(torch.normal(torch.zeros(112 * 112), 1))
        self.net = nn.Sequential(
            nn.Linear(64 * 2, 64),
            Swish(),
            nn.Dropout(),
            nn.Linear(64, 1))

    def normalize_attention(self, attn):
        W = F.softplus(attn)
        W = W / torch.sum(W)
        return W.unsqueeze(0).unsqueeze(0)

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        filter_size = photo.size(1)
        photo = photo.view(batch_size, filter_size, 112 * 112)
        photo_attn = self.normalize_attention(self.photo_attn)
        photo = torch.sum(photo_attn * photo, dim=2)
        sketch = sketch.view(batch_size, filter_size, 112 * 112)
        sketch_attn = self.normalize_attention(self.sketch_attn)
        sketch = torch.sum(sketch_attn * sketch, dim=2)        
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
