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


class PredictorFC6(nn.Module):
    # for FC6 features
    # there isn't any spatial pooling to be done here
    # a total of 1048833 parameters
    def __init__(self, hiddens_dim=128):
        super(PredictorFC6, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096 * 2, hiddens_dim),
            Swish(),
            nn.Dropout(),
            nn.Linear(hiddens_dim, 1))
        self.hiddens_dim = hiddens_dim

    def forward(self, photo, sketch):
        h = torch.cat((photo, sketch), dim=1)
        return self.net(swish(h))


class AttendedSpatialCollapseCONV42(nn.Module):
    # for CONV42 features
    # a total of 1049115 parameters
    def __init__(self):
        super(AttendedSpatialCollapseCONV42, self).__init__()
        self.photo_attn = Parameter(torch.normal(torch.zeros(28 * 28), 1))
        self.sketch_attn = Parameter(torch.normal(torch.zeros(28 * 28), 1))
        self.net = nn.Sequential(
            nn.Linear(512 * 2, 1021),
            Swish(),
            nn.Dropout(),
            nn.Linear(1021, 1))

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
    # for POOL1 features
    # a total. of 1048839 parameters
    def __init__(self):
        super(AttendedSpatialCollapsePOOL1, self).__init__()
        self.photo_attn = Parameter(torch.normal(torch.zeros(112 * 112), 1))
        self.sketch_attn = Parameter(torch.normal(torch.zeros(112 * 112), 1))
        self.net = nn.Sequential(
            nn.Linear(64 * 2, 7875),
            Swish(),
            nn.Dropout(),
            nn.Linear(7875, 1))

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


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)
