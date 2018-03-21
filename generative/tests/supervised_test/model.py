from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchNet(nn.Module):
    """Network that takes as input sketch and photo vectors and computes 
    and adapted & normalized Euclidean distance between each photo and sketch.

    @param n_photos: integer [default: 32]
    """
    def __init__(self, n_photos=32):
        super(SketchNet, self).__init__()
        self.sketch_adaptor = AdaptorNet()
        # there are 32 photos for each sketch
        self.photo_adaptors = nn.ModuleList([AdaptorNet() for _ in xrange(n_photos)])
        self.n_photos = n_photos

    def forward(self, photos, sketch):
        assert photos.size(1) == self.n_photos
        # photos is a torch.Tensor of size batch_size x 32 x 4096 
        # sketch is a torch.Tensor of size batch_size x 4096 (single sketch)
        sketch = self.sketch_adaptor(sketch)
        photos = torch.cat([self.photo_adaptors[i](photos[:, i]).unsqueeze(1) 
                            for i in xrange(self.n_photos)], dim=1)
        # compute euclidean distance from sketch to each photo
        distances = torch.cat([torch.norm(photos[:, i] - sketch, p=2, dim=1).unsqueeze(1) 
                               for i in xrange(self.n_photos)], dim=1)
        # want this to sum to one
        distances = distances / torch.sum(distances, 1).unsqueeze(1)
        return distances


class AdaptorNet(nn.Module):
    """Light network to learn an adapted embedding."""
    def __init__(self):
        super(AdaptorNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            Swish(),
            nn.MaxPool2d(2, stride=2, dilation=1))
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 12 * 12, 1000),
            nn.BatchNorm1d(1000),
            Swish(),
            nn.Dropout(),
            nn.Linear(1000, 1000))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 16 * 12 * 12)
        return self.fc_layers(x)


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941
    def forward(self, x):
        return x * F.sigmoid(x)
