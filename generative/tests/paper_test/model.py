from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SketchNet(nn.Module):
    def __init__(self):
        super(SketchNet, self).__init__()
        self.sketch_adaptor = AdaptorNet()
        self.photo_adaptor = AdaptorNet()
        self.sketch_classifier = CategoryNet()
        self.photo_classifier = CategoryNet()
        self.swish = Swish()

    def forward(self, sketch, good_photo, bad_photo):
        sketch_emb = self.sketch_adaptor(sketch)
        good_photo_emb = self.photo_adaptor(good_photo)
        bad_photo_emb = self.photo_adaptor(bad_photo)
        
        sketch_cls = self.sketch_classifier(sketch)
        good_photo_cls = self.photo_classifier(good_photo)
        bad_photo_cls = self.photo_classifier(bad_photo)
        return (sketch_emb, good_photo_emb, bad_photo_emb, 
                sketch_cls, good_photo_cls, bad_photo_cls)


class CategoryNet(nn.Module):
    def __init__(self):
        super(CategoryNet, self).__init__()
        self.category_head = nn.Linear(1024, 32)

    def forward(self, x):
        return self.category_head(x)


class AdaptorNet(nn.Module):
    def __init__(self):
        super(AdaptorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            Swish(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            Swish(),
            nn.Linear(2048, 1024)) 
    
    def forward(self, input):
        return self.net(input)


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941
    def forward(self, x):
        return x * F.sigmoid(x)

