from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cosine_similarity


class L2EmbedNet(nn.Module):
    def __init__(self):
        super(L2EmbedNet, self).__init__()
        self.photo_adaptor = AdaptorNet(4096, 1000)
        self.sketch_adaptor = AdaptorNet(4096, 1000)
        self.fusenet = FuseClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb)


class AdaptorNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdaptorNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x


class FuseClassifier(nn.Module):
    def forward(self, e1, e2):
        return torch.norm(e1 - e2, p=2, dim=1, keepdim=True)
