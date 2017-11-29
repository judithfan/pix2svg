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
        self.photo_adaptor = AdaptorNet(4096, 2048, 1000)
        self.sketch_adaptor = AdaptorNet(4096, 2048, 1000)
        self.fusenet = FuseClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb)


class AdaptorNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(AdaptorNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.drop1(x)
        x = F.elu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class FuseClassifier(nn.Module):
    def forward(self, e1, e2):
        return torch.norm(e1 - e2, p=2, dim=1, keepdim=True)


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    model = L2EmbedNet()
    model.load_state_dict(checkpoint['state_dict'])
    return model

