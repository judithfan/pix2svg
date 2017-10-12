from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import cosine_similarity


class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.fusenet = FuseClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb)


class AdaptorNet(nn.Module):
    def __init__(self):
        super(AdaptorNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, dilation=1),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2, dilation=1),
        )
        self.net = nn.Sequential(
            nn.Linear(64 * 14 * 14, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(4096, 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class FuseClassifier(nn.Module):
    def forward(self, e1, e2):
        # center cosine similarity (pearson coefficient)
        e1 = e1 - torch.mean(e1, dim=1, keepdim=True)
        e2 = e2 - torch.mean(e2, dim=1, keepdim=True)
        e = cosine_similarity(e1, e2, dim=1).unsqueeze(1)
        return F.sigmoid(e)


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
        checkpoint = torch.load(file_path, map_location=lambda storage, location: storage)
    model = EmbedNet()
    model.load_state_dict(checkpoint['state_dict'])
    return model
