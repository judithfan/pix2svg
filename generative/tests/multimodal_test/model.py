from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cosine_similarity

EMBED_NET_TYPE = 0
CONV_EMBED_NET_TYPE = 1


class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
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


class ConvEmbedNet(nn.Module):
    def __init__(self):
        super(ConvEmbedNet, self).__init__()
        self.photo_adaptor = ConvAdaptorNet(64, 8, 112, 112, 2048, 1000)
        self.sketch_adaptor = ConvAdaptorNet(64, 8, 112, 112, 2048, 1000)
        self.fusenet = FuseClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb)


class ConvAdaptorNet(nn.Module):
    def __init__(self, in_n_filters, out_n_filters, 
                 in_height, in_width, hid_dim, out_dim):
        """Many of the early convolutional networks are too big 
        for us to collapse into a fully connected network. Let's
        shrink down the number of filters + add a max pool. (NCHW)

        :param in_n_filters: number of filters in input tensor
        :param out_n_filters: number of filters in output tensor
        :param in_height: size of height
        :param in_width: size of width
        :param mid_dim: number of dimensions in middle FC layer
        :param out_dim: number of dimensions in output embedding
        """
        super(ConvAdaptorNet, self).__init__()
        self.conv = nn.Conv2d(in_n_filters, out_n_filters, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(size=(2, 2), stride=(2, 2), dilation=(1, 1))
        in_dim = out_n_filters * in_height / 2 * in_width / 2
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


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
    """Return EmbedNet instance"""
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage, location: storage)

    if checkpoint['type'] == EMBED_NET_TYPE:
        model = EmbedNet()
    elif checkpoint['type'] == CONV_EMBED_NET_TYPE:
        model = ConvEmbedNet()
    else:
        raise Exception('Unknown model type %d.' % checkpoint['type'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

