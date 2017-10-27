from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.net = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 1000),
        )

    def forward(self, x):
        x = self.net(x)
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
    if use_cuda:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path, map_location=lambda storage, location: storage)
    model = EmbedNet()
    model.load_state_dict(checkpoint['state_dict'])
    return model


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

