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
        self.predictor = PredictorNet()

    def forward(self, photo, sketch):
        input = torch.cat((photo, sketch), dim=1)
        input = self.predictor(input)
        return F.sigmoid(input)


class PredictorNet(nn.Module):
    def __init__(self):
        super(PredictorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096),
            Swish(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            Swish(),
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000),
            Swish(),
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941
    def forward(self, x):
        return x * F.sigmoid(x)


def pearson_correlation(x1, x2, dim=1, eps=1e-8):
    x1 = x1 - torch.mean(x1, dim=dim, keepdim=True)
    x2 = x2 - torch.mean(x2, dim=dim, keepdim=True)
    return cosine_similarity(x1, x2, dim=dim, eps=eps)


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
