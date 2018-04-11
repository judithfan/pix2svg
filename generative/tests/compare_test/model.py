from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, distance='cosine'):
        super(Classifier, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        if distance == 'cosine':
            self.fusenet = FuseCosineClassifier()
        elif distance == 'euclidean':
            self.fusenet = FuseEuclideanClassifier()
        else:
            raise Exception('distance %s not found.' % distance.upper())
        self.photo_classifier = CategoryClassifier()
        self.sketch_classifier = CategoryClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        photo_pred = self.photo_classifier(photo_emb)
        sketch_pred = self.sketch_classifier(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb), photo_pred, sketch_pred


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.fusenet = FusePredictor()
        self.photo_classifier = CategoryClassifier()
        self.sketch_classifier = CategoryClassifier()

    def forward(self, photo_emb, sketch_emb):
        photo_emb = self.photo_adaptor(photo_emb)
        sketch_emb = self.sketch_adaptor(sketch_emb)
        photo_pred = self.photo_classifier(photo_emb)
        sketch_pred = self.sketch_classifier(sketch_emb)
        return self.fusenet(photo_emb, sketch_emb), photo_pred, sketch_pred


class Soft32Classifier(Classifier):
    def forward(self, photo_32_emb, sketch_emb):
        sketch_emb = self.sketch_adaptor(sketch_emb)
        return torch.cat([self.fusenet(self.photo_adaptor(photo_32_emb[i]), sketch_emb) 
                          for i in xrange(32)], dim=1)


class AdaptorNet(nn.Module):
    def __init__(self):
        super(AdaptorNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Swish(),
            nn.MaxPool2d(2, stride=2, dilation=1),
        )
        self.net = nn.Sequential(
            nn.Linear(64 * 14 * 14, 2048),
            nn.BatchNorm1d(2048),
            Swish(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1000),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class FuseCosineClassifier(nn.Module):
    def forward(self, e1, e2):
        # center cosine similarity (pearson coefficient)
        e1 = e1 - torch.mean(e1, dim=1, keepdim=True)
        e2 = e2 - torch.mean(e2, dim=1, keepdim=True)
        e = cosine_similarity(e1, e2, dim=1).unsqueeze(1)
        return F.sigmoid(e)


class FuseEuclideanClassifier(nn.Module):
    def __init__(self):
        super(FuseEuclideanClassifier, self).__init__()
        self.norm = nn.BatchNorm1d(1)

    def forward(self, e1, e2):
        h = torch.norm(e1 - e2, 2, dim=1).unsqueeze(1)
        return F.sigmoid(self.norm(h))


class FusePredictor(nn.Module):
    def __init__(self):
        super(FusePredictor, self).__init__()
        self.fc = nn.Linear(2000, 1)

    def forward(self, e1, e2):
        h = torch.cat((e1, e2), dim=1)
        return F.softplus(self.fc(h))


class CategoryClassifier(nn.Module):
    def __init__(self):
        super(CategoryClassifier, self).__init__()
        self.fc = nn.Linear(1000, 32)

    def forward(self, x):
        return self.fc(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


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

