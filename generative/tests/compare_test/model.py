from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.adaptor = AdaptorNet()
        self.classifier = CategoryClassifier()

    def forward(self, sketch):
        sketch = swish(self.adaptor(sketch))
        return self.classifier(sketch)


class LabelPredictorFC6(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.adaptor = AdaptorNetFC6()
        self.classifier = CategoryClassifier()

    def forward(self, sketch):
        sketch = swish(self.adaptor(sketch))
        return self.classifier(sketch)


class Label32Predictor(nn.Module):
    def __init__(self):
        super(Label32Predictor, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.classifier = FusePredictor()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, photo, sketch):
        photo = swish(self.photo_adaptor(photo))
        sketch = swish(self.sketch_adaptor(sketch))
        # pred = self.net(input)
        pred = self.classifier(photo, sketch)
        return pred


class AdaptorNet(nn.Module):
    def __init__(self):
        super(AdaptorNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            Swish(),
            nn.MaxPool2d(2, stride=2, dilation=1),
        )
        self.net = nn.Sequential(
            nn.Linear(64 * 14 * 14, 1000),
            # nn.BatchNorm1d(2048),
            # Swish(),
            # nn.Dropout(0.5),
            # nn.Linear(2048, 1000),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.net(x)


class AdaptorNetFC6(nn.Module):
    def __init__(self):
        super(AdaptorNetFC6, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096, 2048),
            Swish(),
            nn.Linear(2048, 1000))

    def forward(self, x):
        return self.net(x)


class CosineClassifier(nn.Module):
    def forward(self, e1, e2):
        # center cosine similarity (pearson coefficient)
        e1 = e1 - torch.mean(e1, dim=1, keepdim=True)
        e2 = e2 - torch.mean(e2, dim=1, keepdim=True)
        e = cosine_similarity(e1, e2, dim=1).unsqueeze(1)
        return F.sigmoid(e)


class FusePredictor(nn.Module):
    def __init__(self):
        super(FusePredictor, self).__init__()
        self.fc1 = nn.Linear(2000, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, e1, e2):
        h = torch.cat((e1, e2), dim=1)
        h = swish(self.fc1(h))
        return self.fc2(h)


class CategoryClassifier(nn.Module):
    def __init__(self):
        super(CategoryClassifier, self).__init__()
        self.fc = nn.Linear(1000, 32)

    def forward(self, x):
        return self.fc(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
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
