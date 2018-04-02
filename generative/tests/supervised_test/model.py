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
    """Network that takes as input sketch and photo vectors and computes 
    and adapted & normalized Euclidean distance between each photo and sketch.

    @param n_photos: integer [default: 32]
    """
    def __init__(self, layer='fc6', n_photos=32):
        super(SketchNet, self).__init__()
        if layer == 'conv42':
            self.sketch_adaptor = Conv42AdaptorNet()
            self.photo_adaptor = Conv42AdaptorNet()
        elif layer == 'fc6':
            self.sketch_adaptor = FC6AdaptorNet()
            self.photo_adaptor = FC6AdaptorNet()
        else:
            raise Exception('%s layer not supported.' % layer)
        self.distance = NNDistance(1000)
        self.n_photos = n_photos
        self.layer = layer

    def forward(self, photos, sketch):
        batch_size = photos.size(0)
        assert photos.size(1) == self.n_photos
        # photos is a torch.Tensor of size batch_size x 32 x 4096 
        # sketch is a torch.Tensor of size batch_size x 4096 (single sketch)
        sketch = self.sketch_adaptor(sketch)
        photos = torch.cat([self.photo_adaptor(photos[:, i]).unsqueeze(1)
                            for i in xrange(self.n_photos)], dim=1)
        # compute distance from sketch to each photo
        distances = torch.cat([self.distance(photos[:, i], sketch)
                               for i in xrange(self.n_photos)], dim=1)
        return distances


class SketchNetHARD(SketchNet):
    def forward(self, photos, sketch):
        batch_size = photos.size(0)
        assert photos.size(1) == self.n_photos
        # photos is a torch.Tensor of size batch_size x 32 x 4096
        # sketch is a torch.Tensor of size batch_size x 4096 (single sketch)
        sketch = self.sketch_adaptor(sketch)
        photos = torch.cat([self.photo_adaptor(photos[:, i]).unsqueeze(1)
                            for i in xrange(self.n_photos)], dim=1)
        distances = torch.cat([pearson_correlation(photos[:, i], sketch, dim=1).unsqueeze(1)
                               for i in xrange(self.n_photos)], dim=1)
        return distances


class SketchNetCATEGORY(SketchNet):
    def __init__(self, layer='fc6', n_photos=32):
        super(SketchNetCATEGORY, self).__init__(layer=layer, n_photos=n_photos)
        self.annotation_net = AnnotationNet()
    
    # this only works for SINGLE photo and SINGLE sketch
    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        sketch = self.sketch_adaptor(sketch)
        photo = self.photo_adaptor(photo)
        output = pearson_correlation(photo, sketch)
        annotation = self.annotation_net(torch.cat((photo, sketch), dim=1)) 
        return F.sigmoid(output), annotation


class SketchNetSOFT(SketchNet):
    def __init__(self, layer='fc6', n_photos=32):
        super(SketchNetSOFT, self).__init__(layer=layer, n_photos=n_photos)
        self.category_net = CategoryNet()

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        sketch = self.sketch_adaptor(sketch)
        photo = self.photo_adaptor(photo)
        output = pearson_correlation(photo, sketch)
        category = self.category_net(sketch) 
        return F.sigmoid(output), category


class SketchNetRAW(nn.Module):
    def __init__(self):
        super(SketchNetRAW, self).__init__()
        self.photo_adaptor = RawAdaptorNet(3)
        self.sketch_adaptor = RawAdaptorNet(1)

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        sketch = self.sketch_adaptor(sketch)
        photo = self.photo_adaptor(photo)
        output = pearson_correlation(photo, sketch)
        return F.sigmoid(output)


class SketchNetDIST(SketchNet):
    def __init__(self, layer='fc6', n_photos=32):
        super(SketchNetDIST, self).__init__(layer=layer, n_photos=n_photos)    
        self.distance = NNDistance(1000)

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        sketch = self.sketch_adaptor(sketch)
        photo = self.photo_adaptor(photo)
        distance = self.distance(photo, sketch)
        return F.sigmoid(distance)


class SketchNetNODIST(SketchNet):
    def __init__(self, layer='fc6', n_photos=32):
        super(SketchNetNODIST, self).__init__(layer=layer, n_photos=n_photos)
        self.net = nn.Sequential(nn.Linear(2000, 2000),
                                 nn.BatchNorm1d(2000),
                                 Swish(),
                                 nn.Linear(2000, 2000))

    def forward(self, photo, sketch):
        batch_size = photo.size(0)
        sketch = self.sketch_adaptor(sketch)
        photo = self.photo_adaptor(photo)
        both = torch.cat((photo, sketch), dim=1)
        both = self.net(both)
        photo, sketch = torch.chunk(both, 2, dim=1)
        return photo, sketch


class RawAdaptorNet(nn.Module):
    def __init__(self, n_inputs):
        super(RawAdaptorNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_inputs, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 1000),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 1000))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return x


class Conv42AdaptorNet(nn.Module):
    """Light network to learn an adapted embedding."""
    def __init__(self):
        super(Conv42AdaptorNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Swish(),
            nn.MaxPool2d(2, stride=2, dilation=1))
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 14 * 14, 4096),
            nn.BatchNorm1d(4096),
            Swish(),
            nn.Dropout(.5),
            nn.Linear(4096, 1000))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 14 * 14)
        return self.fc_layers(x)


class FC6AdaptorNet(nn.Module):
    def __init__(self):
        super(FC6AdaptorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            Swish(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            Swish(),
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000),
            Swish(),
            nn.Linear(1000, 1000))

    def forward(self, x):
        return self.net(x)


class NNDistance(nn.Module):
    def __init__(self, input_dim):
        super(NNDistance, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 1)

    def forward(self, x, y):
        h = torch.cat((x, y), dim=1)
        return self.fc1(h)


class AffineDistance(nn.Module):
    def __init__(self, input_dim):
        super(AffineDistance, self).__init__()
        self.W = Parameter(torch.normal(torch.zeros(2*input_dim), .1))
        self.b = Parameter(torch.normal(torch.zeros(1), .1))

    def forward(self, x, y):
        h = torch.cat((x, y), dim=1)
        h = torch.matmul(h, self.W.unsqueeze(1)) + self.b.unsqueeze(1)
        return h


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941
    def forward(self, x):
        return x * F.sigmoid(x)


class AnnotationNet(nn.Module):
    def __init__(self):
        super(AnnotationNet, self).__init__()
        self.fc = nn.Linear(2000, 32)

    def forward(self, x):
        return self.fc(x)


class CategoryNet(nn.Module):
    def __init__(self):
        super(CategoryNet, self).__init__()
        self.category_head = nn.Linear(1000, 32)

    def forward(self, x):
        return self.category_head(x)


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

