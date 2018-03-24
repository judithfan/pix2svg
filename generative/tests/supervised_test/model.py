from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
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
        # self.combiner = nn.Linear((n_photos + 1) * 100, 32)
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
        # compute euclidean distance from sketch to each photo
        # distances = torch.cat([torch.norm(photos[:, i] - sketch, p=2, dim=1).unsqueeze(1) 
        #                        for i in xrange(self.n_photos)], dim=1)
        distances = torch.cat([pearson_correlation(photos[:, i], sketch, dim=1).unsqueeze(1)
                               for i in xrange(self.n_photos)], dim=1)
        # distances = torch.cat((photos, sketch.unsqueeze(1)), dim=1)
        # distances = distances.view(batch_size, -1)
        # distances = self.combiner(distances)
        log_distances = F.log_softmax(distances, dim=1)
        return log_distances


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
        annotation = self.annotation_net(sketch[:int(batch_size / 4)])  # only sketch
        return F.sigmoid(output), F.log_softmax(annotation, dim=1)


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
            nn.Linear(4096, 2000),
            Swish(),
            nn.Dropout(),
            nn.Linear(2000, 1000))

    def forward(self, x):
        return self.net(x)


class Swish(nn.Module):
    # https://arxiv.org/abs/1710.05941
    def forward(self, x):
        return x * F.sigmoid(x)


class AnnotationNet(nn.Module):
    def __init__(self):
        super(AnnotationNet, self).__init__()
        self.fc = nn.Linear(1000, 32)

    def forward(self, x):
        return self.fc(x)


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

