from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class ModelA(nn.Module):
    # 8192 --> 1
    def __init__(self):
       super(ModelA, self).__init__() 
       self.fc = nn.Linear(8192, 1)

    def forward(self, photo, sketch):
        input = torch.cat((photo, sketch), dim=1)
        input = self.fc(input)
        return F.sigmoid(input)


class ModelB(nn.Module):
    # 8192 --> 256 --> 1
    def __init__(self):
        super(ModelB, self).__init__() 
        self.net = nn.Sequential(
            nn.Linear(8192, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, photo, sketch):
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        return F.sigmoid(input)


class ModelC(nn.Module):
    # 8192 --> 4096 --> 2048 --> 1024 --> 256 --> 1
    def __init__(self):
        super(ModelC, self).__init__() 
        self.net = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, photo, sketch):
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        return F.sigmoid(input)


class ModelD(nn.Module):
    # adaptors --> cat --> 1
    def __init__(self):
        super(ModelD, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.fc = nn.Linear(2000, 1)

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.fc(input)
        return F.sigmoid(input)


class ModelE(nn.Module):
    # adaptors --> swish --> cat --> 1
    def __init__(self):
        super(ModelE, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.fc = nn.Linear(2000, 1)

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = F.leaky_relu(input)
        input = self.fc(input)
        return F.sigmoid(input)


class ModelF(nn.Module):
    # adaptors --> cat --> 256 --> 1
    def __init__(self):
        super(ModelF, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.net = nn.Sequential(
            nn.Linear(2000, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        return F.sigmoid(input)


class ModelG(nn.Module):
    # adaptors --> cat w/ prod --> 1
    def __init__(self):
        super(ModelG, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.fc = nn.Linear(3000, 1)

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch, photo * sketch), dim=1)
        input = self.fc(input)
        return F.sigmoid(input)


class ModelH(nn.Module):
    def __init__(self):
        super(ModelH, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
    
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        dist = pearson_correlation(photo, sketch)
        return F.sigmoid(dist)


class ModelI(nn.Module):
    def __init__(self):
        super(ModelI, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.net = nn.Sequential(
            nn.Linear(2000, 1000), 
            nn.LeakyReLU(),
            nn.Linear(1000, 256),
        )
    
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        photo, sketch = torch.chunk(input, 2, dim=1)
        dist = pearson_correlation(photo, sketch)
        return F.sigmoid(dist)


class ModelJ(nn.Module):
    def __init__(self):
        super(ModelJ, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.M = Parameter(torch.normal(torch.zeros(1000, 1000), 1))
        self.norm = nn.BatchNorm1d(1)
   
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.bmm(torch.matmul(photo, self.M).unsqueeze(1), sketch.unsqueeze(2)).squeeze(1)
        return F.sigmoid(self.norm(input))


class ModelK(nn.Module):
    def __init__(self):
        super(ModelK, self).__init__()
        self.photo_adaptor = AdaptorNet()
        self.sketch_adaptor = AdaptorNet()
        self.norm = nn.BatchNorm1d(1)
   
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.norm(photo - sketch, 2, dim=1).unsqueeze(1)
        return F.sigmoid(self.norm(input))


class AdaptorNet(nn.Module):
    def __init__(self):
        super(AdaptorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1000)) 
    
    def forward(self, input):
        return self.net(input)


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
