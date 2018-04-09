from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# Epoch: 500
# Train Loss: 0.6355    Train Acc: 0.60
# Test  Loss: 0.9016    Test  Acc: 0.47
class ModelA(nn.Module):
    # 8192 --> 1
    def __init__(self, layer):
        super(ModelA, self).__init__() 
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            in_dim = 8192
        elif layer == 'conv42':
            in_dim = 1568
            self.attn = Conv42AttentionNet()
        self.fc = nn.Linear(in_dim, 1)
        self.layer = layer

    def forward(self, photo, sketch):
        if self.layer == 'conv42':
            photo = self.attn(photo)
            sketch = self.attn(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.fc(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.5361     Train Acc: 0.69
# Test  Loss: 1.1756     Test  Acc: 0.46
class ModelB(nn.Module):
    # 8192 --> 256 --> 1
    def __init__(self, layer):
        super(ModelB, self).__init__() 
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            in_dim = 8192
        elif layer == 'conv42':
            in_dim = 1568
            self.attn = Conv42AttentionNet()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.layer = layer

    def forward(self, photo, sketch):
        if self.layer == 'conv42':
            photo = self.attn(photo)
            sketch = self.attn(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.4897     Train Acc: 0.71
# Test  Loss: 2.1062     Test  Acc: 0.49
class ModelC(nn.Module):
    # 8192 --> 4096 --> 2048 --> 1024 --> 256 --> 1
    def __init__(self, layer):
        super(ModelC, self).__init__() 
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
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
        elif layer == 'conv42':
            in_dim = 1568
            self.attn = Conv42AttentionNet()
            self.net = nn.Sequential(
                nn.Linear(1568, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 1),
            )
        self.layer = layer
        
    def forward(self, photo, sketch):
        if self.layer == 'conv42':
            photo = self.attn(photo)
            sketch = self.attn(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.0104    Train Acc: 0.99
# Test  Loss: 4.1160    Test  Acc: 0.48
class ModelD(nn.Module):
    # adaptors --> cat --> 1
    def __init__(self, layer):
        super(ModelD, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
            self.fc = nn.Linear(2000, 1)
        else:
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
            self.fc = nn.Linear(512, 1)
        self.layer = layer

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.fc(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.0079     Train Acc: 0.99
# Test  Loss: 5.0543     Test  Acc: 0.47
class ModelE(nn.Module):
    # adaptors --> swish --> cat --> 1
    def __init__(self, layer):
        super(ModelE, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
            in_dim = 2000
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
            in_dim = 512
        self.fc = nn.Linear(in_dim, 1)
        self.layer = layer

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = F.leaky_relu(input)
        input = self.fc(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.0075    Train Acc: 0.99
# Test  Loss: 6.9078    Test Acc : 0.48
class ModelF(nn.Module):
    # adaptors --> cat --> 256 --> 1
    def __init__(self, layer):
        super(ModelF, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
            in_dim = 2000
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
            in_dim = 512
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.layer = layer

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.0240    Train Acc: 0.99
# Test  Loss: 3.7667    Test  Acc: 0.46
class ModelG(nn.Module):
    # adaptors --> cat w/ prod --> 1
    def __init__(self, layer):
        super(ModelG, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
            in_dim = 3000
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
            in_dim = 768
        self.fc = nn.Linear(in_dim, 1)
        self.layer = layer

    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch, photo * sketch), dim=1)
        input = self.fc(input)
        return F.sigmoid(input)


# Epoch: 500
# Train Loss: 0.3420    Train Acc: 0.97
# Test  Loss: 0.8223    Test  Acc: 0.46
class ModelH(nn.Module):
    def __init__(self, layer):
        super(ModelH, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
        self.layer = layer
    
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        dist = pearson_correlation(photo, sketch)
        return F.sigmoid(dist)


# Epoch: 383
# Train Loss: 0.4019    Train Acc: 0.91
# Test  Loss: 0.8413    Test  Acc: 0.47
class ModelI(nn.Module):
    def __init__(self, layer):
        super(ModelI, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
            self.net = nn.Sequential(
                nn.Linear(2000, 1000), 
                nn.LeakyReLU(),
                nn.Linear(1000, 256),
            )
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
            self.net = nn.Sequential(
                nn.Linear(512, 512), 
                nn.LeakyReLU(),
                nn.Linear(512, 256),
            )
        self.layer = layer
    
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.cat((photo, sketch), dim=1)
        input = self.net(input)
        photo, sketch = torch.chunk(input, 2, dim=1)
        dist = pearson_correlation(photo, sketch)
        return F.sigmoid(dist)


# Epoch: 500
# Train Loss: 0.0108    Train Acc: 0.99
# Test  Loss: 2.2515    Test  Acc: 0.46
class ModelJ(nn.Module):
    def __init__(self, layer):
        super(ModelJ, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
            in_dim = 1000
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
            in_dim = 256
        self.M = Parameter(torch.normal(torch.zeros(in_dim, in_dim), 1))
        self.norm = nn.BatchNorm1d(1)
        self.layer = layer
   
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.bmm(torch.matmul(photo, self.M).unsqueeze(1), sketch.unsqueeze(2)).squeeze(1)
        return F.sigmoid(self.norm(input))


# Epoch: 500
# Train Loss: 0.0116    Train Acc: 0.99
# Test  Loss: 2.3577    Test  Acc: 0.46
class ModelK(nn.Module):
    def __init__(self, layer):
        super(ModelK, self).__init__()
        assert layer in ['fc6', 'conv42']
        if layer == 'fc6':
            self.photo_adaptor = FC6AdaptorNet()
            self.sketch_adaptor = FC6AdaptorNet()
        elif layer == 'conv42':
            self.photo_adaptor = Conv42AdaptorNet()
            self.sketch_adaptor = Conv42AdaptorNet()
        self.norm = nn.BatchNorm1d(1)
        self.layer = layer
   
    def forward(self, photo, sketch):
        photo = self.photo_adaptor(photo)
        sketch = self.sketch_adaptor(sketch)
        input = torch.norm(photo - sketch, 2, dim=1).unsqueeze(1)
        return F.sigmoid(self.norm(input))


class FC6AdaptorNet(nn.Module):
    def __init__(self):
        super(FC6AdaptorNet, self).__init__()
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


class Conv42AttentionNet(nn.Module):
    def __init__(self):
        super(Conv42AttentionNet, self).__init__()
        self.attention = Parameter(torch.normal(torch.zeros(512), 1))

    def forward(self, input):
        batch_size = len(input)
        attention = self.attention.unsqueeze(0).unsqueeze(2).unsqueeze(2).expand(batch_size, 512, 1, 1)
        input = attention * input
        input = torch.sum(input, dim=1)
        return input.view(batch_size, -1)


class Conv42AdaptorNet(nn.Module):
    def __init__(self):
        super(Conv42AdaptorNet, self).__init__()
        self.attn = Conv42AttentionNet()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256))

    def forward(self, input):
        input = self.attn(input)
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
