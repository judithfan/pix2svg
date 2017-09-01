from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np

import torch.nn as nn


class ResNet152Embeddings(nn.Module):
    """Splits ResNet152 into layers so that we can get
    feature embeddings from each section. For ResNet152,
    we define 7 layers:
        0 - early cnn; after maxpool
        1 - layer 1 (contains 3 bottleneck layers)
        2 - layer 2 (contains 7 bottleneck layers)
        3 - layer 3 (contains 36 bottleneck layers)
        4 - layer 4 (cotains 3 bottleneck layers)
        5 - average pool
        6 - fully connected
    :param resnet152: pretrained resnet152 instance
    :param layer_index: number from -1 to 6 (where -1 is a list of all)
    """
    def __init__(self, resnet152, layer_index=-1):
        super(ResNet152Embeddings, self).__init__()
        assert layer >= -1 and layer < 7
        self.maxpool = nn.Sequential(*(list(resnet152.children())[slice(0, 4)]))
        self.layer1 = nn.Sequential(*(list(resnet152.children())[4]))
        self.layer2 = nn.Sequential(*(list(resnet152.children())[5]))
        self.layer3 = nn.Sequential(*(list(resnet152.children())[6]))
        self.layer4 = nn.Sequential(*(list(resnet152.children())[7]))
        self.avgpool = nn.Sequential(list(resnet152.children())[8])
        self.linear = nn.Sequential(list(resnet152.children())[9])
        self.layer_index = layer_index

    def _flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x_maxpool = self.maxpool(x)
        if self.layer_index == 0:
            return [self._flatten(x_maxpool)]
        x_layer1 = self.layer1(x_maxpool)
        if self.layer_index == 1:
            return [self._flatten(x_layer1)]
        x_layer2 = self.layer2(x_layer1)
        if self.layer_index == 2:
            return [self._flatten(x_layer2)]
        x_layer3 = self.layer3(x_layer2)
        if self.layer_index == 3:
            return [self._flatten(x_layer3)]
        x_layer4 = self.layer4(x_layer3)
        if self.layer_index == 4:
            return [self._flatten(x_layer4)]
        x_avgpool = self.avgpool(x_layer4)
        if self.layer_index == 5:
            return [self._flatten(x_avgpool)]
        x_linear = self.linear(x_avgpool)
        if self.layer_index == 6:
            return [self._flatten(x_linear)]
        return [self._flatten(x_maxpool), self._flatten(x_layer1), self._flatten(x_layer2),
                self._flatten(x_layer3), self._flatten(x_layer4), self._flatten(x_avgpool),
                self._flatten(x_linear)]


class AlexNetEmbeddings(nn.Module):
    """Splits AlexNet into layers to get embeddings for
    each group of Convolutional/Pooling/Nonlinear layers.
    For AlexNet, we define 7 layers:
        0 - after maxpool (2)
        1 - after maxpool (5)
        2 - after relu (7)
        3 - after relu (9)
        4 - after maxpool (12)
        5 - after relu (2)
        6 - after relu (5)
        7 - after linear (6)
    :param alexnet: pretrained alexnet instance
    :param layer_index: number from -1 to 7 (where -1 is a list of all)
    """
    def __init__(self, alexnet, layer_index=-1):
        super(AlexNetEmbeddings, self).__init__()
        assert layer >= -1 and layer < 8
        self.group1 = nn.Sequential(*(list(alexnet.features.children())[slice(0, 3)]))
        self.group2 = nn.Sequential(*(list(alexnet.features.children())[slice(3, 6)]))
        self.group3 = nn.Sequential(*(list(alexnet.features.children())[slice(6, 8)]))
        self.group4 = nn.Sequential(*(list(alexnet.features.children())[slice(8, 10)]))
        self.group5 = nn.Sequential(*(list(alexnet.features.children())[slice(10, 13)]))
        self.group6 = nn.Sequential(*(list(alexnet.classifier.children())[slice(0, 4)]))
        self.group7 = nn.Sequential(*(list(alexnet.classifier.children())[slice(4, 6)]))
        self.group8 = nn.Sequential(*(list(alexnet.classifier.children())[6]))
        self.layer_index = layer_index

    def _flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x_group1 = self.group1(x)
        if self.layer_index == 0:
            return [self._flatten(x_group1)]
        x_group2 = self.group2(x_group1)
        if self.layer_index == 1:
            return [self._flatten(x_group2)]
        x_group3 = self.group3(x_group2)
        if self.layer_index == 2:
            return [self._flatten(x_group3)]
        x_group4 = self.group4(x_group3)
        if self.layer_index == 3:
            return [self._flatten(x_group4)]
        x_group5 = self.group5(x_group4)
        if self.layer_index == 4:
            return [self._flatten(x_group5)]
        x_group6 = self.group6(x_group5)
        if self.layer_index == 5:
            return [self._flatten(x_group6)]
        x_group7 = self.group7(x_group6)
        if self.layer_index == 6:
            return [self._flatten(x_group7)]
        x_group8 = self.group7(x_group7)
        if self.layer_index == 7:
            return [self._flatten(x_group8)]
        return [self._flatten(x_group1), self._flatten(x_group2), self._flatten(x_group3),
                self._flatten(x_group4), self._flatten(x_group5), self._flatten(x_group6),
                self._flatten(x_group7), self._flatten(x_group8)]


class VGG19Embeddings(nn.Module):
    """Splits vgg19 into separate sections so that we can get
    feature embeddings from each section.

    :param vgg19: traditional vgg19 model
    """
    def __init__(self, vgg19, layer_index=-1):
        super(VGG19Embeddings, self).__init__()
        self.conv1 = nn.Sequential(*(list(vgg19.features.children())[slice(0, 5)]))
        self.conv2 = nn.Sequential(*(list(vgg19.features.children())[slice(5, 10)]))
        self.conv3 = nn.Sequential(*(list(vgg19.features.children())[slice(10, 19)]))
        self.conv4 = nn.Sequential(*(list(vgg19.features.children())[slice(19, 28)]))
        self.conv5 = nn.Sequential(*(list(vgg19.features.children())[slice(28, 37)]))
        self.linear1 = nn.Sequential(*(list(vgg19.classifier.children())[slice(0, 2)]))
        self.linear2 = nn.Sequential(*(list(vgg19.classifier.children())[slice(3, 5)]))
        self.linear3 = nn.Sequential(list(vgg19.classifier.children())[-1])
        assert layer_index >= -1 and layer_index < 8
        self.layer_index = layer_index

    def _flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        # build in this ugly way so we don't have to evaluate things we don't need to.
        x_conv1 = self.conv1(x)
        if self.layer_index == 0:
            return [self._flatten(x_conv1)]
        x_conv2 = self.conv2(x_conv1)
        if self.layer_index == 1:
            return [self._flatten(x_conv2)]
        x_conv3 = self.conv3(x_conv2)
        if self.layer_index == 2:
            return [self._flatten(x_conv3)]
        x_conv4 = self.conv4(x_conv3)
        if self.layer_index == 3:
            return [self._flatten(x_conv4)]
        x_conv5 = self.conv5(x_conv4)
        x_conv5_flat = self._flatten(x_conv5)
        if self.layer_index == 4:
            return [x_conv5_flat]
        x_linear1 = self.linear1(x_conv5_flat)
        if self.layer_index == 5:
            return [x_linear1]
        x_linear2 = self.linear2(x_linear1)
        if self.layer_index == 6:
            return [x_linear2]
        x_linear3 = self.linear3(x_linear2)
        if self.layer_index == 7:
            return [x_linear3]
        return [self._flatten(x_conv1), self._flatten(x_conv2),
                self._flatten(x_conv3), self._flatten(x_conv4),
                self._flatten(x_conv5), x_linear1, x_linear2, x_linear3]


def vgg_convert_to_avg_pool(vgg):
    """Replace MaxPool2d with AvgPool2d"""
    vgg_copy = copy.deepcopy(vgg)
    max_pool_indices = [4, 9, 18, 27, 36]
    for ii in max_pool_indices:
        setattr(vgg_copy.features, str(ii),
                nn.AvgPool2d((2, 2), stride=(2, 2)))
    return vgg_copy
