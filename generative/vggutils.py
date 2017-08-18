from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np

import torch.nn as nn
import torchvision.models as models


class VGG19Split(nn.Module):
    """Splits vgg19 into separate sections so that we can get
    feature embeddings from each section.

    :param vgg19: traditional vgg19 model
    """
    def __init__(self, vgg19):
        super(VGG19Split, self).__init__()
        self.conv1 = nn.Sequential(*(list(vgg19.features.children())[slice(0, 5)]))
        self.conv2 = nn.Sequential(*(list(vgg19.features.children())[slice(5, 10)]))
        self.conv3 = nn.Sequential(*(list(vgg19.features.children())[slice(10, 19)]))
        self.conv4 = nn.Sequential(*(list(vgg19.features.children())[slice(19, 28)]))
        self.conv5 = nn.Sequential(*(list(vgg19.features.children())[slice(28, 37)]))
        self.linear1 = nn.Sequential(*(list(vgg19.classifier.children())[slice(0, 2)]))
        self.linear2 = nn.Sequential(*(list(vgg19.classifier.children())[slice(3, 5)]))
        self.linear3 = nn.Sequential(list(vgg19.classifier.children())[-1])

    def _flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv5_flat = self._flatten(x_conv5)
        x_linear1 = self.linear1(x_conv5_flat)
        x_linear2 = self.linear2(x_linear1)
        x_linear3 = self.linear3(x_linear2)
        return (self._flatten(x_conv1), self._flatten(x_conv2),
                self._flatten(x_conv3), self._flatten(x_conv4),
                self._flatten(x_conv5), x_linear1, x_linear2, x_linear3)


def build_vgg19_feature_extractor(vgg, chop_index=42):
    """Take features from the activations of the hidden layer
    immediately before the VGG's object classifier (4096 size).
    The last linear layer and last dropout layer are removed,
    preserving the ReLU. The activations are L2 normalized.

    :param vgg: trained vgg19 model
    :param chop_index: vgg19 has 44 total layers (37 of them are in the features container)
                       7 of them are in the classifier container. By default, we use the
                       last rectified linear layer before projecting into 1000 classes.
    :return: 2 PyTorch Sequential models
             - 1 to generate features
             - 1 to run the rest of the network
    """
    vgg_copy = copy.deepcopy(vgg)
    vgg_residual = copy.deepcopy(vgg)
    if chop_index > 37:  # we can keep the features container as is
        vgg_copy.classifier = nn.Sequential(*list(vgg.classifier.children())[:chop_index-37])
        # the residual vgg doesn't need the features then
        vgg_residual.features = nn.Sequential()
        vgg_residual.classifier = nn.Sequential(*list(vgg.classifier.children())[chop_index-37:])
    else:
        vgg_copy.features = nn.Sequential(*list(vgg.features.children())[:chop_index])
        vgg_copy.classifier = nn.Sequential()
        vgg_residual.features = nn.Sequential(*list(vgg.features.children())[chop_index:])

    return vgg_copy, vgg_residual


def vgg_convert_to_avg_pool(vgg):
    """Replace MaxPool2d with AvgPool2d"""
    vgg_copy = copy.deepcopy(vgg)
    max_pool_indices = [4, 9, 18, 27, 36]
    for ii in max_pool_indices:
        setattr(vgg_copy.features, str(ii),
                nn.AvgPool2d((2, 2), stride=(2, 2)))
    return vgg_copy
