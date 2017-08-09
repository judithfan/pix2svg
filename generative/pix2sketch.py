from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable


def chop_vgg19(vgg):
    """Take features from the activations of the hidden layer
    immediately before the VGG's object classifier (4096 size).
    The last linear layer and last dropout layer are removed,
    preserving the ReLU. The activations are L2 normalized.

    :param vgg: trained vgg19 model
    :return: PyTorch Sequential model
    """
    vgg_copy = copy.deepcopy(vgg)
    classifier = nn.Sequential(*list(vgg.classifier.children())[:-2])
    vgg_copy.classifier = classifier
    return vgg_copy


if __name__ == '__main__':
    import os
    import json
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--imagepath', type=str, help='path to image file')
    args = parser.parse_args()

    with open('class_index.json') as fp:
        class_idx = json.load(fp)
        idx2label = [class_idx[str(k)][1]
                     for k in range(len(class_idx))]

    # pretrained on imagenet
    vgg19 = models.vgg19(pretrained=True)

    # needed for imagenet
    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(args['imagepath'])
    data = Variable(preprocessing(img).unsqueeze(0))

    # cut off part of the net to generate features
    vgg_ext = chop_vgg19(vgg19)

    def extract_features(data):
        features = vgg_ext(data).data[0].numpy()
        return features / np.linalg.norm(features)

    def extract_scores(data):
        return vgg19(data).data[0].numpy()
