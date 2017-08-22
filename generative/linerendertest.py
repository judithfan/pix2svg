from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

from pix2sketch import (sketch_loss, sample_endpoint_gaussian2d)
from linerender import RenderNet
from vggutils import (VGG19Split, vgg_convert_to_avg_pool)


if __name__ == "__main__":
    vgg19 = models.vgg19(pretrained=True)
    vgg19 = VGG19Split(vgg19, -1)
    vgg19.eval()

    for p in vgg19.parameters():
        p.requires_grad = False

    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    img = Image.open('./data/car_natural_1.jpg')
    natural = Variable(preprocessing(img).unsqueeze(0))

    distractors = []
    for i in os.listdir('./data/distractors/'):
        distractor_path = os.path.join('./data/distractors/', i)
        distractor = Image.open(distractor_path)
        distractors.append(distractor)
    distractors = Variable(torch.cat([preprocessing(image).unsqueeze(0)
                                      for image in distractors]))

    natural_emb = vgg19(natural)
    distractor_embs = vgg19(distractors)

    coord_samples = sample_endpoint_gaussian2d(112, 112, std=20, size=1,
                                               min_x=0, max_x=224, min_y=0, max_y=224)

    x_samples, y_samples = coord_samples[:, 0], coord_samples[:, 1]
    renderer = RenderNet(112, 112, x_samples[0], y_samples[0], imsize=224)
    optimizer = optim.SGD(renderer.parameters(), lr=1, momentum=0.9)

    def train(epoch):
        renderer.train()
        params = list(renderer.parameters())
        optimizer.zero_grad()
        sketch = renderer()
        sketch_embs = vgg19(sketch)
        loss = sketch_loss(natural_emb, sketch_embs, distractor_embs)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} \tLoss: {:.6f} \tParams: ({}, {})'.format(
            epoch, loss.data[0], params[0].data[0], params[1].data[0]))

    for i in range(5):
        train(i)
