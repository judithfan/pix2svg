"""Pixel-wise loss against a sketch of a vertical line and a
generated sketch. The generated sketch is given the ground
truth anchor point (x0, y0) and prior point (x1, y1) close to
the true 2nd point. The test confirms that via gradient descent,
we can converge to the correct point.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

import sys; sys.path.append('..')
from pix2sketch import mean_pixel_sketch_loss
from linerender import RenderNet


def preprocess_tensor(t):
    """Imagenet normalization on tensor"""
    t[0] = (t[0] - 0.485) / 0.229
    t[1] = (t[1] - 0.456) / 0.224
    t[2] = (t[2] - 0.406) / 0.225
    return t


def gen_ground_truth():
    image = torch.zeros(3, 224, 224)
    image[:, 112:, 112] = 1
    image = preprocess_tensor(image)
    return Variable(torch.unsqueeze(image, dim=0))


if __name__ == "__main__":
    gt_sketch = gen_ground_truth()
    x0, y0 = 112, 112  # use gt for seed anchor
    x1, y1 = 152, 180

    renderer = RenderNet(x0, y0, x1, y1, imsize=224)
    optimizer = optim.SGD(renderer.parameters(), lr=1e10, momentum=0.9)

    def train(epoch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()

        sketch[:, 0] = (sketch[:, 0] - 0.485) / 0.229
        sketch[:, 1] = (sketch[:, 1] - 0.456) / 0.224
        sketch[:, 2] = (sketch[:, 2] - 0.406) / 0.225

        loss = mean_pixel_sketch_loss(gt_sketch, sketch)
        loss.backward()
        optimizer.step()

        params = list(renderer.parameters())
        print('Train Epoch: {} \tLoss: {:.6f}: \tGrad: ({}, {}) \tParams: ({}, {})'.format(
            epoch, loss.data[0], params[0].grad.data.numpy()[0],
            params[1].grad.data.numpy()[0],
            params[0].data.numpy()[0], params[1].data.numpy()[0]))

    for i in range(20):
        train(i)
