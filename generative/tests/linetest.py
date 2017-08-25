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


def gen_ground_truth():
    image = torch.zeros(1, 11, 11)
    image[:, 5:, 5] = 1
    return Variable(torch.unsqueeze(image, dim=0))


if __name__ == "__main__":
    gt_sketch = gen_ground_truth()

    def train(renderer, optimizer, epoch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()
        loss = torch.sum(torch.pow(gt_sketch - sketch, 2))  # l2 loss
        loss.backward()
        optimizer.step()
        params = list(renderer.parameters())
        if epoch % 10 == 0:
            print('Train Epoch: {} \tLoss: {:.6f} \tParams: ({}, {})'.format(
                  epoch, loss.data[0], params[0].data.numpy()[0], params[1].data.numpy()[0]))

    # TEST 1: provide a nearby guess (i found we already need a big fuzz...)
    renderer = RenderNet(5, 5, 7, 9, imsize=11, fuzz=3.0)
    optimizer = optim.SGD(renderer.parameters(), lr=1e-2, momentum=0.5)

    for i in range(500):
        train(renderer, optimizer, i)

    params = list(renderer.named_parameters())
    print('\nTEST 1:')
    print(params)

    # TEST 2: provide the ground truth and make sure it doesn't deviate
    renderer = RenderNet(5, 5, 5, 10, imsize=11, fuzz=3.0)
    optimizer = optim.SGD(renderer.parameters(), lr=1e-2, momentum=0.5)

    for i in range(100):
        train(renderer, optimizer, i)

    params = list(renderer.named_parameters())
    print('\nTEST 2:')
    print(params)
