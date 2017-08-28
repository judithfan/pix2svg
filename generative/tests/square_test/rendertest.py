from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
import sys; sys.path.append('../..')
from linerender import RenderNet


def gen_ground_truth():
    image = torch.ones(1, 11, 11)
    image[:, 2:9, 2] = 0
    image[:, 2:9, 8] = 0
    image[:, 2, 2:9] = 0
    image[:, 8, 2:9] = 0
    return Variable(torch.unsqueeze(image, dim=0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gt_sketch = gen_ground_truth()

    def train(renderer, optimizer, epoch, residual_sketch):
        renderer.train()
        optimizer.zero_grad()
        sketch = renderer()
        sketch = torch.add(sketch, residual_sketch)
        loss = torch.sum(torch.pow(gt_sketch - sketch, 2))  # l2 loss
        loss.backward()
        optimizer.step()
        params = list(renderer.parameters())
        if epoch % 100 == 0:
            print('Train Epoch: {} \tLoss: {:.6f} \tParams: ({}, {})'.format(
                  epoch, loss.data[0], params[0].data.numpy()[0], params[1].data.numpy()[0]))

    sketch = Variable(torch.zeros((1, 1, 11, 11)))

    renderer = RenderNet(2, 2, 5, 5, imsize=11, fuzz=0.1)
    optimizer = optim.SGD(renderer.parameters(), lr=1e-3)
    for j in xrange(500):
        # break tape on residual_sketch on purpose
        train(renderer, optimizer, j, residual_sketch=Variable(sketch.data))
    _sketch = renderer()
    sketch += _sketch

    # x_gt = [2, 2, 8, 8]
    # y_gt = [2, 8, 8, 2]
    # x0, y0 = x_gt[0], y_gt[0]
    # x1, y1 = x_gt[0], y_gt[0]

    # for i in xrange(1, 4):
    #     x2 = np.random.normal(loc=x_gt[i], scale=1)
    #     y2 = np.random.normal(loc=y_gt[i], scale=1)

    #     print('Param Initialization: ({}, {})'.format(x2, y2))
    #     renderer = RenderNet(x1, y1, x2, y2, imsize=11, fuzz=0.1)
    #     optimizer = optim.SGD(renderer.parameters(), lr=1e-3)

    #     x1, y1 = x_gt[i], y_gt[i]

    #     print('Training for point {}'.format(i))
    #     for j in xrange(500):
    #         # break tape on residual_sketch on purpose
    #         train(renderer, optimizer, j, residual_sketch=Variable(sketch.data))

    #     _sketch = renderer()
    #     sketch += _sketch
    #     print('')

    # connect the 3rd point to the first
    # renderer = RenderNet(x1, y1, x0, y0, imsize=11)
    # _sketch = renderer()
    # sketch += _sketch

    # normalize sketch
    min_sketch = torch.min(sketch)
    max_sketch = torch.max(sketch)
    sketch = (sketch - min_sketch) / (max_sketch - min_sketch)
    plt.matshow(sketch[0][0].data.numpy())
    plt.savefig('./sketch.png')
