from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class RenderNet(nn.Module):
    """Renders an image as a CxHxW matrix given 2 points
    such that it is differentiable. This is based on a
    simplified Bresenham's line algorithm: calculate slope and
    round to stay close to the slope.

    :param x0: fixed starting x coordinate
    :param y0: fixed starting y coordinate
    :param x1: initialization for ending x coordinate
    :param y1: initialization for ending y coordinate
    :param imsize: image size to generate
    """
    def __init__(self, x0, y0, x1, y1, imsize=256, linewidth=1):
        super(RenderNet, self).__init__()
        self.x0 = Variable(torch.FloatTensor([x0]), requires_grad=False)
        self.y0 = Variable(torch.FloatTensor([y0]), requires_grad=False)
        self.x1 = Parameter(torch.FloatTensor([x1]))
        self.y1 = Parameter(torch.FloatTensor([y1]))
        self.imsize = imsize
        self.linewidth = linewidth

    def forward(self):
        dx = torch.abs(self.x1 - self.x0)
        dy = torch.abs(self.y1 - self.y0)
        m = dy / dx

        # evenly space out the elements (linspace)
        xpix = Variable(torch.arange(0, dx.data[0] + 1)) + x0
        ypix = Variable(torch.arange(0, dy.data[0] + 1)) + y0

        # sort xpix in increasing order
        xpix, sorted_indices = torch.sort(xpix)
        ypix = ypix[sorted_indices]

        # render the points as a CHW Tensor
        xmin = int(torch.min(xpix).data[0])
        xmax = int(torch.max(xpix).data[0])

        templates = []  # hold lots of vectors
        for y in ypix.split(1):
            y = y.long()
            r = Variable(torch.zeros(self.imsize))
            r[y] = 255  # one hot the relevant index
            templates.append(r)

        templates = torch.t(torch.stack(templates))
        templates = torch.stack((templates, templates, templates), dim=0)

        # prepend and postpend unmanipulated sections
        if xmin > 0:
            prefix = Variable(torch.zeros((3, self.imsize, xmin)))
            templates = torch.stack((prefix, templates))
        if xmax < self.imsize:
            postfix = Variable(torch.zeros((3, self.imsize,
                                           self.imsize - xmax - 1)))
            templates = torch.cat((templates, postfix), dim=2)

        return templates
